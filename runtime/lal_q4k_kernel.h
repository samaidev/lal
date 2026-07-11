/* lal_q4k_kernel.h — Q4_K matmul kernel (llama.cpp Q4_K_M compatible format)
 *
 * Q4_K block: 144 bytes per 256 elements
 *   [2B d fp16][2B dmin fp16][12B scales+mins][128B packed q4]
 *
 * The 12-byte scales encode 8 sub-block scales (6-bit) + 8 sub-block mins (6-bit).
 * Total: 16 × 6-bit = 96 bits = 12 bytes.
 *
 * Decompression:
 *   For sub-block j (32 elements):
 *     scale_j = d * scale_6bit[j] / 63
 *     min_j = -dmin * min_6bit[j] / 63
 *     w[i] = scale_j * q4[i] + min_j  (q4 in 0..15)
 *
 * This gives near-Q8 accuracy because each 32-element sub-block has its own
 * scale and min, adapting to local weight distributions.
 */
#ifndef LAL_Q4K_KERNEL_H
#define LAL_Q4K_KERNEL_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX before including lal_q4k_kernel.h"
#endif

#if defined(__AVX2__) && defined(__F16C__)
/* Unpack 16 × 6-bit values from 12 bytes into 16 uint8 values */
static inline void unpack_scales_6bit(const uint8_t *src, uint8_t *out16) {
    /* 16 × 6-bit = 96 bits packed into 12 bytes */
    uint64_t lo = 0, hi = 0;
    /* Read 12 bytes into 96-bit value (split into 64+32) */
    lo = *(const uint64_t*)src;
    hi = *(const uint32_t*)(src + 8);
    /* Extract 16 × 6-bit values */
    uint64_t all = lo; /* lower 64 bits have 10 values (60 bits) + 4 bits of 11th */
    for (int i = 0; i < 10; i++) {
        out16[i] = (all >> (i * 6)) & 0x3F;
    }
    /* bits 60-63 from lo + bits 0-7 from hi = 12 bits for value 10 */
    out16[10] = ((lo >> 60) | (hi << 4)) & 0x3F;
    /* remaining 5 values from hi (bits 2-31) */
    for (int i = 0; i < 5; i++) {
        out16[11 + i] = (hi >> (2 + i * 6)) & 0x3F;
    }
}
#endif

/* Q4_K matmul: y[out_dim] = q4k_W[out_dim, in_dim] @ x[in_dim] + b[out_dim]
 * q4k_W: 144 bytes per 256 elements (superblock)
 * in_dim must be multiple of 256.
 *
 * Uses cvtepi32_ps technique (no per-block hsum).
 */
static inline void lal_matmul_q4_k(float *y,
                                     const uint8_t *q4k_W,
                                     const float *x,
                                     const float *b,
                                     int in_dim, int out_dim) {
#if defined(__AVX2__) && defined(__F16C__)
    /* Quantize x to Q8 per 32-element block (matching Q4_K sub-blocks) */
    int n_super = in_dim / 256;   /* superblocks per row */
    int row_stride = n_super * 144;

    /* Pre-quantize x to int8, one scale per 32-element sub-block */
    int8_t xq[XQ_MAX] __attribute__((aligned(32)));
    float x_scales[XQ_MAX / 32];  /* per 32-elem scale */
    int n_sub = in_dim / 32;
    for (int sb = 0; sb < n_sub; sb++) {
        const float *xb = x + sb * 32;
        float x_max = 0;
        for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > x_max) x_max = a; }
        float scale = x_max / 127.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        x_scales[sb] = scale;
        float inv = 1.0f / scale;
        for (int i = 0; i < 32; i++) {
            int v = (int)lroundf(xb[i] * inv);
            xq[sb * 32 + i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    }

    __m256i ones = _mm256_set1_epi16(1);
    __m256i m4 = _mm256_set1_epi8(0x0F);

    /* Process 1 row at a time (streaming, 1 memory stream) */
    for (int j = 0; j < out_dim; j++) {
        const uint8_t *row = q4k_W + (size_t)j * row_stride;
        __m256 acc = _mm256_setzero_ps();
        float acc_min = 0.0f;

        for (int s = 0; s < n_super; s++) {
            const uint8_t *sb = row + s * 144;
            /* Load d and dmin (fp16) */
            uint16_t d_u16 = *(const uint16_t*)(sb);
            uint16_t dmin_u16 = *(const uint16_t*)(sb + 2);
            __m128i d_raw = _mm_set1_epi16((short)d_u16);
            __m128i dmin_raw = _mm_set1_epi16((short)dmin_u16);
            float d = _mm_cvtss_f32(_mm_cvtph_ps(d_raw));
            float dmin = _mm_cvtss_f32(_mm_cvtph_ps(dmin_raw));

            /* Unpack 8 scales + 8 mins from 12 bytes */
            uint8_t scales_mins[16];
            unpack_scales_6bit(sb + 4, scales_mins);
            /* scales_mins[0..7] = sub-block scales, [8..15] = sub-block mins */

            /* For each of 8 sub-blocks (32 elements each): */
            const uint8_t *qs = sb + 16;  /* 128 bytes of packed q4 */
            for (int sub = 0; sub < 8; sub++) {
                int qoff = sub * 16;  /* 16 bytes = 32 q4 values */
                int xoff = (s * 8 + sub) * 32;
                float x_scale = x_scales[s * 8 + sub];

                /* Sub-block scale and min */
                float sb_scale = d * scales_mins[sub] / 63.0f;
                float sb_min = -dmin * scales_mins[8 + sub] / 63.0f;

                /* Combined scale for dot product */
                float combined = sb_scale * x_scale;
                __m256 d_v = _mm256_set1_ps(combined);

                /* Load 16 bytes of q4 → 32 values (low nibbles + high nibbles) */
                __m128i q4bits = _mm_loadu_si128((__m128i*)(qs + qoff));
                __m256i q4l = _mm256_cvtepu8_epi16(_mm_and_si128(q4bits, _mm_set1_epi8(0x0F)));
                __m256i q4h = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(q4bits, 4), _mm_set1_epi8(0x0F)));

                /* Load 32 int8 x values (split into 2 × 16 → 2 × 16 int16) */
                __m256i xv_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(xq + xoff)));
                __m256i xv_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(xq + xoff + 16)));

                /* Dot products: q4 (uint8) × xq (int8) → int16 → int32 → float
                 * q4 values are 0..15 (unsigned), xq is -127..127 (signed)
                 * maddubs: uint8 × int8 → int16 (correct for unsigned×signed) */
                __m256i p16_lo = _mm256_maddubs_epi16(q4l, xv_lo);
                __m256i p16_hi = _mm256_maddubs_epi16(q4h, xv_hi);
                __m256i sum32_lo = _mm256_madd_epi16(p16_lo, ones);
                __m256i sum32_hi = _mm256_madd_epi16(p16_hi, ones);

                /* Convert to float and accumulate (NO hsum per block!) */
                __m256 f_lo = _mm256_cvtepi32_ps(sum32_lo);
                __m256 f_hi = _mm256_cvtepi32_ps(sum32_hi);
                acc = _mm256_fmadd_ps(d_v, f_lo, acc);
                acc = _mm256_fmadd_ps(d_v, f_hi, acc);

                /* Min correction: acc_min += sb_min * sum(xq) * x_scale
                 * sum(xq) = hsum of the 32 int8 values */
                /* Compute sum of xq for this sub-block */
                int32_t xq_sum = 0;
                for (int i = 0; i < 32; i++) xq_sum += xq[xoff + i];
                acc_min += sb_min * x_scale * (float)xq_sum;
            }
        }

        /* Final hsum of acc (8 lanes → 1) */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        y[j] = _mm_cvtss_f32(s) + acc_min + (b ? b[j] : 0);
    }
#else
    /* Scalar fallback */
    int n_super = in_dim / 256;
    int row_stride = n_super * 144;
    for (int j = 0; j < out_dim; j++) {
        const uint8_t *row = q4k_W + (size_t)j * row_stride;
        float acc = 0.0f, acc_min = 0.0f;
        for (int s = 0; s < n_super; s++) {
            const uint8_t *sb = row + s * 144;
            uint16_t d_u16 = *(const uint16_t*)(sb);
            uint16_t dmin_u16 = *(const uint16_t*)(sb + 2);
            /* fp16→fp32 manual */
            float d, dmin;
            #define FP16_TO_F32(u16, out) do { \
                uint32_t sign=(u16>>15)&1, exp=(u16>>10)&0x1F, frac=u16&0x3FF; \
                if (exp==0) { out = frac ? (frac/1024.0f/524288.0f)*(sign?-1:1) : 0; } \
                else if (exp==31) { out = frac?(float)NAN:(sign?-(float)INFINITY:(float)INFINITY); } \
                else { float f=(1.0f+frac/1024.0f)*ldexpf(1.0f,(int)exp-15); out=sign?-f:f; } \
            } while(0)
            FP16_TO_F32(d_u16, d);
            FP16_TO_F32(dmin_u16, dmin);
            uint8_t scales_mins[16];
            unpack_scales_6bit(sb + 4, scales_mins);
            const uint8_t *qs = sb + 16;
            for (int sub = 0; sub < 8; sub++) {
                float sb_scale = d * scales_mins[sub] / 63.0f;
                float sb_min = -dmin * scales_mins[8 + sub] / 63.0f;
                int xoff = (s * 8 + sub) * 32;
                int qoff = sub * 16;
                int32_t dot = 0, xq_sum = 0;
                for (int i = 0; i < 32; i++) {
                    uint8_t q4 = (qs[qoff + i/2] >> ((i%2)*4)) & 0xF;
                    dot += q4 * (int)xq[xoff + i];
                    xq_sum += (int)xq[xoff + i];
                }
                acc += sb_scale * (float)dot;
                acc_min += sb_min * (float)xq_sum;
            }
        }
        y[j] = acc + acc_min + (b ? b[j] : 0);
    }
#endif
}

#endif /* LAL_Q4K_KERNEL_H */
