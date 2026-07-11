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
    /* Pre-quantize x to int8 (shared by AVX2 and scalar paths) */
    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;
    int8_t xq[XQ_MAX] __attribute__((aligned(32)));
    float x_scales[XQ_MAX / 32];

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

#if defined(__AVX2__) && defined(__F16C__)
    int row_stride = n_super * 144;

    __m128i mask_0f = _mm_set1_epi8(0x0F);
    __m128i ones128 = _mm_set1_epi16(1);
    float inv_63 = 1.0f / 63.0f;

    /* 4-row parallel: balance between xq sharing and register pressure.
     * 8-row caused 137 stack spills. 4-row uses 4 YMM acc + 4 float am = much less. */
    int j = 0;
    for (; j + 4 <= out_dim; j += 4) {
        const uint8_t *rows[4];
        for (int r = 0; r < 4; r++) rows[r] = q4k_W + (size_t)(j+r) * row_stride;

        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        float am0 = 0, am1 = 0, am2 = 0, am3 = 0;

        for (int s = 0; s < n_super; s++) {
            /* Load d, dmin + unpack scales for all 4 rows */
            float sb_sc[4][8], sb_mn[4][8];
            for (int r = 0; r < 4; r++) {
                const uint8_t *sb = rows[r] + s*144;
                uint16_t d_u16 = *(const uint16_t*)(sb);
                uint16_t dmin_u16 = *(const uint16_t*)(sb+2);
                __m128i d_raw = _mm_set1_epi16((short)d_u16);
                __m128i dmin_raw = _mm_set1_epi16((short)dmin_u16);
                float d = _mm_cvtss_f32(_mm_cvtph_ps(d_raw));
                float dmin = _mm_cvtss_f32(_mm_cvtph_ps(dmin_raw));
                uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);
                for (int k = 0; k < 8; k++) {
                    sb_sc[r][k] = d * (float)sm[k] * inv_63;
                    sb_mn[r][k] = dmin * (float)sm[8+k] * inv_63;
                }
            }

            for (int sub = 0; sub < 8; sub++) {
                int xoff = (s*8+sub)*32;
                float x_scale = x_scales[s*8+sub];

                /* Load xq ONCE — shared by all 4 rows */
                __m128i xv_lo = _mm_loadu_si128((__m128i*)(xq + xoff));
                __m128i xv_hi = _mm_loadu_si128((__m128i*)(xq + xoff + 16));

                /* xq_sum ONCE */
                __m128i xq_s16_la = _mm_cvtepi8_epi16(xv_lo);
                __m128i xq_s16_lb = _mm_cvtepi8_epi16(_mm_srli_si128(xv_lo, 8));
                __m128i xs_lo = _mm_add_epi32(_mm_madd_epi16(xq_s16_la, ones128), _mm_madd_epi16(xq_s16_lb, ones128));
                __m128i xq_s16_ha = _mm_cvtepi8_epi16(xv_hi);
                __m128i xq_s16_hb = _mm_cvtepi8_epi16(_mm_srli_si128(xv_hi, 8));
                __m128i xs_hi = _mm_add_epi32(_mm_madd_epi16(xq_s16_ha, ones128), _mm_madd_epi16(xq_s16_hb, ones128));
                __m128i xs128 = _mm_add_epi32(xs_lo, xs_hi);
                xs128 = _mm_hadd_epi32(xs128, xs128);
                xs128 = _mm_hadd_epi32(xs128, xs128);
                int32_t xq_sum = _mm_cvtsi128_si32(xs128);
                float xq_sum_f = (float)xq_sum * x_scale;

                /* Process each of 4 rows */
                #define PROC_ROW_K4(r, acc_v, am_v) do { \
                    __m128i q4bits = _mm_loadu_si128((__m128i*)(rows[r] + s*144 + 16 + sub*16)); \
                    __m128i q4l = _mm_and_si128(q4bits, mask_0f); \
                    __m128i q4h = _mm_and_si128(_mm_srli_epi16(q4bits, 4), mask_0f); \
                    __m128i p16l = _mm_maddubs_epi16(q4l, xv_lo); \
                    __m128i p16h = _mm_maddubs_epi16(q4h, xv_hi); \
                    __m128i s32l = _mm_madd_epi16(p16l, ones128); \
                    __m128i s32h = _mm_madd_epi16(p16h, ones128); \
                    float combined = sb_sc[r][sub] * x_scale; \
                    __m256 dv = _mm256_set1_ps(combined); \
                    __m256 s32a = _mm256_cvtepi32_ps(_mm256_set_m128i(s32h, s32l)); \
                    acc_v = _mm256_fmadd_ps(dv, s32a, acc_v); \
                    am_v -= sb_mn[r][sub] * xq_sum_f; \
                } while(0)

                PROC_ROW_K4(0, acc0, am0); PROC_ROW_K4(1, acc1, am1);
                PROC_ROW_K4(2, acc2, am2); PROC_ROW_K4(3, acc3, am3);
                #undef PROC_ROW_K4
            }
        }

        #define HSUM_ROW4(r, acc_v, am_v) do { \
            __m128 lo = _mm256_castps256_ps128(acc_v); \
            __m128 hi = _mm256_extractf128_ps(acc_v, 1); \
            __m128 s = _mm_add_ps(lo, hi); \
            s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s); \
            y[j+r] = _mm_cvtss_f32(s) + am_v + (b ? b[j+r] : 0); \
        } while(0)
        HSUM_ROW4(0, acc0, am0); HSUM_ROW4(1, acc1, am1);
        HSUM_ROW4(2, acc2, am2); HSUM_ROW4(3, acc3, am3);
        #undef HSUM_ROW4
    }

    /* Tail: single-row for remaining outputs */
    for (; j < out_dim; j++) {
        const uint8_t *row = q4k_W + (size_t)j * row_stride;
        __m256 acc = _mm256_setzero_ps();
        float acc_min = 0.0f;

        for (int s = 0; s < n_super; s++) {
            const uint8_t *sb = row + s * 144;
            uint16_t d_u16 = *(const uint16_t*)(sb);
            uint16_t dmin_u16 = *(const uint16_t*)(sb + 2);
            __m128i d_raw = _mm_set1_epi16((short)d_u16);
            __m128i dmin_raw = _mm_set1_epi16((short)dmin_u16);
            float d = _mm_cvtss_f32(_mm_cvtph_ps(d_raw));
            float dmin = _mm_cvtss_f32(_mm_cvtph_ps(dmin_raw));

            uint8_t sm[16]; unpack_scales_6bit(sb + 4, sm);
            float sb_sc[8], sb_mn[8];
            for (int k = 0; k < 8; k++) {
                sb_sc[k] = d * (float)sm[k] * inv_63;
                sb_mn[k] = dmin * (float)sm[8+k] * inv_63;
            }

            const uint8_t *qs = sb + 16;
            for (int sub = 0; sub < 8; sub++) {
                int qoff = sub * 16;
                int xoff = (s*8+sub)*32;
                float x_scale = x_scales[s*8+sub];
                float combined = sb_sc[sub] * x_scale;
                __m256 d_v = _mm256_set1_ps(combined);

                __m128i q4bits = _mm_loadu_si128((__m128i*)(qs + qoff));
                __m128i q4l = _mm_and_si128(q4bits, mask_0f);
                __m128i q4h = _mm_and_si128(_mm_srli_epi16(q4bits, 4), mask_0f);
                __m128i xv_lo = _mm_loadu_si128((__m128i*)(xq + xoff));
                __m128i xv_hi = _mm_loadu_si128((__m128i*)(xq + xoff + 16));
                __m128i p16l = _mm_maddubs_epi16(q4l, xv_lo);
                __m128i p16h = _mm_maddubs_epi16(q4h, xv_hi);
                __m128i s32l = _mm_madd_epi16(p16l, ones128);
                __m128i s32h = _mm_madd_epi16(p16h, ones128);
                __m256 s32a = _mm256_cvtepi32_ps(_mm256_set_m128i(s32h, s32l));
                acc = _mm256_fmadd_ps(d_v, s32a, acc);

                __m128i xq_s16_la = _mm_cvtepi8_epi16(xv_lo);
                __m128i xq_s16_lb = _mm_cvtepi8_epi16(_mm_srli_si128(xv_lo, 8));
                __m128i xs_lo = _mm_add_epi32(_mm_madd_epi16(xq_s16_la, ones128), _mm_madd_epi16(xq_s16_lb, ones128));
                __m128i xq_s16_ha = _mm_cvtepi8_epi16(xv_hi);
                __m128i xq_s16_hb = _mm_cvtepi8_epi16(_mm_srli_si128(xv_hi, 8));
                __m128i xs_hi = _mm_add_epi32(_mm_madd_epi16(xq_s16_ha, ones128), _mm_madd_epi16(xq_s16_hb, ones128));
                __m128i xs128 = _mm_add_epi32(xs_lo, xs_hi);
                xs128 = _mm_hadd_epi32(xs128, xs128);
                xs128 = _mm_hadd_epi32(xs128, xs128);
                int32_t xq_sum = _mm_cvtsi128_si32(xs128);
                acc_min -= sb_mn[sub] * x_scale * (float)xq_sum;
            }
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        y[j] = _mm_cvtss_f32(s) + acc_min + (b ? b[j] : 0);
    }
#else
    /* Scalar fallback */
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
                float sb_min = dmin * scales_mins[8 + sub] / 63.0f;
                int xoff = (s * 8 + sub) * 32;
                int qoff = sub * 16;
                float xs = x_scales[s * 8 + sub];
                int32_t dot = 0, xq_sum = 0;
                for (int i = 0; i < 32; i++) {
                    uint8_t q4 = (qs[qoff + i/2] >> ((i%2)*4)) & 0xF;
                    dot += q4 * (int)xq[xoff + i];
                    xq_sum += (int)xq[xoff + i];
                }
                acc += sb_scale * xs * (float)dot;
                acc_min -= sb_min * xs * (float)xq_sum;
            }
        }
        y[j] = acc + acc_min + (b ? b[j] : 0);
    }
#endif
}

#endif /* LAL_Q4K_KERNEL_H */
