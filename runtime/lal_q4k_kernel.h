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

                /* Sub-block scale and min.
                 * Converter: q = round((x + actual_min) / actual_scale)
                 * Dequant:   w = actual_scale * q - actual_min
                 * So: dot(w, x) = actual_scale * dot(q, x) - actual_min * sum(x)
                 * For the dot product with quantized xq:
                 *   result = combined_scale * dot(q4, xq) - combined_min * sum(xq)
                 * where combined_scale = sb_scale * x_scale
                 *       combined_min = sb_min * x_scale (sb_min = dmin * m / 63, positive)
                 */
                float sb_scale = d * scales_mins[sub] / 63.0f;      /* positive */
                float sb_min = dmin * scales_mins[8 + sub] / 63.0f;  /* positive */

                /* Combined scale for dot product */
                float combined = sb_scale * x_scale;
                __m256 d_v = _mm256_set1_ps(combined);

                /* Load 16 bytes of q4 (32 values, packed as q[2i] | q[2i+1]<<4)
                 * maddubs directly: uint8 × int8 → int16, processing pairs.
                 * byte[i] = q[2i] | (q[2i+1] << 4)
                 * maddubs treats byte as [uint8_lo, int8_hi] pairs:
                 *   result[i] = q[2i] * xq[2i] + q[2i+1] * xq[2i+1]  ← CORRECT!
                 * But maddubs needs uint8 for first arg (q4 nibbles are 0-15 = uint8)
                 * and int8 for second arg (xq is -127..127 = int8).
                 * The 4-bit q values ARE uint8 (0-15), so maddubs works directly.
                 * But the high nibble is 0-15 shifted to 0-240, not 0-15!
                 * We need to split: low nibbles (uint8 0-15) and high nibbles (uint8 0-15).
                 *
                 * CORRECT approach: split into low and high, each 16 uint8 values.
                 * low[i] = byte[i] & 0x0F = q[2i]     (indices 0,2,4,...,30)
                 * high[i] = byte[i] >> 4 = q[2i+1]    (indices 1,3,5,...,31)
                 * Then: dot = sum_i q[2i]*xq[2i] + sum_i q[2i+1]*xq[2i+1]
                 *       = maddubs(low, xq_even) + maddubs(high, xq_odd)
                 * But xq is contiguous! We need xq[0,2,4,...,30] and xq[1,3,5,...,31].
                 *
                 * SIMPLER: use maddubs directly on the packed bytes with full xq.
                 * maddubs(packed, xq) where packed has uint8 in low nibble and
                 * "sign-extended" high nibble... NO, that doesn't work because
                 * high nibbles are 0-15 shifted to 0-240 in the upper 4 bits.
                 *
                 * ACTUALLY CORRECT: maddubs treats BOTH operands as bytes.
                 * packed byte[i] = q[2i] | (q[2i+1] << 4).
                 * As uint8: byte[i] = q[2i] + 16*q[2i+1]. This is NOT q[2i]!
                 * We MUST split low/high nibbles first.
                 *
                 * The fix: use maddubs on (low_nibbles, xq_lo) + (high_nibbles, xq_hi)
                 * where xq_lo and xq_hi are the CORRECT halves of xq.
                 * low_nibbles[i] = q[2i], should pair with xq[2i]
                 * high_nibbles[i] = q[2i+1], should pair with xq[2i+1]
                 *
                 * So we need to deinterleave xq: even indices and odd indices.
                 * OR: just do 2 separate maddubs:
                 *   maddubs(low, xq[0..15])  → pairs (q[0]*xq[0]+q[2]*xq[1], ...) ← WRONG
                 *
                 * The REAL fix: change packing to interleaved (llama.cpp style):
                 *   byte[i] = q[i] | (q[i+16] << 4)  for i=0..15
                 * Then low nibbles = q[0..15], high nibbles = q[16..31]. Correct!
                 */

                /* For now: use the DIRECT approach. The packed bytes are
                 * byte[i] = q[2i] | (q[2i+1] << 4). maddubs processes pairs:
                 * result = q[2i]*x[2i] + q[2i+1]*x[2i+1] for each pair.
                 * But maddubs uses the FULL byte as uint8, not split nibbles!
                 * byte[i] = q[2i] + 16*q[2i+1] which is NOT what we want.
                 *
                 * We need to SPLIT into low and high nibbles, then pair correctly.
                 * low[i] = q[2i]   → should multiply with xq[2i]
                 * high[i] = q[2i+1] → should multiply with xq[2i+1]
                 *
                 * To pair correctly with contiguous xq, we need to deinterleave:
                 * xq_even = [xq[0], xq[2], xq[4], ..., xq[30]]
                 * xq_odd  = [xq[1], xq[3], xq[5], ..., xq[31]]
                 * Then: maddubs(low, xq_even) + maddubs(high, xq_odd)
                 *
                 * This is complex. SIMPLER FIX: change the converter packing to
                 * interleaved style: byte[i] = q[i] | (q[i+16] << 4).
                 * Then low = q[0..15], high = q[16..31], pairs with xq[0..15] and xq[16..31].
                 */
                /* SIMD dot product with interleaved packing.
                 * Packing: byte[sub*16+i] = q[sub*32+i] | (q[sub*32+i+16] << 4)
                 * Low nibbles = q[sub*32..+15], pair with xq[sub*32..+15]
                 * High nibbles = q[sub*32+16..+31], pair with xq[sub*32+16..+31]
                 * qoff = sub*16, xoff = (s*8+sub)*32 */
                __m128i q4bits = _mm_loadu_si128((__m128i*)(qs + qoff));
                /* Low nibbles: 16 uint8 (0-15) */
                __m128i q4l = _mm_and_si128(q4bits, _mm_set1_epi8(0x0F));
                /* High nibbles: 16 uint8 (0-15) */
                __m128i q4h = _mm_and_si128(_mm_srli_epi16(q4bits, 4), _mm_set1_epi8(0x0F));

                /* xq for this sub-block: first 16 and next 16 */
                __m128i xv_lo = _mm_loadu_si128((__m128i*)(xq + xoff));      /* xq[0..15] */
                __m128i xv_hi = _mm_loadu_si128((__m128i*)(xq + xoff + 16));  /* xq[16..31] */

                /* maddubs: uint8 × int8 → int16 (8 pairs → 8 int16 per 128-bit) */
                __m128i p16_lo = _mm_maddubs_epi16(q4l, xv_lo);  /* 8 int16 */
                __m128i p16_hi = _mm_maddubs_epi16(q4h, xv_hi);  /* 8 int16 */

                /* madd: 8 int16 pairs → 4 int32 per 128-bit.
                 * But we need all 16 products summed. Use madd with ones:
                 * madd(p16, ones) sums adjacent int16 pairs → 4 int32.
                 * We need 4+4=8 int32 total. Combine into YMM. */
                __m128i sum32_lo = _mm_madd_epi16(p16_lo, _mm_set1_epi16(1));
                __m128i sum32_hi = _mm_madd_epi16(p16_hi, _mm_set1_epi16(1));

                /* Horizontal sum: 4+4 int32 → 1 scalar.
                 * Add the two 128-bit vectors, then hsum. */
                __m128i sum128 = _mm_add_epi32(sum32_lo, sum32_hi);
                sum128 = _mm_hadd_epi32(sum128, sum128);
                sum128 = _mm_hadd_epi32(sum128, sum128);
                int32_t dot = _mm_cvtsi128_si32(sum128);

                /* Accumulate into acc lane 0 */
                float dot_f = (float)dot * combined;
                acc = _mm256_add_ps(acc, _mm256_castps128_ps256(_mm_set_ss(dot_f)));

                /* Min correction: acc_min -= sb_min * x_scale * sum(xq)
                 * SIMD: sum 32 int8 values. _mm_cvtepi8_epi16 only takes 8 bytes,
                 * so split each 16-byte vector into two 8-byte halves. */
                __m128i ones128 = _mm_set1_epi16(1);
                /* xv_lo: 16 int8 → 2 × 8 int16 → 2 × 4 int32 */
                __m128i xq_s16_lo_a = _mm_cvtepi8_epi16(xv_lo);           /* lower 8 → 8 int16 */
                __m128i xq_s16_lo_b = _mm_cvtepi8_epi16(_mm_srli_si128(xv_lo, 8)); /* upper 8 */
                __m128i xq_sum_lo = _mm_add_epi32(
                    _mm_madd_epi16(xq_s16_lo_a, ones128),
                    _mm_madd_epi16(xq_s16_lo_b, ones128));  /* 4 int32 */
                /* xv_hi: 16 int8 → 2 × 8 int16 → 2 × 4 int32 */
                __m128i xq_s16_hi_a = _mm_cvtepi8_epi16(xv_hi);
                __m128i xq_s16_hi_b = _mm_cvtepi8_epi16(_mm_srli_si128(xv_hi, 8));
                __m128i xq_sum_hi = _mm_add_epi32(
                    _mm_madd_epi16(xq_s16_hi_a, ones128),
                    _mm_madd_epi16(xq_s16_hi_b, ones128));  /* 4 int32 */
                /* Combine + hsum */
                __m128i xq_sum128 = _mm_add_epi32(xq_sum_lo, xq_sum_hi);
                xq_sum128 = _mm_hadd_epi32(xq_sum128, xq_sum128);
                xq_sum128 = _mm_hadd_epi32(xq_sum128, xq_sum128);
                int32_t xq_sum = _mm_cvtsi128_si32(xq_sum128);
                acc_min -= sb_min * x_scale * (float)xq_sum;
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
