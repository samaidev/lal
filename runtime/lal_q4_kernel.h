#ifndef LAL_Q4_KERNEL_H
#define LAL_Q4_KERNEL_H

/*
 * lal_q4_kernel.h - Q4_0 matmul kernel (mistral.rs / llama.cpp style).
 *
 * Q4_0 block format:
 *   - Block size: 32 elements
 *   - Per block: 1 fp16 scale (2 bytes) + 16 bytes (32 x 4-bit packed, 2/byte)
 *   - Total: 18 bytes per 32 elements
 *   - 4-bit values are unsigned (0..15), offset by 8: signed_value = q4 - 8
 *
 * Matmul: y[out_dim] = q4_W[out_dim, in_dim] @ x[in_dim] + b[out_dim]
 *
 * Key optimization (from llama.cpp ggml_vec_dot_q4_0):
 *   - Use int32 accumulators (not fp32) — cheaper, no broadcast per block
 *   - Sum scale * dot at the END per row, not per block
 *   - 8-row parallel: 8 accumulators in registers, x loaded once per block
 *
 * Requires AVX2 + F16C.
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX (max in_dim) before including lal_q4_kernel.h"
#endif

/* Quantize x to int8 with per-tensor scale (same as Q8 sign-trick). */
static inline float lal_q4_quantize_x(const float *x, int8_t *xq, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)lroundf(x[i] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        xq[i] = (int8_t)v;
    }
    return scale;
}

/* Q4_0 matmul: y[out_dim] = q4_W[out_dim, in_dim] @ x[in_dim] + b[out_dim]
 *
 * q4_W is packed as blocks: for row r, block b covers elements [b*32..b*32+31].
 * Each block = 2 bytes fp16 scale + 16 bytes packed q4 = 18 bytes total.
 * Total size = out_dim * (in_dim/32) * 18 bytes.
 *
 * in_dim must be a multiple of 32.
 *
 * 8-row parallel: 8 int32 accumulators in YMM registers, x loaded once per block.
 * Scale applied at the END (not per block) using a separate fp32 sum.
 */
static inline void lal_matmul_q4_0(float *y,
                                    const uint8_t *q4_W,
                                    const float *x,
                                    const float *b,
                                    int in_dim, int out_dim) {
#if defined(__AVX2__) && defined(__F16C__)
    /* Quantize x to int8 once */
    int8_t xq[XQ_MAX] __attribute__((aligned(32)));
    float x_scale = lal_q4_quantize_x(x, xq, in_dim);

    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 18;  /* bytes per row */

    __m256i ones = _mm256_set1_epi16(1);
    __m256i eight_v = _mm256_set1_epi8(8);
    __m128i mask_0f = _mm_set1_epi8(0x0F);

    int j = 0;
    /* 8-row parallel */
    for (; j + 8 <= out_dim; j += 8) {
        /* Per-row accumulators: int32 partial dots + fp32 scale*dot sum */
        __m256i iacc0 = _mm256_setzero_si256();
        __m256i iacc1 = _mm256_setzero_si256();
        __m256i iacc2 = _mm256_setzero_si256();
        __m256i iacc3 = _mm256_setzero_si256();
        __m256i iacc4 = _mm256_setzero_si256();
        __m256i iacc5 = _mm256_setzero_si256();
        __m256i iacc6 = _mm256_setzero_si256();
        __m256i iacc7 = _mm256_setzero_si256();
        /* fp32 sum of (scale * dot) per block, 8 lanes each */
        __m256 facc0 = _mm256_setzero_ps();
        __m256 facc1 = _mm256_setzero_ps();
        __m256 facc2 = _mm256_setzero_ps();
        __m256 facc3 = _mm256_setzero_ps();
        __m256 facc4 = _mm256_setzero_ps();
        __m256 facc5 = _mm256_setzero_ps();
        __m256 facc6 = _mm256_setzero_ps();
        __m256 facc7 = _mm256_setzero_ps();

        const uint8_t *row0 = q4_W + (size_t)(j+0) * row_stride;
        const uint8_t *row1 = q4_W + (size_t)(j+1) * row_stride;
        const uint8_t *row2 = q4_W + (size_t)(j+2) * row_stride;
        const uint8_t *row3 = q4_W + (size_t)(j+3) * row_stride;
        const uint8_t *row4 = q4_W + (size_t)(j+4) * row_stride;
        const uint8_t *row5 = q4_W + (size_t)(j+5) * row_stride;
        const uint8_t *row6 = q4_W + (size_t)(j+6) * row_stride;
        const uint8_t *row7 = q4_W + (size_t)(j+7) * row_stride;

        for (int blk = 0; blk < blocks_per_row; blk++) {
            int offset = blk * 18;
            int x_off = blk * 32;

            /* Load x int8 for this block (32 bytes = 1 YMM) */
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + x_off));
            __m256i ax = _mm256_abs_epi8(xv);  /* |x|, uint8 range */

            /* Process each row: load scale (fp16->fp32), load q4, expand,
             * sign-trick dot, accumulate int32; then add scale*dot to fp32 acc. */
            #define PROCESS_ROW(row_ptr, iacc, facc) do { \
                /* Load fp16 scale (2 bytes) -> fp32 scalar */ \
                uint16_t scale_u16 = *(const uint16_t*)(row_ptr + offset); \
                __m128i scale_raw = _mm_set1_epi16((short)scale_u16); \
                __m128 scale_f32 = _mm_cvtph_ps(scale_raw); \
                float scale = _mm_cvtss_f32(scale_f32); \
                /* Load 16 bytes of q4 packed */ \
                __m128i packed = _mm_loadu_si128((__m128i*)(row_ptr + offset + 2)); \
                /* Expand to 32 int8 (signed -8..7) */ \
                __m128i lo = _mm_and_si128(packed, mask_0f); \
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f); \
                __m128i inter0 = _mm_unpacklo_epi8(lo, hi); \
                __m128i inter1 = _mm_unpackhi_epi8(lo, hi); \
                __m256i wv = _mm256_inserti128_si256(_mm256_castsi128_si256(inter0), inter1, 1); \
                wv = _mm256_sub_epi8(wv, eight_v); \
                /* Sign-trick: sw = sign(x) * w, then maddubs(ax, sw) */ \
                __m256i sw = _mm256_sign_epi8(wv, xv); \
                __m256i prod = _mm256_maddubs_epi16(ax, sw); \
                __m256i sum32 = _mm256_madd_epi16(prod, ones); \
                /* Horizontal sum to scalar int32 */ \
                __m128i lo128 = _mm256_castsi256_si128(sum32); \
                __m128i hi128 = _mm256_extracti128_si256(sum32, 1); \
                __m128i s = _mm_add_epi32(lo128, hi128); \
                s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s); \
                int32_t dot = _mm_cvtsi128_si32(s); \
                /* Accumulate scale * dot as fp32 (broadcast + FMA) */ \
                facc = _mm256_fmadd_ps(_mm256_set1_ps(scale), \
                                       _mm256_set1_ps((float)dot), facc); \
            } while(0)

            PROCESS_ROW(row0, iacc0, facc0);
            PROCESS_ROW(row1, iacc1, facc1);
            PROCESS_ROW(row2, iacc2, facc2);
            PROCESS_ROW(row3, iacc3, facc3);
            PROCESS_ROW(row4, iacc4, facc4);
            PROCESS_ROW(row5, iacc5, facc5);
            PROCESS_ROW(row6, iacc6, facc6);
            PROCESS_ROW(row7, iacc7, facc7);
            #undef PROCESS_ROW
        }

        /* Horizontal sum each fp32 acc (8 lanes -> 1) and write output */
        #define HSUM_PS(v) ({ \
            __m128 lo = _mm256_castps256_ps128(v); \
            __m128 hi = _mm256_extractf128_ps(v, 1); \
            __m128 s = _mm_add_ps(lo, hi); \
            s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s); \
            _mm_cvtss_f32(s); \
        })
        y[j+0] = HSUM_PS(facc0) * x_scale + (b ? b[j+0] : 0);
        y[j+1] = HSUM_PS(facc1) * x_scale + (b ? b[j+1] : 0);
        y[j+2] = HSUM_PS(facc2) * x_scale + (b ? b[j+2] : 0);
        y[j+3] = HSUM_PS(facc3) * x_scale + (b ? b[j+3] : 0);
        y[j+4] = HSUM_PS(facc4) * x_scale + (b ? b[j+4] : 0);
        y[j+5] = HSUM_PS(facc5) * x_scale + (b ? b[j+5] : 0);
        y[j+6] = HSUM_PS(facc6) * x_scale + (b ? b[j+6] : 0);
        y[j+7] = HSUM_PS(facc7) * x_scale + (b ? b[j+7] : 0);
        #undef HSUM_PS
    }

    /* Tail: remaining rows (out_dim not multiple of 8) */
    for (; j < out_dim; j++) {
        const uint8_t *row = q4_W + (size_t)j * row_stride;
        float acc = 0.0f;
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int offset = blk * 18;
            int x_off = blk * 32;
            /* Load fp16 scale */
            uint16_t scale_u16 = *(const uint16_t*)(row + offset);
            __m128i scale_raw = _mm_set1_epi16((short)scale_u16);
            float scale = _mm_cvtss_f32(_mm_cvtph_ps(scale_raw));
            /* Load 16 bytes q4, expand to 32 int8 */
            __m128i packed = _mm_loadu_si128((__m128i*)(row + offset + 2));
            __m128i lo = _mm_and_si128(packed, mask_0f);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f);
            __m128i inter0 = _mm_unpacklo_epi8(lo, hi);
            __m128i inter1 = _mm_unpackhi_epi8(lo, hi);
            __m256i wv = _mm256_inserti128_si256(_mm256_castsi128_si256(inter0), inter1, 1);
            wv = _mm256_sub_epi8(wv, eight_v);
            /* Dot with x */
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + x_off));
            __m256i sw = _mm256_sign_epi8(wv, xv);
            __m256i ax = _mm256_abs_epi8(xv);
            __m256i prod = _mm256_maddubs_epi16(ax, sw);
            __m256i sum32 = _mm256_madd_epi16(prod, ones);
            __m128i lo128 = _mm256_castsi256_si128(sum32);
            __m128i hi128 = _mm256_extracti128_si256(sum32, 1);
            __m128i s = _mm_add_epi32(lo128, hi128);
            s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
            int32_t dot = _mm_cvtsi128_si32(s);
            acc += scale * (float)dot;
        }
        y[j] = acc * x_scale + (b ? b[j] : 0);
    }
#else
    /* Scalar fallback */
    int8_t xq[XQ_MAX];
    float x_scale = lal_q4_quantize_x(x, xq, in_dim);
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 18;

    for (int j = 0; j < out_dim; j++) {
        const uint8_t *row = q4_W + (size_t)j * row_stride;
        float acc = 0.0f;
        for (int blk = 0; blk < blocks_per_row; blk++) {
            const uint8_t *block = row + blk * 18;
            uint16_t scale_raw = *(uint16_t*)block;
            uint32_t sign = (scale_raw >> 15) & 1;
            uint32_t exp = (scale_raw >> 10) & 0x1F;
            uint32_t frac = scale_raw & 0x3FF;
            float scale;
            if (exp == 0) {
                if (frac == 0) scale = sign ? -0.0f : 0.0f;
                else {
                    float f = frac / 1024.0f / 524288.0f;
                    scale = sign ? -f : f;
                }
            } else if (exp == 31) {
                scale = frac ? NAN : (sign ? -INFINITY : INFINITY);
            } else {
                float f = (1.0f + frac / 1024.0f) * ldexpf(1.0f, (int)exp - 15);
                scale = sign ? -f : f;
            }
            int32_t dot = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t byte = block[2 + i];
                int q0 = (byte & 0x0F) - 8;
                int q1 = (byte >> 4) - 8;
                dot += q0 * (int)xq[blk*32 + 2*i];
                dot += q1 * (int)xq[blk*32 + 2*i + 1];
            }
            acc += scale * (float)dot;
        }
        y[j] = acc * x_scale + (b ? b[j] : 0);
    }
#endif
}

#endif /* LAL_Q4_KERNEL_H */
