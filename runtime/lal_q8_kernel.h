#ifndef LAL_Q8_KERNEL_H
#define LAL_Q8_KERNEL_H

/*
 * lal_q8_kernel.h - Q8 matmul kernels.
 *
 * Two formats supported:
 *   qtype=1: per-row scale Q8 (legacy, lal_matmul_q8_signtrick)
 *   qtype=3: Q8_0 block format (new, lal_matmul_q8_0) — matches llama.cpp
 *
 * Q8_0 block format (cache-friendly, matches llama.cpp ggml):
 *   - Block size: 32 elements
 *   - Per block: 1 fp16 scale (2 bytes) + 32 bytes (32 x int8 values)
 *   - Total: 34 bytes per 32 elements = 1.0625 bytes/elem
 *   - Scale is INLINE with data → no separate scale array access
 *   - Sequential block access → hardware prefetcher works perfectly
 *
 * mistral.rs sign-trick (used in both kernels):
 *   ax = sign_epi8(x, x) = |x|             (uint8 range)
 *   sw = sign_epi8(w, x) = sign(x) * w     (int8 range)
 *   maddubs(ax, sw) = sum_i x[i] * w[i]    (correct signed dot, no zero-point)
 *
 * Key optimizations vs old kernel:
 *   1. Q8_0 block layout: scale inline, sequential access (prefetcher-friendly)
 *   2. 4-row parallel (not 8): less register pressure, better OOO execution
 *   3. Explicit _mm_prefetch for next block
 *   4. Accumulate scale*dot in fp32 per block (no int32 hsum per block)
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX (max in_dim) before including lal_q8_kernel.h"
#endif

/* === Legacy: per-row scale Q8 matmul (sign-trick, AVX2) ===
 * Kept for backward compatibility with existing GPQ8 files (qtype=1).
 *
 * y[out_dim] = q8_T[out_dim, in_dim] @ x[in_dim] + b[out_dim]
 * q8_T is TRANSPOSED (row-major [out_dim, in_dim]).
 * scale[out_dim] is per-row scale = max|row|/127.
 */
static inline void lal_matmul_q8_signtrick(float *y,
                                           const int8_t *q8_T,
                                           const float *scale,
                                           const float *x,
                                           const float *b,
                                           int in_dim, int out_dim) {
#if defined(__AVX2__)
    /* Quantize x to int8 directly (no +128 offset) */
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    int8_t xq[XQ_MAX] __attribute__((aligned(32)));
    float inv = 1.0f / x_scale;
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * inv);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }

    __m256i ones = _mm256_set1_epi16(1);
    int j = 0;
    /* 8-output parallel: 8 accumulators in registers, x loaded once per chunk */
    for (; j + 8 <= out_dim; j += 8) {
        const int8_t *w0=q8_T+(size_t)(j+0)*in_dim, *w1=q8_T+(size_t)(j+1)*in_dim;
        const int8_t *w2=q8_T+(size_t)(j+2)*in_dim, *w3=q8_T+(size_t)(j+3)*in_dim;
        const int8_t *w4=q8_T+(size_t)(j+4)*in_dim, *w5=q8_T+(size_t)(j+5)*in_dim;
        const int8_t *w6=q8_T+(size_t)(j+6)*in_dim, *w7=q8_T+(size_t)(j+7)*in_dim;
        __m256i a0=_mm256_setzero_si256(),a1=_mm256_setzero_si256();
        __m256i a2=_mm256_setzero_si256(),a3=_mm256_setzero_si256();
        __m256i a4=_mm256_setzero_si256(),a5=_mm256_setzero_si256();
        __m256i a6=_mm256_setzero_si256(),a7=_mm256_setzero_si256();
        for (int i = 0; i < in_dim; i += 32) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
            __m256i ax = _mm256_sign_epi8(xv, xv);
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(w0+i)), xv)), ones));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(w1+i)), xv)), ones));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(w2+i)), xv)), ones));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(w3+i)), xv)), ones));
            a4 = _mm256_add_epi32(a4, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(w4+i)), xv)), ones));
            a5 = _mm256_add_epi32(a5, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(w5+i)), xv)), ones));
            a6 = _mm256_add_epi32(a6, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(w6+i)), xv)), ones));
            a7 = _mm256_add_epi32(a7, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(w7+i)), xv)), ones));
        }
        #define LAL_HSUM32(v) ({ __m128i _lo=_mm256_castsi256_si128(v), _hi=_mm256_extracti128_si256(v,1); __m128i _s=_mm_add_epi32(_lo,_hi); _s=_mm_hadd_epi32(_s,_s); _s=_mm_hadd_epi32(_s,_s); _mm_cvtsi128_si32(_s); })
        y[j+0]=(float)(LAL_HSUM32(a0))*x_scale*scale[j+0]+(b?b[j+0]:0);
        y[j+1]=(float)(LAL_HSUM32(a1))*x_scale*scale[j+1]+(b?b[j+1]:0);
        y[j+2]=(float)(LAL_HSUM32(a2))*x_scale*scale[j+2]+(b?b[j+2]:0);
        y[j+3]=(float)(LAL_HSUM32(a3))*x_scale*scale[j+3]+(b?b[j+3]:0);
        y[j+4]=(float)(LAL_HSUM32(a4))*x_scale*scale[j+4]+(b?b[j+4]:0);
        y[j+5]=(float)(LAL_HSUM32(a5))*x_scale*scale[j+5]+(b?b[j+5]:0);
        y[j+6]=(float)(LAL_HSUM32(a6))*x_scale*scale[j+6]+(b?b[j+6]:0);
        y[j+7]=(float)(LAL_HSUM32(a7))*x_scale*scale[j+7]+(b?b[j+7]:0);
        #undef LAL_HSUM32
    }
    /* Tail: remaining outputs (out_dim not multiple of 8) */
    for (; j < out_dim; j++) {
        const int8_t *w = q8_T + (size_t)j * in_dim;
        __m256i acc32 = _mm256_setzero_si256();
        for (int i = 0; i < in_dim; i += 32) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
            __m256i ax = _mm256_sign_epi8(xv, xv);
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(w+i)), xv)), ones));
        }
        #define LAL_HSUM32_TAIL(v) ({ __m128i _lo=_mm256_castsi256_si128(v), _hi=_mm256_extracti128_si256(v,1); __m128i _s=_mm_add_epi32(_lo,_hi); _s=_mm_hadd_epi32(_s,_s); _s=_mm_hadd_epi32(_s,_s); _mm_cvtsi128_si32(_s); })
        y[j] = (float)(LAL_HSUM32_TAIL(acc32)) * x_scale * scale[j] + (b ? b[j] : 0);
        #undef LAL_HSUM32_TAIL
    }
#else
    /* Scalar fallback (no AVX2) */
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    int8_t xq[XQ_MAX];
    float inv = 1.0f / x_scale;
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * inv);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }
    for (int j = 0; j < out_dim; j++) {
        const int8_t *w = q8_T + (size_t)j * in_dim;
        int32_t dot = 0;
        for (int i = 0; i < in_dim; i++) dot += (int)xq[i] * (int)w[i];
        y[j] = (float)dot * x_scale * scale[j] + (b ? b[j] : 0);
    }
#endif
}

#if defined(__AVX2__) && defined(__F16C__)
/* Helper: compute int8 dot of 32 elements, return 8-lane fp32 (no hsum).
 * Uses sign-trick + maddubs + madd + cvtepi32_ps (llama.cpp technique).
 * This is the key optimization: no horizontal sum per block! */
static inline __m256 lal_dot32_f32(const int8_t *w, const int8_t *xq, __m256i ones) {
    __m256i wv = _mm256_loadu_si256((__m256i*)w);
    __m256i xv = _mm256_loadu_si256((__m256i*)xq);
    __m256i ax = _mm256_sign_epi8(xv, xv);
    __m256i sw = _mm256_sign_epi8(wv, xv);
    __m256i dot16 = _mm256_maddubs_epi16(ax, sw);
    __m256i dot32 = _mm256_madd_epi16(dot16, ones);
    return _mm256_cvtepi32_ps(dot32);
}
#endif

/* === NEW: Q8_0 block-format matmul (llama.cpp technique) ===
 *
 * y[out_dim] = q8_0_W[out_dim, in_dim] @ x[in_dim] + b[out_dim]
 *
 * q8_0_W is packed as blocks: for row r, block b covers elements [b*32..b*32+31].
 * Each block = 2 bytes fp16 scale + 32 bytes int8 = 34 bytes total.
 *
 * KEY TECHNIQUE (from llama.cpp ggml_vec_dot_q8_0_q8_0):
 *   - Use lal_dot32_f32: int8 dot → 8-lane fp32 (NO hsum per block!)
 *   - _mm256_cvtepi32_ps converts int32 partial sums to fp32 without hsum
 *   - fmadd accumulates scale*partial in fp32 across blocks
 *   - Only ONE hsum at the very end per row
 *
 * This is 10-20x faster than per-block hsum approach.
 *
 * 8-row parallel: x is quantized to Q8_0 blocks too (shared across rows),
 * each row has its own fp32 accumulator YMM.
 */
static inline void lal_matmul_q8_0(float *y,
                                    const uint8_t *q8_0_W,
                                    const float *x,
                                    const float *b,
                                    int in_dim, int out_dim) {
#if defined(__AVX2__) && defined(__F16C__)
    /* Quantize x to Q8_0 blocks (per-block scale, matching weight format) */
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 34;

    /* xq8_0: packed Q8_0 blocks for x (scale + 32 int8 per block) */
    uint8_t xq8_0[XQ_MAX / 32 * 34] __attribute__((aligned(32)));
    for (int blk = 0; blk < blocks_per_row; blk++) {
        const float *xb = x + blk * 32;
        float x_max = 0;
        for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > x_max) x_max = a; }
        float scale = x_max / 127.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        float inv = 1.0f / scale;
        /* Store fp16 scale */
        __m128 scale_f32 = _mm_set1_ps(scale);
        __m128i scale_fp16 = _mm_cvtps_ph(scale_f32, 0);
        uint16_t s16 = _mm_extract_epi16(scale_fp16, 0);
        xq8_0[blk * 34 + 0] = s16 & 0xFF;
        xq8_0[blk * 34 + 1] = (s16 >> 8) & 0xFF;
        /* Quantize 32 values to int8 */
        int8_t *xb_q = (int8_t*)(xq8_0 + blk * 34 + 2);
        for (int i = 0; i < 32; i++) {
            int v = (int)lroundf(xb[i] * inv);
            xb_q[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    }

    __m256i ones = _mm256_set1_epi16(1);
    int j = 0;

    /* 8-row parallel: 8 fp32 YMM accumulators */
    for (; j + 8 <= out_dim; j += 8) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        const uint8_t *row0 = q8_0_W + (size_t)(j+0) * row_stride;
        const uint8_t *row1 = q8_0_W + (size_t)(j+1) * row_stride;
        const uint8_t *row2 = q8_0_W + (size_t)(j+2) * row_stride;
        const uint8_t *row3 = q8_0_W + (size_t)(j+3) * row_stride;
        const uint8_t *row4 = q8_0_W + (size_t)(j+4) * row_stride;
        const uint8_t *row5 = q8_0_W + (size_t)(j+5) * row_stride;
        const uint8_t *row6 = q8_0_W + (size_t)(j+6) * row_stride;
        const uint8_t *row7 = q8_0_W + (size_t)(j+7) * row_stride;

        for (int blk = 0; blk < blocks_per_row; blk++) {
            int w_off = blk * 34;
            int x_off = blk * 34;

            /* Load x scale (fp16) once, shared across 8 rows */
            uint16_t x_s16 = *(const uint16_t*)(xq8_0 + x_off);
            __m128i x_sraw = _mm_set1_epi16((short)x_s16);
            __m128 x_sf = _mm_cvtph_ps(x_sraw);
            float x_scale_f = _mm_cvtss_f32(x_sf);

            /* Prefetch next block for all 8 rows */
            if (blk + 2 < blocks_per_row) {
                _mm_prefetch((const char*)(row0 + w_off + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row1 + w_off + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row2 + w_off + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row3 + w_off + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row4 + w_off + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row5 + w_off + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row6 + w_off + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row7 + w_off + 2*34), _MM_HINT_T0);
            }

            /* For each row: load w scale, compute combined scale, dot, fmadd */
            #define PROC_ROW(row_ptr, acc) do { \
                uint16_t w_s16 = *(const uint16_t*)(row_ptr + w_off); \
                __m128i w_sraw = _mm_set1_epi16((short)w_s16); \
                __m128 w_sf = _mm_cvtph_ps(w_sraw); \
                float combined = _mm_cvtss_f32(w_sf) * x_scale_f; \
                __m256 d = _mm256_set1_ps(combined); \
                __m256 q = lal_dot32_f32((const int8_t*)(row_ptr + w_off + 2), \
                                         (const int8_t*)(xq8_0 + x_off + 2), ones); \
                acc = _mm256_fmadd_ps(d, q, acc); \
            } while(0)

            PROC_ROW(row0, acc0);
            PROC_ROW(row1, acc1);
            PROC_ROW(row2, acc2);
            PROC_ROW(row3, acc3);
            PROC_ROW(row4, acc4);
            PROC_ROW(row5, acc5);
            PROC_ROW(row6, acc6);
            PROC_ROW(row7, acc7);
            #undef PROC_ROW
        }

        /* Horizontal sum each 8-lane fp32 acc to scalar, write output */
        #define HSUM_F32_8(v) ({ \
            __m128 lo = _mm256_castps256_ps128(v); \
            __m128 hi = _mm256_extractf128_ps(v, 1); \
            __m128 s = _mm_add_ps(lo, hi); \
            s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s); \
            _mm_cvtss_f32(s); \
        })
        y[j+0] = HSUM_F32_8(acc0) + (b ? b[j+0] : 0);
        y[j+1] = HSUM_F32_8(acc1) + (b ? b[j+1] : 0);
        y[j+2] = HSUM_F32_8(acc2) + (b ? b[j+2] : 0);
        y[j+3] = HSUM_F32_8(acc3) + (b ? b[j+3] : 0);
        y[j+4] = HSUM_F32_8(acc4) + (b ? b[j+4] : 0);
        y[j+5] = HSUM_F32_8(acc5) + (b ? b[j+5] : 0);
        y[j+6] = HSUM_F32_8(acc6) + (b ? b[j+6] : 0);
        y[j+7] = HSUM_F32_8(acc7) + (b ? b[j+7] : 0);
        #undef HSUM_F32_8
    }

    /* Tail: remaining rows */
    for (; j < out_dim; j++) {
        const uint8_t *row = q8_0_W + (size_t)j * row_stride;
        __m256 acc = _mm256_setzero_ps();
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int w_off = blk * 34;
            int x_off = blk * 34;
            uint16_t w_s16 = *(const uint16_t*)(row + w_off);
            uint16_t x_s16 = *(const uint16_t*)(xq8_0 + x_off);
            __m128i w_sraw = _mm_set1_epi16((short)w_s16);
            __m128i x_sraw = _mm_set1_epi16((short)x_s16);
            float combined = _mm_cvtss_f32(_mm_cvtph_ps(w_sraw)) *
                             _mm_cvtss_f32(_mm_cvtph_ps(x_sraw));
            __m256 d = _mm256_set1_ps(combined);
            __m256 q = lal_dot32_f32((const int8_t*)(row + w_off + 2),
                                     (const int8_t*)(xq8_0 + x_off + 2), ones);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        y[j] = _mm_cvtss_f32(s) + (b ? b[j] : 0);
    }
#else
    /* Scalar fallback */
    int8_t xq[XQ_MAX];
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    float inv = 1.0f / x_scale;
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * inv);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 34;
    for (int j = 0; j < out_dim; j++) {
        const uint8_t *row = q8_0_W + (size_t)j * row_stride;
        float acc = 0.0f;
        for (int blk = 0; blk < blocks_per_row; blk++) {
            const uint8_t *block = row + blk * 34;
            uint16_t scale_raw = *(uint16_t*)block;
            uint32_t sign = (scale_raw >> 15) & 1;
            uint32_t exp = (scale_raw >> 10) & 0x1F;
            uint32_t frac = scale_raw & 0x3FF;
            float scale;
            if (exp == 0) {
                if (frac == 0) scale = sign ? -0.0f : 0.0f;
                else { float f = frac / 1024.0f / 524288.0f; scale = sign ? -f : f; }
            } else if (exp == 31) {
                scale = frac ? NAN : (sign ? -INFINITY : INFINITY);
            } else {
                float f = (1.0f + frac / 1024.0f) * ldexpf(1.0f, (int)exp - 15);
                scale = sign ? -f : f;
            }
            int32_t dot = 0;
            for (int i = 0; i < 32; i++)
                dot += (int)xq[blk*32 + i] * (int)((int8_t)block[2 + i]);
            acc += scale * (float)dot;
        }
        y[j] = acc * x_scale + (b ? b[j] : 0);
    }
#endif
}

/* === Quantize a float row to int8 with per-row scale (sign-trick compatible)
 * (legacy, for qtype=1 GPQ8 files)
 */
static inline void lal_quantize_q8_per_row(const float *W, int8_t *q8_T,
                                           float *scale,
                                           int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        const float *row = W + (size_t)j * in_dim;
        float mx = 0;
        for (int i = 0; i < in_dim; i++) { float a = fabsf(row[i]); if (a > mx) mx = a; }
        scale[j] = mx / 127.0f;
        if (scale[j] < 1e-8f) scale[j] = 1e-8f;
        float inv = 1.0f / scale[j];
        for (int i = 0; i < in_dim; i++) {
            int v = (int)lroundf(row[i] * inv);
            q8_T[(size_t)j * in_dim + i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    }
}

/* === Quantize x to int8 for LM head dot product === */
static inline float lal_quantize_x_int8(const float *x, int8_t *xq, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)(x[i] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        xq[i] = (int8_t)v;
    }
    return scale;
}

/* === LM head int8 dot product over a vocab range (sign-trick, AVX2) ===
 * 优化 v2:
 *  1. abs_xq 预计算提到 OpenMP 外 (避免每个线程重复计算 + false sharing)
 *  2. 软件 prefetch 下一个 vocab row (减少 L2 miss stall)
 *  3. 新增 lal_compute_abs_xq() 供调用方预计算
 */
static inline void lal_compute_abs_xq(const int8_t *xq, uint8_t *ax, int n_embd) {
#if defined(__AVX2__)
    for (int i = 0; i < n_embd; i += 32) {
        __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
        __m256i abs_xq = _mm256_sign_epi8(xv, xv);
        _mm256_storeu_si256((__m256i*)(ax + i), abs_xq);
    }
#else
    for (int i = 0; i < n_embd; i++) ax[i] = (uint8_t)(xq[i] < 0 ? -xq[i] : xq[i]);
#endif
}

/* 旧接口保留向后兼容: 内部调用 lal_compute_abs_xq + lal_lm_head_int8_range_abs */
static inline void lal_lm_head_int8_range(float *logits,
                                          const int8_t *xq,
                                          float scale_x,
                                          const int8_t *wte_q,
                                          const float *scale_w,
                                          int v_start, int v_end, int n_embd) {
#if defined(__AVX2__)
    static uint8_t ax[XQ_MAX] __attribute__((aligned(32)));
    lal_compute_abs_xq(xq, ax, n_embd);
    __m256i ones = _mm256_set1_epi16(1);
    for (int v = v_start; v < v_end; v++) {
        const int8_t *wv = wte_q + (size_t)v * n_embd;
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= n_embd; i += 32) {
            __m256i ax_v = _mm256_loadu_si256((__m256i*)(ax + i));
            __m256i sw_v = _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(wv + i)),
                                            _mm256_loadu_si256((__m256i*)(xq + i)));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(ax_v, sw_v), ones));
        }
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        int32_t dot = _mm_cvtsi128_si32(s);
        for (; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
#else
    for (int v = v_start; v < v_end; v++) {
        const int8_t *wv = wte_q + (size_t)v * n_embd;
        int32_t dot = 0;
        for (int i = 0; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
#endif
}

/* 新接口: 接受预计算的 abs_xq, 避免 OpenMP 线程内重复计算
 * 注意: 不含 prefetch — 实测 Xeon Platinum 硬件预取器已足够, 软件 prefetch 反而干扰
 * 注意: AVX-512 VNNI 不适合此处 — dpbusd 需要 unsigned×signed, 而 sign-trick
 *       用 abs(x)×sign_adjusted(w), 两者数学等价但 VNNI 无法直接表达 signed×signed.
 *       详见另一位 agent 的 commit cc896f5 (VNNI 实验失败). */
static inline void lal_lm_head_int8_range_abs(float *logits,
                                               const int8_t *xq,
                                               const uint8_t *ax,
                                               float scale_x,
                                               const int8_t *wte_q,
                                               const float *scale_w,
                                               int v_start, int v_end, int n_embd) {
#if defined(__AVX2__)
    __m256i ones = _mm256_set1_epi16(1);
    for (int v = v_start; v < v_end; v++) {
        const int8_t *wv = wte_q + (size_t)v * n_embd;
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= n_embd; i += 32) {
            __m256i ax_v = _mm256_loadu_si256((__m256i*)(ax + i));
            __m256i sw_v = _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(wv + i)),
                                            _mm256_loadu_si256((__m256i*)(xq + i)));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(ax_v, sw_v), ones));
        }
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        int32_t dot = _mm_cvtsi128_si32(s);
        for (; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
#else
    for (int v = v_start; v < v_end; v++) {
        const int8_t *wv = wte_q + (size_t)v * n_embd;
        int32_t dot = 0;
        for (int i = 0; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
#endif
}

#endif /* LAL_Q8_KERNEL_H */
