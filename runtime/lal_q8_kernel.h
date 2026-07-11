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

/* === NEW: Q8_0 block-format matmul (cache-friendly, matches llama.cpp) ===
 *
 * y[out_dim] = q8_0_W[out_dim, in_dim] @ x[in_dim] + b[out_dim]
 *
 * q8_0_W is packed as blocks: for row r, block b covers elements [b*32..b*32+31].
 * Each block = 2 bytes fp16 scale + 32 bytes int8 = 34 bytes total.
 * Total size = out_dim * (in_dim/32) * 34 bytes.
 *
 * in_dim must be a multiple of 32.
 *
 * Optimizations vs legacy kernel:
 *   1. Scale inline with data → no separate scale array access (saves cache)
 *   2. 4-row parallel (not 8) → less register pressure, better IPC
 *   3. Explicit prefetch for next block's data
 *   4. fp32 accumulation of scale*dot per block (avoids int32 hsum per block)
 *   5. Sequential memory access pattern → HW prefetcher loves it
 */
static inline void lal_matmul_q8_0(float *y,
                                    const uint8_t *q8_0_W,
                                    const float *x,
                                    const float *b,
                                    int in_dim, int out_dim) {
#if defined(__AVX2__) && defined(__F16C__)
    /* Quantize x to int8 once */
    int8_t xq[XQ_MAX] __attribute__((aligned(32)));
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
    int row_stride = blocks_per_row * 34;  /* bytes per row */

    __m256i ones = _mm256_set1_epi16(1);

    int j = 0;
    /* 8-row parallel: 8 scalar fp32 accumulators (one per row).
     * Per block: load 8 weights, compute 8 dot products (each hsum'd to scalar),
     * multiply by 8 scales, add to 8 accumulators.
     *
     * This matches the legacy kernel's structure but with per-block scale.
     * The hsum-to-scalar per block is 3 instructions (hadd x3), cheap on modern CPUs. */
    for (; j + 8 <= out_dim; j += 8) {
        float facc0 = 0, facc1 = 0, facc2 = 0, facc3 = 0;
        float facc4 = 0, facc5 = 0, facc6 = 0, facc7 = 0;

        const uint8_t *row0 = q8_0_W + (size_t)(j+0) * row_stride;
        const uint8_t *row1 = q8_0_W + (size_t)(j+1) * row_stride;
        const uint8_t *row2 = q8_0_W + (size_t)(j+2) * row_stride;
        const uint8_t *row3 = q8_0_W + (size_t)(j+3) * row_stride;
        const uint8_t *row4 = q8_0_W + (size_t)(j+4) * row_stride;
        const uint8_t *row5 = q8_0_W + (size_t)(j+5) * row_stride;
        const uint8_t *row6 = q8_0_W + (size_t)(j+6) * row_stride;
        const uint8_t *row7 = q8_0_W + (size_t)(j+7) * row_stride;

        for (int blk = 0; blk < blocks_per_row; blk++) {
            int offset = blk * 34;
            int x_off = blk * 32;

            /* Prefetch 2 blocks ahead (only first 4 rows to limit prefetch traffic) */
            if (blk + 2 < blocks_per_row) {
                _mm_prefetch((const char*)(row0 + offset + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row1 + offset + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row2 + offset + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row3 + offset + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row4 + offset + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row5 + offset + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row6 + offset + 2*34), _MM_HINT_T0);
                _mm_prefetch((const char*)(row7 + offset + 2*34), _MM_HINT_T0);
            }

            /* Load x int8 for this block (shared across 8 rows) */
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + x_off));
            __m256i ax = _mm256_sign_epi8(xv, xv);  /* |x| */

            /* Process each row: load scale + weight, dot, hsum, multiply, accumulate */
            #define PROC_ROW(row_ptr, facc) do { \
                /* Load fp16 scale -> fp32 scalar */ \
                uint16_t scale_u16 = *(const uint16_t*)(row_ptr + offset); \
                __m128i scale_raw = _mm_set1_epi16((short)scale_u16); \
                __m128 scale_f32 = _mm_cvtph_ps(scale_raw); \
                float scale = _mm_cvtss_f32(scale_f32); \
                /* Load 32 bytes int8 weight */ \
                __m256i wv = _mm256_loadu_si256((__m256i*)(row_ptr + offset + 2)); \
                /* Sign-trick dot */ \
                __m256i sw = _mm256_sign_epi8(wv, xv); \
                __m256i prod = _mm256_maddubs_epi16(ax, sw); \
                __m256i sum32 = _mm256_madd_epi16(prod, ones); \
                /* Horizontal sum to scalar int32 */ \
                __m128i lo128 = _mm256_castsi256_si128(sum32); \
                __m128i hi128 = _mm256_extracti128_si256(sum32, 1); \
                __m128i s = _mm_add_epi32(lo128, hi128); \
                s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s); \
                int32_t dot = _mm_cvtsi128_si32(s); \
                /* Accumulate scale * dot as fp32 scalar */ \
                facc += scale * (float)dot; \
            } while(0)

            PROC_ROW(row0, facc0);
            PROC_ROW(row1, facc1);
            PROC_ROW(row2, facc2);
            PROC_ROW(row3, facc3);
            PROC_ROW(row4, facc4);
            PROC_ROW(row5, facc5);
            PROC_ROW(row6, facc6);
            PROC_ROW(row7, facc7);
            #undef PROC_ROW
        }

        y[j+0] = facc0 * x_scale + (b ? b[j+0] : 0);
        y[j+1] = facc1 * x_scale + (b ? b[j+1] : 0);
        y[j+2] = facc2 * x_scale + (b ? b[j+2] : 0);
        y[j+3] = facc3 * x_scale + (b ? b[j+3] : 0);
        y[j+4] = facc4 * x_scale + (b ? b[j+4] : 0);
        y[j+5] = facc5 * x_scale + (b ? b[j+5] : 0);
        y[j+6] = facc6 * x_scale + (b ? b[j+6] : 0);
        y[j+7] = facc7 * x_scale + (b ? b[j+7] : 0);
    }

    /* Tail: remaining rows (out_dim not multiple of 4) */
    for (; j < out_dim; j++) {
        const uint8_t *row = q8_0_W + (size_t)j * row_stride;
        float acc = 0.0f;
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int offset = blk * 34;
            int x_off = blk * 32;
            __m128i scale_raw = _mm_loadl_epi16((__m128i*)(row + offset));
            float scale = _mm_cvtss_f32(_mm_cvtph_ps(scale_raw));
            __m256i wv = _mm256_loadu_si256((__m256i*)(row + offset + 2));
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + x_off));
            __m256i sw = _mm256_sign_epi8(wv, xv);
            __m256i ax = _mm256_sign_epi8(xv, xv);
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
            /* fp16->fp32 manual */
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

/* === LM head int8 dot product over a vocab range (sign-trick, AVX2) === */
static inline void lal_lm_head_int8_range(float *logits,
                                          const int8_t *xq,
                                          float scale_x,
                                          const int8_t *wte_q,
                                          const float *scale_w,
                                          int v_start, int v_end, int n_embd) {
#if defined(__AVX2__)
    static uint8_t ax[XQ_MAX] __attribute__((aligned(32)));
    for (int i = 0; i < n_embd; i += 32) {
        __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
        __m256i abs_xq = _mm256_sign_epi8(xv, xv);
        _mm256_storeu_si256((__m256i*)(ax + i), abs_xq);
    }
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
