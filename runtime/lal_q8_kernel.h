#ifndef LAL_Q8_KERNEL_H
#define LAL_Q8_KERNEL_H

/*
 * lal_q8_kernel.h - Reusable Q8 matmul kernels with mistral.rs sign-trick.
 *
 * Included by tools/server/gpt2_server.c and tools/server/qwen_server.c.
 * Each server defines its own XQ_MAX (max in_dim, e.g. 4096 for GPT-2,
 * 4864 for Qwen2.5-0.5B) before including this header.
 *
 * The sign-trick (mistral.rs / llama.cpp):
 *   ax = sign_epi8(x, x) = |x|             (uint8 range)
 *   sw = sign_epi8(w, x) = sign(x) * w     (int8 range)
 *   maddubs(ax, sw) = sum_i x[i] * w[i]    (correct signed dot, no zero-point)
 *
 * This avoids the +128 offset and -128*w_sums subtraction of the older
 * approach, saving 1 add per element during x quantization and 1 sub per
 * output at accumulation end. No w_sums pre-computation needed.
 *
 * Requires AVX2. Falls back to scalar on other architectures.
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX (max in_dim) before including lal_q8_kernel.h"
#endif

/* === mistral.rs sign-trick Q8 matmul (AVX2) ===
 *
 * y[out_dim] = q8_T[out_dim, in_dim] @ x[in_dim] + b[out_dim]
 *
 * q8_T is TRANSPOSED (row-major [out_dim, in_dim]) for contiguous SIMD loads.
 * scale[out_dim] is per-row scale = max|row|/127.
 * x is float; quantized on-the-fly to int8 (no +128 offset).
 *
 * 8-output parallel: 8 accumulators in YMM registers, x loaded once per chunk.
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
            __m256i ax = _mm256_sign_epi8(xv, xv);  /* |x|, uint8 range */
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

/* === Quantize a float row to int8 with per-row scale (sign-trick compatible)
 *
 * W is [out_dim, in_dim] row-major. q8_T gets [out_dim, in_dim] int8.
 * scale[out_dim] = max|row|/127.
 *
 * No w_sums computation (sign-trick doesn't need zero-point correction).
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

/* === Quantize x to int8 for LM head dot product ===
 *
 * Returns the per-row scale. xq must be int8 buffer of size n.
 * Same scheme as lal_matmul_q8_signtrick: pure int8, no +128 offset.
 */
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
 *
 * logits[v] = scale_x * scale_w[v] * dot(xq, wte_q[v])
 *
 * wte_q is [vocab, n_embd] int8, scale_w is [vocab] float.
 * xq is [n_embd] int8 (from lal_quantize_x_int8).
 *
 * Hoisted |xq| prep: computed once per call (not per v).
 */
static inline void lal_lm_head_int8_range(float *logits,
                                          const int8_t *xq,
                                          float scale_x,
                                          const int8_t *wte_q,
                                          const float *scale_w,
                                          int v_start, int v_end, int n_embd) {
#if defined(__AVX2__)
    /* Hoist |xq| prep out of the per-v loop */
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
        /* Horizontal sum 8 int32 lanes -> 1 scalar */
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
