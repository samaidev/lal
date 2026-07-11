#ifndef LAL_DEQUANT_H
#define LAL_DEQUANT_H
/*
 * lal_dequant.h - SIMD int8-to-float dequantization helpers.
 *
 * Used for embedding lookup and LM head rerank after float weights are
 * freed (the "free-float-weights" optimization). Converts a row of int8
 * values to float32 by multiplying by a per-row scale.
 *
 * Requires AVX2. Falls back to scalar on other architectures.
 */

#include <immintrin.h>

/* Dequantize n int8 values to float32: out[i] = q[i] * scale
 * AVX2 processes 8 elements per iteration (8x faster than scalar). */
static inline void lal_dequant_row_f32(const int8_t *q, float *out,
                                       float scale, int n) {
#if defined(__AVX2__)
    __m256 v_scale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i bytes = _mm_loadl_epi64((const __m128i*)(q + i));
        __m256i i32 = _mm256_cvtepi8_epi32(bytes);
        __m256 f32 = _mm256_cvtepi32_ps(i32);
        __m256 result = _mm256_mul_ps(f32, v_scale);
        _mm256_storeu_ps(out + i, result);
    }
    for (; i < n; i++) out[i] = q[i] * scale;
#else
    for (int i = 0; i < n; i++) out[i] = q[i] * scale;
#endif
}

/* Dequantize + add to existing float array: out[i] += q[i] * scale
 * (used for embedding lookup with position embeddings: x = wte[tok] + wpe[pos]) */
static inline void lal_dequant_add_f32(const int8_t *q, float *out,
                                       float scale, int n) {
#if defined(__AVX2__)
    __m256 v_scale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i bytes = _mm_loadl_epi64((const __m128i*)(q + i));
        __m256i i32 = _mm256_cvtepi8_epi32(bytes);
        __m256 f32 = _mm256_cvtepi32_ps(i32);
        __m256 scaled = _mm256_mul_ps(f32, v_scale);
        __m256 existing = _mm256_loadu_ps(out + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(existing, scaled));
    }
    for (; i < n; i++) out[i] += q[i] * scale;
#else
    for (int i = 0; i < n; i++) out[i] += q[i] * scale;
#endif
}

#endif /* LAL_DEQUANT_H */
