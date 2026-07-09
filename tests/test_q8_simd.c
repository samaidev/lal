/* test_q8_simd.c — Q8 matmul SIMD (AVX2 VPMADDUBSW + VPMADDWD)
 *
 * Layout: W stored as [out_dim, in_dim] (transposed, row-major per output).
 * This makes each output's weights contiguous — cache-friendly for matmul.
 *
 * y[j] = scale[j] * sum_i(q8_W[j*in_dim+i] * x[i]) + bias[j]
 *
 * SIMD: q8_W is signed int8, x quantized to unsigned int8 [0,255] (offset 128).
 * VPMADDUBSW: unsigned_byte × signed_byte → int16 (saturated)
 * Then VPMADDWD: int16 × int16 → int32
 *
 * Trick: x quantized to uint8 with zero-point=128 (like llama.cpp Q8_0 x).
 *   x_u8[i] = round(x[i] / x_scale) + 128
 *   q8_w is signed [-127, 127]
 *   VPMADDUBSW: x_u8 (unsigned) × q8_w (signed) → int16
 *   This computes (x_u8 - 128) × q8_w = x_quant × q8_w (after offset removal)
 *
 * Compare:
 *   A) Scalar (current matmul_q8)
 *   B) SIMD with transposed W
 *   C) SIMD without transpose (strided, for comparison)
 *
 * Compile: gcc -O3 -mavx2 -mfma -o test_q8_simd test_q8_simd.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#define IN_DIM 768
#define OUT_DIM 768
#define N_TRIALS 500

/* A) Scalar: W is [in, out] row-major (current server layout) */
static void matmul_q8_scalar(float *y, const int8_t *q8, const float *scale,
                             const float *x, const float *b, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float s = b ? b[j] : 0;
        for (int i = 0; i < in_dim; i++)
            s += (float)q8[(size_t)i * out_dim + j] * x[i];
        y[j] = s * scale[j] + (b ? b[j] : 0);
    }
}

/* B) SIMD with transposed W: q8_T is [out_dim, in_dim] row-major
 * Each output j has contiguous in_dim int8 weights. */
static void matmul_q8_simd_transposed(float *y, const int8_t *q8_T, const float *scale,
                                      const float *x, const float *b, int in_dim, int out_dim) {
    /* Quantize x to uint8 with zero-point 128 (unsigned, for VPMADDUBSW) */
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    uint8_t xq[4096];
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] / x_scale) + 128;
        xq[i] = (uint8_t)(v > 255 ? 255 : (v < 0 ? 0 : v));
    }

    for (int j = 0; j < out_dim; j++) {
        const int8_t *w = q8_T + (size_t)j * in_dim;  /* contiguous! */
        __m256i acc32 = _mm256_setzero_si256();
        for (int i = 0; i < in_dim; i += 32) {
            /* Load 32 uint8 from xq and 32 int8 from w */
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
            __m256i wv = _mm256_loadu_si256((__m256i*)(w + i));
            /* VPMADDUBSW: 32 unsigned_byte × 32 signed_byte → 16 int16 (saturated)
             * Each pair: xq[2k]*w[2k] + xq[2k+1]*w[2k+1] → int16 */
            __m256i prod16 = _mm256_maddubs_epi16(xv, wv);
            /* VPMADDWD: 16 int16 × 16 int16(=1) → 8 int32
             * We need sum of int16, so multiply by 1:
             * _mm256_madd_epi16(prod16, ones) where ones = [1,1,1,1,...] */
            __m256i ones = _mm256_set1_epi16(1);
            __m256i sum32 = _mm256_madd_epi16(prod16, ones);
            acc32 = _mm256_add_epi32(acc32, sum32);
        }
        /* Horizontal sum 8 int32 */
        __m128i lo = _mm256_castsi256_si128(acc32);
        __m128i hi = _mm256_extracti128_si256(acc32, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        int32_t dot = _mm_cvtsi128_si32(s);
        /* Remove zero-point offset: sum(xq * w) = sum((x_quant+128)*w)
         * = sum(x_quant*w) + 128*sum(w)
         * So actual dot = dot - 128 * sum(w) */
        /* Precompute sum(w) per output during quantization (omitted here for clarity,
         * in practice store w_sum[j]). For test, compute on the fly. */
        int32_t w_sum = 0;
        for (int i = 0; i < in_dim; i++) w_sum += w[i];
        dot -= 128 * w_sum;
        y[j] = (float)dot * x_scale * scale[j] + (b ? b[j] : 0);
    }
}

/* B-optimized: precompute w_sum during quantization, avoid per-call loop */
static void matmul_q8_simd_opt(float *y, const int8_t *q8_T, const float *scale,
                               const int32_t *w_sums, const float *x,
                               const float *b, int in_dim, int out_dim) {
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    uint8_t xq[4096];
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] / x_scale) + 128;
        xq[i] = (uint8_t)(v > 255 ? 255 : (v < 0 ? 0 : v));
    }
    __m256i ones = _mm256_set1_epi16(1);

    for (int j = 0; j < out_dim; j++) {
        const int8_t *w = q8_T + (size_t)j * in_dim;
        __m256i acc32 = _mm256_setzero_si256();
        for (int i = 0; i < in_dim; i += 32) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
            __m256i wv = _mm256_loadu_si256((__m256i*)(w + i));
            __m256i prod16 = _mm256_maddubs_epi16(xv, wv);
            __m256i sum32 = _mm256_madd_epi16(prod16, ones);
            acc32 = _mm256_add_epi32(acc32, sum32);
        }
        __m128i lo = _mm256_castsi256_si128(acc32);
        __m128i hi = _mm256_extracti128_si256(acc32, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        int32_t dot = _mm_cvtsi128_si32(s);
        dot -= 128 * w_sums[j];  /* remove zero-point offset */
        y[j] = (float)dot * x_scale * scale[j] + (b ? b[j] : 0);
    }
}

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void) {
    srand(42);
    /* q8: [in, out] row-major (server layout) */
    int8_t *q8 = malloc((size_t)IN_DIM*OUT_DIM);
    /* q8_T: [out, in] row-major (transposed, SIMD-friendly) */
    int8_t *q8_T = malloc((size_t)IN_DIM*OUT_DIM);
    float *scale = malloc(OUT_DIM*sizeof(float));
    int32_t *w_sums = malloc(OUT_DIM*sizeof(int32_t));
    float *x = malloc(IN_DIM*sizeof(float));
    float *b = malloc(OUT_DIM*sizeof(float));
    float *y1 = malloc(OUT_DIM*sizeof(float));
    float *y2 = malloc(OUT_DIM*sizeof(float));
    float *y3 = malloc(OUT_DIM*sizeof(float));

    for (int i = 0; i < IN_DIM; i++) x[i] = (rand()/(float)RAND_MAX - 0.5f) * 4;
    for (int j = 0; j < OUT_DIM; j++) { scale[j] = 0.001f + 0.001f*j; b[j] = 0.01f*j; }
    for (size_t i = 0; i < (size_t)IN_DIM*OUT_DIM; i++) q8[i] = (rand() % 255) - 127;
    /* Transpose: q8_T[j*IN_DIM + i] = q8[i*OUT_DIM + j] */
    for (int j = 0; j < OUT_DIM; j++)
        for (int i = 0; i < IN_DIM; i++)
            q8_T[(size_t)j*IN_DIM + i] = q8[(size_t)i*OUT_DIM + j];
    /* Precompute w_sums */
    for (int j = 0; j < OUT_DIM; j++) {
        int32_t s = 0;
        for (int i = 0; i < IN_DIM; i++) s += q8_T[(size_t)j*IN_DIM + i];
        w_sums[j] = s;
    }

    /* Correctness */
    matmul_q8_scalar(y1, q8, scale, x, b, IN_DIM, OUT_DIM);
    matmul_q8_simd_transposed(y2, q8_T, scale, x, b, IN_DIM, OUT_DIM);
    matmul_q8_simd_opt(y3, q8_T, scale, w_sums, x, b, IN_DIM, OUT_DIM);
    float d2=0, d3=0;
    for (int j = 0; j < OUT_DIM; j++) {
        d2 = fmaxf(d2, fabsf(y1[j]-y2[j]));
        d3 = fmaxf(d3, fabsf(y1[j]-y3[j]));
    }
    printf("Correctness (vs scalar):\n");
    printf("  simd_transposed: max diff=%.4f\n", d2);
    printf("  simd_opt:        max diff=%.4f\n", d3);

    /* Benchmark */
    double t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) matmul_q8_scalar(y1, q8, scale, x, b, IN_DIM, OUT_DIM);
    double t_s = now_sec() - t0;
    t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) matmul_q8_simd_transposed(y2, q8_T, scale, x, b, IN_DIM, OUT_DIM);
    double t_t = now_sec() - t0;
    t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) matmul_q8_simd_opt(y3, q8_T, scale, w_sums, x, b, IN_DIM, OUT_DIM);
    double t_o = now_sec() - t0;

    printf("\nBenchmark (%d trials, 768x768):\n", N_TRIALS);
    printf("  scalar:           %.3f ms\n", t_s/N_TRIALS*1000);
    printf("  simd_transposed:  %.3f ms  (%.2fx)\n", t_t/N_TRIALS*1000, t_s/t_t);
    printf("  simd_opt:         %.3f ms  (%.2fx)\n", t_o/N_TRIALS*1000, t_s/t_o);

    /* Estimate end-to-end: 12 layers × 4 matrices = 48 matmuls per token
     * Plus attention + LayerNorm + LM head overhead (~30% of matmul) */
    double ms_per_matmul = t_o / N_TRIALS * 1000;
    double ms_per_token = ms_per_matmul * 48 * 1.3;  /* 48 matmuls + 30% overhead */
    printf("\nEstimated per-token: %.1f ms = %.1f tok/s\n", ms_per_token, 1000.0/ms_per_token);

    free(q8);free(q8_T);free(scale);free(w_sums);free(x);free(b);free(y1);free(y2);free(y3);
    return 0;
}
