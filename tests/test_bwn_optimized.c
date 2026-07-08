/* test_bwn_optimized.c — BWN matmul optimized with OpenBLAS-inspired techniques
 *
 * Techniques applied:
 *   1. Pre-packed sign(W) as contiguous float array (no LUT indirect)
 *   2. Multi-output register blocking: process 8 outputs j at once,
 *      sharing x loads (OpenBLAS micro-kernel pattern)
 *   3. AVX2 FMA intrinsics for the inner loop
 *   4. Cache-blocking on x (load x once, use for many j)
 *
 * Compare:
 *   - Original: LUT-based, single output, scalar dot
 *   - sign_float: pre-packed, single output, 8x unrolled
 *   - Optimized: pre-packed, 8 outputs parallel, AVX2 FMA
 *
 * Compile: gcc -O3 -mavx2 -mfma -o test_bwn_opt test_bwn_optimized.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#define IN_DIM 768
#define OUT_DIM 2304
#define N_TRIALS 200

/* Original: LUT-based sign unpack, single output */
static void bwn_orig(float *y, const uint64_t *wbits, int n_words,
                     const float *x, const float *alpha, const float *bias,
                     int in_dim, int out_dim) {
    static float sign_lut[256][8];
    static int init = 0;
    if (!init) {
        for (int b = 0; b < 256; b++)
            for (int i = 0; i < 8; i++)
                sign_lut[b][i] = (b >> i) & 1 ? 1.0f : -1.0f;
        init = 1;
    }
    for (int j = 0; j < out_dim; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        float dot = 0.0f;
        for (int wi = 0; wi < n_words; wi++) {
            uint64_t w = wb[wi];
            int base = wi * 64;
            for (int bi = 0; bi < 8; bi++) {
                int idx = base + bi * 8;
                if (idx + 7 < in_dim) {
                    const float *sw = sign_lut[(w >> (bi*8)) & 0xFF];
                    dot += x[idx+0]*sw[0] + x[idx+1]*sw[1] + x[idx+2]*sw[2] + x[idx+3]*sw[3];
                    dot += x[idx+4]*sw[4] + x[idx+5]*sw[5] + x[idx+6]*sw[6] + x[idx+7]*sw[7];
                }
            }
        }
        y[j] = dot * alpha[j] + bias[j];
    }
}

/* Optimized: pre-packed sign float, 8 outputs parallel with AVX2 FMA */
static void bwn_optimized(float *y, const float *w_sign, /* [out_dim * in_dim] */
                          const float *x, const float *alpha, const float *bias,
                          int in_dim, int out_dim) {
    /* Process 8 outputs at a time, sharing x loads.
     * Each output j computes dot[j] = sum_i w_sign[j*in_dim+i] * x[i]
     * With AVX2: load x[0..7] once, multiply by 8 different w_sign rows,
     * accumulate into 8 separate accumulators. */
    int j = 0;
    for (; j + 8 <= out_dim; j += 8) {
        const float *w0 = w_sign + (size_t)(j+0) * in_dim;
        const float *w1 = w_sign + (size_t)(j+1) * in_dim;
        const float *w2 = w_sign + (size_t)(j+2) * in_dim;
        const float *w3 = w_sign + (size_t)(j+3) * in_dim;
        const float *w4 = w_sign + (size_t)(j+4) * in_dim;
        const float *w5 = w_sign + (size_t)(j+5) * in_dim;
        const float *w6 = w_sign + (size_t)(j+6) * in_dim;
        const float *w7 = w_sign + (size_t)(j+7) * in_dim;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        for (int i = 0; i < in_dim; i += 8) {
            __m256 xv = _mm256_loadu_ps(x + i);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i), xv, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i), xv, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + i), xv, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + i), xv, acc3);
            acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(w4 + i), xv, acc4);
            acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(w5 + i), xv, acc5);
            acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(w6 + i), xv, acc6);
            acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(w7 + i), xv, acc7);
        }
        /* Horizontal sum each accumulator */
        float d0 = _mm256_cvtss_f32(_mm256_hadd_ps(acc0, acc0));
        float d1 = _mm256_cvtss_f32(_mm256_hadd_ps(acc1, acc1));
        float d2 = _mm256_cvtss_f32(_mm256_hadd_ps(acc2, acc2));
        float d3 = _mm256_cvtss_f32(_mm256_hadd_ps(acc3, acc3));
        float d4 = _mm256_cvtss_f32(_mm256_hadd_ps(acc4, acc4));
        float d5 = _mm256_cvtss_f32(_mm256_hadd_ps(acc5, acc5));
        float d6 = _mm256_cvtss_f32(_mm256_hadd_ps(acc6, acc6));
        float d7 = _mm256_cvtss_f32(_mm256_hadd_ps(acc7, acc7));
        /* Need proper horizontal sum (hadd only does within lanes) */
        /* Actually do it manually */
        #define HSUM256(v) ({ \
            __m256 t = _mm256_hadd_ps(v, v); \
            t = _mm256_hadd_ps(t, t); \
            __m128 lo = _mm256_castps256_ps128(t); \
            __m128 hi = _mm256_extractf128_ps(t, 1); \
            _mm_cvtss_f32(_mm_add_ss(lo, hi)); })
        d0 = HSUM256(acc0); d1 = HSUM256(acc1); d2 = HSUM256(acc2); d3 = HSUM256(acc3);
        d4 = HSUM256(acc4); d5 = HSUM256(acc5); d6 = HSUM256(acc6); d7 = HSUM256(acc7);
        y[j+0] = d0 * alpha[j+0] + bias[j+0];
        y[j+1] = d1 * alpha[j+1] + bias[j+1];
        y[j+2] = d2 * alpha[j+2] + bias[j+2];
        y[j+3] = d3 * alpha[j+3] + bias[j+3];
        y[j+4] = d4 * alpha[j+4] + bias[j+4];
        y[j+5] = d5 * alpha[j+5] + bias[j+5];
        y[j+6] = d6 * alpha[j+6] + bias[j+6];
        y[j+7] = d7 * alpha[j+7] + bias[j+7];
    }
    /* Tail: remaining outputs */
    for (; j < out_dim; j++) {
        const float *w = w_sign + (size_t)j * in_dim;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < in_dim; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), acc);
        y[j] = HSUM256(acc) * alpha[j] + bias[j];
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    srand(42);
    uint64_t *wbits = malloc((size_t)OUT_DIM * ((IN_DIM+63)/64) * sizeof(uint64_t));
    float *w_sign = malloc((size_t)OUT_DIM * IN_DIM * sizeof(float));
    float *x = malloc(IN_DIM * sizeof(float));
    float *alpha = malloc(OUT_DIM * sizeof(float));
    float *bias = malloc(OUT_DIM * sizeof(float));
    float *y1 = malloc(OUT_DIM * sizeof(float));
    float *y2 = malloc(OUT_DIM * sizeof(float));
    int n_words = (IN_DIM + 63) / 64;

    /* Init random weights */
    for (size_t i = 0; i < (size_t)OUT_DIM * ((IN_DIM+63)/64); i++) wbits[i] = ((uint64_t)rand() << 32) | rand();
    for (int i = 0; i < IN_DIM; i++) x[i] = (rand()/(float)RAND_MAX - 0.5f) * 4;
    for (int j = 0; j < OUT_DIM; j++) { alpha[j] = 0.1f; bias[j] = 0.01f * j; }

    /* Pack sign(W) as float from wbits */
    for (int j = 0; j < OUT_DIM; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        float *ws = w_sign + (size_t)j * IN_DIM;
        for (int wi = 0; wi < n_words; wi++) {
            uint64_t w = wb[wi];
            for (int bi = 0; bi < 64; bi++) {
                int i = wi * 64 + bi;
                if (i < IN_DIM) ws[i] = (w >> bi) & 1 ? 1.0f : -1.0f;
            }
        }
    }

    /* Correctness: compare orig vs optimized */
    bwn_orig(y1, wbits, n_words, x, alpha, bias, IN_DIM, OUT_DIM);
    bwn_optimized(y2, w_sign, x, alpha, bias, IN_DIM, OUT_DIM);
    float max_diff = 0;
    for (int j = 0; j < OUT_DIM; j++) {
        float d = fabsf(y1[j] - y2[j]);
        if (d > max_diff) max_diff = d;
    }
    printf("Correctness: max diff = %.6e (should be ~0)\n", max_diff);

    /* Benchmark */
    double t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) bwn_orig(y1, wbits, n_words, x, alpha, bias, IN_DIM, OUT_DIM);
    double t_orig = now_sec() - t0;

    t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) bwn_optimized(y2, w_sign, x, alpha, bias, IN_DIM, OUT_DIM);
    double t_opt = now_sec() - t0;

    printf("Original (LUT):      %.3f ms/trial (%.1f MOps/s)\n",
           t_orig/N_TRIALS*1000, (double)IN_DIM*OUT_DIM*N_TRIALS/t_orig/1e6);
    printf("Optimized (AVX2 8-out): %.3f ms/trial (%.1f MOps/s)\n",
           t_opt/N_TRIALS*1000, (double)IN_DIM*OUT_DIM*N_TRIALS/t_opt/1e6);
    printf("Speedup: %.2fx\n", t_orig/t_opt);

    free(wbits); free(w_sign); free(x); free(alpha); free(bias); free(y1); free(y2);
    return 0;
}
