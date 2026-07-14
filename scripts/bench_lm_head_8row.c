/* bench_lm_head_8row.c — 对比 1-row vs 8-row LM head kernel
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_lmh8 scripts/bench_lm_head_8row.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define N_EMBD 3584
#define VOCAB_SIZE 152064
#define XQ_MAX 18944

#include "runtime/lal_q8_kernel.h"
#include "runtime/lal_simd_optim.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    int n_threads = 2;
    omp_set_num_threads(n_threads);

    int8_t *lm_head_q = _mm_malloc((size_t)VOCAB_SIZE * N_EMBD, 32);
    float *lm_head_s = _mm_malloc(VOCAB_SIZE * sizeof(float), 32);
    float *x = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *logits1 = _mm_malloc(VOCAB_SIZE * sizeof(float), 32);
    float *logits8 = _mm_malloc(VOCAB_SIZE * sizeof(float), 32);
    int8_t *xq = _mm_malloc(N_EMBD, 32);
    uint8_t *abs_xq = _mm_malloc(N_EMBD, 32);

    srand(42);
    for (size_t i = 0; i < (size_t)VOCAB_SIZE * N_EMBD; i++)
        lm_head_q[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < VOCAB_SIZE; i++) lm_head_s[i] = 0.01f;
    for (int i = 0; i < N_EMBD; i++) x[i] = (rand()/((float)RAND_MAX)-0.5f) * 0.3f;

    float scale_x = lal_quantize_x_int8(x, xq, N_EMBD);
    lal_compute_abs_xq(xq, abs_xq, N_EMBD);

    /* Correctness: run both and compare */
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        int n = omp_get_num_threads();
        int v_per = (VOCAB_SIZE + n - 1) / n;
        int v_start = tid * v_per;
        int v_end = v_start + v_per;
        if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
        if (v_start < VOCAB_SIZE) {
            lal_lm_head_int8_range_abs(logits1, xq, abs_xq, scale_x, lm_head_q, lm_head_s, v_start, v_end, N_EMBD);
            lal_lm_head_int8_range_abs_8row(logits8, xq, abs_xq, scale_x, lm_head_q, lm_head_s, v_start, v_end, N_EMBD);
        }
    }
    float max_err = 0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        float d = fabsf(logits1[i] - logits8[i]);
        if (d > max_err) max_err = d;
    }
    printf("=== Correctness (1-row vs 8-row) ===\n");
    printf("max_err = %.2e %s\n\n", max_err, max_err < 1e-4f ? "✅ PASS" : "❌ FAIL");

    /* Benchmark 1-row */
    int n_iter = 5;
    double t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int n = omp_get_num_threads();
            int v_per = (VOCAB_SIZE + n - 1) / n;
            int v_start = tid * v_per;
            int v_end = v_start + v_per;
            if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
            if (v_start < VOCAB_SIZE)
                lal_lm_head_int8_range_abs(logits1, xq, abs_xq, scale_x, lm_head_q, lm_head_s, v_start, v_end, N_EMBD);
        }
    }
    double dt_1row = (now_ms() - t0) / n_iter;

    /* Benchmark 8-row */
    t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int n = omp_get_num_threads();
            int v_per = (VOCAB_SIZE + n - 1) / n;
            int v_start = tid * v_per;
            int v_end = v_start + v_per;
            if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
            if (v_start < VOCAB_SIZE)
                lal_lm_head_int8_range_abs_8row(logits8, xq, abs_xq, scale_x, lm_head_q, lm_head_s, v_start, v_end, N_EMBD);
        }
    }
    double dt_8row = (now_ms() - t0) / n_iter;

    double mem_read = (double)VOCAB_SIZE * N_EMBD + N_EMBD * 4;
    printf("=== LM Head: 1-row vs 8-row (%d threads) ===\n", n_threads);
    printf("1-row: %.2f ms  (%.1f GB/s)\n", dt_1row, mem_read/dt_1row/1e6);
    printf("8-row: %.2f ms  (%.1f GB/s)\n", dt_8row, mem_read/dt_8row/1e6);
    printf("speedup: %.2fx\n", dt_1row/dt_8row);
    printf("\nIn full forward (~480ms):\n");
    printf("  1-row LM head: %.2f ms (%.1f%%)\n", dt_1row, dt_1row/480*100);
    printf("  8-row LM head: %.2f ms (%.1f%%)\n", dt_8row, dt_8row/480*100);
    printf("  saved: %.2f ms/forward\n", dt_1row - dt_8row);

    _mm_free(lm_head_q); _mm_free(lm_head_s); _mm_free(x);
    _mm_free(logits1); _mm_free(logits8); _mm_free(xq); _mm_free(abs_xq);
    return 0;
}
