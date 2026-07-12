/* bench_matmul_breakdown.c — 详细分解 matmul 时间
 * 测量: prepare_x, Q4_K main loop, scale unpack, tail
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_break scripts/bench_matmul_breakdown.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    int in_dim = 3584, out_dim = 18944;
    int n_super = in_dim / 256;
    int row_stride = n_super * 144;

    srand(42);
    float *x = _mm_malloc(in_dim * sizeof(float), 32);
    uint8_t *q4k = _mm_malloc((size_t)out_dim * row_stride, 32);
    float *y = _mm_malloc(out_dim * sizeof(float), 32);

    for (int i = 0; i < in_dim; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
    for (int j = 0; j < out_dim; j++) {
        uint8_t *row = q4k + (size_t)j * row_stride;
        for (int s = 0; s < n_super; s++) {
            uint8_t *sb = row + s*144;
            *(uint16_t*)sb = 0x3C00; *(uint16_t*)(sb+2) = 0x0000;
            memset(sb+4, 0x20, 12);
            for (int i = 0; i < 128; i++) sb[16+i] = rand() & 0xFF;
        }
    }

    int n_threads = 2;
    omp_set_num_threads(n_threads);
    int n_iter = 10;

    /* Full matmul (baseline) */
    lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
    double t0 = now_ms();
    for (int it = 0; it < n_iter; it++) lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
    double dt_full = (now_ms() - t0) / n_iter;

    /* prepare_x only (x quantization + bsums + xq_arr) */
    int8_t xq[XQ_MAX] __attribute__((aligned(32)));
    int16_t bsums[XQ_MAX/32] __attribute__((aligned(32)));
    int8_t xq_arr[XQ_MAX] __attribute__((aligned(32)));
    t0 = now_ms();
    for (int it = 0; it < n_iter; it++) lal_q4k_prepare_x(x, in_dim, xq, bsums, xq_arr);
    double dt_prepare = (now_ms() - t0) / n_iter;

    /* prepared matmul (skip prepare_x) */
    float x_scale;
    lal_q4k_prepare_x(x, in_dim, xq, bsums, xq_arr);
    /* need x_scale too */
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) { float a = fabsf(x[i]); if (a > x_max) x_max = a; }
    x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;

    t0 = now_ms();
    for (int it = 0; it < n_iter; it++) lal_matmul_q4_k_prepared(y, q4k, x, NULL, in_dim, out_dim, xq, bsums, xq_arr, x_scale);
    double dt_prepared = (now_ms() - t0) / n_iter;

    /* Multi-threaded full matmul */
    t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int chunk = (out_dim + n_threads - 1) / n_threads;
            int start = tid * chunk;
            int end = start + chunk;
            if (end > out_dim) end = out_dim;
            if (start < out_dim)
                lal_matmul_q4_k(y + start, q4k + (size_t)start * row_stride, x, NULL, in_dim, end - start);
        }
    }
    double dt_mt = (now_ms() - t0) / n_iter;

    /* Multi-threaded prepared matmul */
    t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int chunk = (out_dim + n_threads - 1) / n_threads;
            int start = tid * chunk;
            int end = start + chunk;
            if (end > out_dim) end = out_dim;
            if (start < out_dim)
                lal_matmul_q4_k_prepared(y + start, q4k + (size_t)start * row_stride, x, NULL, in_dim, end - start, xq, bsums, xq_arr, x_scale);
        }
    }
    double dt_mt_prep = (now_ms() - t0) / n_iter;

    double mem_read = (double)out_dim * row_stride + in_dim * 4;

    printf("=== Q4_K Matmul Breakdown (gate [18944, 3584], %d threads) ===\n\n", n_threads);
    printf("Full matmul (1T):        %.2f ms  (%.1f GB/s)\n", dt_full, mem_read/dt_full/1e6);
    printf("prepare_x only (1T):     %.2f ms  (%.1f%% of full)\n", dt_prepare, dt_prepare/dt_full*100);
    printf("Prepared matmul (1T):    %.2f ms  (%.1f GB/s)\n", dt_prepared, mem_read/dt_prepared/1e6);
    printf("Full matmul (%dT):       %.2f ms  (%.1f GB/s)  %.2fx\n", n_threads, dt_mt, mem_read/dt_mt/1e6, dt_full/dt_mt);
    printf("Prepared matmul (%dT):   %.2f ms  (%.1f GB/s)  %.2fx\n", n_threads, dt_mt_prep, mem_read/dt_mt_prep/1e6, dt_full/dt_mt_prep);
    printf("\nShared prepare savings (2T): %.2f ms/call\n", dt_mt - dt_mt_prep);
    printf("Per forward (28 layers × 3 matmuls): %.2f ms saved\n", (dt_mt - dt_mt_prep) * 28 * 3);

    _mm_free(x); _mm_free(q4k); _mm_free(y);
    return 0;
}
