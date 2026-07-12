/* bench_q4k_v2.c — 对比原始 vs v2 (预计算 scales) Q4_K kernel
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_q4k_v2 scripts/bench_q4k_v2.c -lm -lgomp
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
#include "runtime/lal_q4k_kernel_v2.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int main(int argc, char **argv) {
    int n_threads = argc > 1 ? atoi(argv[1]) : 1;
    struct { const char *name; int in_dim, out_dim; } tests[] = {
        {"gate    [18944, 3584]", 3584, 18944},
        {"down    [3584, 18944]", 18944, 3584},
    };
    int n_tests = sizeof(tests)/sizeof(tests[0]);
    srand(42);

    printf("=== Q4_K: Original vs V2 (precomputed scales) ===\n");
    printf("Threads: %d\n\n", n_threads);
    printf("%-22s  %-10s  %-10s  %-8s  %-10s  %-10s  %-8s\n",
           "test", "orig(ms)", "v2(ms)", "speedup", "orig GB/s", "v2 GB/s", "bw_up");

    for (int t = 0; t < n_tests; t++) {
        int in_dim = tests[t].in_dim, out_dim = tests[t].out_dim;
        int n_super = in_dim / 256;
        int row_stride = n_super * 144;

        float *x = malloc(in_dim * sizeof(float));
        uint8_t *q4k = malloc((size_t)out_dim * row_stride);
        float *y = malloc(out_dim * sizeof(float));

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

        /* Warmup */
        lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
        lal_matmul_q4_k_v2(y, q4k, x, NULL, in_dim, out_dim);

        int n_iter = 5;
        double mem_read = (double)out_dim * row_stride + in_dim * 4;

        /* Single-threaded comparison */
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++) lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
        double dt_orig = (now_s() - t0) / n_iter;

        t0 = now_s();
        for (int it = 0; it < n_iter; it++) lal_matmul_q4_k_v2(y, q4k, x, NULL, in_dim, out_dim);
        double dt_v2 = (now_s() - t0) / n_iter;

        /* Multi-threaded comparison */
        if (n_threads > 1) {
            t0 = now_s();
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
            dt_orig = (now_s() - t0) / n_iter;

            t0 = now_s();
            for (int it = 0; it < n_iter; it++) {
                #pragma omp parallel num_threads(n_threads)
                {
                    int tid = omp_get_thread_num();
                    int chunk = (out_dim + n_threads - 1) / n_threads;
                    int start = tid * chunk;
                    int end = start + chunk;
                    if (end > out_dim) end = out_dim;
                    if (start < out_dim)
                        lal_matmul_q4_k_v2(y + start, q4k + (size_t)start * row_stride, x, NULL, in_dim, end - start);
                }
            }
            dt_v2 = (now_s() - t0) / n_iter;
        }

        printf("%-22s  %-10.2f  %-10.2f  %-8.2fx  %-10.1f  %-10.1f  %-8.2fx\n",
               tests[t].name, dt_orig*1000, dt_v2*1000, dt_orig/dt_v2,
               mem_read/dt_orig/1e9, mem_read/dt_v2/1e9, (mem_read/dt_v2)/(mem_read/dt_orig));

        free(x); free(q4k); free(y);
    }

    return 0;
}
