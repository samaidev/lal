/* bench_q4k_opt_vs_orig.c — 对比原始 vs 优化 Q4_K kernel
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_q4k_cmp scripts/bench_q4k_opt_vs_orig.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel.h"
#include "runtime/lal_q4k_kernel_opt.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int main() {
    struct { const char *name; int in_dim, out_dim; } tests[] = {
        {"q_proj  [512, 3584]",  3584, 512},
        {"gate    [18944, 3584]", 3584, 18944},
        {"down    [3584, 18944]", 18944, 3584},
    };
    int n_tests = sizeof(tests)/sizeof(tests[0]);
    srand(42);

    printf("=== Original vs Optimized Q4_K Kernel ===\n");
    printf("%-22s  %-12s  %-12s  %-8s  %-12s  %-12s  %-8s\n",
           "test", "orig(ms)", "opt(ms)", "speedup", "orig GB/s", "opt GB/s", "bw_up");

    for (int t = 0; t < n_tests; t++) {
        int in_dim = tests[t].in_dim, out_dim = tests[t].out_dim;
        int n_super = in_dim / 256;
        int row_stride = n_super * 144;

        float *x = malloc(in_dim * sizeof(float));
        uint8_t *q4k = malloc((size_t)out_dim * row_stride);
        float *y = malloc(out_dim * sizeof(float));

        for (int i = 0; i < in_dim; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        /* 生成随机 Q4_K 数据 */
        for (int j = 0; j < out_dim; j++) {
            uint8_t *row = q4k + (size_t)j * row_stride;
            for (int s = 0; s < n_super; s++) {
                uint8_t *sb = row + s*144;
                *(uint16_t*)sb = 0x3C00;     /* fp16 1.0 */
                *(uint16_t*)(sb+2) = 0x0000; /* fp16 0.0 */
                memset(sb+4, 0x20, 12);      /* scales */
                for (int i = 0; i < 128; i++) sb[16+i] = rand() & 0xFF;
            }
        }

        /* Warmup */
        lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
        lal_matmul_q4_k_opt(y, q4k, x, NULL, in_dim, out_dim);

        int n_iter = 5;
        double mem_read = (double)out_dim * row_stride + in_dim * 4;

        /* Original */
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++) lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
        double dt_orig = (now_s() - t0) / n_iter;

        /* Optimized */
        t0 = now_s();
        for (int it = 0; it < n_iter; it++) lal_matmul_q4_k_opt(y, q4k, x, NULL, in_dim, out_dim);
        double dt_opt = (now_s() - t0) / n_iter;

        double bw_orig = mem_read / dt_orig / 1e9;
        double bw_opt = mem_read / dt_opt / 1e9;

        printf("%-22s  %-12.2f  %-12.2f  %-8.2fx  %-12.1f  %-12.1f  %-8.2fx\n",
               tests[t].name, dt_orig*1000, dt_opt*1000, dt_orig/dt_opt,
               bw_orig, bw_opt, bw_opt/bw_orig);

        free(x); free(q4k); free(y);
    }

    /* 多线程测试 */
    printf("\n=== Multi-threaded (2 threads) ===\n");
    printf("%-22s  %-12s  %-12s  %-8s\n", "test", "orig(ms)", "opt(ms)", "speedup");

    for (int t = 1; t < n_tests; t++) {  /* skip q_proj (too small) */
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

        int n_iter = 5;

        /* Original parallel */
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++) {
            #pragma omp parallel num_threads(2)
            {
                int tid = omp_get_thread_num();
                int chunk = (out_dim + 1) / 2;
                int start = tid * chunk;
                int end = start + chunk;
                if (end > out_dim) end = out_dim;
                if (start < out_dim)
                    lal_matmul_q4_k(y + start, q4k + (size_t)start * row_stride, x, NULL, in_dim, end - start);
            }
        }
        double dt_orig = (now_s() - t0) / n_iter;

        /* Optimized parallel */
        t0 = now_s();
        for (int it = 0; it < n_iter; it++) {
            #pragma omp parallel num_threads(2)
            {
                int tid = omp_get_thread_num();
                int chunk = (out_dim + 1) / 2;
                int start = tid * chunk;
                int end = start + chunk;
                if (end > out_dim) end = out_dim;
                if (start < out_dim)
                    lal_matmul_q4_k_opt(y + start, q4k + (size_t)start * row_stride, x, NULL, in_dim, end - start);
            }
        }
        double dt_opt = (now_s() - t0) / n_iter;

        printf("%-22s  %-12.2f  %-12.2f  %-8.2fx\n",
               tests[t].name, dt_orig*1000, dt_opt*1000, dt_orig/dt_opt);

        free(x); free(q4k); free(y);
    }

    return 0;
}
