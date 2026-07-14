/* bench_q4k_prefetch.c — 测试 Q4_K kernel 不同 prefetch 距离
 * 构建: gcc -O3 -march=native -fopenmp -I. -o bench_q4k_pf scripts/bench_q4k_prefetch.c -lm -lgomp
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
#define IN_DIM  3584
#define OUT_DIM 18944

#include "runtime/lal_q4k_kernel.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    float *x = _mm_malloc(IN_DIM * sizeof(float), 32);
    float *y = _mm_malloc(OUT_DIM * sizeof(float), 32);
    int n_super = IN_DIM / 256;
    int row_stride = n_super * 144;
    uint8_t *w = _mm_malloc((size_t)OUT_DIM * row_stride, 32);

    srand(42);
    for (int i = 0; i < IN_DIM; i++) x[i] = (float)(rand() % 200 - 100) / 50.0f;
    for (size_t i = 0; i < (size_t)OUT_DIM * row_stride; i++) w[i] = rand() & 0xFF;
    for (int r = 0; r < OUT_DIM; r++)
        for (int s = 0; s < n_super; s++) {
            uint8_t *sb = w + (size_t)r * row_stride + s * 144;
            *(uint16_t*)sb = 0x2E66; *(uint16_t*)(sb+2) = 0x25C3;
            memset(sb+4, 0x20, 12);
        }

    const int ITERS = 10;
    volatile float sink = 0;

    printf("=== Q4_K gate matmul [18944×3584] (%d iters, 2 threads) ===\n", ITERS);

    /* baseline (当前 prefetch 1 ahead = 144 bytes) */
    {
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            lal_matmul_q4_k(y, w, x, NULL, IN_DIM, OUT_DIM);
            sink += y[it % OUT_DIM];
        }
        printf("baseline (pf=1): %.2f ms/iter\n", (now_ms() - t0) / ITERS);
    }

    /* 多线程 */
    printf("\n--- 2 threads ---\n");
    {
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            #pragma omp parallel num_threads(2)
            {
                int tid = omp_get_thread_num();
                int chunk = OUT_DIM / 2;
                int start = tid * chunk;
                lal_matmul_q4_k(y + start, w + (size_t)start * row_stride,
                                x, NULL, IN_DIM, chunk);
            }
            sink += y[it % OUT_DIM];
        }
        printf("2 threads: %.2f ms/iter\n", (now_ms() - t0) / ITERS);
    }

    /* 4 threads (超线程) */
    printf("\n--- 4 threads ---\n");
    {
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            #pragma omp parallel num_threads(4)
            {
                int tid = omp_get_thread_num();
                int chunk = (OUT_DIM + 3) / 4;
                int start = tid * chunk;
                int end = start + chunk;
                if (end > OUT_DIM) end = OUT_DIM;
                if (start < OUT_DIM)
                    lal_matmul_q4_k(y + start, w + (size_t)start * row_stride,
                                    x, NULL, IN_DIM, end - start);
            }
            sink += y[it % OUT_DIM];
        }
        printf("4 threads: %.2f ms/iter\n", (now_ms() - t0) / ITERS);
    }

    if (sink < -1e30) printf("impossible\n");
    return 0;
}
