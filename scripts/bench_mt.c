/* bench_mt.c — multi-threaded kernel benchmark
 * Tests if OpenMP parallelism helps for large matrices.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <omp.h>

#define XQ_MAX 18944
#include "runtime/lal_runtime.h"
#include "runtime/lal_q8_kernel.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

static void bench(const char *name, int in_dim, int out_dim, int n_iter, int n_threads) {
    int8_t *q8_T = memalign(32, (size_t)out_dim * in_dim);
    float   *scale = memalign(32, out_dim * sizeof(float));
    float   *x = memalign(32, in_dim * sizeof(float));
    float   *y = memalign(32, out_dim * sizeof(float));

    for (size_t i = 0; i < (size_t)out_dim * in_dim; i++) q8_T[i] = (int8_t)(rand() & 0x7F);
    for (int i = 0; i < out_dim; i++) scale[i] = 0.01f;
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 100) / 100.0f;

    /* warmup */
    lal_matmul_q8_signtrick(y, q8_T, scale, x, NULL, in_dim, out_dim);

    double t0 = now_s();
    for (int it = 0; it < n_iter; it++) {
        if (n_threads <= 1) {
            lal_matmul_q8_signtrick(y, q8_T, scale, x, NULL, in_dim, out_dim);
        } else {
            #pragma omp parallel num_threads(n_threads)
            {
                int tid = omp_get_thread_num();
                int n   = omp_get_num_threads();
                int chunk = (out_dim + n - 1) / n;
                int start = tid * chunk;
                int end = start + chunk;
                if (end > out_dim) end = out_dim;
                if (start < out_dim) {
                    lal_matmul_q8_signtrick(y + start,
                                            q8_T + (size_t)start * in_dim,
                                            scale + start, x, NULL,
                                            in_dim, end - start);
                }
            }
        }
    }
    double t1 = now_s();
    double dt = (t1 - t0) / n_iter;
    double bytes = (double)out_dim * in_dim + out_dim * 4 + in_dim * 4;
    double gb_s = bytes / dt / 1e9;
    printf("  %-24s [%d, %d] %d threads: %7.3f ms  %6.1f GB/s\n",
           name, out_dim, in_dim, n_threads, dt*1000, gb_s);
    free(q8_T); free(scale); free(x); free(y);
}

int main(void) {
    srand(42);
    printf("=== Multi-threaded Q8 kernel benchmark ===\n\n");
    printf("--- gate_proj [18944, 3584] (the dominant 7B matmul) ---\n");
    bench("gate_proj", 3584, 18944, 10, 1);
    bench("gate_proj", 3584, 18944, 10, 2);
    printf("\n--- down_proj [3584, 18944] ---\n");
    bench("down_proj", 18944, 3584, 10, 1);
    bench("down_proj", 18944, 3584, 10, 2);
    printf("\n--- q_proj [3584, 3584] ---\n");
    bench("q_proj", 3584, 3584, 20, 1);
    bench("q_proj", 3584, 3584, 20, 2);
    printf("\n--- lm_head [152064, 3584] ---\n");
    bench("lm_head", 3584, 152064, 5, 1);
    bench("lm_head", 3584, 152064, 5, 2);
    return 0;
}
