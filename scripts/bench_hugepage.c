/* bench_hugepage.c — 测试 huge pages 对 Q4_K kernel 性能的影响
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_hp scripts/bench_hugepage.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <immintrin.h>
#include <omp.h>

#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* Allocate aligned memory with huge pages hint */
static void *hp_alloc(size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, 4096, size) != 0) return NULL;
    madvise(p, size, MADV_HUGEPAGE);
    return p;
}

int main(int argc, char **argv) {
    int n_threads = argc > 1 ? atoi(argv[1]) : 2;
    int use_hp = argc > 2 ? atoi(argv[2]) : 0;  /* 0=malloc, 1=hugepage */
    omp_set_num_threads(n_threads);
    srand(42);

    struct { const char *name; int in_dim, out_dim; } tests[] = {
        {"gate    [18944, 3584]", 3584, 18944},
        {"down    [3584, 18944]", 18944, 3584},
    };

    printf("=== Huge Pages test (threads=%d, hugepages=%s) ===\n",
           n_threads, use_hp ? "YES" : "NO");

    for (int t = 0; t < 2; t++) {
        int in_dim = tests[t].in_dim, out_dim = tests[t].out_dim;
        int n_super = in_dim / 256;
        int row_stride = n_super * 144;

        float *x = use_hp ? hp_alloc(in_dim * sizeof(float)) : malloc(in_dim * sizeof(float));
        uint8_t *q4k = use_hp ? hp_alloc((size_t)out_dim * row_stride) : malloc((size_t)out_dim * row_stride);
        float *y = use_hp ? hp_alloc(out_dim * sizeof(float)) : malloc(out_dim * sizeof(float));

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

        int n_iter = 5;
        double mem_read = (double)out_dim * row_stride + in_dim * 4;

        double t0 = now_s();
        for (int it = 0; it < n_iter; it++) {
            if (n_threads > 1) {
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
            } else {
                lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
            }
        }
        double dt = (now_s() - t0) / n_iter;

        printf("%-22s  %.2f ms  %.1f GB/s\n", tests[t].name, dt*1000, mem_read/dt/1e9);

        free(x); free(q4k); free(y);
    }

    /* Full forward estimate */
    printf("\n=== E2E estimate (28 layers, 7 matmuls/layer) ===\n");
    /* gate+up: [18944, 3584] = 38.2MB each
     * down: [3584, 18944] = 38.2MB
     * q+k+v+o: [3584, 3584] + [512, 3584]*2 + [3584, 3584] = 7.2+1.0+1.0+7.2 = 16.4MB
     * Total per layer: 38.2*3 + 16.4 = 131MB
     * × 28 layers = 3.67 GB */
    double total_data = 3.67e9;  /* bytes */
    /* Estimate from single matmul bandwidth */
    /* gate at ~8 GB/s (2 threads) → 131MB / (8e9/131e6*3.67e9/131e6) ... */
    /* Just use the measured gate time × 28 layers × 3 (gate+up+down) + small matmuls */
    /* From the test above, gate ≈ 4.7ms with 2 threads. Total = 4.7 * 28 * 3 + small ≈ 400+ ms */
    printf("Total weight data: %.1f MB\n", total_data / 1e6);

    return 0;
}
