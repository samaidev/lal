/* bench_avx512_vs_avx2_q4k.c — 对比 AVX-512 vs AVX2 Q4_K kernel
 * 构建: gcc -O3 -march=native -mavx512f -mavx512bw -DLAL_FORCE_AVX512 -fopenmp -I. -o bench_avx512_q4k scripts/bench_avx512_vs_avx2_q4k.c -lm -lgomp
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

/* 先包含 AVX2 kernel */
#include "runtime/lal_q4k_kernel.h"

/* 强制启用 AVX-512 */
#define LAL_FORCE_AVX512
#if defined(__AVX512BW__) && defined(__AVX512F__)
  #define LAL_HAVE_AVX512 1
  #include "runtime/lal_q4k_kernel_avx512.h"
#endif

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    float *x = _mm_malloc(IN_DIM * sizeof(float), 64);
    float *y_avx2 = _mm_malloc(OUT_DIM * sizeof(float), 64);
    float *y_avx512 = _mm_malloc(OUT_DIM * sizeof(float), 64);
    int n_super = IN_DIM / 256;
    int row_stride = n_super * 144;
    uint8_t *w = _mm_malloc((size_t)OUT_DIM * row_stride, 64);

    srand(42);
    for (int i = 0; i < IN_DIM; i++) x[i] = (float)(rand() % 200 - 100) / 50.0f;
    for (size_t i = 0; i < (size_t)OUT_DIM * row_stride; i++) w[i] = rand() & 0xFF;
    /* 设置合理的 d/dmin */
    for (int r = 0; r < OUT_DIM; r++)
        for (int s = 0; s < n_super; s++) {
            uint8_t *sb = w + (size_t)r * row_stride + s * 144;
            *(uint16_t*)sb = 0x2E66; *(uint16_t*)(sb+2) = 0x25C3;
            memset(sb+4, 0x20, 12);
        }

    const int ITERS = 20;
    volatile float sink = 0;

    printf("=== Q4_K AVX2 vs AVX-512 (IN=%d OUT=%d, %d iters) ===\n", IN_DIM, OUT_DIM, ITERS);

    /* AVX2 */
    {
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            lal_matmul_q4_k(y_avx2, w, x, NULL, IN_DIM, OUT_DIM);
            sink += y_avx2[it % OUT_DIM];
        }
        printf("AVX2:     %.2f ms/iter\n", (now_ms() - t0) / ITERS);
    }

    /* AVX-512 */
#ifdef LAL_HAVE_AVX512
    {
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            lal_matmul_q4_k_avx512(y_avx512, w, x, NULL, IN_DIM, OUT_DIM);
            sink += y_avx512[it % OUT_DIM];
        }
        printf("AVX-512:  %.2f ms/iter\n", (now_ms() - t0) / ITERS);
    }

    /* 正确性验证 */
    lal_matmul_q4_k(y_avx2, w, x, NULL, IN_DIM, OUT_DIM);
    lal_matmul_q4_k_avx512(y_avx512, w, x, NULL, IN_DIM, OUT_DIM);
    double max_err = 0, sum_abs = 0;
    for (int i = 0; i < OUT_DIM; i++) {
        double err = fabs(y_avx2[i] - y_avx512[i]);
        if (err > max_err) max_err = err;
        sum_abs += fabs(y_avx2[i]);
    }
    printf("\n正确性: max_err=%.6e avg_abs=%.4f rel_err=%.4f%%\n",
           max_err, sum_abs/OUT_DIM, max_err/(sum_abs/OUT_DIM)*100);
#else
    printf("AVX-512 not available\n");
#endif

    if (sink < -1e30) printf("impossible\n");
    return 0;
}
