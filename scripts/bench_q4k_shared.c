/* bench_q4k_shared.c — 对比 gate+up 共享预处理 vs 独立预处理
 * 模拟 fused_swiglu 中 gate 和 up 用同一个 x 的场景
 *
 * 构建: gcc -O3 -march=native -fopenmp -I. -o bench_q4k_shared scripts/bench_q4k_shared.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define XQ_MAX 18944
#define IN_DIM  3584
#define OUT_DIM 18944

#include "runtime/lal_q4k_kernel.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    /* 分配数据 */
    float *x = malloc(IN_DIM * sizeof(float));
    float *y_gate_old = malloc(OUT_DIM * sizeof(float));
    float *y_up_old   = malloc(OUT_DIM * sizeof(float));
    float *y_gate_new = malloc(OUT_DIM * sizeof(float));
    float *y_up_new   = malloc(OUT_DIM * sizeof(float));
    /* Q4_K 权重: 每行 n_super * 144 bytes, IN_DIM/256 = 14 superblocks */
    int n_super = IN_DIM / 256;
    int row_stride = n_super * 144;
    uint8_t *q4k_gate = malloc((size_t)OUT_DIM * row_stride);
    uint8_t *q4k_up   = malloc((size_t)OUT_DIM * row_stride);

    /* 初始化数据 — 用有效的 Q4_K 格式避免 NaN */
    srand(42);
    for (int i = 0; i < IN_DIM; i++) x[i] = (float)(rand() % 200 - 100) / 50.0f;
    for (size_t i = 0; i < (size_t)OUT_DIM * row_stride; i++) {
        q4k_gate[i] = rand() & 0xFF;
        q4k_up[i]   = rand() & 0xFF;
    }
    /* 修正 d/dmin 为合理的 fp16 值 (每行前 4 字节 = 2 个 fp16)
     * 设置 d=0.1, dmin=0.01 避免溢出 */
    for (int r = 0; r < OUT_DIM; r++) {
        for (int s = 0; s < n_super; s++) {
            uint8_t *sb_gate = q4k_gate + (size_t)r * row_stride + s * 144;
            uint8_t *sb_up   = q4k_up   + (size_t)r * row_stride + s * 144;
            /* fp16 0.1 = 0x2E66, fp16 0.01 = 0x25C3 */
            *(uint16_t*)sb_gate = 0x2E66;       /* d = 0.1 */
            *(uint16_t*)(sb_gate + 2) = 0x25C3; /* dmin = 0.01 */
            *(uint16_t*)sb_up = 0x2E66;
            *(uint16_t*)(sb_up + 2) = 0x25C3;
        }
    }

    const int ITERS = 20;
    volatile float sink = 0;

    printf("=== Q4_K gate+up: 独立预处理 vs 共享预处理 ===\n");
    printf("IN_DIM=%d OUT_DIM=%d ITERS=%d\n\n", IN_DIM, OUT_DIM, ITERS);

    /* Baseline: gate 和 up 各自独立预处理 (当前 lal_matmul_q4_k) */
    {
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            lal_matmul_q4_k(y_gate_old, q4k_gate, x, NULL, IN_DIM, OUT_DIM);
            lal_matmul_q4_k(y_up_old,   q4k_up,   x, NULL, IN_DIM, OUT_DIM);
            sink += y_gate_old[it % OUT_DIM] + y_up_old[it % OUT_DIM];
        }
        double t = (now_ms() - t0) / ITERS;
        printf("独立预处理 (baseline):  %.2f ms/iter (gate+up)\n", t);
    }

    /* 优化版: 共享预处理 (lal_q4k_prepare_x + lal_matmul_q4_k_prepared) */
    {
        static int8_t xq[XQ_MAX] __attribute__((aligned(32)));
        static int16_t bsums[XQ_MAX / 32] __attribute__((aligned(32)));
        static int8_t xq_arr[XQ_MAX] __attribute__((aligned(32)));

        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            float x_scale = lal_q4k_prepare_x(x, IN_DIM, xq, bsums, xq_arr);
            lal_matmul_q4_k_prepared(y_gate_new, q4k_gate, x, NULL, IN_DIM, OUT_DIM,
                                      xq, bsums, xq_arr, x_scale);
            lal_matmul_q4_k_prepared(y_up_new, q4k_up, x, NULL, IN_DIM, OUT_DIM,
                                      xq, bsums, xq_arr, x_scale);
            sink += y_gate_new[it % OUT_DIM] + y_up_new[it % OUT_DIM];
        }
        double t = (now_ms() - t0) / ITERS;
        printf("共享预处理 (optimized): %.2f ms/iter (gate+up)\n", t);
    }

    /* 正确性验证 */
    printf("\n=== 正确性验证 ===\n");
    lal_matmul_q4_k(y_gate_old, q4k_gate, x, NULL, IN_DIM, OUT_DIM);
    lal_matmul_q4_k(y_up_old,   q4k_up,   x, NULL, IN_DIM, OUT_DIM);
    static int8_t xq[XQ_MAX] __attribute__((aligned(32)));
    static int16_t bsums[XQ_MAX / 32] __attribute__((aligned(32)));
    static int8_t xq_arr[XQ_MAX] __attribute__((aligned(32)));
    float x_scale = lal_q4k_prepare_x(x, IN_DIM, xq, bsums, xq_arr);
    lal_matmul_q4_k_prepared(y_gate_new, q4k_gate, x, NULL, IN_DIM, OUT_DIM,
                              xq, bsums, xq_arr, x_scale);
    lal_matmul_q4_k_prepared(y_up_new, q4k_up, x, NULL, IN_DIM, OUT_DIM,
                              xq, bsums, xq_arr, x_scale);

    double max_err_gate = 0, max_err_up = 0;
    double sum_abs_gate = 0, sum_abs_up = 0;
    for (int i = 0; i < OUT_DIM; i++) {
        double eg = fabs(y_gate_old[i] - y_gate_new[i]);
        double eu = fabs(y_up_old[i] - y_up_new[i]);
        if (eg > max_err_gate) max_err_gate = eg;
        if (eu > max_err_up) max_err_up = eu;
        sum_abs_gate += fabs(y_gate_old[i]);
        sum_abs_up   += fabs(y_up_old[i]);
    }
    printf("gate: max_err=%.6e avg_abs=%.4f rel_err=%.4f%%\n",
           max_err_gate, sum_abs_gate/OUT_DIM, max_err_gate/(sum_abs_gate/OUT_DIM)*100);
    printf("up:   max_err=%.6e avg_abs=%.4f rel_err=%.4f%%\n",
           max_err_up, sum_abs_up/OUT_DIM, max_err_up/(sum_abs_up/OUT_DIM)*100);

    if (sink < -1e30) printf("impossible %f\n", (double)sink);

    free(x); free(y_gate_old); free(y_up_old); free(y_gate_new); free(y_up_new);
    free(q4k_gate); free(q4k_up);
    return 0;
}
