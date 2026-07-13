/* bench_q3k.c — Q3_K 正确性验证 + 性能对比 (简化版)
 * Build: gcc -O3 -march=native -I. -o bench_q3k scripts/bench_q3k.c -lm
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
#include "runtime/lal_q3k_kernel.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    int in_dim = 3584, out_dim = 512;
    int n_super = in_dim / 256;
    srand(42);

    /* Generate random float weights */
    float *w = malloc((size_t)out_dim * in_dim * sizeof(float));
    float *x = malloc(in_dim * sizeof(float));
    float *y_ref = malloc(out_dim * sizeof(float));
    float *y_q3k = malloc(out_dim * sizeof(float));

    for (size_t i = 0; i < (size_t)out_dim * in_dim; i++)
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < in_dim; i++)
        x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;

    /* Reference: float matmul */
    for (int j = 0; j < out_dim; j++) {
        float dot = 0;
        for (int i = 0; i < in_dim; i++)
            dot += w[j * in_dim + i] * x[i];
        y_ref[j] = dot;
    }

    /* Q3_K quantize + matmul */
    int q3k_stride = n_super * Q3K_SUPERBLOCK_BYTES;
    uint8_t *q3k_W = malloc((size_t)out_dim * q3k_stride);
    for (int j = 0; j < out_dim; j++)
        quantize_q3_k_row(w + j * in_dim, q3k_W + j * q3k_stride, in_dim);
    lal_matmul_q3_k(y_q3k, q3k_W, x, NULL, in_dim, out_dim);

    /* Correctness */
    float max_err = 0, sum_err = 0, max_ref = 0;
    for (int j = 0; j < out_dim; j++) {
        float err = fabsf(y_ref[j] - y_q3k[j]);
        if (err > max_err) max_err = err;
        sum_err += err;
        if (fabsf(y_ref[j]) > max_ref) max_ref = fabsf(y_ref[j]);
    }
    printf("=== Q3_K Correctness (in_dim=%d, out_dim=%d) ===\n", in_dim, out_dim);
    printf("Reference max |y|: %.4f\n", max_ref);
    printf("Q3_K max error:    %.6f (%.2f%%)\n", max_err, max_err/max_ref*100);
    printf("Q3_K avg error:    %.6f (%.2f%%)\n", sum_err/out_dim, (sum_err/out_dim)/max_ref*100);

    /* Data size comparison */
    double q4k_data = (double)out_dim * n_super * 144;
    double q3k_data = (double)out_dim * n_super * Q3K_SUPERBLOCK_BYTES;
    printf("\n=== Data Size ===\n");
    printf("Q4_K: %.1f KB (%.4f bytes/elem)\n", q4k_data/1024, q4k_data/(out_dim*in_dim));
    printf("Q3_K: %.1f KB (%.4f bytes/elem)\n", q3k_data/1024, q3k_data/(out_dim*in_dim));
    printf("Reduction: %.1f%%\n", (1 - q3k_data/q4k_data) * 100);

    /* Performance test (gate size) */
    printf("\n=== Performance (gate [18944, 3584]) ===\n");
    int gate_out = 18944;
    int gate_in = 3584;
    int n_super_g = gate_in / 256;

    float *xg = malloc(gate_in * sizeof(float));
    float *yg = malloc(gate_out * sizeof(float));
    uint8_t *q4k_g = malloc((size_t)gate_out * n_super_g * 144);
    uint8_t *q3k_g = malloc((size_t)gate_out * n_super_g * Q3K_SUPERBLOCK_BYTES);

    for (int i = 0; i < gate_in; i++) xg[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
    for (int j = 0; j < gate_out; j++) {
        uint8_t *row4 = q4k_g + (size_t)j * n_super_g * 144;
        for (int s = 0; s < n_super_g; s++) {
            uint8_t *sb = row4 + s*144;
            *(uint16_t*)sb = 0x3C00; *(uint16_t*)(sb+2) = 0x0000;
            memset(sb+4, 0x20, 12);
            for (int i = 0; i < 128; i++) sb[16+i] = rand() & 0xFF;
        }
        uint8_t *row3 = q3k_g + (size_t)j * n_super_g * Q3K_SUPERBLOCK_BYTES;
        for (int s = 0; s < n_super_g; s++) {
            uint8_t *sb = row3 + s*Q3K_SUPERBLOCK_BYTES;
            *(uint16_t*)sb = 0x3C00; *(uint16_t*)(sb+2) = 0x0000;
            memset(sb+4, 0x20, 12);
            for (int i = 0; i < 96; i++) sb[16+i] = rand() & 0xFF;
        }
    }

    int n_iter = 3;
    /* Q4_K */
    lal_matmul_q4_k(yg, q4k_g, xg, NULL, gate_in, gate_out);
    double t0 = now_ms();
    for (int it = 0; it < n_iter; it++) lal_matmul_q4_k(yg, q4k_g, xg, NULL, gate_in, gate_out);
    double dt_q4 = (now_ms() - t0) / n_iter;

    /* Q3_K */
    lal_matmul_q3_k(yg, q3k_g, xg, NULL, gate_in, gate_out);
    t0 = now_ms();
    for (int it = 0; it < n_iter; it++) lal_matmul_q3_k(yg, q3k_g, xg, NULL, gate_in, gate_out);
    double dt_q3 = (now_ms() - t0) / n_iter;

    double q4_mem = (double)gate_out * n_super_g * 144;
    double q3_mem = (double)gate_out * n_super_g * Q3K_SUPERBLOCK_BYTES;
    printf("Q4_K: %.2f ms (%.1f GB/s, %.1f MB data)\n", dt_q4, q4_mem/dt_q4/1e6, q4_mem/1e6);
    printf("Q3_K: %.2f ms (%.1f GB/s, %.1f MB data)\n", dt_q3, q3_mem/dt_q3/1e6, q3_mem/1e6);
    printf("Note: Q3_K kernel is scalar (not yet SIMD optimized)\n");

    free(w); free(x); free(y_ref); free(y_q3k);
    free(q3k_W); free(q4k_g); free(q3k_g);
    free(xg); free(yg);
    return 0;
}
