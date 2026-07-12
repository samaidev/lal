/* bench_kernel.c — micro-benchmark LAL matmul kernels
 *
 * Tests:
 *   1. lal_matmul_q8_signtrick (legacy per-row Q8, qtype=1)
 *   2. lal_matmul_q8_0 (new block Q8_0, qtype=3)
 *
 * Reports: time, GB/s memory bandwidth, GFLOPS.
 *
 * Build: gcc -O3 -mavx2 -mfma -I. -o bench_kernel bench_kernel.c
 *        runtime/lal_runtime.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <immintrin.h>

#define XQ_MAX 18944
#include "runtime/lal_runtime.h"
#include "runtime/lal_q8_kernel.h"
#include "runtime/lal_q4_kernel.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* Test legacy per-row Q8 matmul */
static void bench_q8_legacy(const char *name, int in_dim, int out_dim, int n_iter) {
    int8_t *q8_T = memalign(32, (size_t)out_dim * in_dim);
    float   *scale = memalign(32, out_dim * sizeof(float));
    float   *x = memalign(32, in_dim * sizeof(float));
    float   *y = memalign(32, out_dim * sizeof(float));

    for (size_t i = 0; i < (size_t)out_dim * in_dim; i++) q8_T[i] = (int8_t)(rand() & 0x7F);
    for (int i = 0; i < out_dim; i++) scale[i] = 0.01f;
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 100) / 100.0f;

    lal_matmul_q8_signtrick(y, q8_T, scale, x, NULL, in_dim, out_dim);

    double t0 = now_s();
    for (int i = 0; i < n_iter; i++)
        lal_matmul_q8_signtrick(y, q8_T, scale, x, NULL, in_dim, out_dim);
    double t1 = now_s();
    double dt = (t1 - t0) / n_iter;

    double bytes = (double)out_dim * in_dim + out_dim * 4 + in_dim * 4 + out_dim * 4;
    double gb_s = bytes / dt / 1e9;
    double flops = 2.0 * out_dim * in_dim;
    double gflops = flops / dt / 1e9;

    printf("  %-28s [%d, %d] @ [in=%d]: %7.3f ms  %6.1f GB/s  %6.1f GFLOPS\n",
           name, out_dim, in_dim, in_dim, dt*1000, gb_s, gflops);

    free(q8_T); free(scale); free(x); free(y);
}

/* Test new Q8_0 block-format matmul */
static void bench_q8_0(const char *name, int in_dim, int out_dim, int n_iter) {
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 34;
    uint8_t *q8_0_W = memalign(32, (size_t)out_dim * row_stride);
    float   *x = memalign(32, in_dim * sizeof(float));
    float   *y = memalign(32, out_dim * sizeof(float));

    for (int j = 0; j < out_dim; j++) {
        uint8_t *row = q8_0_W + (size_t)j * row_stride;
        for (int b = 0; b < blocks_per_row; b++) {
            uint8_t *block = row + b * 34;
            uint16_t scale_u16 = 0x2C14;
            block[0] = scale_u16 & 0xFF;
            block[1] = (scale_u16 >> 8) & 0xFF;
            for (int i = 0; i < 32; i++) block[2 + i] = (uint8_t)(rand() & 0x7F);
        }
    }
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 100) / 100.0f;

    lal_matmul_q8_0(y, q8_0_W, x, NULL, in_dim, out_dim);

    double t0 = now_s();
    for (int i = 0; i < n_iter; i++)
        lal_matmul_q8_0(y, q8_0_W, x, NULL, in_dim, out_dim);
    double t1 = now_s();
    double dt = (t1 - t0) / n_iter;

    double bytes = (double)out_dim * blocks_per_row * 34 + in_dim * 4 + out_dim * 4;
    double gb_s = bytes / dt / 1e9;
    double flops = 2.0 * out_dim * in_dim;
    double gflops = flops / dt / 1e9;

    printf("  %-28s [%d, %d] @ [in=%d]: %7.3f ms  %6.1f GB/s  %6.1f GFLOPS\n",
           name, out_dim, in_dim, in_dim, dt*1000, gb_s, gflops);

    free(q8_0_W); free(x); free(y);
}

/* Test Q4_0 block-format matmul */
static void bench_q4_0(const char *name, int in_dim, int out_dim, int n_iter) {
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 18;
    uint8_t *q4_W = memalign(32, (size_t)out_dim * row_stride);
    float   *x = memalign(32, in_dim * sizeof(float));
    float   *y = memalign(32, out_dim * sizeof(float));

    for (int j = 0; j < out_dim; j++) {
        uint8_t *row = q4_W + (size_t)j * row_stride;
        for (int b = 0; b < blocks_per_row; b++) {
            uint8_t *block = row + b * 18;
            uint16_t scale_u16 = 0x2C14;
            block[0] = scale_u16 & 0xFF;
            block[1] = (scale_u16 >> 8) & 0xFF;
            for (int i = 0; i < 16; i++) block[2 + i] = (uint8_t)(rand() & 0xFF);
        }
    }
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 100) / 100.0f;

    lal_matmul_q4_0(y, q4_W, x, NULL, in_dim, out_dim);

    double t0 = now_s();
    for (int i = 0; i < n_iter; i++)
        lal_matmul_q4_0(y, q4_W, x, NULL, in_dim, out_dim);
    double t1 = now_s();
    double dt = (t1 - t0) / n_iter;

    /* Q4_0: 18 bytes per 32 elements = 0.5625 bytes/elem */
    double bytes = (double)out_dim * blocks_per_row * 18 + in_dim * 4 + out_dim * 4;
    double gb_s = bytes / dt / 1e9;
    double flops = 2.0 * out_dim * in_dim;
    double gflops = flops / dt / 1e9;

    printf("  %-28s [%d, %d] @ [in=%d]: %7.3f ms  %6.1f GB/s  %6.1f GFLOPS\n",
           name, out_dim, in_dim, in_dim, dt*1000, gb_s, gflops);

    free(q4_W); free(x); free(y);
}

/* Test Q4_0 single-row streaming matmul (llama.cpp approach) */
static void bench_q4_0_streaming(const char *name, int in_dim, int out_dim, int n_iter) {
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 18;
    uint8_t *q4_W = memalign(32, (size_t)out_dim * row_stride);
    float   *x = memalign(32, in_dim * sizeof(float));
    float   *y = memalign(32, out_dim * sizeof(float));

    for (int j = 0; j < out_dim; j++) {
        uint8_t *row = q4_W + (size_t)j * row_stride;
        for (int b = 0; b < blocks_per_row; b++) {
            uint8_t *block = row + b * 18;
            uint16_t scale_u16 = 0x2C14;
            block[0] = scale_u16 & 0xFF;
            block[1] = (scale_u16 >> 8) & 0xFF;
            for (int i = 0; i < 16; i++) block[2 + i] = (uint8_t)(rand() & 0xFF);
        }
    }
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 100) / 100.0f;

    lal_matmul_q4_0_streaming(y, q4_W, x, NULL, in_dim, out_dim);

    double t0 = now_s();
    for (int i = 0; i < n_iter; i++)
        lal_matmul_q4_0_streaming(y, q4_W, x, NULL, in_dim, out_dim);
    double t1 = now_s();
    double dt = (t1 - t0) / n_iter;

    double bytes = (double)out_dim * blocks_per_row * 18 + in_dim * 4 + out_dim * 4;
    double gb_s = bytes / dt / 1e9;
    double flops = 2.0 * out_dim * in_dim;
    double gflops = flops / dt / 1e9;

    printf("  %-28s [%d, %d] @ [in=%d]: %7.3f ms  %6.1f GB/s  %6.1f GFLOPS\n",
           name, out_dim, in_dim, in_dim, dt*1000, gb_s, gflops);

    free(q4_W); free(x); free(y);
}

/* Test Q4_0A cache-line aligned matmul (32-byte blocks) */
static void bench_q4_0a(const char *name, int in_dim, int out_dim, int n_iter) {
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 32;  /* 32 bytes per block (aligned!) */
    uint8_t *q4a_W = memalign(32, (size_t)out_dim * row_stride);
    float   *x = memalign(32, in_dim * sizeof(float));
    float   *y = memalign(32, out_dim * sizeof(float));

    for (int j = 0; j < out_dim; j++) {
        uint8_t *row = q4a_W + (size_t)j * row_stride;
        for (int b = 0; b < blocks_per_row; b++) {
            uint8_t *block = row + b * 32;
            uint16_t scale_u16 = 0x2C14;
            block[0] = scale_u16 & 0xFF;
            block[1] = (scale_u16 >> 8) & 0xFF;
            for (int i = 0; i < 16; i++) block[2 + i] = (uint8_t)(rand() & 0xFF);
            /* bytes 18-31 are padding (ignored by kernel) */
        }
    }
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 100) / 100.0f;

    lal_matmul_q4_0a(y, q4a_W, x, NULL, in_dim, out_dim);

    double t0 = now_s();
    for (int i = 0; i < n_iter; i++)
        lal_matmul_q4_0a(y, q4a_W, x, NULL, in_dim, out_dim);
    double t1 = now_s();
    double dt = (t1 - t0) / n_iter;

    /* Q4_0A: 32 bytes per 32 elements = 1 byte/elem (with padding) */
    double bytes = (double)out_dim * blocks_per_row * 32 + in_dim * 4 + out_dim * 4;
    double gb_s = bytes / dt / 1e9;
    double flops = 2.0 * out_dim * in_dim;
    double gflops = flops / dt / 1e9;

    printf("  %-28s [%d, %d] @ [in=%d]: %7.3f ms  %6.1f GB/s  %6.1f GFLOPS\n",
           name, out_dim, in_dim, in_dim, dt*1000, gb_s, gflops);

    free(q4a_W); free(x); free(y);
}

int main(void) {
    srand(42);
    printf("=== LAL matmul kernel micro-benchmark ===\n\n");

    printf("--- Legacy Q8 (per-row scale, 8-row parallel) ---\n");
    bench_q8_legacy("q_proj",   3584, 3584, 20);
    bench_q8_legacy("k_proj",   3584,  512, 50);
    bench_q8_legacy("o_proj",   3584, 3584, 20);
    bench_q8_legacy("gate_proj",3584,18944, 10);
    bench_q8_legacy("down_proj",18944,3584, 10);
    bench_q8_legacy("lm_head",  3584, 152064, 5);

    printf("\n--- Q4_0 8-row parallel (18B blocks, MISALIGNED) ---\n");
    bench_q4_0("q_proj",   3584, 3584, 20);
    bench_q4_0("k_proj",   3584,  512, 50);
    bench_q4_0("o_proj",   3584, 3584, 20);
    bench_q4_0("gate_proj",3584,18944, 10);
    bench_q4_0("down_proj",18944,3584, 10);
    bench_q4_0("lm_head",  3584, 152064, 5);

    printf("\n--- Q4_0A 8-row parallel (32B blocks, ALIGNED!) ---\n");
    bench_q4_0a("q_proj",   3584, 3584, 20);
    bench_q4_0a("k_proj",   3584,  512, 50);
    bench_q4_0a("o_proj",   3584, 3584, 20);
    bench_q4_0a("gate_proj",3584,18944, 10);
    bench_q4_0a("down_proj",18944,3584, 10);
    bench_q4_0a("lm_head",  3584, 152064, 5);

    printf("\n--- Q8_0 block-format (34B blocks, inline fp16 scale) ---\n");
    bench_q8_0("q_proj",   3584, 3584, 20);
    bench_q8_0("k_proj",   3584,  512, 50);
    bench_q8_0("o_proj",   3584, 3584, 20);
    bench_q8_0("gate_proj",3584,18944, 10);
    bench_q8_0("down_proj",18944,3584, 10);
    bench_q8_0("lm_head",  3584, 152064, 5);

    return 0;
}
