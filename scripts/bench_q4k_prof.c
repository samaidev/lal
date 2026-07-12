/* bench_q4k_prof.c — Profile Q4_K: per-layer vs lm_head time
 * Build: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -I. -o bench_q4k_prof scripts/bench_q4k_prof.c runtime/lal_runtime.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <immintrin.h>
#include <omp.h>

#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel.h"
#include "runtime/lal_q8_kernel.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* Quantize w to Q8 per-row (matches lal_quantize_q8_per_row) */
static void quantize_q8_per_row(const float *w, int8_t *q, float *s, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        const float *wr = w + j*in_dim;
        float amax = 0;
        for (int i = 0; i < in_dim; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float scale = amax / 127.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        s[j] = scale;
        float inv = 1.0f / scale;
        for (int i = 0; i < in_dim; i++) {
            int v = (int)lroundf(wr[i] * inv);
            q[j*in_dim + i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    }
}

/* Quantize x to int8, return scale */
static float quantize_x_int8(const float *x, int8_t *xq, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    float scale = amax / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)lroundf(x[i] * inv);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }
    return scale;
}

int main() {
    /* Simulate one forward pass: 28 layers × 7 matmuls + 1 lm_head */
    srand(42);
    int n_embd = 3584, n_layer = 28;
    int vocab_real = 152064;  /* real vocab for time extrapolation */
    int vocab = 4096;         /* small vocab to fit in memory (extrapolate) */
    int q_dim = 512, kv_dim = 512, hid = 18944;

    /* Allocate x and quantized x */
    float *x = memalign(32, n_embd * sizeof(float));
    int8_t *xq = memalign(32, n_embd);
    for (int i = 0; i < n_embd; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
    float scale_x = quantize_x_int8(x, xq, n_embd);

    /* Allocate Q4_K weights for each layer matmul */
    struct { int in_dim, out_dim; const char *name; } mats[] = {
        {n_embd, q_dim,  "q_proj"},
        {n_embd, kv_dim, "k_proj"},
        {n_embd, kv_dim, "v_proj"},
        {q_dim,  n_embd, "o_proj"},
        {n_embd, hid,    "gate"},
        {n_embd, hid,    "up"},
        {hid,    n_embd, "down"},
    };
    int n_mats = 7;

    /* Allocate and fill Q4_K weights — make valid blocks to avoid NaN */
    uint8_t *q4k_w[7];
    for (int m = 0; m < n_mats; m++) {
        int in_dim = mats[m].in_dim, out_dim = mats[m].out_dim;
        int n_super = in_dim / 256;
        int row_stride = n_super * 144;
        size_t total = (size_t)out_dim * row_stride;
        printf("[alloc] %s: %zu MB\n", mats[m].name, total / (1024*1024)); fflush(stdout);
        q4k_w[m] = memalign(32, total);
        if (!q4k_w[m]) { printf("OOM %s\n", mats[m].name); return 1; }
        /* Fill each 144-byte block with valid Q4_K data */
        for (int j = 0; j < out_dim; j++) {
            uint8_t *row = q4k_w[m] + (size_t)j * row_stride;
            for (int s = 0; s < n_super; s++) {
                uint8_t *sb = row + s * 144;
                *(uint16_t*)(sb) = 0x3C00;     /* d = 1.0 */
                *(uint16_t*)(sb+2) = 0x0000;   /* dmin = 0.0 */
                memset(sb+4, 0, 12);
                uint64_t *sl = (uint64_t*)(sb+4);
                uint32_t *sh = (uint32_t*)(sb+12);
                *sl = 0x4210842108421ULL;
                *sh = 0x10842;
                memset(sb+16, 0x88, 128);
            }
        }
    }

    /* LM head: int8 [vocab, n_embd] — 545MB, use malloc */
    printf("[alloc] lm_head: %zu MB\n", (size_t)vocab * n_embd / (1024*1024)); fflush(stdout);
    int8_t *lm_head_q = memalign(32, (size_t)vocab * n_embd);
    float *lm_head_s = memalign(32, vocab * sizeof(float));
    if (!lm_head_q) { printf("OOM lm_head\n"); return 1; }
    for (size_t i = 0; i < (size_t)vocab * n_embd; i++)
        lm_head_q[i] = (int8_t)(rand() & 0xFF);
    for (int i = 0; i < vocab; i++) lm_head_s[i] = 0.001f;

    /* Output buffers */
    float *y = memalign(32, hid * sizeof(float));  /* max out_dim */

    /* Warmup */
    for (int m = 0; m < n_mats; m++) {
        lal_matmul_q4_k(y, q4k_w[m], x, NULL, mats[m].in_dim, mats[m].out_dim);
    }

    /* Profile: per-matmul time × 28 layers */
    printf("=== Per-matmul time (1 thread, averaged over 5 runs) ===\n");
    double layer_total = 0;
    for (int m = 0; m < n_mats; m++) {
        int in_dim = mats[m].in_dim, out_dim = mats[m].out_dim;
        int n_iter = 5;
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++)
            lal_matmul_q4_k(y, q4k_w[m], x, NULL, in_dim, out_dim);
        double dt = (now_s() - t0) / n_iter;
        int n_super = in_dim / 256;
        double mem_read = (double)out_dim * n_super * 144 + in_dim * 4;
        printf("  %-10s [%d, %d]:  %.2f ms  (%.1f GB/s)\n",
               mats[m].name, out_dim, in_dim, dt*1000, mem_read/dt/1e9);
        layer_total += dt;
    }
    printf("  Total per layer: %.2f ms\n", layer_total * 1000);
    printf("  Total 28 layers: %.2f ms\n", layer_total * 28 * 1000);

    /* Profile lm_head (1 thread) — extrapolate to real vocab */
    {
        int n_iter = 5;
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++)
            lal_lm_head_int8_range(y, xq, scale_x, lm_head_q, lm_head_s, 0, vocab, n_embd);
        double dt = (now_s() - t0) / n_iter;
        double mem_read = (double)vocab * n_embd + n_embd * 4;
        printf("\n  lm_head     [%d, %d]:  %.2f ms  (%.1f GB/s)\n",
               vocab, n_embd, dt*1000, mem_read/dt/1e9);
        printf("  Extrapolated to vocab=%d: %.2f ms\n", vocab_real, dt*1000 * vocab_real / vocab);
    }
    /* Profile lm_head (2 threads) */
    {
        int n_iter = 5;
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++) {
            #pragma omp parallel num_threads(2)
            {
                int tid = omp_get_thread_num();
                int v_per = (vocab + 1) / 2;
                int vs = tid * v_per, ve = vs + v_per;
                if (ve > vocab) ve = vocab;
                lal_lm_head_int8_range(y, xq, scale_x, lm_head_q, lm_head_s, vs, ve, n_embd);
            }
        }
        double dt = (now_s() - t0) / n_iter;
        printf("  lm_head (2 threads, vocab=%d): %.2f ms  → extrapolated: %.2f ms\n",
               vocab, dt*1000, dt*1000 * vocab_real / vocab);
    }

    printf("\n=== Total time per token (1 thread) ===\n");
    double total = layer_total * 28 + 0;  /* + lm_head 1-thread */
    printf("  Layers: %.2f ms\n", layer_total * 28 * 1000);

    free(x); free(xq);
    for (int m = 0; m < n_mats; m++) free(q4k_w[m]);
    free(lm_head_q); free(lm_head_s); free(y);
    return 0;
}
