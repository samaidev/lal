/* bench_e2e_mt.c — 端到端 forward 基准测试 (多线程 OpenMP 版)
 * 使用 parallel_matmul_q4_k 模拟真实 server 的多线程行为
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_e2e_mt scripts/bench_e2e_mt.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define N_EMBD    3584
#define N_LAYER   28
#define N_HEAD    28
#define N_KV_HEAD 4
#define HEAD_DIM  128
#define KV_DIM    (N_KV_HEAD * HEAD_DIM)
#define Q_DIM     (N_HEAD * HEAD_DIM)
#define MLP_DIM   18944
#define N_CTX     4096
#define RMS_EPS   1e-6f
#define N_Q_PER_KV (N_HEAD / N_KV_HEAD)
#define XQ_MAX 18944

#include "runtime/lal_q4k_kernel.h"
#include "runtime/lal_simd_optim.h"

static int g_n_threads = 2;

/* Parallel Q4_K matmul — 从 qwen7b_server.c 复制 */
static inline void parallel_matmul_q4_k(float *y, const uint8_t *q4k_W,
                                         const float *x, const float *b,
                                         int in_dim, int out_dim) {
    if (g_n_threads <= 1 || out_dim < 2048) {
        lal_matmul_q4_k(y, q4k_W, x, b, in_dim, out_dim);
        return;
    }
    int n_super = in_dim / 256;
    int row_stride = n_super * 144;
    #pragma omp parallel num_threads(g_n_threads)
    {
        int tid = omp_get_thread_num();
        int n   = omp_get_num_threads();
        int chunk = (out_dim + n - 1) / n;
        int start = tid * chunk;
        int end = start + chunk;
        if (end > out_dim) end = out_dim;
        if (start < out_dim) {
            lal_matmul_q4_k(y + start, q4k_W + (size_t)start * row_stride,
                            x, b ? b + start : NULL, in_dim, end - start);
        }
    }
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

#define GEN_Q4K(name, out_dim, in_dim) \
    uint8_t *name = _mm_malloc((size_t)out_dim * ((in_dim)/256*144), 32); \
    do { \
        for (int j = 0; j < out_dim; j++) { \
            uint8_t *row = name + (size_t)j * ((in_dim)/256*144); \
            for (int s = 0; s < (in_dim)/256; s++) { \
                uint8_t *sb = row + s*144; \
                *(uint16_t*)sb = 0x3C00; \
                *(uint16_t*)(sb+2) = 0x0000; \
                memset(sb+4, 0x20, 12); \
                for (int i = 0; i < 128; i++) sb[16+i] = rand() & 0xFF; \
            } \
        } \
    } while(0)

int main(int argc, char **argv) {
    g_n_threads = argc > 1 ? atoi(argv[1]) : 2;
    int pos = argc > 2 ? atoi(argv[2]) : 128;
    int n_iter = argc > 3 ? atoi(argv[3]) : 3;
    omp_set_num_threads(g_n_threads);
    srand(42);

    float *x   = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *ln  = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *norm_w = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *q_buf = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *k_buf = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *v_buf = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *attn_out = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *proj = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *gate_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *up_buf   = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *act_buf  = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *mlp_out  = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *kc = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *vc = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);

    GEN_Q4K(w_q, Q_DIM, N_EMBD);
    GEN_Q4K(w_k, KV_DIM, N_EMBD);
    GEN_Q4K(w_v, KV_DIM, N_EMBD);
    GEN_Q4K(w_o, N_EMBD, Q_DIM);
    GEN_Q4K(w_gate, MLP_DIM, N_EMBD);
    GEN_Q4K(w_up, MLP_DIM, N_EMBD);
    GEN_Q4K(w_down, N_EMBD, MLP_DIM);

    for (int i = 0; i < N_EMBD; i++) { x[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; norm_w[i] = 0.9f; }

    printf("=== E2E Forward (OpenMP %d threads, pos=%d) ===\n", g_n_threads, pos);

    /* Warmup */
    for (int l = 0; l < 1; l++) {
        lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
        parallel_matmul_q4_k(q_buf, w_q, ln, NULL, N_EMBD, Q_DIM);
        parallel_matmul_q4_k(k_buf, w_k, ln, NULL, N_EMBD, KV_DIM);
        parallel_matmul_q4_k(v_buf, w_v, ln, NULL, N_EMBD, KV_DIM);
        lal_gqa_attn_simd(attn_out, q_buf, k_buf, v_buf, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
        parallel_matmul_q4_k(proj, w_o, attn_out, NULL, Q_DIM, N_EMBD);
        lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
        parallel_matmul_q4_k(gate_buf, w_gate, ln, NULL, N_EMBD, MLP_DIM);
        parallel_matmul_q4_k(up_buf, w_up, ln, NULL, N_EMBD, MLP_DIM);
        for (int i = 0; i < MLP_DIM; i++) { float g = gate_buf[i]; act_buf[i] = (g/(1.0f+expf(-g)))*up_buf[i]; }
        parallel_matmul_q4_k(mlp_out, w_down, act_buf, NULL, MLP_DIM, N_EMBD);
    }

    /* Benchmark */
    double t0 = now_s();
    for (int it = 0; it < n_iter; it++) {
        for (int l = 0; l < N_LAYER; l++) {
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
            parallel_matmul_q4_k(q_buf, w_q, ln, NULL, N_EMBD, Q_DIM);
            parallel_matmul_q4_k(k_buf, w_k, ln, NULL, N_EMBD, KV_DIM);
            parallel_matmul_q4_k(v_buf, w_v, ln, NULL, N_EMBD, KV_DIM);
            lal_gqa_attn_simd(attn_out, q_buf, k_buf, v_buf, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
            parallel_matmul_q4_k(proj, w_o, attn_out, NULL, Q_DIM, N_EMBD);
            lal_residual_add_simd(x, proj, N_EMBD);
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
            parallel_matmul_q4_k(gate_buf, w_gate, ln, NULL, N_EMBD, MLP_DIM);
            parallel_matmul_q4_k(up_buf, w_up, ln, NULL, N_EMBD, MLP_DIM);
            for (int i = 0; i < MLP_DIM; i++) { float g = gate_buf[i]; act_buf[i] = (g/(1.0f+expf(-g)))*up_buf[i]; }
            parallel_matmul_q4_k(mlp_out, w_down, act_buf, NULL, MLP_DIM, N_EMBD);
            lal_residual_add_simd(x, mlp_out, N_EMBD);
        }
    }
    double dt = (now_s() - t0) / n_iter;

    double per_layer = (Q_DIM + KV_DIM + KV_DIM + N_EMBD + MLP_DIM + MLP_DIM) * (N_EMBD/256*144) + N_EMBD * (MLP_DIM/256*144);
    double matmul_bytes = N_LAYER * per_layer;

    printf("Total: %.2f ms/token  (%.2f tok/s)\n", dt*1000, 1.0/dt);
    printf("BW: %.1f GB/s  (data: %.0f MB)\n", matmul_bytes/dt/1e9, matmul_bytes/1e6);

    return 0;
}
