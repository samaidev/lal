/* bench_profile.c — 分项计时 forward pass, 找出真正瓶颈
 * 构建: gcc -O3 -march=native -fopenmp -I. -o bench_profile scripts/bench_profile.c -lm -lgomp
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

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
    int pos = argc > 1 ? atoi(argv[1]) : 128;
    int iters = argc > 2 ? atoi(argv[2]) : 10;

    if (!getenv("OMP_PROC_BIND")) setenv("OMP_PROC_BIND", "spread", 0);
    omp_set_dynamic(0);

    srand(42);

    /* 分配对齐内存 */
    float *x = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *ln = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *norm_w = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *q_buf = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *k_buf = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *v_buf = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *attn_out = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *proj = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *gate_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *up_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *act_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *mlp_out = _mm_malloc(N_EMBD * sizeof(float), 32);

    /* KV cache */
    float *kc = _mm_malloc((size_t)N_CTX * KV_DIM * sizeof(float), 32);
    float *vc = _mm_malloc((size_t)N_CTX * KV_DIM * sizeof(float), 32);

    /* 权重 (Q4_K 格式, 14 superblocks × 144 bytes/row) */
    int n_super = N_EMBD / 256;
    int row_stride = n_super * 144;
    #define ALLOC_W(name, out_d) uint8_t *name = _mm_malloc((size_t)(out_d) * row_stride, 32); \
        memset(name, 0x20, (size_t)(out_d) * row_stride); \
        for (int r=0; r<(out_d); r++) for (int s=0; s<n_super; s++) { \
            uint8_t *sb=name+(size_t)r*row_stride+s*144; \
            *(uint16_t*)sb=0x2E66; *(uint16_t*)(sb+2)=0x25C3; \
            memset(sb+4, 0x20, 12); }

    ALLOC_W(w_q, Q_DIM)
    ALLOC_W(w_k, KV_DIM)
    ALLOC_W(w_v, KV_DIM)
    ALLOC_W(w_o, N_EMBD)
    ALLOC_W(w_gate, MLP_DIM)
    ALLOC_W(w_up, MLP_DIM)
    ALLOC_W(w_down, N_EMBD)

    /* 初始化 x */
    for (int i = 0; i < N_EMBD; i++) x[i] = (float)(rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < N_EMBD; i++) norm_w[i] = 1.0f;

    /* 计时累计 */
    double t_rms = 0, t_qkv = 0, t_attn = 0, t_oproj = 0, t_mlp = 0, t_total = 0;

    printf("=== Forward 分项计时 (pos=%d, %d iters, %d threads) ===\n", pos, iters, g_n_threads);

    for (int it = 0; it < iters; it++) {
        double t0 = now_us();

        /* Pre-attn RMSNorm */
        double ts = now_us();
        for (int l = 0; l < N_LAYER; l++)
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
        t_rms += now_us() - ts;

        /* Q/K/V matmul */
        ts = now_us();
        for (int l = 0; l < N_LAYER; l++) {
            parallel_matmul_q4_k(q_buf, w_q, ln, NULL, N_EMBD, Q_DIM);
            parallel_matmul_q4_k(k_buf, w_k, ln, NULL, N_EMBD, KV_DIM);
            parallel_matmul_q4_k(v_buf, w_v, ln, NULL, N_EMBD, KV_DIM);
        }
        t_qkv += now_us() - ts;

        /* Attention (含 KV cache 写入) */
        ts = now_us();
        for (int l = 0; l < N_LAYER; l++)
            lal_gqa_attn_simd(attn_out, q_buf, k_buf, v_buf, kc, vc, pos,
                              N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
        t_attn += now_us() - ts;

        /* O proj */
        ts = now_us();
        for (int l = 0; l < N_LAYER; l++)
            parallel_matmul_q4_k(proj, w_o, attn_out, NULL, Q_DIM, N_EMBD);
        t_oproj += now_us() - ts;

        /* Pre-MLP RMSNorm + gate/up + SiLU + down (细分) */
        double t_gate=0, t_up=0, t_silu=0, t_down=0, t_rms2=0;
        for (int l = 0; l < N_LAYER; l++) {
            ts = now_us();
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
            t_rms2 += now_us() - ts;
            ts = now_us();
            parallel_matmul_q4_k(gate_buf, w_gate, ln, NULL, N_EMBD, MLP_DIM);
            t_gate += now_us() - ts;
            ts = now_us();
            parallel_matmul_q4_k(up_buf, w_up, ln, NULL, N_EMBD, MLP_DIM);
            t_up += now_us() - ts;
            ts = now_us();
            lal_silu_mul_simd(act_buf, gate_buf, up_buf, MLP_DIM);
            t_silu += now_us() - ts;
            ts = now_us();
            parallel_matmul_q4_k(mlp_out, w_down, act_buf, NULL, MLP_DIM, N_EMBD);
            t_down += now_us() - ts;
        }
        t_mlp += t_gate + t_up + t_silu + t_down + t_rms2;

        t_total += now_us() - t0;
    }

    /* 输出 */
    printf("\n=== 分项占比 (28 layers × %d iters) ===\n", iters);
    printf("  RMSNorm (×56):  %8.2f ms  (%.1f%%)\n", t_rms/1000, t_rms/t_total*100);
    printf("  Q/K/V matmul:   %8.2f ms  (%.1f%%)\n", t_qkv/1000, t_qkv/t_total*100);
    printf("  Attention:      %8.2f ms  (%.1f%%)\n", t_attn/1000, t_attn/t_total*100);
    printf("  O proj:         %8.2f ms  (%.1f%%)\n", t_oproj/1000, t_oproj/t_total*100);
    printf("  MLP 细分:\n");
    printf("    gate matmul:  %8.2f ms  (%.1f%%)\n", 0.0, 0.0);
    printf("    up matmul:    %8.2f ms  (%.1f%%)\n", 0.0, 0.0);
    printf("    SiLU:         %8.2f ms  (%.1f%%)\n", 0.0, 0.0);
    printf("    down matmul:  %8.2f ms  (%.1f%%)\n", 0.0, 0.0);
    printf("  MLP (gate+up+SiLU+down): %8.2f ms  (%.1f%%)\n", t_mlp/1000, t_mlp/t_total*100);
    printf("  ─────────────────────────────────\n");
    printf("  Total:          %8.2f ms  (%.1f%%)\n", t_total/1000, 100.0);
    printf("  Per token:      %8.2f ms\n", t_total/iters/1000);
    printf("  Tok/s:          %8.2f\n", iters / (t_total/1e6));

    return 0;
}
