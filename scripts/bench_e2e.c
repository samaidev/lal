/* bench_e2e.c — 端到端 forward 基准测试
 * 模拟完整 forward pass: matmul + RMSNorm + Attention + SwiGLU + Residual
 * 使用 OpenMP 多线程，测量实际 tok/s
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_e2e scripts/bench_e2e.c -lm -lgomp
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
#define VOCAB_SIZE 152064
#define N_CTX     4096
#define RMS_EPS   1e-6f
#define N_Q_PER_KV (N_HEAD / N_KV_HEAD)
#define XQ_MAX 18944

#include "runtime/lal_q4k_kernel.h"
#include "runtime/lal_simd_optim.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int main(int argc, char **argv) {
    int n_threads = argc > 1 ? atoi(argv[1]) : 2;
    int pos = argc > 2 ? atoi(argv[2]) : 128;
    int n_iter = argc > 3 ? atoi(argv[3]) : 5;
    omp_set_num_threads(n_threads);

    srand(42);

    /* 分配所有 forward 所需 buffers */
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

    /* 生成 Q4_K 权重 (每层 7 个 matmul) */
    int n_super_3584 = N_EMBD / 256;
    int n_super_18944 = MLP_DIM / 256;
    int row_stride_3584 = n_super_3584 * 144;
    int row_stride_18944 = n_super_18944 * 144;

    /* 生成随机 Q4_K 权重 */
    #define GEN_Q4K(name, out_dim, in_dim) \
        uint8_t *name = _mm_malloc((size_t)out_dim * ((in_dim)/256*144), 32); \
        do { \
            float *tmp = malloc((size_t)out_dim*in_dim*sizeof(float)); \
            for (size_t i = 0; i < (size_t)out_dim*in_dim; i++) tmp[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f; \
            for (int j = 0; j < out_dim; j++) { \
                /* 简化量化: 随机填入有效 Q4_K 数据 */ \
                uint8_t *row = name + (size_t)j * ((in_dim)/256*144); \
                for (int s = 0; s < (in_dim)/256; s++) { \
                    uint8_t *sb = row + s*144; \
                    *(uint16_t*)sb = 0x3C00; /* fp16 1.0 */ \
                    *(uint16_t*)(sb+2) = 0x0000; /* fp16 0.0 */ \
                    memset(sb+4, 0x20, 12); /* scales */ \
                    for (int i = 0; i < 128; i++) sb[16+i] = rand() & 0xFF; \
                } \
            } \
            free(tmp); \
        } while(0)

    GEN_Q4K(w_q, Q_DIM, N_EMBD);       /* [3584, 3584] */
    GEN_Q4K(w_k, KV_DIM, N_EMBD);      /* [512, 3584] */
    GEN_Q4K(w_v, KV_DIM, N_EMBD);      /* [512, 3584] */
    GEN_Q4K(w_o, N_EMBD, Q_DIM);       /* [3584, 3584] */
    GEN_Q4K(w_gate, MLP_DIM, N_EMBD);  /* [18944, 3584] */
    GEN_Q4K(w_up, MLP_DIM, N_EMBD);    /* [18944, 3584] */
    GEN_Q4K(w_down, N_EMBD, MLP_DIM);  /* [3584, 18944] */

    /* 初始化输入 */
    for (int i = 0; i < N_EMBD; i++) { x[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; norm_w[i] = 0.9f; }

    printf("=== E2E Forward Benchmark ===\n");
    printf("CPU: %d threads, pos=%d, %d iters\n", n_threads, pos, n_iter);
    printf("Simulating %d layers × (Q/K/V/O proj + Attention + Gate/Up/Down MLP)\n\n", N_LAYER);

    /* Warmup */
    for (int l = 0; l < 2; l++) {
        lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
        lal_matmul_q4_k(q_buf, w_q, ln, NULL, N_EMBD, Q_DIM);
        lal_matmul_q4_k(k_buf, w_k, ln, NULL, N_EMBD, KV_DIM);
        lal_matmul_q4_k(v_buf, w_v, ln, NULL, N_EMBD, KV_DIM);
        lal_gqa_attn_simd(attn_out, q_buf, k_buf, v_buf, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
        lal_matmul_q4_k(proj, w_o, attn_out, NULL, Q_DIM, N_EMBD);
        lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
        lal_matmul_q4_k(gate_buf, w_gate, ln, NULL, N_EMBD, MLP_DIM);
        lal_matmul_q4_k(up_buf, w_up, ln, NULL, N_EMBD, MLP_DIM);
        for (int i = 0; i < MLP_DIM; i++) { float g = gate_buf[i]; act_buf[i] = (g/(1.0f+expf(-g)))*up_buf[i]; }
        lal_matmul_q4_k(mlp_out, w_down, act_buf, NULL, MLP_DIM, N_EMBD);
    }

    /* Benchmark */
    double t0 = now_s();
    for (int it = 0; it < n_iter; it++) {
        for (int l = 0; l < N_LAYER; l++) {
            /* Attention block */
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
            lal_matmul_q4_k(q_buf, w_q, ln, NULL, N_EMBD, Q_DIM);
            lal_matmul_q4_k(k_buf, w_k, ln, NULL, N_EMBD, KV_DIM);
            lal_matmul_q4_k(v_buf, w_v, ln, NULL, N_EMBD, KV_DIM);
            lal_gqa_attn_simd(attn_out, q_buf, k_buf, v_buf, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
            lal_matmul_q4_k(proj, w_o, attn_out, NULL, Q_DIM, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) x[i] += proj[i];
            /* MLP block */
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
            lal_matmul_q4_k(gate_buf, w_gate, ln, NULL, N_EMBD, MLP_DIM);
            lal_matmul_q4_k(up_buf, w_up, ln, NULL, N_EMBD, MLP_DIM);
            for (int i = 0; i < MLP_DIM; i++) { float g = gate_buf[i]; act_buf[i] = (g/(1.0f+expf(-g)))*up_buf[i]; }
            lal_matmul_q4_k(mlp_out, w_down, act_buf, NULL, MLP_DIM, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) x[i] += mlp_out[i];
        }
    }
    double dt = (now_s() - t0) / n_iter;

    /* 计算理论 matmul 数据量 */
    double matmul_bytes = 0;
    /* 每层: q(3584×3584/256×144) + k(512×3584/256×144) + v(512×...) + o(3584×...) + gate(18944×...) + up(18944×...) + down(3584×18944/256×144) */
    double per_layer = (Q_DIM + KV_DIM + KV_DIM + N_EMBD + MLP_DIM + MLP_DIM) * (N_EMBD/256*144)
                     + N_EMBD * (MLP_DIM/256*144);
    matmul_bytes = N_LAYER * per_layer;

    printf("Total forward time: %.2f ms\n", dt * 1000);
    printf("Estimated tok/s: %.2f\n", 1.0 / dt);
    printf("Matmul data read: %.1f MB\n", matmul_bytes / 1e6);
    printf("Effective bandwidth: %.1f GB/s\n", matmul_bytes / dt / 1e9);
    printf("\nBreakdown per layer:\n");
    printf("  7 matmuls: %.2f ms (estimated)\n", per_layer / 6.5e9 * 1000);

    /* 单项 matmul 时间 */
    double t1 = now_s();
    for (int it = 0; it < n_iter; it++) lal_matmul_q4_k(gate_buf, w_gate, ln, NULL, N_EMBD, MLP_DIM);
    double dt_gate = (now_s() - t1) / n_iter;
    printf("\nSingle matmul (gate [18944,3584]): %.2f ms\n", dt_gate * 1000);

    t1 = now_s();
    for (int it = 0; it < n_iter; lal_matmul_q4_k(mlp_out, w_down, act_buf, NULL, MLP_DIM, N_EMBD), it++);
    double dt_down = (now_s() - t1) / n_iter;
    printf("Single matmul (down [3584,18944]): %.2f ms\n", dt_down * 1000);

    t1 = now_s();
    for (int it = 0; it < n_iter * N_LAYER; it++)
        lal_gqa_attn_simd(attn_out, q_buf, k_buf, v_buf, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
    double dt_attn = (now_s() - t1) / (n_iter * N_LAYER);
    printf("Single attention (pos=%d): %.2f ms\n", pos, dt_attn * 1000);

    return 0;
}
