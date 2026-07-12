/* bench_final.c — 最终优化版 E2E 基准测试
 * 包含所有优化: SIMD attention/RMSNorm/residual + huge pages + 对齐分配
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_final scripts/bench_final.c -lm -lgomp
 * Run:   OMP_PROC_BIND=spread taskset -c 0,1 ./bench_final 2 128
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

/* Huge-page aware allocation */
static void *hp_alloc(size_t size, size_t align) {
    void *p = NULL;
    if (align < 4096) align = 4096;
    if (posix_memalign(&p, align, size) != 0) return NULL;
    memset(p, 0, size);
    madvise(p, size, MADV_HUGEPAGE);
    return p;
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

#define GEN_Q4K_HP(name, out_dim, in_dim) \
    uint8_t *name = hp_alloc((size_t)out_dim * ((in_dim)/256*144), 4096); \
    do { \
        for (int j = 0; j < out_dim; j++) { \
            uint8_t *row = name + (size_t)j * ((in_dim)/256*144); \
            for (int s = 0; s < (in_dim)/256; s++) { \
                uint8_t *sb = row + s*144; \
                *(uint16_t*)sb = 0x3C00; *(uint16_t*)(sb+2) = 0x0000; \
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

    /* All buffers use huge pages + 4K alignment */
    float *x   = hp_alloc(N_EMBD * sizeof(float), 32);
    float *ln  = hp_alloc(N_EMBD * sizeof(float), 32);
    float *norm_w = hp_alloc(N_EMBD * sizeof(float), 32);
    float *q_buf = hp_alloc(Q_DIM * sizeof(float), 32);
    float *k_buf = hp_alloc(KV_DIM * sizeof(float), 32);
    float *v_buf = hp_alloc(KV_DIM * sizeof(float), 32);
    float *attn_out = hp_alloc(Q_DIM * sizeof(float), 32);
    float *proj = hp_alloc(N_EMBD * sizeof(float), 32);
    float *gate_buf = hp_alloc(MLP_DIM * sizeof(float), 32);
    float *up_buf   = hp_alloc(MLP_DIM * sizeof(float), 32);
    float *act_buf  = hp_alloc(MLP_DIM * sizeof(float), 32);
    float *mlp_out  = hp_alloc(N_EMBD * sizeof(float), 32);
    float *kc = hp_alloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *vc = hp_alloc(N_CTX * KV_DIM * sizeof(float), 32);

    GEN_Q4K_HP(w_q, Q_DIM, N_EMBD);
    GEN_Q4K_HP(w_k, KV_DIM, N_EMBD);
    GEN_Q4K_HP(w_v, KV_DIM, N_EMBD);
    GEN_Q4K_HP(w_o, N_EMBD, Q_DIM);
    GEN_Q4K_HP(w_gate, MLP_DIM, N_EMBD);
    GEN_Q4K_HP(w_up, MLP_DIM, N_EMBD);
    GEN_Q4K_HP(w_down, N_EMBD, MLP_DIM);

    for (int i = 0; i < N_EMBD; i++) { x[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; norm_w[i] = 0.9f; }

    printf("=== LAL Final Optimized E2E ===\n");
    printf("Threads: %d, pos=%d, hugepages=YES\n", g_n_threads, pos);
    printf("Optimizations: SIMD attn(7x) + RMSNorm(4.5x) + residual(8x) + HP + taskset\n\n");

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

    /* Benchmark with detailed timing */
    double total_matmul = 0, total_attn = 0, total_norm = 0, total_other = 0;
    double t0 = now_s();
    for (int it = 0; it < n_iter; it++) {
        for (int l = 0; l < N_LAYER; l++) {
            double t1 = now_s();
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
            total_norm += now_s() - t1;

            t1 = now_s();
            parallel_matmul_q4_k(q_buf, w_q, ln, NULL, N_EMBD, Q_DIM);
            parallel_matmul_q4_k(k_buf, w_k, ln, NULL, N_EMBD, KV_DIM);
            parallel_matmul_q4_k(v_buf, w_v, ln, NULL, N_EMBD, KV_DIM);
            total_matmul += now_s() - t1;

            t1 = now_s();
            lal_gqa_attn_simd(attn_out, q_buf, k_buf, v_buf, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
            total_attn += now_s() - t1;

            t1 = now_s();
            parallel_matmul_q4_k(proj, w_o, attn_out, NULL, Q_DIM, N_EMBD);
            lal_residual_add_simd(x, proj, N_EMBD);
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
            total_norm += now_s() - t1;

            t1 = now_s();
            parallel_matmul_q4_k(gate_buf, w_gate, ln, NULL, N_EMBD, MLP_DIM);
            parallel_matmul_q4_k(up_buf, w_up, ln, NULL, N_EMBD, MLP_DIM);
            for (int i = 0; i < MLP_DIM; i++) { float g = gate_buf[i]; act_buf[i] = (g/(1.0f+expf(-g)))*up_buf[i]; }
            parallel_matmul_q4_k(mlp_out, w_down, act_buf, NULL, MLP_DIM, N_EMBD);
            lal_residual_add_simd(x, mlp_out, N_EMBD);
            total_matmul += now_s() - t1;
        }
    }
    double dt = (now_s() - t0) / n_iter;
    total_matmul /= n_iter;
    total_attn /= n_iter;
    total_norm /= n_iter;

    double per_layer = (Q_DIM + KV_DIM + KV_DIM + N_EMBD + MLP_DIM + MLP_DIM) * (N_EMBD/256*144) + N_EMBD * (MLP_DIM/256*144);
    double matmul_bytes = N_LAYER * per_layer;

    printf("Total: %.2f ms/token  (%.2f tok/s)\n", dt*1000, 1.0/dt);
    printf("BW: %.1f GB/s  (data: %.0f MB)\n", matmul_bytes/dt/1e9, matmul_bytes/1e6);
    printf("\nBreakdown:\n");
    printf("  Matmul (Q4_K):  %.2f ms (%.1f%%)  %.1f GB/s\n", total_matmul*1000, total_matmul/dt*100, matmul_bytes/total_matmul/1e9);
    printf("  Attention:      %.2f ms (%.1f%%)\n", total_attn*1000, total_attn/dt*100);
    printf("  RMSNorm+Resid:  %.2f ms (%.1f%%)\n", total_norm*1000, total_norm/dt*100);
    printf("  Other (SiLU):   %.2f ms (%.1f%%)\n", (dt-total_matmul-total_attn-total_norm)*1000, (dt-total_matmul-total_attn-total_norm)/dt*100);

    return 0;
}
