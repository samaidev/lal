/* bench_forward.c — 模拟完整 forward pass 的性能对比
 * 测量标量版 vs SIMD 版在真实 forward 场景下的速度差异
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_forward scripts/bench_forward.c -lm -lgomp
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

#include "runtime/lal_simd_optim.h"

/* 标量实现 */
static void rms_norm_scalar(float *out, const float *x, const float *w, int n) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * ms * w[i];
}

static void gqa_attn_scalar(float *out, const float *Q, const float *Kn, const float *Vn,
                            float *k_cache, float *v_cache, int pos) {
    memcpy(k_cache + pos * KV_DIM, Kn, KV_DIM * sizeof(float));
    memcpy(v_cache + pos * KV_DIM, Vn, KV_DIM * sizeof(float));
    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    static float scores[N_CTX];
    for (int h = 0; h < N_HEAD; h++) {
        const float *qh = Q + h * HEAD_DIM;
        int kvh = h / N_Q_PER_KV;
        float max_score = -1e30f;
        for (int t = 0; t <= pos; t++) {
            const float *kt = k_cache + t * KV_DIM + kvh * HEAD_DIM;
            float dot = 0;
            for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * kt[d];
            scores[t] = dot * inv_sqrt;
            if (scores[t] > max_score) max_score = scores[t];
        }
        float sum = 0;
        for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t]-max_score); sum += scores[t]; }
        float *oh = out + h * HEAD_DIM;
        memset(oh, 0, HEAD_DIM * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float w = scores[t] / sum;
            const float *vt = v_cache + t * KV_DIM + kvh * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) oh[d] += w * vt[d];
        }
    }
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int main() {
    /* 模拟一次 forward 中的非 matmul 部分：RMSNorm + RoPE + Attention × 28 layers */
    int positions[] = {32, 128, 512, 1024, 2048};
    int n_pos = sizeof(positions)/sizeof(positions[0]);

    float *x = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *w = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *ln = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *Q = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *K = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *V = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *attn_out = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *kc = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *vc = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *rope_cos = _mm_malloc(HEAD_DIM/2 * sizeof(float), 32);
    float *rope_sin = _mm_malloc(HEAD_DIM/2 * sizeof(float), 32);

    for (int i = 0; i < N_EMBD; i++) { x[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; w[i] = 0.9f + (rand()/((float)RAND_MAX)-0.5f)*0.2f; }
    for (int i = 0; i < Q_DIM; i++) Q[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f;
    for (int i = 0; i < KV_DIM; i++) { K[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; V[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; }
    for (int i = 0; i < HEAD_DIM/2; i++) { rope_cos[i] = cosf(i*0.01f); rope_sin[i] = sinf(i*0.01f); }

    printf("=== Forward 非 matmul 部分性能对比 (28 layers) ===\n");
    printf("%-10s  %-12s  %-12s  %-8s  %-10s\n", "pos", "scalar(ms)", "SIMD(ms)", "speedup", "saved(ms)");

    for (int pi = 0; pi < n_pos; pi++) {
        int pos = positions[pi];
        int n_iter = pos <= 128 ? 20 : (pos <= 512 ? 5 : 2);

        /* 标量版 forward (RMSNorm×3 + RoPE + Attention) × 28 layers */
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++) {
            for (int l = 0; l < N_LAYER; l++) {
                rms_norm_scalar(ln, x, w, N_EMBD);       /* pre-attn */
                /* 假设 Q/K/V 已经算好 */
                gqa_attn_scalar(attn_out, Q, K, V, kc, vc, pos);
                rms_norm_scalar(ln, x, w, N_EMBD);       /* pre-MLP */
            }
            rms_norm_scalar(ln, x, w, N_EMBD);  /* final */
        }
        double dt_scalar = (now_s() - t0) / n_iter;

        /* SIMD 版 forward */
        t0 = now_s();
        for (int it = 0; it < n_iter; it++) {
            for (int l = 0; l < N_LAYER; l++) {
                lal_rms_norm_simd(ln, x, w, N_EMBD, RMS_EPS);
                lal_gqa_attn_simd(attn_out, Q, K, V, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
                lal_rms_norm_simd(ln, x, w, N_EMBD, RMS_EPS);
            }
            lal_rms_norm_simd(ln, x, w, N_EMBD, RMS_EPS);
        }
        double dt_simd = (now_s() - t0) / n_iter;

        double speedup = dt_scalar / dt_simd;
        double saved_ms = (dt_scalar - dt_simd) * 1000;
        printf("%-10d  %-12.2f  %-12.2f  %-8.2f  %-10.2f\n", pos, dt_scalar*1000, dt_simd*1000, speedup, saved_ms);
    }

    printf("\n注: 此基准只测量 forward 中的 RMSNorm + Attention 部分（不含 matmul）\n");
    printf("    matmul 是主要瓶颈，但 attention 在长 context 时占比显著\n");

    _mm_free(x); _mm_free(w); _mm_free(ln); _mm_free(Q); _mm_free(K); _mm_free(V);
    _mm_free(attn_out); _mm_free(kc); _mm_free(vc); _mm_free(rope_cos); _mm_free(rope_sin);
    return 0;
}
