/* bench_full_forward.c — 完整 forward 非matmul 部分性能对比
 * 包含: RMSNorm + RoPE + Attention + SwiGLU + Residual × 28 layers
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_full scripts/bench_full_forward.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

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

static void silu_mul_scalar(float *act, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        act[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

static void residual_add_scalar(float *x, const float *y, int n) {
    for (int i = 0; i < n; i++) x[i] += y[i];
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int main() {
    int positions[] = {32, 128, 512, 1024};
    int n_pos = sizeof(positions)/sizeof(positions[0]);

    float *x = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *w = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *ln = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *Q = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *K = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *V = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *attn_out = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *proj = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *gate_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *up_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *act_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *mlp_out = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *kc = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *vc = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);

    for (int i = 0; i < N_EMBD; i++) { x[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; w[i] = 0.9f; }
    for (int i = 0; i < Q_DIM; i++) Q[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f;
    for (int i = 0; i < KV_DIM; i++) { K[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; V[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; }
    for (int i = 0; i < MLP_DIM; i++) { gate_buf[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; up_buf[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; }
    for (int i = 0; i < N_EMBD; i++) { proj[i] = mlp_out[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; }

    printf("=== 完整 forward 非 matmul 部分 (28 layers, 含 SwiGLU+Residual) ===\n");
    printf("%-8s  %-12s  %-12s  %-8s  %-10s\n", "pos", "scalar(ms)", "SIMD(ms)", "speedup", "saved(ms)");

    for (int pi = 0; pi < n_pos; pi++) {
        int pos = positions[pi];
        int n_iter = pos <= 128 ? 10 : 3;

        /* 标量版 */
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++) {
            for (int l = 0; l < N_LAYER; l++) {
                rms_norm_scalar(ln, x, w, N_EMBD);
                gqa_attn_scalar(attn_out, Q, K, V, kc, vc, pos);
                residual_add_scalar(x, proj, N_EMBD);
                rms_norm_scalar(ln, x, w, N_EMBD);
                silu_mul_scalar(act_buf, gate_buf, up_buf, MLP_DIM);
                residual_add_scalar(x, mlp_out, N_EMBD);
            }
            rms_norm_scalar(ln, x, w, N_EMBD);
        }
        double dt_scalar = (now_s() - t0) / n_iter;

        /* SIMD 版 */
        t0 = now_s();
        for (int it = 0; it < n_iter; it++) {
            for (int l = 0; l < N_LAYER; l++) {
                lal_rms_norm_simd(ln, x, w, N_EMBD, RMS_EPS);
                lal_gqa_attn_simd(attn_out, Q, K, V, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
                lal_residual_add_simd(x, proj, N_EMBD);
                lal_rms_norm_simd(ln, x, w, N_EMBD, RMS_EPS);
                lal_silu_mul_simd(act_buf, gate_buf, up_buf, MLP_DIM);
                lal_residual_add_simd(x, mlp_out, N_EMBD);
            }
            lal_rms_norm_simd(ln, x, w, N_EMBD, RMS_EPS);
        }
        double dt_simd = (now_s() - t0) / n_iter;

        printf("%-8d  %-12.2f  %-12.2f  %-8.2f  %-10.2f\n", pos, dt_scalar*1000, dt_simd*1000, dt_scalar/dt_simd, (dt_scalar-dt_simd)*1000);
    }

    /* 单独测 SwiGLU 和 Residual */
    printf("\n=== 单项测试 (1000 iters) ===\n");
    int n_iter = 1000;
    double t0 = now_s();
    for (int i = 0; i < n_iter; i++) silu_mul_scalar(act_buf, gate_buf, up_buf, MLP_DIM);
    double dt_s = (now_s() - t0) / n_iter;
    t0 = now_s();
    for (int i = 0; i < n_iter; i++) lal_silu_mul_simd(act_buf, gate_buf, up_buf, MLP_DIM);
    double dt_v = (now_s() - t0) / n_iter;
    printf("  SwiGLU (MLP_DIM=%d):  scalar %.2f us  SIMD %.2f us  speedup %.2fx\n", MLP_DIM, dt_s*1e6, dt_v*1e6, dt_s/dt_v);

    t0 = now_s();
    for (int i = 0; i < n_iter; i++) residual_add_scalar(x, proj, N_EMBD);
    dt_s = (now_s() - t0) / n_iter;
    t0 = now_s();
    for (int i = 0; i < n_iter; i++) lal_residual_add_simd(x, proj, N_EMBD);
    dt_v = (now_s() - t0) / n_iter;
    printf("  Residual (N_EMBD=%d):  scalar %.2f us  SIMD %.2f us  speedup %.2fx\n", N_EMBD, dt_s*1e6, dt_v*1e6, dt_s/dt_v);

    _mm_free(x); _mm_free(w); _mm_free(ln); _mm_free(Q); _mm_free(K); _mm_free(V);
    _mm_free(attn_out); _mm_free(proj); _mm_free(gate_buf); _mm_free(up_buf);
    _mm_free(act_buf); _mm_free(mlp_out); _mm_free(kc); _mm_free(vc);
    return 0;
}
