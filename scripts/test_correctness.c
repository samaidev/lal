/* test_correctness.c — 验证 SIMD 优化后 forward 输出正确性
 *
 * 比较标量实现 vs SIMD 实现在相同输入下的输出是否一致
 * Build: gcc -O3 -march=native -I. -o test_correctness scripts/test_correctness.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define N_EMBD    3584
#define N_HEAD    28
#define N_KV_HEAD 4
#define HEAD_DIM  128
#define KV_DIM    (N_KV_HEAD * HEAD_DIM)
#define N_Q_PER_KV (N_HEAD / N_KV_HEAD)
#define N_CTX     4096
#define RMS_EPS   1e-6f

#include "runtime/lal_simd_optim.h"

/* 标量实现 (原版) */
static void rms_norm_scalar(float *out, const float *x, const float *w, int n) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * ms * w[i];
}

static void rope_apply_scalar(float *q, float *k, int pos, const float *rc, const float *rs) {
    for (int h = 0; h < N_HEAD; h++) {
        float *qh = q + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float c = rc[d], s = rs[d];
            float q0 = qh[d], q1 = qh[d + HEAD_DIM/2];
            qh[d] = q0*c - q1*s; qh[d+HEAD_DIM/2] = q0*s + q1*c;
        }
    }
    for (int h = 0; h < N_KV_HEAD; h++) {
        float *kh = k + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float c = rc[d], s = rs[d];
            float k0 = kh[d], k1 = kh[d + HEAD_DIM/2];
            kh[d] = k0*c - k1*s; kh[d+HEAD_DIM/2] = k0*s + k1*c;
        }
    }
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

static float max_abs_diff(const float *a, const float *b, int n) {
    float max_diff = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

int main() {
    srand(42);
    int pos = 64;

    /* 分配 buffers */
    float *x = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *w = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *out_s = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *out_v = _mm_malloc(N_EMBD * sizeof(float), 32);

    /* RMSNorm 测试 */
    printf("=== RMSNorm 正确性测试 ===\n");
    for (int i = 0; i < N_EMBD; i++) { x[i] = (rand()/((float)RAND_MAX)-0.5f)*2.0f; w[i] = (rand()/((float)RAND_MAX)-0.5f)*0.5f + 0.8f; }
    rms_norm_scalar(out_s, x, w, N_EMBD);
    lal_rms_norm_simd(out_v, x, w, N_EMBD, RMS_EPS);
    float rms_diff = max_abs_diff(out_s, out_v, N_EMBD);
    printf("  最大差异: %.2e %s\n", rms_diff, rms_diff < 1e-5f ? "✅ PASS" : "❌ FAIL");

    /* RoPE 测试 */
    printf("\n=== RoPE 正确性测试 ===\n");
    float *Q_s = _mm_malloc(N_HEAD * HEAD_DIM * sizeof(float), 32);
    float *Q_v = _mm_malloc(N_HEAD * HEAD_DIM * sizeof(float), 32);
    float *K_s = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *K_v = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *rc = _mm_malloc(HEAD_DIM/2 * sizeof(float), 32);
    float *rs = _mm_malloc(HEAD_DIM/2 * sizeof(float), 32);

    for (int i = 0; i < N_HEAD*HEAD_DIM; i++) Q_s[i] = Q_v[i] = (rand()/((float)RAND_MAX)-0.5f)*2.0f;
    for (int i = 0; i < KV_DIM; i++) K_s[i] = K_v[i] = (rand()/((float)RAND_MAX)-0.5f)*2.0f;
    for (int i = 0; i < HEAD_DIM/2; i++) { rc[i] = cosf(i * 0.01f); rs[i] = sinf(i * 0.01f); }

    rope_apply_scalar(Q_s, K_s, pos, rc, rs);
    lal_rope_apply_simd(Q_v, K_v, N_HEAD, N_KV_HEAD, HEAD_DIM, pos, rc, rs);
    float q_diff = max_abs_diff(Q_s, Q_v, N_HEAD * HEAD_DIM);
    float k_diff = max_abs_diff(K_s, K_v, KV_DIM);
    printf("  Q 最大差异: %.2e %s\n", q_diff, q_diff < 1e-5f ? "✅ PASS" : "❌ FAIL");
    printf("  K 最大差异: %.2e %s\n", k_diff, k_diff < 1e-5f ? "✅ PASS" : "❌ FAIL");

    /* Attention 测试 */
    printf("\n=== GQA Attention 正确性测试 (pos=%d) ===\n", pos);
    float *Q = _mm_malloc(N_HEAD * HEAD_DIM * sizeof(float), 32);
    float *K = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *V = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *attn_s = _mm_malloc(N_HEAD * HEAD_DIM * sizeof(float), 32);
    float *attn_v = _mm_malloc(N_HEAD * HEAD_DIM * sizeof(float), 32);
    float *kc_s = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *vc_s = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *kc_v = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *vc_v = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);

    for (int i = 0; i < N_HEAD*HEAD_DIM; i++) Q[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f;
    for (int i = 0; i < KV_DIM; i++) { K[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; V[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; }

    /* 填充一些历史 cache */
    for (int i = 0; i < pos * KV_DIM; i++) { kc_s[i] = kc_v[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; }
    for (int i = 0; i < pos * KV_DIM; i++) { vc_s[i] = vc_v[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; }

    gqa_attn_scalar(attn_s, Q, K, V, kc_s, vc_s, pos);
    lal_gqa_attn_simd(attn_v, Q, K, V, kc_v, vc_v, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
    float attn_diff = max_abs_diff(attn_s, attn_v, N_HEAD * HEAD_DIM);
    printf("  Attention 最大差异: %.2e %s\n", attn_diff, attn_diff < 1e-4f ? "✅ PASS" : "❌ FAIL");

    /* 验证 cache 也一致 */
    float kc_diff = max_abs_diff(kc_s, kc_v, (pos+1) * KV_DIM);
    float vc_diff = max_abs_diff(vc_s, vc_v, (pos+1) * KV_DIM);
    printf("  K cache 差异: %.2e %s\n", kc_diff, kc_diff < 1e-6f ? "✅ PASS" : "❌ FAIL");
    printf("  V cache 差异: %.2e %s\n", vc_diff, vc_diff < 1e-6f ? "✅ PASS" : "❌ FAIL");

    _mm_free(x); _mm_free(w); _mm_free(out_s); _mm_free(out_v);
    _mm_free(Q_s); _mm_free(Q_v); _mm_free(K_s); _mm_free(K_v); _mm_free(rc); _mm_free(rs);
    _mm_free(Q); _mm_free(K); _mm_free(V); _mm_free(attn_s); _mm_free(attn_v);
    _mm_free(kc_s); _mm_free(vc_s); _mm_free(kc_v); _mm_free(vc_v);

    printf("\n=== 总结 ===\n");
    int ok = (rms_diff < 1e-5f) && (q_diff < 1e-5f) && (k_diff < 1e-5f) && (attn_diff < 1e-4f);
    printf(ok ? "✅ 所有测试通过，SIMD 优化保持正确性\n" : "❌ 有测试失败\n");
    return ok ? 0 : 1;
}
