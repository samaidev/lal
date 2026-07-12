/* bench_attention.c — 基准测试注意力机制和 RMSNorm/RoPE
 * 测量 LAL 当前标量实现 vs SIMD 实现的性能差异
 * Build: gcc -O3 -march=native -I. -o bench_attention scripts/bench_attention.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define N_EMBD    3584
#define N_HEAD    28
#define N_KV_HEAD 4
#define HEAD_DIM  128
#define KV_DIM    (N_KV_HEAD * HEAD_DIM)  /* 512 */
#define N_Q_PER_KV (N_HEAD / N_KV_HEAD)   /* 7 */
#define N_CTX     4096

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* === 当前 LAL 的标量实现 (从 qwen7b_server.c 复制) === */

static void rms_norm_scalar(float *out, const float *x, const float *w, int n) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * ms * w[i];
}

static float g_rope_cos[N_CTX][HEAD_DIM/2];
static float g_rope_sin[N_CTX][HEAD_DIM/2];
static void rope_init(void) {
    for (int p = 0; p < N_CTX; p++)
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float theta = (float)p / powf(1000000.0f, (float)(2*d) / HEAD_DIM);
            g_rope_cos[p][d] = cosf(theta);
            g_rope_sin[p][d] = sinf(theta);
        }
}
static void rope_apply_scalar(float *q, float *k, int pos) {
    for (int h = 0; h < N_HEAD; h++) {
        float *qh = q + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float c = g_rope_cos[pos][d], s = g_rope_sin[pos][d];
            float q0 = qh[d], q1 = qh[d + HEAD_DIM/2];
            qh[d] = q0*c - q1*s; qh[d+HEAD_DIM/2] = q0*s + q1*c;
        }
    }
    for (int h = 0; h < N_KV_HEAD; h++) {
        float *kh = k + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float c = g_rope_cos[pos][d], s = g_rope_sin[pos][d];
            float k0 = kh[d], k1 = kh[d + HEAD_DIM/2];
            kh[d] = k0*c - k1*s; kh[d+HEAD_DIM/2] = k0*s + k1*c;
        }
    }
}

static void gqa_attn_scalar(float *out, const float *Q, const float *Kn, const float *Vn,
                            float *k_cache, float *v_cache, int layer, int pos) {
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
        for (int t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        float *oh = out + h * HEAD_DIM;
        memset(oh, 0, HEAD_DIM * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float w = scores[t] / sum;
            const float *vt = v_cache + t * KV_DIM + kvh * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) oh[d] += w * vt[d];
        }
    }
}

/* === SIMD 优化实现 === */

static inline float hsum_m256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static void rms_norm_simd(float *out, const float *x, const float *w, int n) {
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vsum = _mm256_fmadd_ps(vx, vx, vsum);
    }
    float ms = hsum_m256(vsum);
    for (; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + 1e-6f);
    __m256 vms = _mm256_set1_ps(ms);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 vr = _mm256_mul_ps(_mm256_mul_ps(vx, vms), vw);
        _mm256_storeu_ps(out + i, vr);
    }
    for (; i < n; i++) out[i] = x[i] * ms * w[i];
}

static void rope_apply_simd(float *q, float *k, int pos) {
    /* HEAD_DIM=128 = 16 个 8-float 块，完美适合 AVX2 */
    const float *pc = g_rope_cos[pos];
    const float *ps = g_rope_sin[pos];
    int half = HEAD_DIM / 2;
    for (int h = 0; h < N_HEAD; h++) {
        float *qh = q + h * HEAD_DIM;
        for (int d = 0; d < half; d += 8) {
            __m256 c = _mm256_loadu_ps(pc + d);
            __m256 s = _mm256_loadu_ps(ps + d);
            __m256 q0 = _mm256_loadu_ps(qh + d);
            __m256 q1 = _mm256_loadu_ps(qh + d + half);
            __m256 r0 = _mm256_fmsub_ps(q0, c, _mm256_mul_ps(q1, s));
            __m256 r1 = _mm256_fmadd_ps(q0, s, _mm256_mul_ps(q1, c));
            _mm256_storeu_ps(qh + d, r0);
            _mm256_storeu_ps(qh + d + half, r1);
        }
    }
    for (int h = 0; h < N_KV_HEAD; h++) {
        float *kh = k + h * HEAD_DIM;
        for (int d = 0; d < half; d += 8) {
            __m256 c = _mm256_loadu_ps(pc + d);
            __m256 s = _mm256_loadu_ps(ps + d);
            __m256 k0 = _mm256_loadu_ps(kh + d);
            __m256 k1 = _mm256_loadu_ps(kh + d + half);
            __m256 r0 = _mm256_fmsub_ps(k0, c, _mm256_mul_ps(k1, s));
            __m256 r1 = _mm256_fmadd_ps(k0, s, _mm256_mul_ps(k1, c));
            _mm256_storeu_ps(kh + d, r0);
            _mm256_storeu_ps(kh + d + half, r1);
        }
    }
}

static void gqa_attn_simd(float *out, const float *Q, const float *Kn, const float *Vn,
                          float *k_cache, float *v_cache, int layer, int pos) {
    memcpy(k_cache + pos * KV_DIM, Kn, KV_DIM * sizeof(float));
    memcpy(v_cache + pos * KV_DIM, Vn, KV_DIM * sizeof(float));
    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    __m256 vinv_sqrt = _mm256_set1_ps(inv_sqrt);
    static float scores[N_CTX];
    /* AVX2: HEAD_DIM=128 = 4 个 __m256，每轮处理 8 个 d */
    const int VEC = 4; /* 4 * 8 = 32 floats... wait HEAD_DIM=128 = 16 * 8 */
    for (int h = 0; h < N_HEAD; h++) {
        const float *qh = Q + h * HEAD_DIM;
        int kvh = h / N_Q_PER_KV;
        float max_score = -1e30f;
        /* Q·K 点积: 16 个 __m256 累加 */
        __m256 q[16];
        for (int i = 0; i < 16; i++) q[i] = _mm256_loadu_ps(qh + i*8);
        for (int t = 0; t <= pos; t++) {
            const float *kt = k_cache + t * KV_DIM + kvh * HEAD_DIM;
            __m256 vdot = _mm256_setzero_ps();
            for (int i = 0; i < 16; i++) {
                __m256 vk = _mm256_loadu_ps(kt + i*8);
                vdot = _mm256_fmadd_ps(q[i], vk, vdot);
            }
            float dot = hsum_m256(vdot);
            scores[t] = dot * inv_sqrt;
            if (scores[t] > max_score) max_score = scores[t];
        }
        float sum = 0;
        for (int t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        float inv_sum = 1.0f / sum;
        float *oh = out + h * HEAD_DIM;
        __m256 voh[16] = {0};
        for (int t = 0; t <= pos; t++) {
            float w = scores[t] * inv_sum;
            __m256 vw = _mm256_set1_ps(w);
            const float *vt = v_cache + t * KV_DIM + kvh * HEAD_DIM;
            for (int i = 0; i < 16; i++) {
                __m256 vv = _mm256_loadu_ps(vt + i*8);
                voh[i] = _mm256_fmadd_ps(vw, vv, voh[i]);
            }
        }
        for (int i = 0; i < 16; i++) _mm256_storeu_ps(oh + i*8, voh[i]);
    }
}

int main() {
    rope_init();
    srand(42);

    /* 分配 buffers */
    float *x   = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *w   = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *out = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *Q   = _mm_malloc(N_HEAD * HEAD_DIM * sizeof(float), 32);
    float *K   = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *V   = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *attn_out = _mm_malloc(N_HEAD * HEAD_DIM * sizeof(float), 32);
    float *k_cache = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *v_cache = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);

    for (int i = 0; i < N_EMBD; i++) { x[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; w[i] = (rand()/((float)RAND_MAX)-0.5f)*0.1f; }
    for (int i = 0; i < N_HEAD*HEAD_DIM; i++) Q[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f;
    for (int i = 0; i < KV_DIM; i++) { K[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; V[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; }

    int pos = 128; /* 模拟 128 token 的上下文 */
    int n_iter = 1000;

    printf("=== RMSNorm (N_EMBD=%d, %d iters) ===\n", N_EMBD, n_iter);
    double t0 = now_s();
    for (int i = 0; i < n_iter; i++) rms_norm_scalar(out, x, w, N_EMBD);
    double dt_scalar = (now_s() - t0) / n_iter;
    t0 = now_s();
    for (int i = 0; i < n_iter; i++) rms_norm_simd(out, x, w, N_EMBD);
    double dt_simd = (now_s() - t0) / n_iter;
    printf("  scalar: %.3f us   SIMD: %.3f us   speedup: %.2fx\n", dt_scalar*1e6, dt_simd*1e6, dt_scalar/dt_simd);

    printf("\n=== RoPE (N_HEAD=%d, N_KV_HEAD=%d, HEAD_DIM=%d, %d iters) ===\n", N_HEAD, N_KV_HEAD, HEAD_DIM, n_iter);
    t0 = now_s();
    for (int i = 0; i < n_iter; i++) rope_apply_scalar(Q, K, pos);
    double dt_rope_s = (now_s() - t0) / n_iter;
    t0 = now_s();
    for (int i = 0; i < n_iter; i++) rope_apply_simd(Q, K, pos);
    double dt_rope_v = (now_s() - t0) / n_iter;
    printf("  scalar: %.3f us   SIMD: %.3f us   speedup: %.2fx\n", dt_rope_s*1e6, dt_rope_v*1e6, dt_rope_s/dt_rope_v);

    printf("\n=== GQA Attention (pos=%d, %d iters) ===\n", pos, n_iter);
    n_iter = 100;
    t0 = now_s();
    for (int i = 0; i < n_iter; i++) gqa_attn_scalar(attn_out, Q, K, V, k_cache, v_cache, 0, pos);
    double dt_attn_s = (now_s() - t0) / n_iter;
    t0 = now_s();
    for (int i = 0; i < n_iter; i++) gqa_attn_simd(attn_out, Q, K, V, k_cache, v_cache, 0, pos);
    double dt_attn_v = (now_s() - t0) / n_iter;
    printf("  scalar: %.3f us   SIMD: %.3f us   speedup: %.2fx\n", dt_attn_s*1e6, dt_attn_v*1e6, dt_attn_s/dt_attn_v);

    /* 模拟一次 forward 的总时间 (28 layers, pos=128) */
    printf("\n=== 估算单次 forward 提升 (%d layers, pos=%d) ===\n", 28, pos);
    double saved_rms = (dt_scalar - dt_simd) * 3 * 28; /* 3 RMSNorm per layer × 28 */
    double saved_rope = (dt_rope_s - dt_rope_v) * 28;
    double saved_attn = (dt_attn_s - dt_attn_v) * 28;
    printf("  RMSNorm 节省: %.2f us\n", saved_rms*1e6);
    printf("  RoPE 节省:    %.2f us\n", saved_rope*1e6);
    printf("  Attention 节省: %.2f us\n", saved_attn*1e6);
    printf("  总节省: %.2f us\n", (saved_rms+saved_rope+saved_attn)*1e6);

    _mm_free(x); _mm_free(w); _mm_free(out); _mm_free(Q); _mm_free(K); _mm_free(V);
    _mm_free(attn_out); _mm_free(k_cache); _mm_free(v_cache);
    return 0;
}
