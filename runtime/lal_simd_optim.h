/* lal_simd_optim.h — SIMD 优化版 RMSNorm / RoPE / GQA Attention
 *
 * 性能提升（基于 bench_attention 实测）：
 *   RMSNorm:   2.69x（标量 2.79μs → SIMD 1.04μs，N_EMBD=3584）
 *   RoPE:      ~1.0x（编译器已自动向量化）
 *   GQA Attention: 4.76x（标量 213.6μs → SIMD 44.9μs，pos=128）
 *
 * 每 forward (28 layers, pos=128) 总节省约 4.87ms
 * 对 1.4 tok/s baseline (~714ms/token) 提升约 0.7%
 * 对长 context (pos=512+) 提升更大，因为 attention 是 O(n²)
 *
 * 关键技术：
 *   - AVX2 256-bit FMADD 向量化点积（HEAD_DIM=128 = 16 个 __m256）
 *   - V 累加用 16 个寄存器避免反复 load/store
 *   - RMSNorm 两阶段：SIMD 求平方和 + SIMD 乘法
 *   - RoPE 用 FMSUB/FMADD 替代标量乘减
 */
#ifndef LAL_SIMD_OPTIM_H
#define LAL_SIMD_OPTIM_H

#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#if defined(__AVX2__) && defined(__FMA__)

static inline float lal_hsum_m256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* SIMD 优化版 RMSNorm — 2.69x faster than scalar */
static inline void lal_rms_norm_simd(float * __restrict__ out,
                                      const float * __restrict__ x,
                                      const float * __restrict__ w,
                                      int n, float eps) {
    /* Phase 1: SIMD sum of squares */
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    int n8 = n & ~7;
    for (; i < n8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vsum = _mm256_fmadd_ps(vx, vx, vsum);
    }
    float ms = lal_hsum_m256(vsum);
    /* tail */
    for (; i < n; i++) ms += x[i] * x[i];

    float inv_rms = 1.0f / sqrtf(ms / n + eps);
    __m256 vinv = _mm256_set1_ps(inv_rms);

    /* Phase 2: SIMD (x * inv_rms) * w */
    i = 0;
    for (; i < n8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 vr = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw);
        _mm256_storeu_ps(out + i, vr);
    }
    for (; i < n; i++) out[i] = x[i] * inv_rms * w[i];
}

/* SIMD 优化版 RoPE — 对 HEAD_DIM=128 完美向量化 (16 个 __m256) */
static inline void lal_rope_apply_simd(float * __restrict__ q,
                                        float * __restrict__ k,
                                        int n_head, int n_kv_head, int head_dim,
                                        int pos,
                                        const float * __restrict__ rope_cos,
                                        const float * __restrict__ rope_sin) {
    const float *pc = rope_cos;
    const float *ps = rope_sin;
    int half = head_dim / 2;
    int half8 = half & ~7;

    /* Apply to Q */
    for (int h = 0; h < n_head; h++) {
        float *qh = q + (size_t)h * head_dim;
        int d = 0;
        for (; d < half8; d += 8) {
            __m256 c = _mm256_loadu_ps(pc + d);
            __m256 s = _mm256_loadu_ps(ps + d);
            __m256 q0 = _mm256_loadu_ps(qh + d);
            __m256 q1 = _mm256_loadu_ps(qh + d + half);
            /* r0 = q0*c - q1*s; r1 = q0*s + q1*c */
            __m256 r0 = _mm256_fmsub_ps(q0, c, _mm256_mul_ps(q1, s));
            __m256 r1 = _mm256_fmadd_ps(q0, s, _mm256_mul_ps(q1, c));
            _mm256_storeu_ps(qh + d, r0);
            _mm256_storeu_ps(qh + d + half, r1);
        }
        for (; d < half; d++) {
            float c = pc[d], s = ps[d];
            float q0 = qh[d], q1 = qh[d + half];
            qh[d] = q0*c - q1*s;
            qh[d+half] = q0*s + q1*c;
        }
    }
    /* Apply to K */
    for (int h = 0; h < n_kv_head; h++) {
        float *kh = k + (size_t)h * head_dim;
        int d = 0;
        for (; d < half8; d += 8) {
            __m256 c = _mm256_loadu_ps(pc + d);
            __m256 s = _mm256_loadu_ps(ps + d);
            __m256 k0 = _mm256_loadu_ps(kh + d);
            __m256 k1 = _mm256_loadu_ps(kh + d + half);
            __m256 r0 = _mm256_fmsub_ps(k0, c, _mm256_mul_ps(k1, s));
            __m256 r1 = _mm256_fmadd_ps(k0, s, _mm256_mul_ps(k1, c));
            _mm256_storeu_ps(kh + d, r0);
            _mm256_storeu_ps(kh + d + half, r1);
        }
        for (; d < half; d++) {
            float c = pc[d], s = ps[d];
            float k0 = kh[d], k1 = kh[d + half];
            kh[d] = k0*c - k1*s;
            kh[d+half] = k0*s + k1*c;
        }
    }
}

/* SIMD 优化版 GQA Attention — 4.76x faster than scalar
 *
 * 关键优化：
 * 1. Q·K 点积用 16 个 __m256 累加（HEAD_DIM=128 完美填充）
 * 2. V 加权累加用 16 个寄存器，避免反复 load/store
 * 3. 预计算 inv_sum 用乘法替代除法
 *
 * 参数说明：
 *   out:       [n_head * head_dim] 输出
 *   Q:         [n_head * head_dim] query
 *   Kn:        [kv_dim] 新的 key (n_kv_head * head_dim)
 *   Vn:        [kv_dim] 新的 value
 *   k_cache:   [N_CTX * kv_dim] key cache
 *   v_cache:   [N_CTX * kv_dim] value cache
 *   pos:       当前位置（0-indexed）
 *   n_head, n_kv_head, head_dim, n_q_per_kv: GQA 参数
 */
static inline void lal_gqa_attn_simd(float * __restrict__ out,
                                      const float * __restrict__ Q,
                                      const float * __restrict__ Kn,
                                      const float * __restrict__ Vn,
                                      float * __restrict__ k_cache,
                                      float * __restrict__ v_cache,
                                      int pos,
                                      int n_head, int n_kv_head, int head_dim,
                                      int n_q_per_kv, int kv_dim, int n_ctx) {
    /* Store new K, V into cache */
    memcpy(k_cache + (size_t)pos * kv_dim, Kn, kv_dim * sizeof(float));
    memcpy(v_cache + (size_t)pos * kv_dim, Vn, kv_dim * sizeof(float));

    float inv_sqrt = 1.0f / sqrtf((float)head_dim);
    int half = head_dim / 2;
    /* HEAD_DIM=128 → 16 个 8-float 块 */
    int nvec = head_dim / 8;

    /* 使用静态 scores 数组避免栈溢出 */
    static float scores[8192]; /* 支持 n_ctx <= 8192 */
    if (pos >= 8192) pos = 8191; /* safety */

    for (int h = 0; h < n_head; h++) {
        const float *qh = Q + (size_t)h * head_dim;
        int kvh = h / n_q_per_kv;

        /* Load Q vectors once per head */
        __m256 qv[32]; /* max head_dim/8 = 32 (for head_dim=256) */
        for (int i = 0; i < nvec; i++)
            qv[i] = _mm256_loadu_ps(qh + i * 8);

        float max_score = -1e30f;

        /* Phase 1: Q·K dot product + find max */
        for (int t = 0; t <= pos; t++) {
            const float *kt = k_cache + (size_t)t * kv_dim + kvh * head_dim;
            __m256 vdot = _mm256_setzero_ps();
            for (int i = 0; i < nvec; i++) {
                __m256 vk = _mm256_loadu_ps(kt + i * 8);
                vdot = _mm256_fmadd_ps(qv[i], vk, vdot);
            }
            float dot = lal_hsum_m256(vdot);
            scores[t] = dot * inv_sqrt;
            if (scores[t] > max_score) max_score = scores[t];
        }

        /* Phase 2: softmax (exp + sum) */
        float sum = 0;
        for (int t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        float inv_sum = 1.0f / sum;

        /* Phase 3: weighted V accumulation — keep 16 accumulators in registers */
        float *oh = out + (size_t)h * head_dim;
        __m256 vacc[32];
        for (int i = 0; i < nvec; i++) vacc[i] = _mm256_setzero_ps();

        for (int t = 0; t <= pos; t++) {
            float w = scores[t] * inv_sum;
            __m256 vw = _mm256_set1_ps(w);
            const float *vt = v_cache + (size_t)t * kv_dim + kvh * head_dim;
            for (int i = 0; i < nvec; i++) {
                __m256 vv = _mm256_loadu_ps(vt + i * 8);
                vacc[i] = _mm256_fmadd_ps(vw, vv, vacc[i]);
            }
        }

        /* Store accumulators to output */
        for (int i = 0; i < nvec; i++)
            _mm256_storeu_ps(oh + i * 8, vacc[i]);
    }
}

#else
/* Fallback: 标量实现（非 x86 平台） */
#warning "AVX2 not available, using scalar fallback for SIMD optimizations"

static inline void lal_rms_norm_simd(float *out, const float *x, const float *w, int n, float eps) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * ms * w[i];
}

static inline void lal_rope_apply_simd(float *q, float *k, int n_head, int n_kv_head,
                                        int head_dim, int pos,
                                        const float *rope_cos, const float *rope_sin) {
    for (int h = 0; h < n_head; h++) {
        float *qh = q + (size_t)h * head_dim;
        for (int d = 0; d < head_dim/2; d++) {
            float c = rope_cos[d], s = rope_sin[d];
            float q0 = qh[d], q1 = qh[d + head_dim/2];
            qh[d] = q0*c - q1*s; qh[d+head_dim/2] = q0*s + q1*c;
        }
    }
    for (int h = 0; h < n_kv_head; h++) {
        float *kh = k + (size_t)h * head_dim;
        for (int d = 0; d < head_dim/2; d++) {
            float c = rope_cos[d], s = rope_sin[d];
            float k0 = kh[d], k1 = kh[d + head_dim/2];
            kh[d] = k0*c - k1*s; kh[d+head_dim/2] = k0*s + k1*c;
        }
    }
}

static inline void lal_gqa_attn_simd(float *out, const float *Q, const float *Kn, const float *Vn,
                                      float *k_cache, float *v_cache, int pos,
                                      int n_head, int n_kv_head, int head_dim,
                                      int n_q_per_kv, int kv_dim, int n_ctx) {
    memcpy(k_cache + (size_t)pos * kv_dim, Kn, kv_dim * sizeof(float));
    memcpy(v_cache + (size_t)pos * kv_dim, Vn, kv_dim * sizeof(float));
    float inv_sqrt = 1.0f / sqrtf((float)head_dim);
    static float scores[8192];
    if (pos >= 8192) pos = 8191;
    for (int h = 0; h < n_head; h++) {
        const float *qh = Q + (size_t)h * head_dim;
        int kvh = h / n_q_per_kv;
        float max_score = -1e30f;
        for (int t = 0; t <= pos; t++) {
            const float *kt = k_cache + (size_t)t * kv_dim + kvh * head_dim;
            float dot = 0;
            for (int d = 0; d < head_dim; d++) dot += qh[d] * kt[d];
            scores[t] = dot * inv_sqrt;
            if (scores[t] > max_score) max_score = scores[t];
        }
        float sum = 0;
        for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t]-max_score); sum += scores[t]; }
        float *oh = out + (size_t)h * head_dim;
        memset(oh, 0, head_dim * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float w = scores[t] / sum;
            const float *vt = v_cache + (size_t)t * kv_dim + kvh * head_dim;
            for (int d = 0; d < head_dim; d++) oh[d] += w * vt[d];
        }
    }
}
#endif

#endif /* LAL_SIMD_OPTIM_H */
