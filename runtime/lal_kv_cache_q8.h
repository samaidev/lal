/* lal_kv_cache_q8.h — KV Cache Q8 量化
 *
 * 当前: KV cache 是 F32, 28 layers × 4096 ctx × 512 kv_dim × 4 bytes × 2 (K+V) = 448MB
 * Q8 后: 28 × 4096 × 512 × 1 byte × 2 + scales = 112MB + 28×4096×2×4 = ~125MB
 *
 * 好处:
 *   1. 内存带宽减少 4x → attention Q·K 和 V 加权读取快 4x
 *   2. cache 命中率提升 → 更多数据在 L2/L3
 *   3. 长 context 时 attention 是瓶颈，4x 带宽减少 → 显著加速
 *
 * 代价:
 *   1. 精度损失 (~1% perplexity 增加, 对生成质量影响小)
 *   2. 读取时需要反量化 (但可以融合到 dot product 中)
 *
 * 格式: 每 32 个 float 一组，存 1 个 fp16 scale + 32 个 int8 = 34 bytes (Q8_0 格式)
 */
#ifndef LAL_KV_CACHE_Q8_H
#define LAL_KV_CACHE_Q8_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Q8_0 block: 32 个 float 量化为 1 fp16 scale + 32 int8 = 34 bytes */
#define KV_Q8_BLOCK 32
#define KV_Q8_BYTES(kv_dim) ((kv_dim) / KV_Q8_BLOCK * 34)

/* 将一个 kv_dim 的 float 向量量化到 Q8_0 格式
 * dst: 34 * (kv_dim/32) bytes
 * src: kv_dim floats
 */
static inline void kv_quantize_q8_0(uint8_t * __restrict__ dst,
                                      const float * __restrict__ src,
                                      int dim) {
    int n_blocks = dim / KV_Q8_BLOCK;
    for (int b = 0; b < n_blocks; b++) {
        const float *sb = src + b * KV_Q8_BLOCK;
        uint8_t *db = dst + b * 34;

        /* 求 abs max */
        float amax = 0;
        for (int i = 0; i < KV_Q8_BLOCK; i++) {
            float a = fabsf(sb[i]);
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        float inv = 1.0f / scale;

        /* 存 fp16 scale */
        __m128 v = _mm_set1_ps(scale);
        __m128i h = _mm_cvtps_ph(v, 0);
        *(uint16_t*)db = _mm_extract_epi16(h, 0);

        /* 量化 32 个值到 int8 */
        int8_t *q = (int8_t*)(db + 2);
        for (int i = 0; i < KV_Q8_BLOCK; i++) {
            int v = (int)lroundf(sb[i] * inv);
            q[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    }
}

/* 从 Q8_0 cache 读取一个向量并反量化到 float
 * 用于 V 加权累加时读取 V
 */
static inline void kv_dequantize_q8_0(float * __restrict__ dst,
                                       const uint8_t * __restrict__ src,
                                       int dim) {
    int n_blocks = dim / KV_Q8_BLOCK;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *sb = src + b * 34;
        float scale = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)sb)));
        const int8_t *q = (const int8_t*)(sb + 2);
        float *db = dst + b * KV_Q8_BLOCK;
        __m256 vs = _mm256_set1_ps(scale);
        /* 32 = 4 × 8, 用 SIMD 反量化 */
        for (int i = 0; i < KV_Q8_BLOCK; i += 8) {
            __m128i qi = _mm_loadl_epi64((__m128i*)(q + i)); /* 8 int8 */
            __m256i qi32 = _mm256_cvtepi8_epi32(qi); /* 8 int32 */
            __m256 vf = _mm256_cvtepi32_ps(qi32);
            __m256 vr = _mm256_mul_ps(vf, vs);
            _mm256_storeu_ps(db + i, vr);
        }
    }
}

/* 融合的 Q·K dot product: 直接用 Q8_0 格式的 K cache，无需反量化到 float
 * 返回 Q · K (float)
 * Q: head_dim floats (float)
 * K_cache: Q8_0 格式, head_dim/32 个 block
 */
static inline float kv_dot_q8_0(const float * __restrict__ q,
                                 const uint8_t * __restrict__ k_q8,
                                 int head_dim) {
    int n_blocks = head_dim / KV_Q8_BLOCK;
    __m256 vsum = _mm256_setzero_ps();
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *kb = k_q8 + b * 34;
        float scale = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)kb)));
        const int8_t *qi = (const int8_t*)(kb + 2);
        const float *qb = q + b * KV_Q8_BLOCK;
        __m256 vs = _mm256_set1_ps(scale);
        /* 32 = 4 × 8 */
        for (int i = 0; i < KV_Q8_BLOCK; i += 8) {
            __m128i ki = _mm_loadl_epi64((__m128i*)(qi + i));
            __m256i ki32 = _mm256_cvtepi8_epi32(ki);
            __m256 kf = _mm256_cvtepi32_ps(ki32);
            __m256 qf = _mm256_loadu_ps(qb + i);
            vsum = _mm256_fmadd_ps(_mm256_mul_ps(kf, vs), qf, vsum);
        }
    }
    /* hsum */
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* 融合的 V 加权累加: 直接用 Q8_0 格式的 V cache
 * out += weight * V
 * V_cache: Q8_0 格式, head_dim/32 个 block
 */
static inline void kv_vmac_q8_0(float * __restrict__ out,
                                 float weight,
                                 const uint8_t * __restrict__ v_q8,
                                 int head_dim) {
    int n_blocks = head_dim / KV_Q8_BLOCK;
    __m256 vw = _mm256_set1_ps(weight);
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *vb = v_q8 + b * 34;
        float scale = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)vb)));
        const int8_t *qi = (const int8_t*)(vb + 2);
        float *ob = out + b * KV_Q8_BLOCK;
        __m256 vs = _mm256_set1_ps(scale);
        for (int i = 0; i < KV_Q8_BLOCK; i += 8) {
            __m128i vi = _mm_loadl_epi64((__m128i*)(qi + i));
            __m256i vi32 = _mm256_cvtepi8_epi32(vi);
            __m256 vf = _mm256_cvtepi32_ps(vi32);
            __m256 vscaled = _mm256_mul_ps(vf, vs);
            __m256 vws = _mm256_mul_ps(vscaled, vw);
            __m256 vo = _mm256_loadu_ps(ob + i);
            _mm256_storeu_ps(ob + i, _mm256_add_ps(vo, vws));
        }
    }
}

#endif /* LAL_KV_CACHE_Q8_H */
