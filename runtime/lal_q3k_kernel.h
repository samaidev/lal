/* lal_q3k_kernel.h — Q3_K matmul kernel (3-bit quantization)
 *
 * Q3_K 格式 (每 256 elements = 1 superblock):
 *   4 bytes:  fp16 d (superblock scale) + fp16 dmin
 *   12 bytes: 16 × 6-bit packed scales (8 sub-block scales + 8 sub-block mins)
 *   96 bytes: 256 × 3-bit packed q values (8 sub-blocks × 32 values × 3 bits)
 * Total: 112 bytes per 256 elements = 0.4375 bytes/elem
 *
 * vs Q4_K: 144 bytes / 256 = 0.5625 bytes/elem
 * 数据量减少: 112/144 = 77.8% (减少 22.2%)
 *
 * 3-bit 量化: q values 0-7, 需要 dmin 偏移 (w = q * scale - min)
 * 精度: 比 Q4_K 低, 但对 MLP 可能可接受
 *
 * 3-bit packing: 32 values × 3 bits = 96 bits = 12 bytes per sub-block
 *   byte[i] 的 8 bits 包含 2-3 个 3-bit values (跨 byte 边界)
 */
#ifndef LAL_Q3K_KERNEL_H
#define LAL_Q3K_KERNEL_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX before including lal_q3k_kernel.h"
#endif

#include "lal_q4k_kernel.h"  /* 复用 unpack_scales_6bit */

/* Unpack 32 × 3-bit values from 12 bytes into 32 uint8 (values 0-7)
 * 96 bits packed: val[i] = (packed >> (i*3)) & 0x7
 */
static inline void unpack_q3_32(const uint8_t *src, uint8_t *out32) {
    uint64_t lo = *(const uint64_t*)src;
    uint32_t hi = *(const uint32_t*)(src + 8);
    /* 32 × 3 = 96 bits. lo has 64 bits (21 values + 1 bit), hi has 32 bits */
    out32[0]  = lo & 0x7;
    out32[1]  = (lo >> 3) & 0x7;
    out32[2]  = (lo >> 6) & 0x7;
    out32[3]  = (lo >> 9) & 0x7;
    out32[4]  = (lo >> 12) & 0x7;
    out32[5]  = (lo >> 15) & 0x7;
    out32[6]  = (lo >> 18) & 0x7;
    out32[7]  = (lo >> 21) & 0x7;
    out32[8]  = (lo >> 24) & 0x7;
    out32[9]  = (lo >> 27) & 0x7;
    out32[10] = (lo >> 30) & 0x7;
    out32[11] = (lo >> 33) & 0x7;
    out32[12] = (lo >> 36) & 0x7;
    out32[13] = (lo >> 39) & 0x7;
    out32[14] = (lo >> 42) & 0x7;
    out32[15] = (lo >> 45) & 0x7;
    out32[16] = (lo >> 48) & 0x7;
    out32[17] = (lo >> 51) & 0x7;
    out32[18] = (lo >> 54) & 0x7;
    out32[19] = (lo >> 57) & 0x7;
    out32[20] = (lo >> 60) & 0x7;
    out32[21] = ((lo >> 63) | ((uint64_t)hi << 1)) & 0x7;
    out32[22] = (hi >> 2) & 0x7;
    out32[23] = (hi >> 5) & 0x7;
    out32[24] = (hi >> 8) & 0x7;
    out32[25] = (hi >> 11) & 0x7;
    out32[26] = (hi >> 14) & 0x7;
    out32[27] = (hi >> 17) & 0x7;
    out32[28] = (hi >> 20) & 0x7;
    out32[29] = (hi >> 23) & 0x7;
    out32[30] = (hi >> 26) & 0x7;
    out32[31] = (hi >> 29) & 0x7;
}

/* Q3_K superblock: 112 bytes
 * [0..3]:   fp16 d + fp16 dmin
 * [4..15]:  12 bytes 6-bit packed scales (8 scales + 8 mins)
 * [16..111]: 96 bytes packed q3 values (8 sub-blocks × 12 bytes)
 */
#define Q3K_SUPERBLOCK_BYTES 112

/* 简化 Q3_K matmul — 使用标量反量化 + SIMD 点积
 * 这是初始版本, 后续可优化为完全 SIMD
 */
static inline void lal_matmul_q3_k(float * __restrict__ y,
                                     const uint8_t * __restrict__ q3k_W,
                                     const float * __restrict__ x,
                                     const float * __restrict__ b,
                                     int in_dim, int out_dim) {
    int n_super = in_dim / 256;

    /* Quantize x to int8 (same as Q4_K) */
    int8_t xq[XQ_MAX] __attribute__((aligned(32)));
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) { float a = fabsf(x[i]); if (a > x_max) x_max = a; }
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    float x_inv = 1.0f / x_scale;
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * x_inv);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }

    /* bsums for min correction */
    int n_sub = in_dim / 32;
    int16_t bsums[XQ_MAX / 32] __attribute__((aligned(32)));
    for (int sb = 0; sb < n_sub; sb++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) sum += (int)xq[sb * 32 + i];
        bsums[sb] = (int16_t)sum;
    }

    const float inv_63 = 1.0f / 63.0f;
    const float inv_7 = 1.0f / 7.0f;  /* Q3 has 8 levels (0-7), but we use 7 for normalization */

    int row_stride = n_super * Q3K_SUPERBLOCK_BYTES;

    /* Scalar path (correct, will optimize later) */
    for (int j = 0; j < out_dim; j++) {
        const uint8_t *row = q3k_W + (size_t)j * row_stride;
        float sumf = 0;

        for (int s = 0; s < n_super; s++) {
            const uint8_t *sb = row + s * Q3K_SUPERBLOCK_BYTES;
            float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb))));
            float dmin = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2))));
            uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);

            /* Min correction: sum_i min_i * xq_i */
            int32_t mp_sum = 0;
            for (int k = 0; k < 8; k++)
                mp_sum += (int)(int8_t)sm[8+k] * (int)bsums[s*8+k];
            float min_corr = dmin * x_scale * (float)mp_sum * inv_63;

            /* Dot product: w = (q3 * sub_scale) - min
             * dot = sum(w * x) = sum(q3 * sub_scale * x) - sum(min * x)
             *     = sub_scale * sum(q3 * xq * x_scale) - min * sum(xq * x_scale)
             *
             * But xq is quantized x: xq ≈ x / x_scale
             * So: sum(q3 * xq) * sub_scale * x_scale - min * bsum * x_scale
             *
             * sub_scale = d * sm[k] / 63
             * min = dmin * sm[8+k] / 63
             */
            /* Dot product with per-sub-block scaling
             * w[k][i] = q3[k][i] * (d * sm[k] / 63) - (dmin * sm[8+k] / 63)
             * dot = sum_k sum_i w[k][i] * x[i]
             *     = sum_k (d * sm[k] / 63) * sum_i(q3[k][i] * xq[i]) * x_scale
             *       - sum_k (dmin * sm[8+k] / 63) * sum_i(xq[i]) * x_scale
             *     = d * x_scale / 63 * sum_k(sm[k] * subdot_k)
             *       - dmin * x_scale / 63 * sum_k(sm[8+k] * bsums_k)
             */
            float scaled_dot = 0;
            for (int k = 0; k < 8; k++) {
                uint8_t q3[32];
                unpack_q3_32(sb + 16 + k * 12, q3);
                const int8_t *xqs = xq + s*256 + k*32;
                int32_t subdot = 0;
                for (int i = 0; i < 32; i++)
                    subdot += (int)q3[i] * (int)xqs[i];
                scaled_dot += (float)sm[k] * (float)subdot;
            }
            sumf += d * x_scale * inv_63 * scaled_dot - min_corr;
        }
        y[j] = sumf + (b ? b[j] : 0);
    }
}

/* Quantize a float row to Q3_K format (for testing) */
static inline void quantize_q3_k_row(const float *w, uint8_t *out, int in_dim) {
    int n_super = in_dim / 256;
    for (int s = 0; s < n_super; s++) {
        const float *wb = w + s * 256;
        uint8_t *sb = out + s * Q3K_SUPERBLOCK_BYTES;

        /* Find per-sub-block min/max */
        float sub_min[8], sub_max[8], sub_scale[8];
        float max_scale = 0, max_min = 0;
        for (int j = 0; j < 8; j++) {
            sub_min[j] = 1e30f; sub_max[j] = -1e30f;
            for (int i = 0; i < 32; i++) {
                float v = wb[j*32+i];
                if (v < sub_min[j]) sub_min[j] = v;
                if (v > sub_max[j]) sub_max[j] = v;
            }
            /* Q3 has 8 levels (0-7), so scale = (max-min)/7 */
            sub_scale[j] = (sub_max[j] - sub_min[j]) / 7.0f;
            if (sub_scale[j] < 1e-8f) sub_scale[j] = 1e-8f;
            if (sub_scale[j] > max_scale) max_scale = sub_scale[j];
            if (fabsf(sub_min[j]) > max_min) max_min = fabsf(sub_min[j]);
        }
        if (max_scale < 1e-8f) max_scale = 1e-8f;
        if (max_min < 1e-8f) max_min = 1e-8f;

        float d = max_scale, dmin = max_min;
        *(uint16_t*)(sb) = _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(d), 0), 0);
        *(uint16_t*)(sb+2) = _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(dmin), 0), 0);

        /* Pack 6-bit scales and mins (same as Q4_K) */
        uint8_t sc6[8], m6[8];
        for (int j = 0; j < 8; j++) {
            sc6[j] = (uint8_t)(lroundf(sub_scale[j] / d * 63.0f) & 0x3F);
            m6[j] = (uint8_t)(lroundf(fabsf(sub_min[j]) / dmin * 63.0f) & 0x3F);
        }
        __uint128_t bits = 0;
        for (int j = 0; j < 16; j++) {
            uint8_t val = j < 8 ? sc6[j] : m6[j-8];
            bits |= ((__uint128_t)(val & 0x3F)) << (j * 6);
        }
        for (int i = 0; i < 12; i++) sb[4+i] = (bits >> (i*8)) & 0xFF;

        /* Pack 3-bit q values */
        uint8_t *qs = sb + 16;
        for (int j = 0; j < 8; j++) {
            float ascale = d * sc6[j] / 63.0f;
            float amin = dmin * m6[j] / 63.0f;
            uint8_t q3[32];
            for (int i = 0; i < 32; i++) {
                int q = lroundf((wb[j*32+i] + amin) / (ascale + 1e-8f));
                if (q < 0) q = 0; if (q > 7) q = 7;
                q3[i] = (uint8_t)q;
            }
            /* Pack 32 × 3-bit into 12 bytes */
            uint8_t packed[12];
            memset(packed, 0, 12);
            for (int i = 0; i < 32; i++) {
                int bit_pos = i * 3;
                packed[bit_pos / 8] |= (q3[i] & 0x7) << (bit_pos % 8);
                if (bit_pos % 8 > 5) {
                    /* Spills into next byte */
                    packed[bit_pos / 8 + 1] |= (q3[i] >> (8 - bit_pos % 8)) & 0x7;
                }
            }
            memcpy(qs + j * 12, packed, 12);
        }
    }
}

#endif /* LAL_Q3K_KERNEL_H */
