/* lal_q4k_kernel_vnni.h — Q4_K matmul kernel using AVX-512 VNNI (simplified)
 *
 * 用 _mm256_dpbusd_epi32 替代 maddubs+madd, 标量累加结果
 * 每个 sub-block pair 的 dot product 用 VNNI 一次完成
 */
#ifndef LAL_Q4K_KERNEL_VNNI_H
#define LAL_Q4K_KERNEL_VNNI_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX before including lal_q4k_kernel_vnni.h"
#endif

#if !defined(__AVX512VNNI__)
#error "AVX512_VNNI required"
#endif

#include "lal_q4k_kernel.h"

static inline void lal_matmul_q4_k_vnni(float * __restrict__ y,
                                          const uint8_t * __restrict__ q4k_W,
                                          const float * __restrict__ x,
                                          const float * __restrict__ b,
                                          int in_dim, int out_dim) {
    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;

    int8_t xq[XQ_MAX] __attribute__((aligned(64)));
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) { float a = fabsf(x[i]); if (a > x_max) x_max = a; }
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    float x_inv = 1.0f / x_scale;
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * x_inv);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }

    int16_t bsums[XQ_MAX / 32] __attribute__((aligned(64)));
    for (int sb = 0; sb < n_sub; sb++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) sum += (int)xq[sb * 32 + i];
        bsums[sb] = (int16_t)sum;
    }

    /* Pre-arrange xq: even/odd split per sub-block (16 values each) */
    int8_t xq_even[XQ_MAX / 2] __attribute__((aligned(64)));
    int8_t xq_odd[XQ_MAX / 2] __attribute__((aligned(64)));
    for (int s = 0; s < n_super; s++) {
        for (int k = 0; k < 8; k++) {
            const int8_t *src = xq + s*256 + k*32;
            int8_t *dst_e = xq_even + s*128 + k*16;
            int8_t *dst_o = xq_odd + s*128 + k*16;
            for (int i = 0; i < 16; i++) {
                dst_e[i] = src[2*i];
                dst_o[i] = src[2*i + 1];
            }
        }
    }

    int row_stride = n_super * 144;
    const float inv_63 = 1.0f / 63.0f;
    const __m256i vm4 = _mm256_set1_epi8(0x0F);

    int j = 0;
    /* 8-row parallel with VNNI (256-bit) */
    for (; j + 8 <= out_dim; j += 8) {
        const uint8_t * __restrict__ rows[8];
        for (int r = 0; r < 8; r++) rows[r] = q4k_W + (size_t)(j+r) * row_stride;

        /* Use scalar accumulators per row (VNNI gives us the dot product directly) */
        float acc[8] = {0};
        float amin[8] = {0};

        for (int s = 0; s < n_super; s++) {
            __m128i bs_v = _mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);

            for (int r = 0; r < 8; r++) {
                const uint8_t *sb_ptr = rows[r] + s * 144;
                float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb_ptr))));
                float dmin = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb_ptr+2))));
                uint8_t sm[16] __attribute__((aligned(16)));
                unpack_scales_6bit(sb_ptr + 4, sm);

                /* Min correction */
                __m128i mn_v = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)(sm+8)));
                __m128i mp = _mm_madd_epi16(mn_v, bs_v);
                int32_t mp_sum = _mm_cvtsi128_si32(mp) + _mm_extract_epi32(mp,1)
                    + _mm_extract_epi32(mp,2) + _mm_extract_epi32(mp,3);
                amin[r] -= dmin * x_scale * (float)mp_sum * inv_63;

                /* VNNI dot product: 4 iterations of 2 sub-blocks (32 bytes each) */
                const uint8_t *qs = sb_ptr + 16;
                int32_t dot_total = 0;

                for (int iter = 0; iter < 4; iter++) {
                    __m256i q4b = _mm256_loadu_si256((__m256i*)(qs + iter*32));
                    __m256i q4l = _mm256_and_si256(q4b, vm4);
                    __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4b, 4), vm4);
                    __m256i xq_e = _mm256_loadu_si256((__m256i*)(xq_even + s*128 + iter*32));
                    __m256i xq_o = _mm256_loadu_si256((__m256i*)(xq_odd + s*128 + iter*32));

                    __m256i tmp = _mm256_setzero_si256();
                    tmp = _mm256_dpbusd_epi32(tmp, q4l, xq_e);
                    tmp = _mm256_dpbusd_epi32(tmp, q4h, xq_o);

                    /* Scale by sm[iter*2] and sm[iter*2+1] */
                    __m256i sc_v = _mm256_set_epi32(
                        sm[iter*2+1], sm[iter*2+1], sm[iter*2+1], sm[iter*2+1],
                        sm[iter*2],   sm[iter*2],   sm[iter*2],   sm[iter*2]);
                    tmp = _mm256_mullo_epi32(tmp, sc_v);

                    /* Horizontal sum of 8 int32 */
                    int32_t partial = _mm256_cvtsi256_si32(tmp);
                    partial += _mm256_extract_epi32(tmp, 1);
                    partial += _mm256_extract_epi32(tmp, 2);
                    partial += _mm256_extract_epi32(tmp, 3);
                    partial += _mm256_extract_epi32(tmp, 4);
                    partial += _mm256_extract_epi32(tmp, 5);
                    partial += _mm256_extract_epi32(tmp, 6);
                    partial += _mm256_extract_epi32(tmp, 7);
                    dot_total += partial;
                }

                acc[r] += d * x_scale * inv_63 * (float)dot_total;
            }
        }

        for (int r = 0; r < 8; r++)
            y[j+r] = acc[r] + amin[r] + (b ? b[j+r] : 0);
    }

    /* Tail: single row */
    for (; j < out_dim; j++) {
        const uint8_t *sb0 = q4k_W + (size_t)j * row_stride;
        float acc_val = 0, acc_min = 0;
        for (int s = 0; s < n_super; s++) {
            const uint8_t *sb = sb0 + s * 144;
            float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb))));
            float dmin = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2))));
            uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);
            __m128i mn_v = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)(sm+8)));
            __m128i bs_v = _mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);
            __m128i mp = _mm_madd_epi16(mn_v, bs_v);
            mp = _mm_hadd_epi32(mp, mp); mp = _mm_hadd_epi32(mp, mp);
            acc_min -= dmin * x_scale * (float)_mm_cvtsi128_si32(mp) * inv_63;

            const uint8_t *qs = sb + 16;
            int32_t dot_total = 0;
            for (int iter = 0; iter < 4; iter++) {
                __m256i q4b = _mm256_loadu_si256((__m256i*)(qs + iter*32));
                __m256i q4l = _mm256_and_si256(q4b, vm4);
                __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4b, 4), vm4);
                __m256i xq_e = _mm256_loadu_si256((__m256i*)(xq_even + s*128 + iter*32));
                __m256i xq_o = _mm256_loadu_si256((__m256i*)(xq_odd + s*128 + iter*32));
                __m256i tmp = _mm256_setzero_si256();
                tmp = _mm256_dpbusd_epi32(tmp, q4l, xq_e);
                tmp = _mm256_dpbusd_epi32(tmp, q4h, xq_o);
                __m256i sc_v = _mm256_set_epi32(
                    sm[iter*2+1], sm[iter*2+1], sm[iter*2+1], sm[iter*2+1],
                    sm[iter*2],   sm[iter*2],   sm[iter*2],   sm[iter*2]);
                tmp = _mm256_mullo_epi32(tmp, sc_v);
                int32_t partial = _mm256_cvtsi256_si32(tmp);
                for (int i = 1; i < 8; i++) partial += _mm256_extract_epi32(tmp, i);
                dot_total += partial;
            }
            acc_val += d * x_scale * inv_63 * (float)dot_total;
        }
        y[j] = acc_val + acc_min + (b ? b[j] : 0);
    }
}

#endif /* LAL_Q4K_KERNEL_VNNI_H */
