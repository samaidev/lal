/* lal_q4k2_kernel.h — Q4_K2 格式: 预解包 scales 的 Q4_K 变体
 *
 * 核心优化: 将 Q4_K 的 6-bit packed scales (12 bytes) 改为 8-bit unpacked (16 bytes)
 *
 * Q4_K 格式:  4(fp16 d/dmin) + 12(6-bit packed scales) + 128(packed q4) = 144 bytes/256elem
 * Q4_K2 格式: 4(fp16 d/dmin) + 16(8-bit unpacked scales) + 128(packed q4) = 148 bytes/256elem
 *
 * 数据量增加: 148/144 = 2.8% — 几乎可忽略
 * 性能提升: 消除 unpack_scales_6bit 调用 (16次标量 shift+mask → 1次 SIMD load)
 *
 * 理论带宽提升: Q8 kernel 达到 10.9 GB/s (简单反量化), Q4_K 只有 6.5 GB/s
 * Q4_K2 预期接近 Q8 的带宽利用率, 同时保持 Q4 的数据量
 */
#ifndef LAL_Q4K2_KERNEL_H
#define LAL_Q4K2_KERNEL_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX before including lal_q4k2_kernel.h"
#endif

/* Q4_K2 superblock: 148 bytes
 *   [0..1]:   fp16 d (superblock scale)
 *   [2..3]:   fp16 dmin (superblock min scale)
 *   [4..19]:  16 × uint8 scales (8-bit, pre-unpacked from 6-bit)
 *   [20..35]: 16 × uint8 mins (8-bit, pre-unpacked from 6-bit)
 *   [36..163]: 128 bytes packed q4 (256 × 4 bits, ADJACENT packing)
 *
 * 注意: 实际布局调整为 4 + 32 + 128 = 164 bytes (scales+mins 连续 32 bytes)
 * 为了对齐, pad 到 168 bytes? 不, 保持紧凑 164 bytes.
 * 实际: 4 + 16 + 16 + 128 = 164 bytes/256elem = 0.6406 bytes/elem
 * vs Q4_K: 144 bytes/256elem = 0.5625 bytes/elem
 * 增加: 164/144 = 13.9% — 可接受, 因为带宽利用率会大幅提升
 */
#define Q4K2_SUPERBLOCK_BYTES 164

static inline void lal_matmul_q4_k2(float * __restrict__ y,
                                      const uint8_t * __restrict__ q4k2_W,
                                      const float * __restrict__ x,
                                      const float * __restrict__ b,
                                      int in_dim, int out_dim) {
    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;

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

    int16_t bsums[XQ_MAX / 32] __attribute__((aligned(32)));
    for (int sb = 0; sb < n_sub; sb++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) sum += (int)xq[sb * 32 + i];
        bsums[sb] = (int16_t)sum;
    }

#if defined(__AVX2__) && defined(__F16C__)
    int row_stride = n_super * Q4K2_SUPERBLOCK_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const float inv_63 = 1.0f / 63.0f;

    /* Pre-arrange xq for ADJACENT packing */
    int8_t xq_arr[XQ_MAX] __attribute__((aligned(32)));
    for (int s = 0; s < n_super; s++) {
        int8_t *dst_even = xq_arr + s*256;
        int8_t *dst_odd  = xq_arr + s*256 + 128;
        for (int k = 0; k < 8; k++) {
            const int8_t *src = xq + s*256 + k*32;
            for (int i = 0; i < 16; i++) {
                dst_even[k*16 + i] = src[2*i];
                dst_odd[k*16 + i]  = src[2*i + 1];
            }
        }
    }

    int j = 0;
    for (; j + 8 <= out_dim; j += 8) {
        const uint8_t * __restrict__ rows[8];
        for (int r = 0; r < 8; r++) rows[r] = q4k2_W + (size_t)(j+r) * row_stride;

        __m256 acc0=_mm256_setzero_ps(),acc1=_mm256_setzero_ps();
        __m256 acc2=_mm256_setzero_ps(),acc3=_mm256_setzero_ps();
        __m256 acc4=_mm256_setzero_ps(),acc5=_mm256_setzero_ps();
        __m256 acc6=_mm256_setzero_ps(),acc7=_mm256_setzero_ps();
        float am0=0,am1=0,am2=0,am3=0,am4=0,am5=0,am6=0,am7=0;

        for (int s = 0; s < n_super; s++) {
            const __m256i xve0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+0));
            const __m256i xve1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+32));
            const __m256i xve2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+64));
            const __m256i xve3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+96));
            const __m256i xvo0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+128));
            const __m256i xvo1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+160));
            const __m256i xvo2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+192));
            const __m256i xvo3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+224));

            __m128i bs_v=_mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);

            /* Prefetch */
            _mm_prefetch((const char*)(rows[0] + (s+1)*Q4K2_SUPERBLOCK_BYTES + 36), _MM_HINT_T0);
            _mm_prefetch((const char*)(rows[0] + (s+1)*Q4K2_SUPERBLOCK_BYTES + 100), _MM_HINT_T0);

            #define PROC8_K2(r) do { \
                const uint8_t *sb=rows[r]+s*Q4K2_SUPERBLOCK_BYTES; \
                float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb)))); \
                float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2)))); \
                /* 直接 load 16 bytes scales + 16 bytes mins — 无需 unpack_scales_6bit! */ \
                const uint8_t *sc = sb + 4;  /* 16 uint8 scales */ \
                const uint8_t *mn = sb + 20; /* 16 uint8 mins */ \
                /* Min correction */ \
                __m128i mn_bytes = _mm_loadu_si128((__m128i*)mn); /* 16 bytes, 取低 8 */ \
                __m128i mn_v = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)mn)); /* 8 int16 */ \
                __m128i mp=_mm_madd_epi16(mn_v,bs_v); \
                int32_t mp_sum = _mm_cvtsi128_si32(mp) \
                    + _mm_extract_epi32(mp, 1) \
                    + _mm_extract_epi32(mp, 2) \
                    + _mm_extract_epi32(mp, 3); \
                float r_am=dmin*x_scale*(float)mp_sum*inv_63; \
                if(r==0)am0-=r_am;else if(r==1)am1-=r_am;else if(r==2)am2-=r_am; \
                else if(r==3)am3-=r_am;else if(r==4)am4-=r_am;else if(r==5)am5-=r_am; \
                else if(r==6)am6-=r_am;else am7-=r_am; \
                /* 4 iterations: 直接用 sc[] 数组, 无需解包 */ \
                __m256i sumi=_mm256_setzero_si256(); \
                const uint8_t*qs=sb+36; /* packed q4 data starts at offset 36 */ \
                /* iter 0 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)qs); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve0), \
                                                  _mm256_maddubs_epi16(q4h,xvo0)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sc[1]),_mm_set1_epi16((short)sc[0])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+32)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve1), \
                                                  _mm256_maddubs_epi16(q4h,xvo1)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sc[3]),_mm_set1_epi16((short)sc[2])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+64)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve2), \
                                                  _mm256_maddubs_epi16(q4h,xvo2)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sc[5]),_mm_set1_epi16((short)sc[4])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+96)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve3), \
                                                  _mm256_maddubs_epi16(q4h,xvo3)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sc[7]),_mm_set1_epi16((short)sc[6])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                float mult=d*x_scale*inv_63; \
                __m256 *ap=(r==0)?&acc0:(r==1)?&acc1:(r==2)?&acc2:(r==3)?&acc3: \
                            (r==4)?&acc4:(r==5)?&acc5:(r==6)?&acc6:&acc7; \
                *ap=_mm256_fmadd_ps(_mm256_set1_ps(mult),_mm256_cvtepi32_ps(sumi),*ap); \
            } while(0)

            PROC8_K2(0);PROC8_K2(1);PROC8_K2(2);PROC8_K2(3);
            PROC8_K2(4);PROC8_K2(5);PROC8_K2(6);PROC8_K2(7);
            #undef PROC8_K2
        }

        #define HS8_K2(r,a,m) do{__m128 lo=_mm256_castps256_ps128(a),hi=_mm256_extractf128_ps(a,1);\
            __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);\
            y[j+r]=_mm_cvtss_f32(s2)+m+(b?b[j+r]:0);}while(0)
        HS8_K2(0,acc0,am0);HS8_K2(1,acc1,am1);HS8_K2(2,acc2,am2);HS8_K2(3,acc3,am3);
        HS8_K2(4,acc4,am4);HS8_K2(5,acc5,am5);HS8_K2(6,acc6,am6);HS8_K2(7,acc7,am7);
        #undef HS8_K2
    }

    /* Tail: single-row */
    for (; j < out_dim; j++) {
        const uint8_t * __restrict__ sb0 = q4k2_W + (size_t)j * row_stride;
        __m256 acc = _mm256_setzero_ps();
        float acc_min = 0;
        for (int s = 0; s < n_super; s++) {
            const uint8_t *sb = sb0 + s * Q4K2_SUPERBLOCK_BYTES;
            float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb))));
            float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2))));
            const uint8_t *sc = sb + 4;
            const uint8_t *mn = sb + 20;
            __m128i mn_v=_mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)mn));
            __m128i bs_v=_mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);
            __m128i mp=_mm_madd_epi16(mn_v,bs_v);mp=_mm_hadd_epi32(mp,mp);mp=_mm_hadd_epi32(mp,mp);
            acc_min -= dmin*x_scale*(float)_mm_cvtsi128_si32(mp)*inv_63;

            __m256i sumi=_mm256_setzero_si256();
            const uint8_t*qs=sb+36;
            const __m256i xve0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+0));
            const __m256i xve1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+32));
            const __m256i xve2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+64));
            const __m256i xve3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+96));
            const __m256i xvo0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+128));
            const __m256i xvo1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+160));
            const __m256i xvo2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+192));
            const __m256i xvo3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+224));

            #define SR_K2(qoff,xve,xvo,sa,sb_) do{\
                __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+qoff));\
                __m256i q4l=_mm256_and_si256(q4b,m4);\
                __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4);\
                __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve),_mm256_maddubs_epi16(q4h,xvo));\
                __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sb_),_mm_set1_epi16((short)sa));\
                sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16));}while(0)
            SR_K2(0,xve0,xvo0,sc[0],sc[1]);
            SR_K2(32,xve1,xvo1,sc[2],sc[3]);
            SR_K2(64,xve2,xvo2,sc[4],sc[5]);
            SR_K2(96,xve3,xvo3,sc[6],sc[7]);
            #undef SR_K2

            acc=_mm256_fmadd_ps(_mm256_set1_ps(d*x_scale*inv_63),_mm256_cvtepi32_ps(sumi),acc);
        }
        __m128 lo=_mm256_castps256_ps128(acc),hi=_mm256_extractf128_ps(acc,1);
        __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);
        y[j]=_mm_cvtss_f32(s2)+acc_min+(b?b[j]:0);
    }
#else
    #error "Q4_K2 requires AVX2 + F16C"
#endif
}

#endif /* LAL_Q4K2_KERNEL_H */
