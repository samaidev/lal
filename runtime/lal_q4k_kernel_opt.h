/* lal_q4k_kernel_opt.h — 优化版 Q4_K matmul kernel
 *
 * 优化点:
 * 1. Prefetch 2 superblocks ahead (vs 1) — 更好地隐藏 DRAM 延迟
 * 2. 使用 _MM_HINT_NTA 预取权重 — 避免权重数据污染 L2/L3 cache
 *    (权重是流式读取的，每个 weight 只用一次，不应该占用 cache)
 * 3. Prefetch 所有 3 个 cache line (vs 2) — 覆盖完整 144 字节 superblock
 * 4. 在 superblock 循环开始时预取所有 8 行的下一个 superblock
 */
#ifndef LAL_Q4K_KERNEL_OPT_H
#define LAL_Q4K_KERNEL_OPT_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX before including lal_q4k_kernel_opt.h"
#endif

/* 复用原 kernel 的 unpack_scales_6bit */
#include "lal_q4k_kernel.h"

static inline void lal_matmul_q4_k_opt(float * __restrict__ y,
                                        const uint8_t * __restrict__ q4k_W,
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
    int row_stride = n_super * 144;
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
        for (int r = 0; r < 8; r++) rows[r] = q4k_W + (size_t)(j+r) * row_stride;

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

            /* 优化: 预取 2 superblocks ahead, NTA hint (不污染 cache)
             * 权重是流式读取的，每个 weight byte 只用一次 */
            if (s + 2 < n_super) {
                for (int r = 0; r < 8; r++) {
                    const char *pf = (const char*)(rows[r] + (s+2)*144);
                    _mm_prefetch(pf, _MM_HINT_NTA);
                    _mm_prefetch(pf + 64, _MM_HINT_NTA);
                }
            }

            #define PROC8_OPT(r) do { \
                const uint8_t *sb=rows[r]+s*144; \
                float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb)))); \
                float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2)))); \
                uint8_t sm[16] __attribute__((aligned(16))); unpack_scales_6bit(sb+4,sm); \
                __m128i mn_bytes = _mm_loadl_epi64((__m128i*)(sm+8)); \
                __m128i mn_v = _mm_cvtepi8_epi16(mn_bytes); \
                __m128i mp=_mm_madd_epi16(mn_v,bs_v); \
                int32_t mp_sum = _mm_cvtsi128_si32(mp) \
                    + _mm_extract_epi32(mp, 1) \
                    + _mm_extract_epi32(mp, 2) \
                    + _mm_extract_epi32(mp, 3); \
                float r_am=dmin*x_scale*(float)mp_sum*inv_63; \
                if(r==0)am0-=r_am;else if(r==1)am1-=r_am;else if(r==2)am2-=r_am; \
                else if(r==3)am3-=r_am;else if(r==4)am4-=r_am;else if(r==5)am5-=r_am; \
                else if(r==6)am6-=r_am;else am7-=r_am; \
                __m256i sumi=_mm256_setzero_si256(); \
                const uint8_t*qs=sb+16; \
                /* iter 0 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)qs); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve0), \
                                                  _mm256_maddubs_epi16(q4h,xvo0)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[1]),_mm_set1_epi16((short)sm[0])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+32)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve1), \
                                                  _mm256_maddubs_epi16(q4h,xvo1)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[3]),_mm_set1_epi16((short)sm[2])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+64)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve2), \
                                                  _mm256_maddubs_epi16(q4h,xvo2)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[5]),_mm_set1_epi16((short)sm[4])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+96)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve3), \
                                                  _mm256_maddubs_epi16(q4h,xvo3)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[7]),_mm_set1_epi16((short)sm[6])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                float mult=d*x_scale*inv_63; \
                __m256 *ap=(r==0)?&acc0:(r==1)?&acc1:(r==2)?&acc2:(r==3)?&acc3: \
                            (r==4)?&acc4:(r==5)?&acc5:(r==6)?&acc6:&acc7; \
                *ap=_mm256_fmadd_ps(_mm256_set1_ps(mult),_mm256_cvtepi32_ps(sumi),*ap); \
            } while(0)

            PROC8_OPT(0);PROC8_OPT(1);PROC8_OPT(2);PROC8_OPT(3);
            PROC8_OPT(4);PROC8_OPT(5);PROC8_OPT(6);PROC8_OPT(7);
            #undef PROC8_OPT
        }

        #define HS8_OPT(r,a,m) do{__m128 lo=_mm256_castps256_ps128(a),hi=_mm256_extractf128_ps(a,1);\
            __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);\
            y[j+r]=_mm_cvtss_f32(s2)+m+(b?b[j+r]:0);}while(0)
        HS8_OPT(0,acc0,am0);HS8_OPT(1,acc1,am1);HS8_OPT(2,acc2,am2);HS8_OPT(3,acc3,am3);
        HS8_OPT(4,acc4,am4);HS8_OPT(5,acc5,am5);HS8_OPT(6,acc6,am6);HS8_OPT(7,acc7,am7);
        #undef HS8_OPT
    }

    /* Tail: single-row */
    for (; j < out_dim; j++) {
        const uint8_t * __restrict__ row = q4k_W + (size_t)j * row_stride;
        __m256 acc = _mm256_setzero_ps();
        float acc_min = 0;
        for (int s = 0; s < n_super; s++) {
            const uint8_t * __restrict__ sb = row + s * 144;
            float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb))));
            float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2))));
            uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);
            __m128i mn_v=_mm_set_epi16(sm[15],sm[14],sm[13],sm[12],sm[11],sm[10],sm[9],sm[8]);
            __m128i bs_v=_mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);
            __m128i mp=_mm_madd_epi16(mn_v,bs_v);mp=_mm_hadd_epi32(mp,mp);mp=_mm_hadd_epi32(mp,mp);
            acc_min -= dmin*x_scale*(float)_mm_cvtsi128_si32(mp)*inv_63;

            __m256i sumi=_mm256_setzero_si256();
            const uint8_t*qs=sb+16;
            const __m256i xve0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+0));
            const __m256i xve1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+32));
            const __m256i xve2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+64));
            const __m256i xve3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+96));
            const __m256i xvo0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+128));
            const __m256i xvo1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+160));
            const __m256i xvo2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+192));
            const __m256i xvo3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+224));

            #define SR_OPT(qoff,xve,xvo,sa,sb_) do{\
                __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+qoff));\
                __m256i q4l=_mm256_and_si256(q4b,m4);\
                __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4);\
                __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve),_mm256_maddubs_epi16(q4h,xvo));\
                __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sb_),_mm_set1_epi16((short)sa));\
                sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16));}while(0)
            SR_OPT(0,xve0,xvo0,sm[0],sm[1]);
            SR_OPT(32,xve1,xvo1,sm[2],sm[3]);
            SR_OPT(64,xve2,xvo2,sm[4],sm[5]);
            SR_OPT(96,xve3,xvo3,sm[6],sm[7]);
            #undef SR_OPT

            /* NTA prefetch for tail too */
            if (s + 2 < n_super) {
                _mm_prefetch((const char*)(sb + 2*144), _MM_HINT_NTA);
            }

            acc=_mm256_fmadd_ps(_mm256_set1_ps(d*x_scale*inv_63),_mm256_cvtepi32_ps(sumi),acc);
        }
        __m128 lo=_mm256_castps256_ps128(acc),hi=_mm256_extractf128_ps(acc,1);
        __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);
        y[j]=_mm_cvtss_f32(s2)+acc_min+(b?b[j]:0);
    }
#else
    /* Scalar fallback — just call original */
    lal_matmul_q4_k(y, q4k_W, x, b, in_dim, out_dim);
#endif
}

#endif /* LAL_Q4K_KERNEL_OPT_H */
