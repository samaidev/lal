/* lal_q4k_kernel_v2.h — Q4_K matmul kernel v2: 预计算 scales
 *
 * 核心优化: 将 unpack_scales_6bit 从内循环移出, 批量预计算
 *
 * 原始 kernel: 每次 PROC8(r) 调用 unpack_scales_6bit (标量, ~15 cycles)
 *   → 8 rows × 14 superblocks × 15 cycles = 1680 cycles per 8-row block
 *   → 占总 matmul 时间的 54.5%!
 *
 * 优化 kernel: 在主循环前预计算所有 scales, 主循环只加载预计算结果
 *   → 预计算可以用 SIMD 向量化, 主循环更精简
 *   → 预计节省 50%+ 的 scale 解包时间
 */
#ifndef LAL_Q4K_KERNEL_V2_H
#define LAL_Q4K_KERNEL_V2_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX before including lal_q4k_kernel_v2.h"
#endif

#include "lal_q4k_kernel.h"  /* 复用 unpack_scales_6bit */

static inline void lal_matmul_q4_k_v2(float * __restrict__ y,
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

        /* === 预计算所有 scales 和 min corrections ===
         * 为 8 rows × n_super superblocks 预计算:
         * - d (fp16 → float)
         * - dmin (fp16 → float)
         * - 16 个 unpacked scales per superblock
         * - min correction (dmin * sum(mins * bsums))
         */
        float d_arr[8][16];      /* max 16 superblocks for in_dim=3584+ */
        float dmin_arr[8][16];
        uint8_t scales[8][16][16]; /* [row][superblock][16 scales] */
        float min_corr[8][16];   /* precomputed min correction */

        for (int r = 0; r < 8; r++) {
            for (int s = 0; s < n_super && s < 16; s++) {
                const uint8_t *sb = rows[r] + s * 144;
                d_arr[r][s] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb))));
                dmin_arr[r][s] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2))));
                unpack_scales_6bit(sb + 4, scales[r][s]);

                /* Precompute min correction */
                __m128i mn_bytes = _mm_loadl_epi64((__m128i*)(scales[r][s] + 8));
                __m128i mn_v = _mm_cvtepi8_epi16(mn_bytes);
                __m128i bs_v = _mm_set_epi16(
                    bsums[s*8+7], bsums[s*8+6], bsums[s*8+5], bsums[s*8+4],
                    bsums[s*8+3], bsums[s*8+2], bsums[s*8+1], bsums[s*8+0]);
                __m128i mp = _mm_madd_epi16(mn_v, bs_v);
                int32_t mp_sum = _mm_cvtsi128_si32(mp)
                    + _mm_extract_epi32(mp, 1)
                    + _mm_extract_epi32(mp, 2)
                    + _mm_extract_epi32(mp, 3);
                min_corr[r][s] = dmin_arr[r][s] * x_scale * (float)mp_sum * inv_63;
            }
        }

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

            /* Prefetch next superblock (keep original T0, 1-ahead) */
            _mm_prefetch((const char*)(rows[0] + (s+1)*144 + 16), _MM_HINT_T0);
            _mm_prefetch((const char*)(rows[0] + (s+1)*144 + 80), _MM_HINT_T0);

            #define PROC8_V2(r) do { \
                const uint8_t *qs=rows[r]+s*144+16; \
                float mult=d_arr[r][s]*x_scale*inv_63; \
                __m256i sumi=_mm256_setzero_si256(); \
                /* iter 0 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)qs); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve0), \
                                                  _mm256_maddubs_epi16(q4h,xvo0)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)scales[r][s][1]),_mm_set1_epi16((short)scales[r][s][0])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+32)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve1), \
                                                  _mm256_maddubs_epi16(q4h,xvo1)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)scales[r][s][3]),_mm_set1_epi16((short)scales[r][s][2])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+64)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve2), \
                                                  _mm256_maddubs_epi16(q4h,xvo2)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)scales[r][s][5]),_mm_set1_epi16((short)scales[r][s][4])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+96)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve3), \
                                                  _mm256_maddubs_epi16(q4h,xvo3)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)scales[r][s][7]),_mm_set1_epi16((short)scales[r][s][6])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                __m256 *ap=(r==0)?&acc0:(r==1)?&acc1:(r==2)?&acc2:(r==3)?&acc3: \
                            (r==4)?&acc4:(r==5)?&acc5:(r==6)?&acc6:&acc7; \
                *ap=_mm256_fmadd_ps(_mm256_set1_ps(mult),_mm256_cvtepi32_ps(sumi),*ap); \
                /* Apply min correction */ \
                if(r==0)am0-=min_corr[r][s];else if(r==1)am1-=min_corr[r][s]; \
                else if(r==2)am2-=min_corr[r][s];else if(r==3)am3-=min_corr[r][s]; \
                else if(r==4)am4-=min_corr[r][s];else if(r==5)am5-=min_corr[r][s]; \
                else if(r==6)am6-=min_corr[r][s];else am7-=min_corr[r][s]; \
            } while(0)

            PROC8_V2(0);PROC8_V2(1);PROC8_V2(2);PROC8_V2(3);
            PROC8_V2(4);PROC8_V2(5);PROC8_V2(6);PROC8_V2(7);
            #undef PROC8_V2
        }

        #define HS8_V2(r,a,m) do{__m128 lo=_mm256_castps256_ps128(a),hi=_mm256_extractf128_ps(a,1);\
            __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);\
            y[j+r]=_mm_cvtss_f32(s2)+m+(b?b[j+r]:0);}while(0)
        HS8_V2(0,acc0,am0);HS8_V2(1,acc1,am1);HS8_V2(2,acc2,am2);HS8_V2(3,acc3,am3);
        HS8_V2(4,acc4,am4);HS8_V2(5,acc5,am5);HS8_V2(6,acc6,am6);HS8_V2(7,acc7,am7);
        #undef HS8_V2
    }

    /* Tail: single-row (use original kernel) */
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

            #define SR_V2(qoff,xve,xvo,sa,sb_) do{\
                __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+qoff));\
                __m256i q4l=_mm256_and_si256(q4b,m4);\
                __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4);\
                __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve),_mm256_maddubs_epi16(q4h,xvo));\
                __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sb_),_mm_set1_epi16((short)sa));\
                sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16));}while(0)
            SR_V2(0,xve0,xvo0,sm[0],sm[1]);
            SR_V2(32,xve1,xvo1,sm[2],sm[3]);
            SR_V2(64,xve2,xvo2,sm[4],sm[5]);
            SR_V2(96,xve3,xvo3,sm[6],sm[7]);
            #undef SR_V2

            acc=_mm256_fmadd_ps(_mm256_set1_ps(d*x_scale*inv_63),_mm256_cvtepi32_ps(sumi),acc);
        }
        __m128 lo=_mm256_castps256_ps128(acc),hi=_mm256_extractf128_ps(acc,1);
        __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);
        y[j]=_mm_cvtss_f32(s2)+acc_min+(b?b[j]:0);
    }
#else
    lal_matmul_q4_k(y, q4k_W, x, b, in_dim, out_dim);
#endif
}

#endif /* LAL_Q4K_KERNEL_V2_H */
