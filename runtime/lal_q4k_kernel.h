/* lal_q4k_kernel.h — Q4_K matmul kernel (128-bit maddubs, ADJACENT packing)
 *
 * ADJACENT packing (llama.cpp style, matches Python converter):
 *   byte[sub*16+i] = q[sub*32+2i] | (q[sub*32+2i+1] << 4)
 *
 * Per sub-block (16 packed bytes = 32 q4 values):
 *   q4l = q4 & 0xF  → 16 uint8 (even-indexed q values)
 *   q4h = q4 >> 4   → 16 uint8 (odd-indexed q values)
 *   xq  = 32 int8 for this sub-block
 *   Split xq into even/odd halves (16 each)
 *   pl = maddubs(q4l, xq_even) → 8 int16
 *   ph = maddubs(q4h, xq_odd)  → 8 int16
 *   p16 = pl + ph
 *   s = sc * p16 → 4 int32
 *
 * Process 2 sub-blocks per iteration (32 packed bytes = 2×16).
 * 8-row parallel, bsums for min correction, 1 cvtepi32_ps per superblock.
 */
#ifndef LAL_Q4K_KERNEL_H
#define LAL_Q4K_KERNEL_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX before including lal_q4k_kernel.h"
#endif

/* Unpack 16 × 6-bit values from 12 bytes (packed low-to-high) into 16 uint8.
 * Used by both SIMD and scalar paths. */
static inline void unpack_scales_6bit(const uint8_t *src, uint8_t *out16) {
    uint64_t lo = *(const uint64_t*)src;
    uint32_t hi = *(const uint32_t*)(src + 8);
    for (int i = 0; i < 10; i++) out16[i] = (lo >> (i * 6)) & 0x3F;
    out16[10] = ((lo >> 60) | (hi << 4)) & 0x3F;
    for (int i = 0; i < 5; i++) out16[11 + i] = (hi >> (2 + i * 6)) & 0x3F;
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* Use AVX-512 kernel (512-bit maddubs) when available */
#include "lal_q4k_kernel_avx512.h"
/* lal_matmul_q4_k is defined as lal_matmul_q4_k_avx512 in the avx512 header */
#define lal_matmul_q4_k lal_matmul_q4_k_avx512

#else
static inline void lal_matmul_q4_k(float * __restrict__ y,
                                     const uint8_t * __restrict__ q4k_W,
                                     const float * __restrict__ x,
                                     const float * __restrict__ b,
                                     int in_dim, int out_dim) {
    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;

    /* Single-scale x quantization to int8 */
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

    /* Pre-compute bsums (sum of int8 xq per sub-block) for min correction */
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

    /* Pre-arrange xq for ADJACENT packing: split each 32-byte sub-block into
     * 16 even-indexed + 16 odd-indexed elements.
     * Layout per superblock: [sub0-7 even (128 bytes), sub0-7 odd (128 bytes)]
     */
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

    /* 8-row parallel main loop */
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
            /* Pre-load xq_arr for this superblock: 8 YMM (4 even pairs + 4 odd pairs)
             * xq_arr layout: [sub0-7 even (128), sub0-7 odd (128)]
             *   xv0 = [sub0_even, sub1_even]  (for iter 0 even half)
             *   xv1 = [sub2_even, sub3_even]  (for iter 1 even half)
             *   xv2 = [sub4_even, sub5_even]  (for iter 2 even half)
             *   xv3 = [sub6_even, sub7_even]  (for iter 3 even half)
             *   xv4 = [sub0_odd,  sub1_odd]   (for iter 0 odd half)
             *   xv5 = [sub2_odd,  sub3_odd]   (for iter 1 odd half)
             *   xv6 = [sub4_odd,  sub5_odd]   (for iter 2 odd half)
             *   xv7 = [sub6_odd,  sub7_odd]   (for iter 3 odd half)
             */
            const __m256i xv0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+0));
            const __m256i xv1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+32));
            const __m256i xv2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+64));
            const __m256i xv3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+96));
            const __m256i xv4=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+128));
            const __m256i xv5=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+160));
            const __m256i xv6=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+192));
            const __m256i xv7=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+224));

            __m128i bs_v=_mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);

            /* For ADJACENT packing:
             *   q4l = q4 & 0xF → even-indexed q values (16 uint8)
             *   q4h = q4 >> 4  → odd-indexed q values (16 uint8)
             *   xq_even = first 16 bytes of xv (xq[0,2,4,...,30])
             *   xq_odd  = last 16 bytes of xv  (xq[1,3,5,...,31])
             *   maddubs(q4l, xq_even) → 8 int16 partial dots
             *   maddubs(q4h, xq_odd)  → 8 int16 partial dots
             *   p16 = lo + hi → 8 int16 (scaled by sc later)
             *
             * Per iteration: 2 sub-blocks (32 packed bytes)
             *   qs[0..15] = sub_a packed, qs[16..31] = sub_b packed
             *   q4b = load 32 bytes
             *   q4l = q4b & 0xF → [sub_a_even(16), sub_b_even(16)]
             *   q4h = q4b >> 4  → [sub_a_odd(16),  sub_b_odd(16)]
             *   q4ll = low 128 = sub_a_even, q4lh = high 128 = sub_b_even
             *   q4hl = low 128 = sub_a_odd,  q4hh = high 128 = sub_b_odd
             *   xq_a_even = low 128 of xv0 = xq[0,2,..,30] of sub_a
             *   xq_a_odd  = high 128 of xv0 = xq[1,3,..,31] of sub_a
             *   xq_b_even = low 128 of xv1
             *   xq_b_odd  = high 128 of xv1
             */
            #define PROC8(r) do { \
                const uint8_t *sb=rows[r]+s*144; \
                float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb)))); \
                float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2)))); \
                uint8_t sm[16]; unpack_scales_6bit(sb+4,sm); \
                __m128i mn_v=_mm_set_epi16(sm[15],sm[14],sm[13],sm[12],sm[11],sm[10],sm[9],sm[8]); \
                __m128i mp=_mm_madd_epi16(mn_v,bs_v); \
                mp=_mm_hadd_epi32(mp,mp); mp=_mm_hadd_epi32(mp,mp); \
                float r_am=dmin*x_scale*(float)_mm_cvtsi128_si32(mp)*inv_63; \
                if(r==0)am0-=r_am;else if(r==1)am1-=r_am;else if(r==2)am2-=r_am; \
                else if(r==3)am3-=r_am;else if(r==4)am4-=r_am;else if(r==5)am5-=r_am; \
                else if(r==6)am6-=r_am;else am7-=r_am; \
                __m256i sumi=_mm256_setzero_si256(); \
                const uint8_t*qs=sb+16; \
                /* iter 0: sub0+sub1. even=xv0, odd=xv4 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)qs); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m128i q4ll=_mm256_castsi256_si128(q4l),q4lh=_mm256_extracti128_si256(q4l,1); \
                    __m128i q4hl=_mm256_castsi256_si128(q4h),q4hh=_mm256_extracti128_si256(q4h,1); \
                    __m128i xa_e=_mm256_castsi256_si128(xv0),xb_e=_mm256_extracti128_si256(xv0,1); \
                    __m128i xa_o=_mm256_castsi256_si128(xv4),xb_o=_mm256_extracti128_si256(xv4,1); \
                    __m128i pl=_mm_maddubs_epi16(q4ll,xa_e),ph=_mm_maddubs_epi16(q4hl,xa_o); \
                    __m128i pl2=_mm_maddubs_epi16(q4lh,xb_e),ph2=_mm_maddubs_epi16(q4hh,xb_o); \
                    __m128i sc0=_mm_set1_epi16((short)sm[0]),sc1=_mm_set1_epi16((short)sm[1]); \
                    __m128i s0=_mm_add_epi32(_mm_madd_epi16(sc0,pl),_mm_madd_epi16(sc0,ph)); \
                    __m128i s1=_mm_add_epi32(_mm_madd_epi16(sc1,pl2),_mm_madd_epi16(sc1,ph2)); \
                    sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0)); \
                } \
                /* iter 1: sub2+sub3. even=xv1, odd=xv5 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+32)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m128i q4ll=_mm256_castsi256_si128(q4l),q4lh=_mm256_extracti128_si256(q4l,1); \
                    __m128i q4hl=_mm256_castsi256_si128(q4h),q4hh=_mm256_extracti128_si256(q4h,1); \
                    __m128i xa_e=_mm256_castsi256_si128(xv1),xb_e=_mm256_extracti128_si256(xv1,1); \
                    __m128i xa_o=_mm256_castsi256_si128(xv5),xb_o=_mm256_extracti128_si256(xv5,1); \
                    __m128i pl=_mm_maddubs_epi16(q4ll,xa_e),ph=_mm_maddubs_epi16(q4hl,xa_o); \
                    __m128i pl2=_mm_maddubs_epi16(q4lh,xb_e),ph2=_mm_maddubs_epi16(q4hh,xb_o); \
                    __m128i sc0=_mm_set1_epi16((short)sm[2]),sc1=_mm_set1_epi16((short)sm[3]); \
                    __m128i s0=_mm_add_epi32(_mm_madd_epi16(sc0,pl),_mm_madd_epi16(sc0,ph)); \
                    __m128i s1=_mm_add_epi32(_mm_madd_epi16(sc1,pl2),_mm_madd_epi16(sc1,ph2)); \
                    sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0)); \
                } \
                /* iter 2: sub4+sub5. even=xv2, odd=xv6 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+64)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m128i q4ll=_mm256_castsi256_si128(q4l),q4lh=_mm256_extracti128_si256(q4l,1); \
                    __m128i q4hl=_mm256_castsi256_si128(q4h),q4hh=_mm256_extracti128_si256(q4h,1); \
                    __m128i xa_e=_mm256_castsi256_si128(xv2),xb_e=_mm256_extracti128_si256(xv2,1); \
                    __m128i xa_o=_mm256_castsi256_si128(xv6),xb_o=_mm256_extracti128_si256(xv6,1); \
                    __m128i pl=_mm_maddubs_epi16(q4ll,xa_e),ph=_mm_maddubs_epi16(q4hl,xa_o); \
                    __m128i pl2=_mm_maddubs_epi16(q4lh,xb_e),ph2=_mm_maddubs_epi16(q4hh,xb_o); \
                    __m128i sc0=_mm_set1_epi16((short)sm[4]),sc1=_mm_set1_epi16((short)sm[5]); \
                    __m128i s0=_mm_add_epi32(_mm_madd_epi16(sc0,pl),_mm_madd_epi16(sc0,ph)); \
                    __m128i s1=_mm_add_epi32(_mm_madd_epi16(sc1,pl2),_mm_madd_epi16(sc1,ph2)); \
                    sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0)); \
                } \
                /* iter 3: sub6+sub7. even=xv3, odd=xv7 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+96)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m128i q4ll=_mm256_castsi256_si128(q4l),q4lh=_mm256_extracti128_si256(q4l,1); \
                    __m128i q4hl=_mm256_castsi256_si128(q4h),q4hh=_mm256_extracti128_si256(q4h,1); \
                    __m128i xa_e=_mm256_castsi256_si128(xv3),xb_e=_mm256_extracti128_si256(xv3,1); \
                    __m128i xa_o=_mm256_castsi256_si128(xv7),xb_o=_mm256_extracti128_si256(xv7,1); \
                    __m128i pl=_mm_maddubs_epi16(q4ll,xa_e),ph=_mm_maddubs_epi16(q4hl,xa_o); \
                    __m128i pl2=_mm_maddubs_epi16(q4lh,xb_e),ph2=_mm_maddubs_epi16(q4hh,xb_o); \
                    __m128i sc0=_mm_set1_epi16((short)sm[6]),sc1=_mm_set1_epi16((short)sm[7]); \
                    __m128i s0=_mm_add_epi32(_mm_madd_epi16(sc0,pl),_mm_madd_epi16(sc0,ph)); \
                    __m128i s1=_mm_add_epi32(_mm_madd_epi16(sc1,pl2),_mm_madd_epi16(sc1,ph2)); \
                    sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0)); \
                } \
                float mult=d*x_scale*inv_63; \
                __m256 *ap=(r==0)?&acc0:(r==1)?&acc1:(r==2)?&acc2:(r==3)?&acc3: \
                            (r==4)?&acc4:(r==5)?&acc5:(r==6)?&acc6:&acc7; \
                *ap=_mm256_fmadd_ps(_mm256_set1_ps(mult),_mm256_cvtepi32_ps(sumi),*ap); \
            } while(0)

            PROC8(0);PROC8(1);PROC8(2);PROC8(3);
            PROC8(4);PROC8(5);PROC8(6);PROC8(7);
            #undef PROC8
        }

        #define HS8(r,a,m) do{__m128 lo=_mm256_castps256_ps128(a),hi=_mm256_extractf128_ps(a,1);\
            __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);\
            y[j+r]=_mm_cvtss_f32(s2)+m+(b?b[j+r]:0);}while(0)
        HS8(0,acc0,am0);HS8(1,acc1,am1);HS8(2,acc2,am2);HS8(3,acc3,am3);
        HS8(4,acc4,am4);HS8(5,acc5,am5);HS8(6,acc6,am6);HS8(7,acc7,am7);
        #undef HS8
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
            const __m256i xv0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+0));
            const __m256i xv1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+32));
            const __m256i xv2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+64));
            const __m256i xv3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+96));
            const __m256i xv4=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+128));
            const __m256i xv5=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+160));
            const __m256i xv6=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+192));
            const __m256i xv7=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+224));

            #define SR(qoff,xve,xvo,sa,sb_) do{\
                __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+qoff));\
                __m256i q4l=_mm256_and_si256(q4b,m4);\
                __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4);\
                __m128i q4ll=_mm256_castsi256_si128(q4l),q4lh=_mm256_extracti128_si256(q4l,1);\
                __m128i q4hl=_mm256_castsi256_si128(q4h),q4hh=_mm256_extracti128_si256(q4h,1);\
                /* xve = [sub_a_even_xq, sub_b_even_xq], xvo = [sub_a_odd_xq, sub_b_odd_xq] */\
                __m128i xa_e=_mm256_castsi256_si128(xve),xb_e=_mm256_extracti128_si256(xve,1);\
                __m128i xa_o=_mm256_castsi256_si128(xvo),xb_o=_mm256_extracti128_si256(xvo,1);\
                /* sub_a: maddubs(sub_a_even_q, sub_a_even_xq) + maddubs(sub_a_odd_q, sub_a_odd_xq) */\
                __m128i pl=_mm_maddubs_epi16(q4ll,xa_e),ph=_mm_maddubs_epi16(q4hl,xa_o);\
                /* sub_b: same */\
                __m128i pl2=_mm_maddubs_epi16(q4lh,xb_e),ph2=_mm_maddubs_epi16(q4hh,xb_o);\
                __m128i sc0=_mm_set1_epi16((short)sa),sc1=_mm_set1_epi16((short)sb_);\
                __m128i s0=_mm_add_epi32(_mm_madd_epi16(sc0,pl),_mm_madd_epi16(sc0,ph));\
                __m128i s1=_mm_add_epi32(_mm_madd_epi16(sc1,pl2),_mm_madd_epi16(sc1,ph2));\
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0));\
            }while(0)
            SR(0,xv0,xv4,sm[0],sm[1]);SR(32,xv1,xv5,sm[2],sm[3]);
            SR(64,xv2,xv6,sm[4],sm[5]);SR(96,xv3,xv7,sm[6],sm[7]);
            #undef SR

            float mult=d*x_scale*inv_63;
            acc=_mm256_fmadd_ps(_mm256_set1_ps(mult),_mm256_cvtepi32_ps(sumi),acc);
        }
        __m128 lo=_mm256_castps256_ps128(acc),hi=_mm256_extractf128_ps(acc,1);
        __m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);
        y[j]=_mm_cvtss_f32(s)+acc_min+(b?b[j]:0);
    }
#else
    /* Scalar fallback (ADJACENT packing) */
    int row_stride = n_super * 144;
    float inv_63 = 1.0f / 63.0f;
    for (int j = 0; j < out_dim; j++) {
        const uint8_t *row = q4k_W + (size_t)j * row_stride;
        float acc = 0, acc_min = 0;
        for (int s = 0; s < n_super; s++) {
            const uint8_t *sb = row + s * 144;
            uint16_t d_u16=*(const uint16_t*)(sb),dmin_u16=*(const uint16_t*)(sb+2);
            float d,dmin;
            #define F16(u,o) do{uint32_t sg=(u>>15),ex=(u>>10)&0x1F,fr=u&0x3FF;\
                if(ex==0)o=fr?(fr/1024.0f/524288.0f)*(sg?-1:1):0;\
                else if(ex==31)o=fr?(float)NAN:(sg?-(float)INFINITY:(float)INFINITY);\
                else{float f=(1.0f+fr/1024.0f)*ldexpf(1.0f,(int)ex-15);o=sg?-f:f;}}while(0)
            F16(d_u16,d);F16(dmin_u16,dmin);
            uint8_t sm[16];unpack_scales_6bit(sb+4,sm);
            float min_dot=0;for(int k=0;k<8;k++)min_dot+=(float)sm[8+k]*(float)bsums[s*8+k];
            acc_min-=dmin*x_scale*min_dot*inv_63;
            const uint8_t*qs=sb+16;
            /* ADJACENT: byte[sub*16+i] = q[sub*32+2i] | (q[sub*32+2i+1]<<4) */
            for(int sub=0;sub<8;sub++){int xoff=(s*8+sub)*32;
            int32_t dot=0;
            for(int i=0;i<16;i++){uint8_t bv=qs[sub*16+i];
            dot+=(int)(bv&0xF)*(int)xq[xoff+2*i];
            dot+=(int)((bv>>4)&0xF)*(int)xq[xoff+2*i+1];}
            acc+=d*(float)sm[sub]*(float)dot*x_scale*inv_63;}}
        y[j]=acc+acc_min+(b?b[j]:0);
    }
#endif
}

#endif /* AVX512 else branch */
#endif /* LAL_Q4K_KERNEL_H */
