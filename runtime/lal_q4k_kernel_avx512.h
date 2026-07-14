/* lal_q4k_kernel_avx512.h — Q4_K matmul kernel using AVX512_BW (512-bit maddubs)
 *
 * Requires: AVX512F + AVX512BW (NOT VNNI).
 *
 * ADJACENT packing (llama.cpp style, NEW — converter must produce this):
 *   byte[i] = q[2i] | (q[2i+1] << 4)   (within each sub-block of 32 elements)
 *
 *   So 16 packed bytes contain 32 q4 values, all from the SAME sub-block.
 *   q4l = q4 & 0xF  → 16 uint8 (even-indexed q values)
 *   q4h = q4 >> 4   → 16 uint8 (odd-indexed q values)
 *   xq  = 32 int8 for this sub-block: [q0_x, q1_x, ..., q31_x]
 *
 *   maddubs(q4l_16, xq_even_16) = sum of q[2i]*xq[2i] pairs → 8 int16
 *   maddubs(q4h_16, xq_odd_16)  = sum of q[2i+1]*xq[2i+1] → 8 int16
 *
 * For 512-bit (64 bytes packed = 4 sub-blocks × 16 bytes = 128 q values):
 *   q4l = 64 uint8: [sub_a_even×16, sub_b_even×16, sub_c_even×16, sub_d_even×16]
 *   q4h = 64 uint8: [sub_a_odd×16, sub_b_odd×16, sub_c_odd×16, sub_d_odd×16]
 *   xq_lo = 64 int8: [sub_a_even, sub_b_even, sub_c_even, sub_d_even] (16 each)
 *   xq_hi = 64 int8: [sub_a_odd, sub_b_odd, sub_c_odd, sub_d_odd] (16 each)
 *
 *   To get xq_lo/xq_hi from 4 sub-blocks of 32-byte xq:
 *     Each xv (32 bytes): [even×16, odd×16]
 *     For 4 sub-blocks: xv0, xv1, xv2, xv3
 *     xq_lo = concat(xv0.lo16, xv1.lo16, xv2.lo16, xv3.lo16)
 *     xq_hi = concat(xv0.hi16, xv1.hi16, xv2.hi16, xv3.hi16)
 *
 *   Use _mm512_shuffle_i64x2 to combine, or simpler:
 *   Load xq already split into even/odd halves during x quantization.
 *
 * Implementation note: we pre-arrange xq into "even" and "odd" arrays
 * at quantization time. This adds a small one-time cost per token but
 * saves permutes in the hot loop.
 */
#ifndef LAL_Q4K_KERNEL_AVX512_H
#define LAL_Q4K_KERNEL_AVX512_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(XQ_MAX)
#error "Define XQ_MAX before including lal_q4k_kernel_avx512.h"
#endif

/* unpack_scales_6bit is provided by lal_q4k_kernel.h */

#if defined(__AVX512F__) && defined(__AVX512BW__)

/* Pre-arrange xq so that 4 consecutive sub-blocks' even-indexed elements
 * are packed together (32 bytes), then odd-indexed (32 bytes).
 * Layout per superblock (256 elements = 8 sub-blocks):
 *   xq_even[0..127]: sub0_even, sub1_even, sub2_even, sub3_even (4×16=64) + sub4..7 (64)
 *   xq_odd[0..127]:  same for odd indices
 *
 * Actually simpler: just split each sub-block into 16 even + 16 odd.
 * For superblock s, we need:
 *   xv_even_lo = [sub0_even(16), sub1_even(16), sub2_even(16), sub3_even(16)] = 64 bytes
 *   xv_odd_lo  = [sub0_odd,  sub1_odd,  sub2_odd,  sub3_odd] = 64 bytes
 *   xv_even_hi = [sub4_even, sub5_even, sub6_even, sub7_even] = 64 bytes
 *   xv_odd_hi  = [sub4_odd,  sub5_odd,  sub6_odd,  sub7_odd] = 64 bytes
 */
static inline void lal_matmul_q4_k_avx512(float * __restrict__ y,
                                            const uint8_t * __restrict__ q4k_W,
                                            const float * __restrict__ x,
                                            const float * __restrict__ b,
                                            int in_dim, int out_dim) {
    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;

    /* Quantize x to int8 */
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

    /* Pre-arrange xq into even/odd layout per superblock.
     * For each superblock (256 elements = 8 sub-blocks of 32):
     *   Original sub k: xq[k*32 .. k*32+31], even indices = xq[k*32+0,2,4,...,30], odd = xq[k*32+1,3,...,31]
     *   We want: xq_arr[s][0..255] where:
     *     [0..15]  = sub0 even (16 bytes)
     *     [16..31] = sub1 even
     *     [32..47] = sub2 even
     *     [48..63] = sub3 even
     *     [64..79] = sub4 even
     *     [80..95] = sub5 even
     *     [96..111]= sub6 even
     *     [112..127]=sub7 even
     *     [128..255] = same for odd
     */
    int8_t xq_arr[XQ_MAX] __attribute__((aligned(64)));
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

    /* bsums for min correction */
    int16_t bsums[XQ_MAX / 32] __attribute__((aligned(64)));
    for (int sb = 0; sb < n_sub; sb++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) sum += (int)xq[sb * 32 + i];
        bsums[sb] = (int16_t)sum;
    }

    int row_stride = n_super * 144;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const float inv_63 = 1.0f / 63.0f;

    /* 8-row parallel main loop */
    int j = 0;
    for (; j + 8 <= out_dim; j += 8) {
        const uint8_t * __restrict__ rows[8];
        for (int r = 0; r < 8; r++) rows[r] = q4k_W + (size_t)(j+r) * row_stride;

        __m512 acc0=_mm512_setzero_ps(),acc1=_mm512_setzero_ps();
        __m512 acc2=_mm512_setzero_ps(),acc3=_mm512_setzero_ps();
        __m512 acc4=_mm512_setzero_ps(),acc5=_mm512_setzero_ps();
        __m512 acc6=_mm512_setzero_ps(),acc7=_mm512_setzero_ps();
        float am0=0,am1=0,am2=0,am3=0,am4=0,am5=0,am6=0,am7=0;

        for (int s = 0; s < n_super; s++) {
            /* Load pre-arranged xq: 4 zmm = 256 bytes per superblock
             * Layout: [sub0-7 even (128 bytes), sub0-7 odd (128 bytes)]
             *   xv_even_lo = sub0-3 even (64 bytes)
             *   xv_even_hi = sub4-7 even (64 bytes)
             *   xv_odd_lo  = sub0-3 odd  (64 bytes)
             *   xv_odd_hi  = sub4-7 odd  (64 bytes)
             */
            const __m512i xv_even_lo = _mm512_loadu_si512(xq_arr + s*256 + 0);    /* sub0..3 even */
            const __m512i xv_even_hi = _mm512_loadu_si512(xq_arr + s*256 + 64);   /* sub4..7 even */
            const __m512i xv_odd_lo  = _mm512_loadu_si512(xq_arr + s*256 + 128);  /* sub0..3 odd */
            const __m512i xv_odd_hi  = _mm512_loadu_si512(xq_arr + s*256 + 192);  /* sub4..7 odd */

            __m128i bs_v=_mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);

            #define PROC8(r) do { \
                const uint8_t *sb=rows[r]+s*144; \
                float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb)))); \
                float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2)))); \
                uint8_t sm[16]; unpack_scales_6bit(sb+4,sm); \
                /* Min correction */ \
                __m128i mn_v=_mm_set_epi16(sm[15],sm[14],sm[13],sm[12],sm[11],sm[10],sm[9],sm[8]); \
                __m128i mp=_mm_madd_epi16(mn_v,bs_v); \
                mp=_mm_hadd_epi32(mp,mp); mp=_mm_hadd_epi32(mp,mp); \
                float r_am=dmin*x_scale*(float)_mm_cvtsi128_si32(mp)*inv_63; \
                if(r==0)am0-=r_am;else if(r==1)am1-=r_am;else if(r==2)am2-=r_am; \
                else if(r==3)am3-=r_am;else if(r==4)am4-=r_am;else if(r==5)am5-=r_am; \
                else if(r==6)am6-=r_am;else am7-=r_am; \
                __m512i sumi=_mm512_setzero_si512(); \
                const uint8_t*qs=sb+16; \
                /* 优化 sc_v: set1_epi16 broadcast (比 set_epi16 32立即数快) */ \
                __m256i sc01 = _mm256_set_m128i(_mm_set1_epi16((short)sm[1]), _mm_set1_epi16((short)sm[0])); \
                __m256i sc23 = _mm256_set_m128i(_mm_set1_epi16((short)sm[3]), _mm_set1_epi16((short)sm[2])); \
                __m512i sc0123 = _mm512_inserti64x4(_mm512_castsi256_si512(sc01), sc23, 1); \
                __m256i sc45 = _mm256_set_m128i(_mm_set1_epi16((short)sm[5]), _mm_set1_epi16((short)sm[4])); \
                __m256i sc67 = _mm256_set_m128i(_mm_set1_epi16((short)sm[7]), _mm_set1_epi16((short)sm[6])); \
                __m512i sc4567 = _mm512_inserti64x4(_mm512_castsi256_si512(sc45), sc67, 1); \
                /* iter 0 */ \
                { \
                    __m512i q4b=_mm512_loadu_si512(qs); \
                    __m512i q4l=_mm512_and_si512(q4b,m4); \
                    __m512i q4h=_mm512_and_si512(_mm512_srli_epi16(q4b,4),m4); \
                    __m512i p_lo=_mm512_maddubs_epi16(q4l, xv_even_lo); \
                    __m512i p_hi=_mm512_maddubs_epi16(q4h, xv_odd_lo); \
                    __m512i p16=_mm512_adds_epi16(p_lo, p_hi); \
                    sumi=_mm512_add_epi32(sumi, _mm512_madd_epi16(sc0123, p16)); \
                } \
                /* iter 1 */ \
                { \
                    __m512i q4b=_mm512_loadu_si512(qs+64); \
                    __m512i q4l=_mm512_and_si512(q4b,m4); \
                    __m512i q4h=_mm512_and_si512(_mm512_srli_epi16(q4b,4),m4); \
                    __m512i p_lo=_mm512_maddubs_epi16(q4l, xv_even_hi); \
                    __m512i p_hi=_mm512_maddubs_epi16(q4h, xv_odd_hi); \
                    __m512i p16=_mm512_adds_epi16(p_lo, p_hi); \
                    sumi=_mm512_add_epi32(sumi, _mm512_madd_epi16(sc4567, p16)); \
                } \
                float mult=d*x_scale*inv_63; \
                __m512 *ap=(r==0)?&acc0:(r==1)?&acc1:(r==2)?&acc2:(r==3)?&acc3: \
                            (r==4)?&acc4:(r==5)?&acc5:(r==6)?&acc6:&acc7; \
                *ap=_mm512_fmadd_ps(_mm512_set1_ps(mult),_mm512_cvtepi32_ps(sumi),*ap); \
            } while(0)

            PROC8(0);PROC8(1);PROC8(2);PROC8(3);
            PROC8(4);PROC8(5);PROC8(6);PROC8(7);
            #undef PROC8
        }

        /* Horizontal sum + min correction + bias */
        #define HS8(r,a,m) do{ \
            y[j+r]=_mm512_reduce_add_ps(a)+m+(b?b[j+r]:0); \
        }while(0)
        HS8(0,acc0,am0);HS8(1,acc1,am1);HS8(2,acc2,am2);HS8(3,acc3,am3);
        HS8(4,acc4,am4);HS8(5,acc5,am5);HS8(6,acc6,am6);HS8(7,acc7,am7);
        #undef HS8
    }

    /* Tail: single-row scalar */
    for (; j < out_dim; j++) {
        const uint8_t * __restrict__ row = q4k_W + (size_t)j * row_stride;
        float acc = 0, acc_min = 0;
        for (int s = 0; s < n_super; s++) {
            const uint8_t *sb = row + s * 144;
            uint16_t d_u16=*(const uint16_t*)(sb),dmin_u16=*(const uint16_t*)(sb+2);
            float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)d_u16)));
            float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)dmin_u16)));
            uint8_t sm[16]; unpack_scales_6bit(sb+4,sm);
            float min_dot=0;for(int k=0;k<8;k++)min_dot+=(float)sm[8+k]*(float)bsums[s*8+k];
            acc_min-=dmin*x_scale*min_dot*inv_63;
            const uint8_t*qs=sb+16;
            for(int sub=0;sub<8;sub++){int xoff=(s*8+sub)*32;
            int32_t dot=0;
            for(int i=0;i<16;i++){uint8_t bv=qs[sub*16+i];
            dot+=(int)(bv&0xF)*(int)xq[xoff+i];dot+=(int)((bv>>4)&0xF)*(int)xq[xoff+i+16];}
            acc+=d*(float)sm[sub]*(float)dot*x_scale*inv_63;}}
            y[j]=acc+acc_min+(b?b[j]:0);
    }
}
#endif /* __AVX512F__ && __AVX512BW__ */
#endif /* LAL_Q4K_KERNEL_AVX512_H */
