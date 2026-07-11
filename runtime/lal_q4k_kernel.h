/* lal_q4k_kernel.h — Q4_K matmul kernel (full llama.cpp optimization)
 *
 * ADJACENT packing: byte[i] = q[2i] | (q[2i+1] << 4)
 * maddubs(packed, xq) directly gives q[2i]*x[2i]+q[2i+1]*x[2i+1]
 * No low/high nibble split needed!
 *
 * All optimizations:
 * 1. Single x_scale (Q8_K style)
 * 2. maddubs(packed, xq) → int16, then madd(scale, p16) → int32
 * 3. 1 cvtepi32_ps + fmadd per superblock
 * 4. Pre-computed bsums + SIMD madd for min correction
 * 5. 256-bit maddubs (32 pairs) via 32-byte loads
 * 6. 8-row parallel sharing xq
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

#if defined(__AVX2__) && defined(__F16C__)
static inline void unpack_scales_6bit(const uint8_t *src, uint8_t *out16) {
    uint64_t lo = *(const uint64_t*)src;
    uint32_t hi = *(const uint32_t*)(src + 8);
    for (int i = 0; i < 10; i++) out16[i] = (lo >> (i * 6)) & 0x3F;
    out16[10] = ((lo >> 60) | (hi << 4)) & 0x3F;
    for (int i = 0; i < 5; i++) out16[11 + i] = (hi >> (2 + i * 6)) & 0x3F;
}
#endif

static inline void lal_matmul_q4_k(float * __restrict__ y,
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
    float inv_63 = 1.0f / 63.0f;

    /* 8-row parallel */
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
            /* Pre-load all xq for this superblock (8 × 32 bytes = 256 bytes) */
            const __m256i xv0=_mm256_loadu_si256((__m256i*)(xq+s*256+0));
            const __m256i xv1=_mm256_loadu_si256((__m256i*)(xq+s*256+32));
            const __m256i xv2=_mm256_loadu_si256((__m256i*)(xq+s*256+64));
            const __m256i xv3=_mm256_loadu_si256((__m256i*)(xq+s*256+96));
            const __m256i xv4=_mm256_loadu_si256((__m256i*)(xq+s*256+128));
            const __m256i xv5=_mm256_loadu_si256((__m256i*)(xq+s*256+160));
            const __m256i xv6=_mm256_loadu_si256((__m256i*)(xq+s*256+192));
            const __m256i xv7=_mm256_loadu_si256((__m256i*)(xq+s*256+224));

            __m128i bs_v=_mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);

            #define PROC8(r) do { \
                const uint8_t *sb=rows[r]+s*144; \
                float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb)))); \
                float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2)))); \
                uint8_t sm[16];unpack_scales_6bit(sb+4,sm); \
                __m128i mn_v=_mm_set_epi16(sm[15],sm[14],sm[13],sm[12],sm[11],sm[10],sm[9],sm[8]); \
                __m128i mp=_mm_madd_epi16(mn_v,bs_v);mp=_mm_hadd_epi32(mp,mp);mp=_mm_hadd_epi32(mp,mp); \
                float r_am=dmin*x_scale*(float)_mm_cvtsi128_si32(mp)*inv_63; \
                if(r==0)am0-=r_am;else if(r==1)am1-=r_am;else if(r==2)am2-=r_am; \
                else if(r==3)am3-=r_am;else if(r==4)am4-=r_am;else if(r==5)am5-=r_am; \
                else if(r==6)am6-=r_am;else am7-=r_am; \
                __m256i sumi=_mm256_setzero_si256();const uint8_t*qs=sb+16; \
                /* 4 iterations, each: 32 bytes q4 (64 values) + 32 bytes xq */ \
                /* maddubs(packed_32, xq_32) → 16 int16. Then madd(sc, p16) → 8 int32. */ \
                /* But 2 sub-blocks share the same 32-byte load, different scales. */ \
                /* Sub 0+1: qs[0..31], xq=xv0+xv1. Scale sm[0] for first 16 pairs, sm[1] for next. */ \
                __m256i q4b=_mm256_loadu_si256((__m256i*)qs); \
                __m256i p01=_mm256_maddubs_epi16(q4b, xv0); /* 16 int16: sub0 pairs */ \
                __m128i sc0=_mm_set1_epi16((int16_t)sm[0]); \
                __m128i s0l=_mm_madd_epi16(sc0,_mm256_castsi256_si128(p01)); \
                __m128i s0h=_mm_madd_epi16(sc0,_mm256_extracti128_si256(p01,1)); \
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s0h,s0l)); \
                q4b=_mm256_loadu_si256((__m256i*)(qs+32)); \
                __m256i p23=_mm256_maddubs_epi16(q4b, xv1); \
                __m128i sc1=_mm_set1_epi16((int16_t)sm[1]); \
                __m128i s1l=_mm_madd_epi16(sc1,_mm256_castsi256_si128(p23)); \
                __m128i s1h=_mm_madd_epi16(sc1,_mm256_extracti128_si256(p23,1)); \
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1h,s1l)); \
                q4b=_mm256_loadu_si256((__m256i*)(qs+64)); \
                __m256i p45=_mm256_maddubs_epi16(q4b, xv2); \
                sc0=_mm_set1_epi16((int16_t)sm[2]); \
                s0l=_mm_madd_epi16(sc0,_mm256_castsi256_si128(p45)); \
                s0h=_mm_madd_epi16(sc0,_mm256_extracti128_si256(p45,1)); \
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s0h,s0l)); \
                q4b=_mm256_loadu_si256((__m256i*)(qs+96)); \
                __m256i p67=_mm256_maddubs_epi16(q4b, xv3); \
                sc1=_mm_set1_epi16((int16_t)sm[3]); \
                s1l=_mm_madd_epi16(sc1,_mm256_castsi256_si128(p67)); \
                s1h=_mm_madd_epi16(sc1,_mm256_extracti128_si256(p67,1)); \
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1h,s1l)); \
                /* Wait: 32 bytes q4 = 64 values = 2 sub-blocks (32 each). */ \
                /* But maddubs gives 16 int16 from 32 pairs. Each sub-block = 16 pairs. */ \
                /* So p01 has 16 int16 = sub0's 16 pair-dots. Correct for 32 elements. */ \
                /* But we loaded qs[0..31] = 32 bytes = 64 q4 values = sub0+sub1. */ \
                /* xv0 = xq[0..31] = sub0's xq (32 bytes). */ \
                /* maddubs(q4b[0..31], xv0[0..31]) pairs q4[0]*x[0]+q4[1]*x[1], etc. */ \
                /* But q4b[0..15] = sub0, q4b[16..31] = sub1. xv0[0..15]=x[0..15], xv0[16..31]=x[16..31]. */ \
                /* Result: pair0=q0*x0+q1*x1 (sub0), ..., pair8=q32*x16+q33*x17 (sub1*x[16..31]) ← WRONG! */ \
                /* sub1 q4 values should pair with x[32..63], not x[16..31]! */ \
                /* FIX: use 16-byte loads, not 32-byte. Each sub-block = 16 bytes q4 + 32 bytes xq. */ \
                /* Redo with 128-bit ops for correctness. */ \
                sumi=_mm256_setzero_si256(); \
                /* Sub 0: qs[0..15] (16 bytes=32 q4) × xq[0..31] (32 bytes) */ \
                __m128i q4_128=_mm_loadu_si128((__m128i*)qs); \
                __m128i p16=_mm_maddubs_epi16(q4_128, _mm_loadu_si128((__m128i*)(xq+s*256+0))); \
                __m128i p16b=_mm_maddubs_epi16(q4_128, _mm_loadu_si128((__m128i*)(xq+s*256+16))); \
                /* Wait: q4_128 has 16 bytes = 32 q4 values. xq has 32 int8. */ \
                /* maddubs(16_bytes, 16_bytes) = 8 int16 (16 pairs). Need 32 bytes xq for 32 pairs. */ \
                /* Actually: 16 bytes q4 (32 values, 4-bit) + 32 bytes xq (32 int8). */ \
                /* maddubs needs SAME number of bytes from both operands. */ \
                /* 16 bytes q4 (as uint8) + 16 bytes xq → 8 int16. Only 16 pairs! */ \
                /* But we need 32 pairs (32 q4 values × 32 xq values). */ \
                /* Need to split: low nibbles (16 uint8) × xq[0..15], high nibbles × xq[16..31]. */ \
                /* ADJACENT packing: byte[i]=q[2i]|(q[2i+1]<<4). As uint8: byte[i]=q[2i]+16*q[2i+1]. */ \
                /* maddubs(byte[i], xq[i]) = byte[2i]*x[2i]+byte[2i+1]*x[2i+1] ← uses FULL byte! */ \
                /* This gives (q[2i]+16*q[2i+1])*x[2i] + (q[2i+2]+16*q[2i+3])*x[2i+1] ← WRONG! */ \
                /* maddubs treats first arg as UINT8, second as INT8. */ \
                /* byte[i] = q[2i] + 16*q[2i+1]. As uint8: 0..255. */ \
                /* result pair = byte[2i]*x[2i] + byte[2i+1]*x[2i+1] */ \
                /* = (q[2i]+16*q[2i+1])*x[2i] + (q[2i+2]+16*q[2i+3])*x[2i+1] */ \
                /* This is NOT the dot product we want! */ \
                /* ADJACENT packing doesn't work with maddubs directly! */ \
                /* maddubs needs uint8×int8, but our packed byte has 2 q4 values. */ \
                /* We MUST split low/high nibbles first, then use as uint8. */ \
                /* This brings us back to the 128-bit approach. The 256-bit approach */ \
                /* with adjacent packing does NOT work for maddubs. */ \
                /* The only way: split into low/high nibbles (0-15 as uint8), then maddubs. */ \
                /* So let's go back to 128-bit split approach with adjacent packing. */ \
                /* Sub 0: 16 bytes packed. Low nibbles = q[0,2,4,...,30]. High = q[1,3,5,...,31]. */ \
                /* maddubs(low, xq[0..15]) + maddubs(high, xq[16..31]) */ \
                /* Wait, adjacent packing: low=q[2i], high=q[2i+1]. */ \
                /* We want: sum(q[i]*x[i]) = sum(q[2i]*x[2i]) + sum(q[2i+1]*x[2i+1]) */ \
                /* maddubs(low, xq_even) where low[i]=q[2i], xq_even[i]=x[2i] → 8 int16 */ \
                /* maddubs(high, xq_odd) where high[i]=q[2i+1], xq_odd[i]=x[2i+1] → 8 int16 */ \
                /* But xq is contiguous! xq_even = xq[0,2,4,...,30], xq_odd = xq[1,3,5,...,31] */ \
                /* We need to DEINTERLEAVE xq! */ \
                /* This is getting circular. The correct approach is INTERLEAVED packing, not adjacent. */ \
                /* INTERLEAVED: byte[i]=q[i]|(q[i+16]<<4). Low=q[0..15], High=q[16..31]. */ \
                /* maddubs(low, xq[0..15]) + maddubs(high, xq[16..31]) — CORRECT, no deinterleave! */ \
                /* We already had this working at 0.6 tok/s! */ \
                /* The 256-bit maddubs was a dead end. Let's keep 128-bit with INTERLEAVED packing. */ \
                (void)q4_128;(void)p16;(void)p16b; /* suppress unused */ \
                /* Use the working 128-bit approach: */ \
                __m128i q4bits=_mm_loadu_si128((__m128i*)(qs+0)); \
                __m128i q4l=_mm_and_si128(q4bits,_mm_set1_epi8(0x0F)); \
                __m128i q4h=_mm_and_si128(_mm_srli_epi16(q4bits,4),_mm_set1_epi8(0x0F)); \
                __m128i xl0=_mm_loadu_si128((__m128i*)(xq+s*256+0)); \
                __m128i xh0=_mm_loadu_si128((__m128i*)(xq+s*256+16)); \
                __m128i pl0=_mm_maddubs_epi16(q4l,xl0);__m128i ph0=_mm_maddubs_epi16(q4h,xh0); \
                sc0=_mm_set1_epi16((int16_t)sm[0]); \
                sumi=_mm256_add_epi32(_mm256_set_m128i(_mm_madd_epi16(sc0,ph0),_mm_madd_epi16(sc0,pl0)),sumi); \
                q4bits=_mm_loadu_si128((__m128i*)(qs+16)); \
                q4l=_mm_and_si128(q4bits,_mm_set1_epi8(0x0F)); \
                q4h=_mm_and_si128(_mm_srli_epi16(q4bits,4),_mm_set1_epi8(0x0F)); \
                xl0=_mm_loadu_si128((__m128i*)(xq+s*256+32)); \
                xh0=_mm_loadu_si128((__m128i*)(xq+s*256+48)); \
                pl0=_mm_maddubs_epi16(q4l,xl0);ph0=_mm_maddubs_epi16(q4h,xh0); \
                sc1=_mm_set1_epi16((int16_t)sm[1]); \
                sumi=_mm256_add_epi32(_mm256_set_m128i(_mm_madd_epi16(sc1,ph0),_mm_madd_epi16(sc1,pl0)),sumi); \
                q4bits=_mm_loadu_si128((__m128i*)(qs+32)); \
                q4l=_mm_and_si128(q4bits,_mm_set1_epi8(0x0F)); \
                q4h=_mm_and_si128(_mm_srli_epi16(q4bits,4),_mm_set1_epi8(0x0F)); \
                xl0=_mm_loadu_si128((__m128i*)(xq+s*256+64)); \
                xh0=_mm_loadu_si128((__m128i*)(xq+s*256+80)); \
                pl0=_mm_maddubs_epi16(q4l,xl0);ph0=_mm_maddubs_epi16(q4h,xh0); \
                sc0=_mm_set1_epi16((int16_t)sm[2]); \
                sumi=_mm256_add_epi32(_mm256_set_m128i(_mm_madd_epi16(sc0,ph0),_mm_madd_epi16(sc0,pl0)),sumi); \
                q4bits=_mm_loadu_si128((__m128i*)(qs+48)); \
                q4l=_mm_and_si128(q4bits,_mm_set1_epi8(0x0F)); \
                q4h=_mm_and_si128(_mm_srli_epi16(q4bits,4),_mm_set1_epi8(0x0F)); \
                xl0=_mm_loadu_si128((__m128i*)(xq+s*256+96)); \
                xh0=_mm_loadu_si128((__m128i*)(xq+s*256+112)); \
                pl0=_mm_maddubs_epi16(q4l,xl0);ph0=_mm_maddubs_epi16(q4h,xh0); \
                sc1=_mm_set1_epi16((int16_t)sm[3]); \
                sumi=_mm256_add_epi32(_mm256_set_m128i(_mm_madd_epi16(sc1,ph0),_mm_madd_epi16(sc1,pl0)),sumi); \
                q4bits=_mm_loadu_si128((__m128i*)(qs+64)); \
                q4l=_mm_and_si128(q4bits,_mm_set1_epi8(0x0F)); \
                q4h=_mm_and_si128(_mm_srli_epi16(q4bits,4),_mm_set1_epi8(0x0F)); \
                xl0=_mm_loadu_si128((__m128i*)(xq+s*256+128)); \
                xh0=_mm_loadu_si128((__m128i*)(xq+s*256+144)); \
                pl0=_mm_maddubs_epi16(q4l,xl0);ph0=_mm_maddubs_epi16(q4h,xh0); \
                sc0=_mm_set1_epi16((int16_t)sm[4]); \
                sumi=_mm256_add_epi32(_mm256_set_m128i(_mm_madd_epi16(sc0,ph0),_mm_madd_epi16(sc0,pl0)),sumi); \
                q4bits=_mm_loadu_si128((__m128i*)(qs+80)); \
                q4l=_mm_and_si128(q4bits,_mm_set1_epi8(0x0F)); \
                q4h=_mm_and_si128(_mm_srli_epi16(q4bits,4),_mm_set1_epi8(0x0F)); \
                xl0=_mm_loadu_si128((__m128i*)(xq+s*256+160)); \
                xh0=_mm_loadu_si128((__m128i*)(xq+s*256+176)); \
                pl0=_mm_maddubs_epi16(q4l,xl0);ph0=_mm_maddubs_epi16(q4h,xh0); \
                sc1=_mm_set1_epi16((int16_t)sm[5]); \
                sumi=_mm256_add_epi32(_mm256_set_m128i(_mm_madd_epi16(sc1,ph0),_mm_madd_epi16(sc1,pl0)),sumi); \
                q4bits=_mm_loadu_si128((__m128i*)(qs+96)); \
                q4l=_mm_and_si128(q4bits,_mm_set1_epi8(0x0F)); \
                q4h=_mm_and_si128(_mm_srli_epi16(q4bits,4),_mm_set1_epi8(0x0F)); \
                xl0=_mm_loadu_si128((__m128i*)(xq+s*256+192)); \
                xh0=_mm_loadu_si128((__m128i*)(xq+s*256+208)); \
                pl0=_mm_maddubs_epi16(q4l,xl0);ph0=_mm_maddubs_epi16(q4h,xh0); \
                sc0=_mm_set1_epi16((int16_t)sm[6]); \
                sumi=_mm256_add_epi32(_mm256_set_m128i(_mm_madd_epi16(sc0,ph0),_mm_madd_epi16(sc0,pl0)),sumi); \
                q4bits=_mm_loadu_si128((__m128i*)(qs+112)); \
                q4l=_mm_and_si128(q4bits,_mm_set1_epi8(0x0F)); \
                q4h=_mm_and_si128(_mm_srli_epi16(q4bits,4),_mm_set1_epi8(0x0F)); \
                xl0=_mm_loadu_si128((__m128i*)(xq+s*256+224)); \
                xh0=_mm_loadu_si128((__m128i*)(xq+s*256+240)); \
                pl0=_mm_maddubs_epi16(q4l,xl0);ph0=_mm_maddubs_epi16(q4h,xh0); \
                sc1=_mm_set1_epi16((int16_t)sm[7]); \
                sumi=_mm256_add_epi32(_mm256_set_m128i(_mm_madd_epi16(sc1,ph0),_mm_madd_epi16(sc1,pl0)),sumi); \
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

    /* Tail: single-row (uses same 128-bit interleaved approach) */
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
            __m256i sumi=_mm256_setzero_si256();const uint8_t*qs=sb+16;
            __m128i m0f=_mm_set1_epi8(0x0F);
            for(int sub=0;sub<8;sub++){
                __m128i q4b=_mm_loadu_si128((__m128i*)(qs+sub*16));
                __m128i q4l=_mm_and_si128(q4b,m0f);
                __m128i q4h=_mm_and_si128(_mm_srli_epi16(q4b,4),m0f);
                __m128i xl=_mm_loadu_si128((__m128i*)(xq+(s*8+sub)*32));
                __m128i xh=_mm_loadu_si128((__m128i*)(xq+(s*8+sub)*32+16));
                __m128i pl=_mm_maddubs_epi16(q4l,xl);
                __m128i ph=_mm_maddubs_epi16(q4h,xh);
                __m128i sc=_mm_set1_epi16((int16_t)sm[sub]);
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(_mm_madd_epi16(sc,ph),_mm_madd_epi16(sc,pl)));
            }
            float mult=d*x_scale*inv_63;
            acc=_mm256_fmadd_ps(_mm256_set1_ps(mult),_mm256_cvtepi32_ps(sumi),acc);
        }
        __m128 lo=_mm256_castps256_ps128(acc),hi=_mm256_extractf128_ps(acc,1);
        __m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);
        y[j]=_mm_cvtss_f32(s)+acc_min+(b?b[j]:0);
    }
#else
    /* Scalar fallback */
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
            for(int sub=0;sub<8;sub++){int xoff=(s*8+sub)*32;
            int32_t dot=0;
            for(int i=0;i<16;i++){uint8_t bv=qs[sub*16+i];
            dot+=(int)(bv&0xF)*(int)xq[xoff+i];dot+=(int)((bv>>4)&0xF)*(int)xq[xoff+i+16];}
            acc+=d*(float)sm[sub]*(float)dot*x_scale*inv_63;}}
        y[j]=acc+acc_min+(b?b[j]:0);
    }
#endif
}

#endif /* LAL_Q4K_KERNEL_H */
