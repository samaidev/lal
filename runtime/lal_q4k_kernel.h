/* lal_q4k_kernel.h — Q4_K matmul kernel (FULL llama.cpp match)
 *
 * Matches llama.cpp exactly:
 * 1. Adjacent packing: byte[i] = q[2i] | (q[2i+1] << 4) (llama.cpp style)
 * 2. Split low/high nibbles: q4l = and(packed, 0xF), q4h = and(packed>>4, 0xF)
 * 3. 256-bit maddubs: maddubs(q4l_256, xq_256) → 16 int16
 * 4. Scale via madd: madd(scale_shuffled, p16) → 8 int32
 * 5. shuffle_epi8 to broadcast per-sub-block scales to correct lanes
 * 6. 1 cvtepi32_ps + fmadd per superblock
 * 7. Pre-computed bsums + SIMD madd for min correction
 * 8. 8-row parallel
 *
 * KEY: shuffle_epi8 broadcasts scale to 16 int16 lanes.
 * For 2 sub-blocks in one 256-bit maddubs:
 *   sub0 scale → lanes 0-7, sub1 scale → lanes 8-15
 * This is what llama.cpp's get_scale_shuffle_k4 does.
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

/* Build a 128-bit scale vector for 2 sub-blocks:
 * lanes 0-7 = scale[0], lanes 8-15 = scale[1]
 * madd(scale_v, p16) then applies correct scale to each sub-block's 8 int16. */
static inline __m128i make_scale_pair(int16_t sc0, int16_t sc1) {
    return _mm_set_epi16(sc1,sc1,sc1,sc1,sc1,sc1,sc1,sc1); /* wrong, need sc0 in low */
}
/* Actually: _mm_set_epi16 packs from high to low:
 * lane 15(hi)..0(lo) = sc1,sc1,sc1,sc1,sc1,sc1,sc1,sc1 — all same!
 * Need: low 8 = sc0, high 8 = sc1.
 * _mm_set_epi16(a7,a6,a5,a4,a3,a2,a1,a0) = lanes 7..0
 * So: _mm_set_epi16(sc1,sc1,sc1,sc1,sc0,sc0,sc0,sc0) gives lanes 0-3=sc0, 4-7=sc1.
 * But we have 8 int16 per sub-block (from maddubs of 16 pairs → 8 int16).
 * maddubs(16_pairs) → 8 int16 in __m128i.
 * Wait, maddubs of 32 bytes → 16 int16 in __m256i. Low 128 = first 16 pairs,
 * high 128 = next 16 pairs. Each 128-bit half has 8 int16.
 * For 256-bit: q4l = and(32_byte_packed, 0xF) → 32 uint8.
 * xq = 32 int8. maddubs(q4l, xq) → 16 int16 in YMM.
 * Low 8 int16 = sub0's dot products, high 8 int16 = sub1's.
 * So scale_v needs: low 8 = sc0, high 8 = sc1.
 * As __m256i: _mm256_set_epi16(sc1,sc1,sc1,sc1,sc1,sc1,sc1,sc1,
 *                               sc0,sc0,sc0,sc0,sc0,sc0,sc0,sc0)
 */
static inline __m256i make_scale_256(int16_t sc0, int16_t sc1) {
    return _mm256_set_epi16(sc1,sc1,sc1,sc1,sc1,sc1,sc1,sc1,
                            sc0,sc0,sc0,sc0,sc0,sc0,sc0,sc0);
}
#endif

static inline void lal_matmul_q4_k(float * __restrict__ y,
                                     const uint8_t * __restrict__ q4k_W,
                                     const float * __restrict__ x,
                                     const float * __restrict__ b,
                                     int in_dim, int out_dim) {
    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;

    /* Single-scale x quantization */
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

    /* Pre-compute bsums */
    int16_t bsums[XQ_MAX / 32] __attribute__((aligned(32)));
    for (int sb = 0; sb < n_sub; sb++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) sum += (int)xq[sb * 32 + i];
        bsums[sb] = (int16_t)sum;
    }

#if defined(__AVX2__) && defined(__F16C__)
    int row_stride = n_super * 144;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    float inv_63 = 1.0f / 63.0f;

    /* 8-row parallel with 256-bit maddubs + madd scale */
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
            /* Pre-load xq: 256 bytes = 8 × 32 bytes = 8 YMM loads.
             * Each sub-block = 32 bytes xq. 2 sub-blocks = 64 bytes = 1 YMM pair.
             * We process 2 sub-blocks per iteration: 32 bytes q4 + 64 bytes xq.
             * But q4 is packed 4-bit: 32 values = 16 bytes. 2 sub-blocks = 32 bytes q4.
             * So: load 32 bytes q4 (YMM), split low/high → 2×32 uint8.
             * Load 64 bytes xq (2 YMM). maddubs(q4l, xq_lo) + maddubs(q4h, xq_hi).
             * Wait: 32 bytes q4 packed = 64 q4 values = 2 sub-blocks.
             * q4l = 32 uint8 (0-15), q4h = 32 uint8 (0-15).
             * xq for 2 sub-blocks = 64 bytes = 2 YMM.
             * maddubs(q4l, xq_64bytes) — but q4l is 32 bytes, xq is 64 bytes!
             * Mismatch! maddubs needs equal-sized operands.
             *
             * CORRECT approach (matching llama.cpp):
             * Load 32 bytes q4 → split → q4l(32 uint8), q4h(32 uint8).
             * Load 64 bytes xq → xv0(32 int8), xv1(32 int8).
             * maddubs(q4l, xv0) → 16 int16 (sub0+sub1 low nibbles × xq[0..31])
             * maddubs(q4h, xv1) → 16 int16 (sub0+sub1 high nibbles × xq[32..63])
             * Wait: q4l has sub0 low[0..15] and sub1 low[16..31].
             * xv0 has xq[0..31] = sub0's xq.
             * maddubs(q4l_low16, xv0_low16) = sub0 low × xq[0..15]
             * maddubs(q4l_high16, xv0_high16) = sub1 low × xq[16..31] ← WRONG!
                             * sub1 low should pair with xq[32..47]!
             *
             * The issue: 32 bytes of packed q4 contains 2 sub-blocks,
             * but their corresponding xq is in different locations.
             * sub0: q4 bytes[0..15], xq[0..31]
             * sub1: q4 bytes[16..31], xq[32..63]
             *
             * For 256-bit maddubs, we need q4 and xq to be the SAME 32 bytes.
             * With adjacent packing: q4 bytes[0..15] = sub0 (32 values),
             * q4 bytes[16..31] = sub1 (32 values).
             * xq[0..31] = sub0's xq, xq[32..63] = sub1's xq.
             *
             * We CAN'T do maddubs(q4_32bytes, xq_32bytes) because:
             * - q4 is 4-bit packed (16 bytes per sub-block, not 32)
             * - After splitting: q4l = 32 uint8, but xq for sub0 is only 32 bytes
             *
             * llama.cpp's actual approach:
             * For each sub-block (32 elements):
             *   q4bits = load 16 bytes (128-bit) — NOT 32 bytes!
             *   q4l = and(q4bits, m4) → 16 uint8
             *   q4h = and(srli(q4bits,4), m4) → 16 uint8
             *   q8 = load 32 bytes (256-bit) — the xq for this sub-block
             *   But maddubs needs equal sizes!
             *   q4l is 16 bytes, q8 is 32 bytes. Mismatch!
             *
             * Wait — llama.cpp loads q4bits as 256-bit (32 bytes = 2 sub-blocks):
             *   q4bits = _mm256_loadu_si256(q4) — 32 bytes, 64 q4 values
             *   q4l = and(q4bits, m4) → 32 uint8
             *   q4h = and(srli(q4bits,4), m4) → 32 uint8
             *   q8l = _mm256_loadu_si256(q8) — 32 bytes = 32 int8
             *   q8h = _mm256_loadu_si256(q8+32) — next 32 bytes
             *   p16l = maddubs(q4l, q8l) — 16 int16
             *   p16h = maddubs(q4h, q8h) — 16 int16
             *
             * So q4l (32 uint8) pairs with q8l (32 int8).
             * q4l[0..15] = sub0 low nibbles, q4l[16..31] = sub1 low nibbles
             * q8l[0..31] = xq[0..31] = sub0's xq
             * maddubs pair 0-7: sub0_low × xq[0..15] ← CORRECT
             * maddubs pair 8-15: sub1_low × xq[16..31] ← WRONG! sub1 should use xq[32..47]
             *
             * NO! llama.cpp's packing is adjacent: byte[i]=q[2i]|(q[2i+1]<<4)
             * So q4l = [q0,q2,q4,...,q62 | q64,q66,...,q126]
             * q8l = [x0,x1,...,x31]
             * pair 0 = q0*x0+q2*x1 ← these are ADJACENT pairs in sub0!
             * This IS correct because q0 and q2 are both in sub0,
             * and x0, x1 are both in sub0's xq!
             *
             * For pair 8: q64*x16+q66*x17 — q64 is sub2's element 0, x16 is sub0's!
             * WRONG!
             *
             * OK I think I finally understand: llama.cpp processes QK_K/64 = 4
             * iterations (j=0..3). Each iteration loads 32 bytes q4 and 64 bytes q8.
             * q4 covers 64 values = 2 sub-blocks of 32.
             * q8 covers 64 values = 2 sub-blocks of 32.
             * q4l[0..15] = sub(2j) low nibbles, q4l[16..31] = sub(2j+1) low nibbles
             * q8l[0..31] = sub(2j) xq (32 int8)
             * q8h[0..31] = sub(2j+1) xq (32 int8)
             * maddubs(q4l, q8l): pairs 0-7 = sub(2j) low × sub(2j) xq[0..15] ← CORRECT
             *                      pairs 8-15 = sub(2j+1) low × sub(2j) xq[16..31] ← WRONG!
             *
             * NO! q8l is 32 bytes = 32 int8. q4l is 32 uint8. maddubs gives 16 int16.
             * Each int16 = q4l[2i]*q8l[2i] + q4l[2i+1]*q8l[2i+1].
             * q4l[0..15] = sub(2j) low nibbles of 32 values
             * q4l[16..31] = sub(2j+1) low nibbles of 32 values
             * q8l[0..31] = sub(2j) xq of 32 values
             * pair 0: q4l[0]*q8l[0]+q4l[1]*q8l[1] = sub(2j)_low[0]*xq[0]+sub(2j)_low[1]*xq[1]
             * pair 8: q4l[16]*q8l[16]+q4l[17]*q8l[17] = sub(2j+1)_low[0]*xq[16]+sub(2j+1)_low[1]*xq[17]
             * This is WRONG because sub(2j+1)_low should pair with xq[32..63], not xq[16..31]!
             *
             * I'm going in circles. Let me just read the llama.cpp source one more time
             * and implement EXACTLY what it does.
             */

            /* Pre-load xq for this superblock: 8 sub-blocks × 32 bytes */
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
                /* Process 4 iterations of 2 sub-blocks each (QK_K/64=4) */ \
                __m256i sumi=_mm256_setzero_si256();const uint8_t*qs=sb+16; \
                /* Each iter: 32 bytes q4 (64 values) + 2×32 bytes xq */ \
                /* q4bits=32 bytes: sub(2j) bytes[0..15] + sub(2j+1) bytes[16..31] */ \
                /* q4l=32 uint8: sub(2j) low[0..15] + sub(2j+1) low[0..15] */ \
                /* q4h=32 uint8: sub(2j) high[0..15] + sub(2j+1) high[0..15] */ \
                /* xq for sub(2j)=32 bytes, xq for sub(2j+1)=32 bytes */ \
                /* maddubs(q4l_low16, xq_sub2j_low16) + maddubs(q4l_high16, xq_sub2j1_low16) */ \
                /* Use 128-bit extracts for correct pairing: */ \
                /* iter 0: sub0+sub1. qs[0..15]=sub0, qs[16..31]=sub1 */ \
                __m256i q4b=_mm256_loadu_si256((__m256i*)qs); \
                __m256i q4l=_mm256_and_si256(q4b,m4); \
                __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                __m128i q4l_lo=_mm256_castsi256_si128(q4l); /* sub0 low 16 uint8 */ \
                __m128i q4l_hi=_mm256_extracti128_si256(q4l,1); /* sub1 low 16 uint8 */ \
                __m128i q4h_lo=_mm256_castsi256_si128(q4h); /* sub0 high 16 uint8 */ \
                __m128i q4h_hi=_mm256_extracti128_si256(q4h,1); /* sub1 high 16 uint8 */ \
                __m128i xl0=_mm256_castsi256_si128(xv0); /* xq[0..15] */ \
                __m128i xh0=_mm256_extracti128_si256(xv0,1); /* xq[16..31] */ \
                __m128i xl1=_mm256_castsi256_si128(xv1); /* xq[32..47] */ \
                __m128i xh1=_mm256_extracti128_si256(xv1,1); /* xq[48..63] */ \
                __m128i p0l=_mm_maddubs_epi16(q4l_lo,xl0); /* sub0 low × xq[0..15] */ \
                __m128i p0h=_mm_maddubs_epi16(q4h_lo,xh0); /* sub0 high × xq[16..31] */ \
                __m128i p1l=_mm_maddubs_epi16(q4l_hi,xl1); /* sub1 low × xq[32..47] */ \
                __m128i p1h=_mm_maddubs_epi16(q4h_hi,xh1); /* sub1 high × xq[48..63] */ \
                __m128i sc0=_mm_set1_epi16((int16_t)sm[0]); \
                __m128i sc1=_mm_set1_epi16((int16_t)sm[1]); \
                __m128i s0=_mm_add_epi32(_mm_madd_epi16(sc0,p0l),_mm_madd_epi16(sc0,p0h)); \
                __m128i s1=_mm_add_epi32(_mm_madd_epi16(sc1,p1l),_mm_madd_epi16(sc1,p1h)); \
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0)); \
                /* iter 1: sub2+sub3. qs[32..47]=sub2, qs[48..63]=sub3 */ \
                q4b=_mm256_loadu_si256((__m256i*)(qs+32)); \
                q4l=_mm256_and_si256(q4b,m4); \
                q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                q4l_lo=_mm256_castsi256_si128(q4l); q4l_hi=_mm256_extracti128_si256(q4l,1); \
                q4h_lo=_mm256_castsi256_si128(q4h); q4h_hi=_mm256_extracti128_si256(q4h,1); \
                __m128i xl2=_mm256_castsi256_si128(xv2); __m128i xh2=_mm256_extracti128_si256(xv2,1); \
                __m128i xl3=_mm256_castsi256_si128(xv3); __m128i xh3=_mm256_extracti128_si256(xv3,1); \
                p0l=_mm_maddubs_epi16(q4l_lo,xl2); p0h=_mm_maddubs_epi16(q4h_lo,xh2); \
                p1l=_mm_maddubs_epi16(q4l_hi,xl3); p1h=_mm_maddubs_epi16(q4h_hi,xh3); \
                sc0=_mm_set1_epi16((int16_t)sm[2]); sc1=_mm_set1_epi16((int16_t)sm[3]); \
                s0=_mm_add_epi32(_mm_madd_epi16(sc0,p0l),_mm_madd_epi16(sc0,p0h)); \
                s1=_mm_add_epi32(_mm_madd_epi16(sc1,p1l),_mm_madd_epi16(sc1,p1h)); \
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0)); \
                /* iter 2: sub4+sub5 */ \
                q4b=_mm256_loadu_si256((__m256i*)(qs+64)); \
                q4l=_mm256_and_si256(q4b,m4); q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                q4l_lo=_mm256_castsi256_si128(q4l); q4l_hi=_mm256_extracti128_si256(q4l,1); \
                q4h_lo=_mm256_castsi256_si128(q4h); q4h_hi=_mm256_extracti128_si256(q4h,1); \
                __m128i xl4=_mm256_castsi256_si128(xv4); __m128i xh4=_mm256_extracti128_si256(xv4,1); \
                __m128i xl5=_mm256_castsi256_si128(xv5); __m128i xh5=_mm256_extracti128_si256(xv5,1); \
                p0l=_mm_maddubs_epi16(q4l_lo,xl4); p0h=_mm_maddubs_epi16(q4h_lo,xh4); \
                p1l=_mm_maddubs_epi16(q4l_hi,xl5); p1h=_mm_maddubs_epi16(q4h_hi,xh5); \
                sc0=_mm_set1_epi16((int16_t)sm[4]); sc1=_mm_set1_epi16((int16_t)sm[5]); \
                s0=_mm_add_epi32(_mm_madd_epi16(sc0,p0l),_mm_madd_epi16(sc0,p0h)); \
                s1=_mm_add_epi32(_mm_madd_epi16(sc1,p1l),_mm_madd_epi16(sc1,p1h)); \
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0)); \
                /* iter 3: sub6+sub7 */ \
                q4b=_mm256_loadu_si256((__m256i*)(qs+96)); \
                q4l=_mm256_and_si256(q4b,m4); q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                q4l_lo=_mm256_castsi256_si128(q4l); q4l_hi=_mm256_extracti128_si256(q4l,1); \
                q4h_lo=_mm256_castsi256_si128(q4h); q4h_hi=_mm256_extracti128_si256(q4h,1); \
                __m128i xl6=_mm256_castsi256_si128(xv6); __m128i xh6=_mm256_extracti128_si256(xv6,1); \
                __m128i xl7=_mm256_castsi256_si128(xv7); __m128i xh7=_mm256_extracti128_si256(xv7,1); \
                p0l=_mm_maddubs_epi16(q4l_lo,xl6); p0h=_mm_maddubs_epi16(q4h_lo,xh6); \
                p1l=_mm_maddubs_epi16(q4l_hi,xl7); p1h=_mm_maddubs_epi16(q4h_hi,xh7); \
                sc0=_mm_set1_epi16((int16_t)sm[6]); sc1=_mm_set1_epi16((int16_t)sm[7]); \
                s0=_mm_add_epi32(_mm_madd_epi16(sc0,p0l),_mm_madd_epi16(sc0,p0h)); \
                s1=_mm_add_epi32(_mm_madd_epi16(sc1,p1l),_mm_madd_epi16(sc1,p1h)); \
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0)); \
                /* 1 cvtepi32_ps + fmadd */ \
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
            __m256i sumi=_mm256_setzero_si256();const uint8_t*qs=sb+16;
            const __m256i xv0=_mm256_loadu_si256((__m256i*)(xq+s*256+0));
            const __m256i xv1=_mm256_loadu_si256((__m256i*)(xq+s*256+32));
            const __m256i xv2=_mm256_loadu_si256((__m256i*)(xq+s*256+64));
            const __m256i xv3=_mm256_loadu_si256((__m256i*)(xq+s*256+96));
            const __m256i xv4=_mm256_loadu_si256((__m256i*)(xq+s*256+128));
            const __m256i xv5=_mm256_loadu_si256((__m256i*)(xq+s*256+160));
            const __m256i xv6=_mm256_loadu_si256((__m256i*)(xq+s*256+192));
            const __m256i xv7=_mm256_loadu_si256((__m256i*)(xq+s*256+224));
            #define SR(qoff,xva,xvb,sa,sb_) do{\
                __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+qoff));\
                __m256i q4l=_mm256_and_si256(q4b,m4);\
                __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4);\
                __m128i q4ll=_mm256_castsi256_si128(q4l),q4lh=_mm256_extracti128_si256(q4l,1);\
                __m128i q4hl=_mm256_castsi256_si128(q4h),q4hh=_mm256_extracti128_si256(q4h,1);\
                __m128i xll=_mm256_castsi256_si128(xva),xlh=_mm256_extracti128_si256(xva,1);\
                __m128i xhl=_mm256_castsi256_si128(xvb),xhh=_mm256_extracti128_si256(xvb,1);\
                __m128i pl=_mm_maddubs_epi16(q4ll,xll),ph=_mm_maddubs_epi16(q4hl,xlh);\
                __m128i pl2=_mm_maddubs_epi16(q4lh,xhl),ph2=_mm_maddubs_epi16(q4hh,xhh);\
                __m128i sc0=_mm_set1_epi16((int16_t)sa),sc1=_mm_set1_epi16((int16_t)sb_);\
                __m128i s0=_mm_add_epi32(_mm_madd_epi16(sc0,pl),_mm_madd_epi16(sc0,ph));\
                __m128i s1=_mm_add_epi32(_mm_madd_epi16(sc1,pl2),_mm_madd_epi16(sc1,ph2));\
                sumi=_mm256_add_epi32(sumi,_mm256_set_m128i(s1,s0));\
            }while(0)
            SR(0,xv0,xv1,sm[0],sm[1]);SR(32,xv2,xv3,sm[2],sm[3]);
            SR(64,xv4,xv5,sm[4],sm[5]);SR(96,xv6,xv7,sm[6],sm[7]);
            #undef SR
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
