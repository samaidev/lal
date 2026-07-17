/* lal_q4k_kernel.h — Q4_K matmul kernel (256-bit maddubs, ADJACENT packing)
 *
 * ADJACENT packing (llama.cpp style, matches Python converter):
 *   byte[sub*16+i] = q[sub*32+2i] | (q[sub*32+2i+1] << 4)
 *
 * KEY INSIGHT: With ADJACENT packing + pre-arranged xq (even/odd split),
 * we can use TRUE 256-bit maddubs WITHOUT vpermute!
 *
 * Per iteration (2 sub-blocks, 32 packed bytes):
 *   q4b = load 32 bytes = [sub_a_packed(16), sub_b_packed(16)]
 *   q4l = q4b & 0xF = [sub_a_even(16), sub_b_even(16)]   — 32 uint8
 *   q4h = q4b >> 4  = [sub_a_odd(16),  sub_b_odd(16)]    — 32 uint8
 *   xq_even = load 32 bytes = [sub_a_even_xq, sub_b_even_xq]
 *   xq_odd  = load 32 bytes = [sub_a_odd_xq,  sub_b_odd_xq]
 *
 *   pl = maddubs(q4l, xq_even) → 16 int16 = [sub_a_partial(8), sub_b_partial(8)]
 *   ph = maddubs(q4h, xq_odd)  → 16 int16 = [sub_a_partial(8), sub_b_partial(8)]
 *   p16 = pl + ph → 16 int16
 *
 *   scale_v = [sc_a×8, sc_b×8] (via set_m128i)
 *   s = madd_epi16(scale_v, p16) → 8 int32 = [sub_a_dot(4), sub_b_dot(4)]
 *   sumi += s
 *
 * This uses 256-bit maddubs + 256-bit madd — NO extracti128, NO vpermute!
 * 4 iterations per superblock, 8-row parallel.
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
 *
 * Vectorized approach: load 16 bytes, use multiply + shift to extract 6-bit fields.
 * The 12-byte packed data holds 16 × 6 = 96 bits. We pad to 128 bits and use
 * SIMD to extract all 16 values in a few operations.
 *
 * Technique: treat the 12 bytes as a 128-bit number (zero-padded).
 * Multiply by a precomputed vector of shift amounts, then mask.
 * Actually simpler: use the "bit extraction via multiply" trick.
 *
 * For 6-bit fields packed at positions 0, 6, 12, ..., 90:
 *   val[i] = (packed >> (i*6)) & 0x3F
 *
 * Vectorized: load 16 bytes as __m128i, use pmaddubsw + psrlw + pand pattern.
 * But the simplest correct approach is still scalar — compiler optimizes well.
 * Keep scalar but mark as inline (already is).
 */
static inline void unpack_scales_6bit(const uint8_t *src, uint8_t *out16) {
    uint64_t lo = *(const uint64_t*)src;
    uint32_t hi = *(const uint32_t*)(src + 8);
    /* Unrolled for better instruction-level parallelism */
    out16[0]  = lo & 0x3F;
    out16[1]  = (lo >> 6) & 0x3F;
    out16[2]  = (lo >> 12) & 0x3F;
    out16[3]  = (lo >> 18) & 0x3F;
    out16[4]  = (lo >> 24) & 0x3F;
    out16[5]  = (lo >> 30) & 0x3F;
    out16[6]  = (lo >> 36) & 0x3F;
    out16[7]  = (lo >> 42) & 0x3F;
    out16[8]  = (lo >> 48) & 0x3F;
    out16[9]  = (lo >> 54) & 0x3F;
    out16[10] = ((lo >> 60) | (hi << 4)) & 0x3F;
    out16[11] = (hi >> 2) & 0x3F;
    out16[12] = (hi >> 8) & 0x3F;
    out16[13] = (hi >> 14) & 0x3F;
    out16[14] = (hi >> 20) & 0x3F;
    out16[15] = (hi >> 26) & 0x3F;
}

/* === Q4_K 预处理: 量化 x + 计算 bsums + pre-arrange xq_arr (SIMD 优化) ===
 * 提取出来让 gate/up matmul 共享 (fused_swiglu 中 gate 和 up 用同一个 x)
 * SIMD 优化: x_max / 量化 / bsums / xq_arr 重排 全部向量化
 */
static inline float lal_q4k_prepare_x(const float * __restrict__ x,
                                       int in_dim,
                                       int8_t * __restrict__ xq,
                                       int16_t * __restrict__ bsums,
                                       int8_t * __restrict__ xq_arr) {
    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;

#if defined(__AVX2__) && defined(__F16C__)
    /* === SIMD 求 abs max === */
    __m256 vabs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 vmax = _mm256_setzero_ps();
    int i = 0;
    int n8 = in_dim & ~7;
    for (; i < n8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vabs = _mm256_and_ps(vx, vabs_mask);
        vmax = _mm256_max_ps(vmax, vabs);
    }
    /* horizontal max */
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1,0,3,2)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,0,0,1)));
    float x_max = _mm_cvtss_f32(m);
    for (; i < in_dim; i++) { float a = fabsf(x[i]); if (a > x_max) x_max = a; }

    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    float x_inv = 1.0f / x_scale;

    /* === SIMD 量化: lroundf(x*inv) → cvtps_epi32 (round-to-nearest-even) ===
     * 用 _mm256_cvtps_epi32 代替标量 lroundf, 8x faster
     * cvtps_epi32 默认 round-to-nearest-even, 与 lroundf (round-half-away) 微小差异
     * 对 int8 量化影响 < 1 LSB, 可忽略 */
    __m256 vinv = _mm256_set1_ps(x_inv);
    __m256i v127 = _mm256_set1_epi32(127);
    __m256i vn127 = _mm256_set1_epi32(-127);
    i = 0;
    for (; i < n8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vscaled = _mm256_mul_ps(vx, vinv);
        __m256i vi = _mm256_cvtps_epi32(vscaled);
        /* clamp to [-127, 127] */
        vi = _mm256_min_epi32(vi, v127);
        vi = _mm256_max_epi32(vi, vn127);
        /* pack int32 → int8 via 2x packs */
        __m128i lo128 = _mm256_castsi256_si128(vi);
        __m128i hi128 = _mm256_extracti128_si256(vi, 1);
        __m128i packed16 = _mm_packs_epi32(lo128, hi128); /* int16 */
        __m128i packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128()); /* int8 */
        _mm_storel_epi64((__m128i*)(xq + i), packed8);
    }
    for (; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * x_inv);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }

    /* === bsums: 每 32 个 int8 求和 → int16 (标量, 32 次加法 < 1μs) === */
    for (int sb = 0; sb < n_sub; sb++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) sum += (int)xq[sb * 32 + i];
        bsums[sb] = (int16_t)sum;
    }

    /* === SIMD xq_arr 重排: 奇偶分离 ===
     * src 有 32 字节 (一个 sub-block), 要拆成 16 even + 16 odd
     * 用两个 128-bit shuffle: lo(src[0..15]) + hi(src[16..31])
     * each: even = {0,2,4,6,8,10,12,14, 0x80×8}, odd = {1,3,5,7,9,11,13,15, 0x80×8}
     * 然后 unpacklo_epi64 合并两半的偶数/奇数 */
    static const uint8_t even_mask[16] __attribute__((aligned(16))) = {
        0,2,4,6,8,10,12,14, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80
    };
    static const uint8_t odd_mask[16] __attribute__((aligned(16))) = {
        1,3,5,7,9,11,13,15, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80
    };
    __m128i veven_m = _mm_loadu_si128((const __m128i*)even_mask);
    __m128i vodd_m  = _mm_loadu_si128((const __m128i*)odd_mask);
    for (int s = 0; s < n_super; s++) {
        int8_t *dst_even = xq_arr + s*256;
        int8_t *dst_odd  = xq_arr + s*256 + 128;
        for (int k = 0; k < 8; k++) {
            const int8_t *src = xq + s*256 + k*32;
            /* 加载 32 字节为两个 128-bit */
            __m128i lo = _mm_loadu_si128((__m128i*)(src));      /* src[0..15] */
            __m128i hi = _mm_loadu_si128((__m128i*)(src + 16));  /* src[16..31] */
            /* shuffle: 取 lo 的偶数索引 (8 字节) + hi 的偶数索引 (8 字节) */
            __m128i lo_even = _mm_shuffle_epi8(lo, veven_m);  /* 8 even from lo + 8 zero */
            __m128i hi_even = _mm_shuffle_epi8(hi, veven_m);  /* 8 even from hi + 8 zero */
            __m128i even16 = _mm_unpacklo_epi64(lo_even, hi_even); /* 8+8 = 16 even */
            __m128i lo_odd = _mm_shuffle_epi8(lo, vodd_m);
            __m128i hi_odd = _mm_shuffle_epi8(hi, vodd_m);
            __m128i odd16 = _mm_unpacklo_epi64(lo_odd, hi_odd);
            _mm_storeu_si128((__m128i*)(dst_even + k*16), even16);
            _mm_storeu_si128((__m128i*)(dst_odd + k*16), odd16);
        }
    }
#else
    /* 标量 fallback */
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) { float a = fabsf(x[i]); if (a > x_max) x_max = a; }
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    float x_inv = 1.0f / x_scale;
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * x_inv);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }
    for (int sb = 0; sb < n_sub; sb++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) sum += (int)xq[sb * 32 + i];
        bsums[sb] = (int16_t)sum;
    }
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
#endif
    return x_scale;
}

/* === Q4_K matmul 接受预计算数据 (gate/up 共享预处理) === */
static inline void lal_matmul_q4_k_prepared(float * __restrict__ y,
                                              const uint8_t * __restrict__ q4k_W,
                                              const float * __restrict__ x,
                                              const float * __restrict__ b,
                                              int in_dim, int out_dim,
                                              const int8_t * __restrict__ xq,
                                              const int16_t * __restrict__ bsums,
                                              const int8_t * __restrict__ xq_arr,
                                              float x_scale) {
    int n_super = in_dim / 256;

#if defined(__AVX2__) && defined(__F16C__)
    int row_stride = n_super * 144;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const float inv_63 = 1.0f / 63.0f;

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

            #define PROC8_PREPARED(r) do { \
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
                /* prefetch 2 superblocks ahead (实测最优, pf_dist=2 比 pf_dist=1 快 1.5%) */ \
                if (s + 2 < n_super) { \
                    _mm_prefetch((const char*)(qs+288), _MM_HINT_T0); \
                    _mm_prefetch((const char*)(qs+288+64), _MM_HINT_T0); \
                } \
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
            PROC8_PREPARED(0); PROC8_PREPARED(1); PROC8_PREPARED(2); PROC8_PREPARED(3);
            PROC8_PREPARED(4); PROC8_PREPARED(5); PROC8_PREPARED(6); PROC8_PREPARED(7);
            #undef PROC8_PREPARED
        }

        /* Bug fix: 需要 horizontal sum acc 的 8 个 lane (各 lane 值不同) */
        #define HS8P(r,a,m) do{__m128 lo=_mm256_castps256_ps128(a),hi=_mm256_extractf128_ps(a,1);\
            __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);\
            y[j+r]=_mm_cvtss_f32(s2)+m+(b?b[j+r]:0);}while(0)
        HS8P(0,acc0,am0);HS8P(1,acc1,am1);HS8P(2,acc2,am2);HS8P(3,acc3,am3);
        HS8P(4,acc4,am4);HS8P(5,acc5,am5);HS8P(6,acc6,am6);HS8P(7,acc7,am7);
        #undef HS8P
    }
    for (; j < out_dim; j++) {
        const uint8_t *sb = q4k_W + (size_t)j * row_stride;
        float sumf = 0;
        for (int s = 0; s < n_super; s++) {
            float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+s*144))));
            float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+s*144+2))));
            uint8_t sm[16]; unpack_scales_6bit(sb+s*144+4,sm);
            /* Bug fix: sm[8+i] 是 uint8 但代表 int8 (min 值), 需要 sign-extend */
            int32_t mp_sum=0; for(int i=0;i<8;i++) mp_sum+=(int)(int8_t)sm[8+i]*bsums[s*8+i];
            float r_am=dmin*x_scale*(float)mp_sum*inv_63;
            int32_t isum=0;
            for (int k = 0; k < 8; k++) {
                const int8_t *xqs = xq + s*256 + k*32;
                const uint8_t *qs = sb + s*144 + 16 + k*16;
                for (int i = 0; i < 16; i++) {
                    int q4 = (qs[i] & 0x0F);
                    int q4h = (qs[i] >> 4) & 0x0F;
                    /* Bug fix: Q4_K 数学是 w = q4 * scale - min, 点积是 sum(q4 * xq) * scale - min * sum(xq)
                     * 不是 (q4 - scale) * xq. scale 是乘法因子不是减法. */
                    isum += q4 * xqs[2*i] + q4h * xqs[2*i+1];
                }
            }
            sumf += d * x_scale * inv_63 * isum - r_am;
        }
        y[j] = sumf + (b?b[j]:0);
    }
#else
    /* Fallback (no AVX2): reuse lal_matmul_q4_k */
    (void)xq; (void)bsums; (void)xq_arr; (void)x_scale;
    lal_matmul_q4_k(y, q4k_W, x, b, in_dim, out_dim);
#endif
}

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
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const float inv_63 = 1.0f / 63.0f;

    /* Pre-arrange xq for ADJACENT packing: split each 32-byte sub-block into
     * 16 even-indexed + 16 odd-indexed elements.
     * Layout per superblock (256 elements = 8 sub-blocks):
     *   [sub0-7 even (128 bytes), sub0-7 odd (128 bytes)]
     *
     * For 256-bit maddubs, we load 32 bytes at a time = 2 sub-blocks' even/odd:
     *   xq_arr[s*256 + 0..31]   = sub0_even(16) + sub1_even(16)   ← for iter 0
     *   xq_arr[s*256 + 32..63]  = sub2_even(16) + sub3_even(16)   ← for iter 1
     *   xq_arr[s*256 + 64..95]  = sub4_even(16) + sub5_even(16)   ← for iter 2
     *   xq_arr[s*256 + 96..127] = sub6_even(16) + sub7_even(16)   ← for iter 3
     *   xq_arr[s*256 + 128..159]= sub0_odd(16) + sub1_odd(16)     ← for iter 0
     *   ... etc
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

    /* 8-row parallel main loop.
     * Why 8 rows: AVX2 has 16 YMM registers. 8 accumulators (acc0-acc7) use
     * 8 registers, leaving 8 for temporals (xq, q4, p16, sc_v, etc.).
     * 4-row: not enough ILP. 16-row: register spill to stack (slower).
     * See ARCHITECTURE.md §3. */
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
            /* Pre-load xq_arr: 8 YMM (4 even + 4 odd), 32 bytes each */
            const __m256i xve0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+0));    /* sub0+1 even */
            const __m256i xve1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+32));   /* sub2+3 even */
            const __m256i xve2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+64));   /* sub4+5 even */
            const __m256i xve3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+96));   /* sub6+7 even */
            const __m256i xvo0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+128));  /* sub0+1 odd */
            const __m256i xvo1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+160));  /* sub2+3 odd */
            const __m256i xvo2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+192));  /* sub4+5 odd */
            const __m256i xvo3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+224));  /* sub6+7 odd */

            __m128i bs_v=_mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);

            #define PROC8(r) do { \
                const uint8_t *sb=rows[r]+s*144; \
                /* d/dmin are fp16 (2 bytes each). Convert to float for scaling. */ \
                float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb)))); \
                float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2)))); \
                /* Unpack 16 × 6-bit scales+mins. This is the most expensive \
                 * non-matmul operation (~15 instructions). See PITFALLS.md §1. */ \
                uint8_t sm[16] __attribute__((aligned(16))); unpack_scales_6bit(sb+4,sm); \
                /* Min correction: build mn_v from sm[8..15] using cvtepi8_epi16 \
                 * (2 instructions) instead of set_epi16 (7+ instructions). \
                 * See ARCHITECTURE.md §6 for why extract > hadd. */ \
                __m128i mn_bytes = _mm_loadl_epi64((__m128i*)(sm+8)); /* 8 bytes, zero-extended */ \
                __m128i mn_v = _mm_cvtepi8_epi16(mn_bytes); /* 8 int16 */ \
                __m128i mp=_mm_madd_epi16(mn_v,bs_v); \
                /* Sum 4 int32 lanes via extract (parallel, 1 cycle) not hadd (serial, 4 cycles) */ \
                int32_t mp_sum = _mm_cvtsi128_si32(mp) \
                    + _mm_extract_epi32(mp, 1) \
                    + _mm_extract_epi32(mp, 2) \
                    + _mm_extract_epi32(mp, 3); \
                float r_am=dmin*x_scale*(float)mp_sum*inv_63; \
                if(r==0)am0-=r_am;else if(r==1)am1-=r_am;else if(r==2)am2-=r_am; \
                else if(r==3)am3-=r_am;else if(r==4)am4-=r_am;else if(r==5)am5-=r_am; \
                else if(r==6)am6-=r_am;else am7-=r_am; \
                /* 4 iterations: 2 sub-blocks per iter, 256-bit maddubs */ \
                __m256i sumi=_mm256_setzero_si256(); \
                const uint8_t*qs=sb+16; \
                /* Prefetch 2 superblocks ahead (实测最优, pf_dist=2 比 pf_dist=1 快 1.5%) */ \
                if (s + 2 < n_super) { \
                    _mm_prefetch((const char*)(qs+288), _MM_HINT_T0); \
                    _mm_prefetch((const char*)(qs+288+64), _MM_HINT_T0); \
                } \
                /* iter 0: sub0+sub1 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)qs); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve0), \
                                                  _mm256_maddubs_epi16(q4h,xvo0)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[1]),_mm_set1_epi16((short)sm[0])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                /* iter 1: sub2+sub3 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+32)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve1), \
                                                  _mm256_maddubs_epi16(q4h,xvo1)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[3]),_mm_set1_epi16((short)sm[2])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                /* iter 2: sub4+sub5 */ \
                { \
                    __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+64)); \
                    __m256i q4l=_mm256_and_si256(q4b,m4); \
                    __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                    __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve2), \
                                                  _mm256_maddubs_epi16(q4h,xvo2)); \
                    __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[5]),_mm_set1_epi16((short)sm[4])); \
                    sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); \
                } \
                /* iter 3: sub6+sub7 */ \
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

    /* Tail: single-row (same 256-bit approach) */
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

            #define SR(qoff,xve,xvo,sa,sb_) do{\
                __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+qoff));\
                __m256i q4l=_mm256_and_si256(q4b,m4);\
                __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4);\
                __m256i pl=_mm256_maddubs_epi16(q4l,xve);\
                __m256i ph=_mm256_maddubs_epi16(q4h,xvo);\
                __m256i p16=_mm256_add_epi16(pl,ph);\
                __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sb_),_mm_set1_epi16((short)sa));\
                sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16));\
            }while(0)
            SR(0,xve0,xvo0,sm[0],sm[1]);SR(32,xve1,xvo1,sm[2],sm[3]);
            SR(64,xve2,xvo2,sm[4],sm[5]);SR(96,xve3,xvo3,sm[6],sm[7]);
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

#endif /* LAL_Q4K_KERNEL_H */
