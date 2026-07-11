/* lal_q4k_kernel.h — Q4_K matmul kernel (llama.cpp-style optimization)
 *
 * Q4_K block: 144 bytes per 256 elements
 *   [2B d fp16][2B dmin fp16][12B scales+mins][128B packed q4]
 *
 * Key optimizations (learned from llama.cpp):
 * 1. Single x_scale for entire x vector (like Q8_K) → enables int16 scale path
 * 2. madd(sc_v, p16) at int16→int32 (no overflow, no per-sub-block cvtepi32_ps)
 * 3. 1 cvtepi32_ps + 1 fmadd per superblock (not per sub-block)
 * 4. Pre-computed bsums → 1 madd for min correction
 * 5. 4-row parallel sharing xq loads
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

static inline void lal_matmul_q4_k(float *y,
                                     const uint8_t *q4k_W,
                                     const float *x,
                                     const float *b,
                                     int in_dim, int out_dim) {
    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;

    /* Single-scale x quantization (like llama.cpp Q8_K) */
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

    /* Pre-compute bsums: sum of 32 int8 per sub-block (for min correction) */
    int16_t bsums[XQ_MAX / 32];
    for (int sb = 0; sb < n_sub; sb++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) sum += (int)xq[sb * 32 + i];
        bsums[sb] = (int16_t)sum;
    }

#if defined(__AVX2__) && defined(__F16C__)
    int row_stride = n_super * 144;
    __m128i mask_0f = _mm_set1_epi8(0x0F);
    float inv_63 = 1.0f / 63.0f;

    /* 4-row parallel */
    int j = 0;
    for (; j + 4 <= out_dim; j += 4) {
        const uint8_t *rows[4];
        for (int r = 0; r < 4; r++) rows[r] = q4k_W + (size_t)(j+r) * row_stride;

        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        float am0 = 0, am1 = 0, am2 = 0, am3 = 0;

        for (int s = 0; s < n_super; s++) {
            for (int r = 0; r < 4; r++) {
                const uint8_t *sb = rows[r] + s * 144;
                float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb))));
                float dmin = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2))));

                uint8_t sm[16]; unpack_scales_6bit(sb + 4, sm);

                /* Min correction: am -= (dmin * x_scale / 63) * dot(mn, bsums)
                 * Use SIMD: 8 mins × 8 bsums → 1 madd → 4 int32 → hsum */
                __m128i mn_v = _mm_set_epi16(sm[15],sm[14],sm[13],sm[12],sm[11],sm[10],sm[9],sm[8]);
                __m128i bs_v = _mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                                             bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);
                __m128i min_prod = _mm_madd_epi16(mn_v, bs_v);  /* 4 int32 */
                min_prod = _mm_hadd_epi32(min_prod, min_prod);
                min_prod = _mm_hadd_epi32(min_prod, min_prod);
                float r_am = dmin * x_scale * (float)_mm_cvtsi128_si32(min_prod) * inv_63;
                if (r == 0) am0 -= r_am; else if (r == 1) am1 -= r_am;
                else if (r == 2) am2 -= r_am; else am3 -= r_am;

                /* Process 8 sub-blocks with int16 scale via madd */
                __m256i sumi = _mm256_setzero_si256();
                const uint8_t *qs = sb + 16;

                for (int sub = 0; sub < 8; sub++) {
                    __m128i q4bits = _mm_loadu_si128((__m128i*)(qs + sub * 16));
                    __m128i q4l = _mm_and_si128(q4bits, mask_0f);
                    __m128i q4h = _mm_and_si128(_mm_srli_epi16(q4bits, 4), mask_0f);
                    __m128i xv_lo = _mm_loadu_si128((__m128i*)(xq + (s*8+sub)*32));
                    __m128i xv_hi = _mm_loadu_si128((__m128i*)(xq + (s*8+sub)*32 + 16));
                    __m128i p16l = _mm_maddubs_epi16(q4l, xv_lo);
                    __m128i p16h = _mm_maddubs_epi16(q4h, xv_hi);
                    /* madd(sc, p16): sc*pair0 + sc*pair1 → int32 (no overflow!) */
                    __m128i sc_v = _mm_set1_epi16((int16_t)sm[sub]);
                    __m128i s32l = _mm_madd_epi16(sc_v, p16l);
                    __m128i s32h = _mm_madd_epi16(sc_v, p16h);
                    __m256i s32 = _mm256_set_m128i(s32h, s32l);
                    sumi = _mm256_add_epi32(sumi, s32);
                }

                /* 1 cvtepi32_ps + 1 fmadd per superblock! */
                float mult = d * x_scale * inv_63;
                __m256 vd = _mm256_set1_ps(mult);
                __m256 *ap = (r==0)?&acc0:(r==1)?&acc1:(r==2)?&acc2:&acc3;
                *ap = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), *ap);
            }
        }

        #define HS4(r,a,m) do{__m128 lo=_mm256_castps256_ps128(a),hi=_mm256_extractf128_ps(a,1);\
            __m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);\
            y[j+r]=_mm_cvtss_f32(s)+m+(b?b[j+r]:0);}while(0)
        HS4(0,acc0,am0);HS4(1,acc1,am1);HS4(2,acc2,am2);HS4(3,acc3,am3);
        #undef HS4
    }

    /* Tail: single-row */
    for (; j < out_dim; j++) {
        const uint8_t *row = q4k_W + (size_t)j * row_stride;
        __m256 acc = _mm256_setzero_ps();
        float acc_min = 0;

        for (int s = 0; s < n_super; s++) {
            const uint8_t *sb = row + s * 144;
            float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb))));
            float dmin = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2))));
            uint8_t sm[16]; unpack_scales_6bit(sb + 4, sm);

            __m128i mn_v = _mm_set_epi16(sm[15],sm[14],sm[13],sm[12],sm[11],sm[10],sm[9],sm[8]);
            __m128i bs_v = _mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);
            __m128i mp = _mm_madd_epi16(mn_v, bs_v); mp = _mm_hadd_epi32(mp,mp); mp = _mm_hadd_epi32(mp,mp);
            acc_min -= dmin * x_scale * (float)_mm_cvtsi128_si32(mp) * inv_63;
            __m128i mn_v = _mm_set_epi16(sm[15],sm[14],sm[13],sm[12],sm[11],sm[10],sm[9],sm[8]);
            __m128i bs_v = _mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);
            __m128i mp = _mm_madd_epi16(mn_v, bs_v); mp = _mm_hadd_epi32(mp,mp); mp = _mm_hadd_epi32(mp,mp);
            acc_min -= dmin * x_scale * (float)_mm_cvtsi128_si32(mp) * inv_63;
            __m128i mn_v = _mm_set_epi16(sm[15],sm[14],sm[13],sm[12],sm[11],sm[10],sm[9],sm[8]);
            __m128i bs_v = _mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);
            __m128i mp = _mm_madd_epi16(mn_v, bs_v); mp = _mm_hadd_epi32(mp,mp); mp = _mm_hadd_epi32(mp,mp);
            acc_min -= dmin * x_scale * (float)_mm_cvtsi128_si32(mp) * inv_63;

            __m256i sumi = _mm256_setzero_si256();
            const uint8_t *qs = sb + 16;
            for (int sub = 0; sub < 8; sub++) {
                __m128i q4bits = _mm_loadu_si128((__m128i*)(qs + sub * 16));
                __m128i q4l = _mm_and_si128(q4bits, mask_0f);
                __m128i q4h = _mm_and_si128(_mm_srli_epi16(q4bits, 4), mask_0f);
                __m128i xv_lo = _mm_loadu_si128((__m128i*)(xq + (s*8+sub)*32));
                __m128i xv_hi = _mm_loadu_si128((__m128i*)(xq + (s*8+sub)*32 + 16));
                __m128i p16l = _mm_maddubs_epi16(q4l, xv_lo);
                __m128i p16h = _mm_maddubs_epi16(q4h, xv_hi);
                __m128i sc_v = _mm_set1_epi16((int16_t)sm[sub]);
                __m128i s32l = _mm_madd_epi16(sc_v, p16l);
                __m128i s32h = _mm_madd_epi16(sc_v, p16h);
                __m256i s32 = _mm256_set_m128i(s32h, s32l);
                sumi = _mm256_add_epi32(sumi, s32);
            }
            float mult = d * x_scale * inv_63;
            acc = _mm256_fmadd_ps(_mm256_set1_ps(mult), _mm256_cvtepi32_ps(sumi), acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        y[j] = _mm_cvtss_f32(s) + acc_min + (b ? b[j] : 0);
    }
#else
    /* Scalar fallback */
    int row_stride = n_super * 144;
    float inv_63 = 1.0f / 63.0f;
    for (int j = 0; j < out_dim; j++) {
        const uint8_t *row = q4k_W + (size_t)j * row_stride;
        float acc = 0, acc_min = 0;
        for (int s = 0; s < n_super; s++) {
            float min_dot = 0;
            for (int k = 0; k < 8; k++) min_dot += (float)sm[8+k] * (float)bsums[s*8+k];
            acc_min -= dmin * x_scale * min_dot * inv_63;
            float min_dot = 0;
            for (int k = 0; k < 8; k++) min_dot += (float)sm[8+k] * (float)bsums[s*8+k];
            acc_min -= dmin * x_scale * min_dot * inv_63;
            float min_dot = 0;
            for (int k = 0; k < 8; k++) min_dot += (float)sm[8+k] * (float)bsums[s*8+k];
            acc_min -= dmin * x_scale * min_dot * inv_63;
            #define F16(u16,out) do{uint32_t sg=(u16>>15),ex=(u16>>10)&0x1F,fr=u16&0x3FF;\
                if(ex==0)out=fr?(fr/1024.0f/524288.0f)*(sg?-1:1):0;\
                else if(ex==31)out=fr?(float)NAN:(sg?-(float)INFINITY:(float)INFINITY);\
                else{float f=(1.0f+fr/1024.0f)*ldexpf(1.0f,(int)ex-15);out=sg?-f:f;}}while(0)
            F16(d_u16,d); F16(dmin_u16,dmin);
            uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);
            float min_dot = 0;
            for (int k = 0; k < 8; k++) min_dot += (float)sm[8+k] * (float)bsums[s*8+k];
            acc_min -= dmin * x_scale * min_dot * inv_63;
            const uint8_t *qs = sb + 16;
            for (int sub = 0; sub < 8; sub++) {
                int xoff = (s*8+sub)*32, qoff = sub*16;
                int32_t dot = 0;
                for (int i = 0; i < 16; i++) {
                    uint8_t bv = qs[qoff+i];
                    dot += (int)(bv & 0xF) * (int)xq[xoff+i];
                    dot += (int)((bv>>4) & 0xF) * (int)xq[xoff+i+16];
                }
                acc += d * (float)sm[sub] * (float)dot * x_scale * inv_63;
            }
        }
        y[j] = acc + acc_min + (b ? b[j] : 0);
    }
#endif
}

#endif /* LAL_Q4K_KERNEL_H */
