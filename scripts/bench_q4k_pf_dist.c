/* bench_q4k_pf_dist.c — 测试 Q4_K kernel 不同 prefetch 距离
 * 构建: gcc -O3 -march=native -fopenmp -I. -o bench_q4k_pfd scripts/bench_q4k_pf_dist.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define XQ_MAX 18944
#define IN_DIM  3584
#define OUT_DIM 18944

#include "runtime/lal_q4k_kernel.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* 手动测试不同 prefetch 距离的 Q4_K matmul */
static inline void lal_matmul_q4_k_pf(float * __restrict__ y,
                                        const uint8_t * __restrict__ q4k_W,
                                        const float * __restrict__ x,
                                        const float * __restrict__ b,
                                        int in_dim, int out_dim,
                                        int pf_dist) {
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

            #define PROC8_PF(r) do { \
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
                /* 可配置 prefetch 距离 */ \
                if (s + pf_dist < n_super) { \
                    const uint8_t *pf = rows[r] + (s+pf_dist)*144 + 16; \
                    _mm_prefetch((const char*)pf, _MM_HINT_T0); \
                    _mm_prefetch((const char*)(pf+64), _MM_HINT_T0); \
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
            PROC8_PF(0); PROC8_PF(1); PROC8_PF(2); PROC8_PF(3);
            PROC8_PF(4); PROC8_PF(5); PROC8_PF(6); PROC8_PF(7);
            #undef PROC8_PF
        }

        y[j+0]+=_mm256_cvtss_f32(acc0)+am0; y[j+1]+=_mm256_cvtss_f32(acc1)+am1;
        y[j+2]+=_mm256_cvtss_f32(acc2)+am2; y[j+3]+=_mm256_cvtss_f32(acc3)+am3;
        y[j+4]+=_mm256_cvtss_f32(acc4)+am4; y[j+5]+=_mm256_cvtss_f32(acc5)+am5;
        y[j+6]+=_mm256_cvtss_f32(acc6)+am6; y[j+7]+=_mm256_cvtss_f32(acc7)+am7;
    }
    for (; j < out_dim; j++) {
        const uint8_t *sb = q4k_W + (size_t)j * row_stride;
        float sumf = 0;
        for (int s = 0; s < n_super; s++) {
            float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+s*144))));
            float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+s*144+2))));
            uint8_t sm[16]; unpack_scales_6bit(sb+s*144+4,sm);
            int32_t mp_sum=0; for(int i=0;i<8;i++) mp_sum+=(int)(int8_t)sm[8+i]*bsums[s*8+i];
            float r_am=dmin*x_scale*(float)mp_sum*inv_63;
            int32_t isum=0;
            for (int k = 0; k < 8; k++) {
                const int8_t *xqs = xq + s*256 + k*32;
                const uint8_t *qs = sb + s*144 + 16 + k*16;
                for (int i = 0; i < 16; i++) {
                    int q4 = (qs[i] & 0x0F);
                    int q4h = (qs[i] >> 4) & 0x0F;
                    isum += q4 * xqs[2*i] + q4h * xqs[2*i+1];
                }
            }
            sumf += d * x_scale * inv_63 * isum - r_am;
        }
        y[j] += sumf;
    }
#endif
}

int main(void) {
    float *x = _mm_malloc(IN_DIM * sizeof(float), 32);
    float *y = _mm_malloc(OUT_DIM * sizeof(float), 32);
    int n_super = IN_DIM / 256;
    int row_stride = n_super * 144;
    uint8_t *w = _mm_malloc((size_t)OUT_DIM * row_stride, 32);

    srand(42);
    for (int i = 0; i < IN_DIM; i++) x[i] = (float)(rand() % 200 - 100) / 50.0f;
    for (size_t i = 0; i < (size_t)OUT_DIM * row_stride; i++) w[i] = rand() & 0xFF;
    for (int r = 0; r < OUT_DIM; r++)
        for (int s = 0; s < n_super; s++) {
            uint8_t *sb = w + (size_t)r * row_stride + s * 144;
            *(uint16_t*)sb = 0x2E66; *(uint16_t*)(sb+2) = 0x25C3;
            memset(sb+4, 0x20, 12);
        }

    const int ITERS = 20;
    volatile float sink = 0;

    printf("=== Q4_K prefetch 距离测试 (%d iters, 1 thread) ===\n", ITERS);
    int pf_values[] = {0, 1, 2, 4, 8};
    int n_pf = sizeof(pf_values)/sizeof(pf_values[0]);
    for (int pfi = 0; pfi < n_pf; pfi++) {
        int pf = pf_values[pfi];
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            lal_matmul_q4_k_pf(y, w, x, NULL, IN_DIM, OUT_DIM, pf);
            sink += y[it % OUT_DIM];
        }
        printf("pf_dist=%d: %.2f ms/iter\n", pf, (now_ms() - t0) / ITERS);
    }

    if (sink < -1e30) printf("impossible\n");
    return 0;
}
