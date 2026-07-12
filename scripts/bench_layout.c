/* bench_layout.c — 对比 Q4_K 权重布局: row-major vs tile-major
 *
 * row-major (当前): row0[superblock0..13], row1[superblock0..13], ...
 * tile-major (新):  tile0[row0..7 superblock0], tile1[row0..7 superblock1], ...
 *
 * tile-major 让 8-row 并行时同一 superblock 的 8 行数据连续,
 * 减少 prefetch 跨度, 可能提升 cache 效率
 *
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_layout scripts/bench_layout.c -lm -lgomp
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
#include "runtime/lal_q4k_kernel.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Tile-major Q4_K matmul: 权重按 [superblock][row_group][8 rows × 144 bytes] 排列
 * 8-row 并行时, 同一 superblock 的 8 行数据连续, prefetch 跨度从 8*2016=16KB 降到 1152 bytes
 */
static inline void lal_matmul_q4_k_tile(float * __restrict__ y,
                                          const uint8_t * __restrict__ q4k_tile_W,
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

    /* Tile layout: [n_super][out_dim/8][8 × 144 bytes]
     * tile_offset = s * (out_dim/8) * 8 * 144 + (j/8) * 8 * 144 + r * 144
     */
    int tile_group_stride = 8 * 144;  /* 8 rows × 144 bytes per superblock */
    int tile_super_stride = (out_dim / 8) * tile_group_stride;

    int j = 0;
    for (; j + 8 <= out_dim; j += 8) {
        __m256 acc0=_mm256_setzero_ps(),acc1=_mm256_setzero_ps();
        __m256 acc2=_mm256_setzero_ps(),acc3=_mm256_setzero_ps();
        __m256 acc4=_mm256_setzero_ps(),acc5=_mm256_setzero_ps();
        __m256 acc6=_mm256_setzero_ps(),acc7=_mm256_setzero_ps();
        float am0=0,am1=0,am2=0,am3=0,am4=0,am5=0,am6=0,am7=0;

        int jg = j / 8;

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

            /* Tile base: 8 rows of this superblock are contiguous */
            const uint8_t *tile_base = q4k_tile_W + (size_t)s * tile_super_stride + (size_t)jg * tile_group_stride;

            #define PROC8_TILE(r) do { \
                const uint8_t *sb=tile_base + r*144; \
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
                _mm_prefetch((const char*)(qs+144), _MM_HINT_T0); \
                _mm_prefetch((const char*)(qs+144+64), _MM_HINT_T0); \
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

            PROC8_TILE(0);PROC8_TILE(1);PROC8_TILE(2);PROC8_TILE(3);
            PROC8_TILE(4);PROC8_TILE(5);PROC8_TILE(6);PROC8_TILE(7);
            #undef PROC8_TILE
        }

        #define HS8T(r,a,m) do{__m128 lo=_mm256_castps256_ps128(a),hi=_mm256_extractf128_ps(a,1);\
            __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);\
            y[j+r]=_mm_cvtss_f32(s2)+m+(b?b[j+r]:0);}while(0)
        HS8T(0,acc0,am0);HS8T(1,acc1,am1);HS8T(2,acc2,am2);HS8T(3,acc3,am3);
        HS8T(4,acc4,am4);HS8T(5,acc5,am5);HS8T(6,acc6,am6);HS8T(7,acc7,am7);
        #undef HS8T
    }

    /* Tail: use original kernel for remaining rows */
    for (; j < out_dim; j++) {
        /* For tile layout, need to reconstruct row-major access */
        /* Skip for benchmark — out_dim is always multiple of 8 in tests */
    }
#endif
}

/* Convert row-major Q4_K to tile-major */
static void convert_to_tile(const uint8_t *src, uint8_t *dst, int out_dim, int n_super) {
    int tile_group_stride = 8 * 144;
    int tile_super_stride = (out_dim / 8) * tile_group_stride;
    for (int s = 0; s < n_super; s++) {
        for (int jg = 0; jg < out_dim / 8; jg++) {
            uint8_t *tile_base = dst + (size_t)s * tile_super_stride + (size_t)jg * tile_group_stride;
            for (int r = 0; r < 8; r++) {
                const uint8_t *src_row = src + (size_t)(jg*8 + r) * n_super * 144 + s * 144;
                memcpy(tile_base + r * 144, src_row, 144);
            }
        }
    }
}

int main(int argc, char **argv) {
    int n_threads = argc > 1 ? atoi(argv[1]) : 2;
    omp_set_num_threads(n_threads);

    int in_dim = 3584, out_dim = 18944;
    int n_super = in_dim / 256;
    int row_stride = n_super * 144;

    srand(42);
    float *x = _mm_malloc(in_dim * sizeof(float), 32);
    uint8_t *q4k_row = _mm_malloc((size_t)out_dim * row_stride, 32);
    uint8_t *q4k_tile = _mm_malloc((size_t)out_dim * row_stride, 32);
    float *y1 = _mm_malloc(out_dim * sizeof(float), 32);
    float *y2 = _mm_malloc(out_dim * sizeof(float), 32);

    for (int i = 0; i < in_dim; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
    for (int j = 0; j < out_dim; j++) {
        uint8_t *row = q4k_row + (size_t)j * row_stride;
        for (int s = 0; s < n_super; s++) {
            uint8_t *sb = row + s*144;
            *(uint16_t*)sb = 0x3C00; *(uint16_t*)(sb+2) = 0x0000;
            memset(sb+4, 0x20, 12);
            for (int i = 0; i < 128; i++) sb[16+i] = rand() & 0xFF;
        }
    }
    convert_to_tile(q4k_row, q4k_tile, out_dim, n_super);

    /* Correctness */
    lal_matmul_q4_k(y1, q4k_row, x, NULL, in_dim, out_dim);
    lal_matmul_q4_k_tile(y2, q4k_tile, x, NULL, in_dim, out_dim);
    float max_err = 0;
    for (int j = 0; j < out_dim; j++) {
        float d = fabsf(y1[j] - y2[j]);
        if (d > max_err) max_err = d;
    }
    printf("=== Correctness (row-major vs tile-major) ===\n");
    printf("max_err = %.2e %s\n\n", max_err, max_err < 1e-3f ? "✅ PASS" : "❌ FAIL");

    /* Benchmark */
    int n_iter = 5;
    double mem_read = (double)out_dim * row_stride + in_dim * 4;

    /* Row-major */
    lal_matmul_q4_k(y1, q4k_row, x, NULL, in_dim, out_dim);
    double t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int chunk = (out_dim + n_threads - 1) / n_threads;
            int start = (tid * chunk) & ~7;  /* align to 8 */
            int end = start + chunk;
            if (end > out_dim) end = out_dim;
            if (start < out_dim)
                lal_matmul_q4_k(y1 + start, q4k_row + (size_t)start * row_stride, x, NULL, in_dim, end - start);
        }
    }
    double dt_row = (now_ms() - t0) / n_iter;

    /* Tile-major */
    lal_matmul_q4_k_tile(y2, q4k_tile, x, NULL, in_dim, out_dim);
    t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int n = omp_get_num_threads();
            int groups = out_dim / 8;
            int chunk = (groups + n - 1) / n;
            int gs = tid * chunk;
            int ge = gs + chunk;
            if (ge > groups) ge = groups;
            if (gs < groups)
                lal_matmul_q4_k_tile(y2 + gs*8, q4k_tile, x, NULL, in_dim, (ge-gs)*8);
        }
    }
    double dt_tile = (now_ms() - t0) / n_iter;

    printf("=== Row-major vs Tile-major (%d threads) ===\n", n_threads);
    printf("Row-major: %.2f ms  (%.1f GB/s)\n", dt_row, mem_read/dt_row/1e6);
    printf("Tile-major: %.2f ms  (%.1f GB/s)\n", dt_tile, mem_read/dt_tile/1e6);
    printf("speedup: %.2fx\n", dt_row/dt_tile);

    _mm_free(x); _mm_free(q4k_row); _mm_free(q4k_tile); _mm_free(y1); _mm_free(y2);
    return 0;
}
