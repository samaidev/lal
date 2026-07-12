/* bench_avx512_vs_avx2.c — 对比 AVX2 vs AVX512 Q4_K kernel 性能
 * Build: gcc -O3 -march=native -I. -o bench_avx512 scripts/bench_avx512_vs_avx2.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel.h"
#if defined(__AVX512BW__)
  #include "runtime/lal_q4k_kernel_avx512.h"
#endif

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* 从 bench_q4k.c 复制的量化函数 */
static void quantize_q4_k_row(const float *w, uint8_t *out, int in_dim) {
    int n_super = in_dim / 256;
    for (int s = 0; s < n_super; s++) {
        const float *wb = w + s * 256;
        uint8_t *sb = out + s * 144;
        float sub_min[8], sub_max[8], sub_scale[8];
        float max_scale = 0, max_min = 0;
        for (int j = 0; j < 8; j++) {
            sub_min[j] = 1e30f; sub_max[j] = -1e30f;
            for (int i = 0; i < 32; i++) { float v = wb[j*32+i]; if (v<sub_min[j]) sub_min[j]=v; if (v>sub_max[j]) sub_max[j]=v; }
            sub_scale[j] = (sub_max[j] - sub_min[j]) / 15.0f;
            if (sub_scale[j] < 1e-8f) sub_scale[j] = 1e-8f;
            if (sub_scale[j] > max_scale) max_scale = sub_scale[j];
            if (fabsf(sub_min[j]) > max_min) max_min = fabsf(sub_min[j]);
        }
        if (max_scale < 1e-8f) max_scale = 1e-8f;
        if (max_min < 1e-8f) max_min = 1e-8f;
        float d = max_scale, dmin = max_min;
        __m128 d_f = _mm_set1_ps(d);
        __m128i d_h = _mm_cvtps_ph(d_f, 0);
        uint16_t d_u16 = _mm_extract_epi16(d_h, 0);
        __m128 dmin_f = _mm_set1_ps(dmin);
        __m128i dmin_h = _mm_cvtps_ph(dmin_f, 0);
        uint16_t dmin_u16 = _mm_extract_epi16(dmin_h, 0);
        *(uint16_t*)(sb) = d_u16;
        *(uint16_t*)(sb+2) = dmin_u16;
        uint8_t sc6[8], m6[8];
        uint64_t combined[16];
        for (int j = 0; j < 8; j++) {
            sc6[j] = (uint8_t)(lroundf(sub_scale[j] / d * 63.0f) & 0x3F);
            m6[j] = (uint8_t)(lroundf(fabsf(sub_min[j]) / dmin * 63.0f) & 0x3F);
        }
        for (int j = 0; j < 16; j++) { combined[j] = sc6[j]; combined[j+8>15?15:j+8] = m6[j]; }
        for (int j = 0; j < 16; j++) combined[j] = j < 8 ? sc6[j] : m6[j-8];
        __uint128_t bits = 0;
        for (int j = 0; j < 16; j++) bits |= ((__uint128_t)(combined[j] & 0x3F)) << (j * 6);
        for (int b = 0; b < 12; b++) sb[4+b] = (bits >> (b * 8)) & 0xFF;
        uint8_t *qs = sb + 16;
        for (int j = 0; j < 8; j++) {
            float ascale = d * sc6[j] / 63.0f;
            float amin = dmin * m6[j] / 63.0f;
            for (int i = 0; i < 16; i++) {
                int idx_lo = j*32 + i, idx_hi = j*32 + i + 16;
                int q_lo = lroundf((wb[idx_lo] + amin) / (ascale + 1e-8f));
                int q_hi = lroundf((wb[idx_hi] + amin) / (ascale + 1e-8f));
                if (q_lo < 0) q_lo = 0; if (q_lo > 15) q_lo = 15;
                if (q_hi < 0) q_hi = 0; if (q_hi > 15) q_hi = 15;
                qs[j*16 + i] = (uint8_t)q_lo | ((uint8_t)q_hi << 4);
            }
        }
    }
}

int main() {
    struct { const char *name; int in_dim, out_dim; } tests[] = {
        {"q_proj  [512, 3584]",  3584, 512},
        {"gate    [18944, 3584]", 3584, 18944},
        {"down    [3584, 18944]", 18944, 3584},
    };
    int n_tests = sizeof(tests)/sizeof(tests[0]);
    srand(42);

    printf("=== AVX2 vs AVX-512 Q4_K Kernel ===\n");
    printf("%-22s  %-12s  %-12s  %-8s\n", "test", "AVX2(ms)", "AVX512(ms)", "speedup");

    for (int t = 0; t < n_tests; t++) {
        int in_dim = tests[t].in_dim, out_dim = tests[t].out_dim;
        int n_super = in_dim / 256;
        int row_stride = n_super * 144;

        float *w = malloc((size_t)out_dim*in_dim*sizeof(float));
        float *x = malloc(in_dim*sizeof(float));
        uint8_t *q4k = malloc((size_t)out_dim * row_stride);
        float *y = malloc(out_dim*sizeof(float));

        for (size_t i = 0; i < (size_t)out_dim*in_dim; i++) w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        for (int i = 0; i < in_dim; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        for (int j = 0; j < out_dim; j++) quantize_q4_k_row(w + j*in_dim, q4k + (size_t)j*row_stride, in_dim);

        /* AVX2 warmup + bench */
        lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
        int n_iter = 5;
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++) lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
        double dt_avx2 = (now_s() - t0) / n_iter;

#if defined(__AVX512BW__)
        /* AVX-512 warmup + bench */
        lal_matmul_q4_k_avx512(y, q4k, x, NULL, in_dim, out_dim);
        t0 = now_s();
        for (int it = 0; it < n_iter; it++) lal_matmul_q4_k_avx512(y, q4k, x, NULL, in_dim, out_dim);
        double dt_avx512 = (now_s() - t0) / n_iter;
        printf("%-22s  %-12.2f  %-12.2f  %-8.2fx\n", tests[t].name, dt_avx2*1000, dt_avx512*1000, dt_avx2/dt_avx512);
#else
        printf("%-22s  %-12.2f  (AVX512 not available)\n", tests[t].name, dt_avx2*1000);
#endif

        free(w); free(x); free(q4k); free(y);
    }
    return 0;
}
