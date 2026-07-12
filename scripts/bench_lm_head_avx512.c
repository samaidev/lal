/* bench_lm_head_avx512.c — AVX-512 VNNI vs AVX2 LM head 对比
 * 构建: gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vnni -fopenmp -I. -o bench_lm_avx512 scripts/bench_lm_head_avx512.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>
#include <omp.h>

#define VOCAB_SIZE 152064
#define N_EMBD     3584

/* AVX2 baseline (sign-trick) */
static void lm_head_avx2(float *logits, const int8_t *xq, const uint8_t *ax,
                          float scale_x, const int8_t *wte_q, const float *scale_w,
                          int v_start, int v_end, int n_embd) {
    __m256i ones = _mm256_set1_epi16(1);
    for (int v = v_start; v < v_end; v++) {
        const int8_t *wv = wte_q + (size_t)v * n_embd;
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= n_embd; i += 32) {
            __m256i ax_v = _mm256_loadu_si256((__m256i*)(ax + i));
            __m256i sw_v = _mm256_sign_epi8(_mm256_loadu_si256((__m256i*)(wv + i)),
                                            _mm256_loadu_si256((__m256i*)(xq + i)));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(ax_v, sw_v), ones));
        }
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        int32_t dot = _mm_cvtsi128_si32(s);
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
}

/* AVX-512 VNNI 版本 */
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
static void lm_head_avx512_vnni(float *logits, const int8_t *xq, const uint8_t *ax,
                                  float scale_x, const int8_t *wte_q, const float *scale_w,
                                  int v_start, int v_end, int n_embd) {
    for (int v = v_start; v < v_end; v++) {
        const int8_t *wv = wte_q + (size_t)v * n_embd;
        __m512i acc = _mm512_setzero_si512();
        int i = 0;
        for (; i + 64 <= n_embd; i += 64) {
            __m512i vax = _mm512_loadu_si512((__m512i*)(ax + i));
            __m512i vw  = _mm512_loadu_si512((__m512i*)(wv + i));
            acc = _mm512_dpbusd_epi32(acc, vax, vw);
        }
        int32_t dot = _mm512_reduce_add_epi32(acc);
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
}
#endif

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    int8_t *xq    = NULL; posix_memalign((void**)&xq, 64, N_EMBD);
    uint8_t *ax   = NULL; posix_memalign((void**)&ax, 64, N_EMBD);
    int8_t *wte_q = NULL; posix_memalign((void**)&wte_q, 64, (size_t)VOCAB_SIZE * N_EMBD);
    float *scale_w = malloc(VOCAB_SIZE * sizeof(float));
    float *logits = malloc(VOCAB_SIZE * sizeof(float));

    srand(42);
    for (int i = 0; i < N_EMBD; i++) { xq[i] = (int8_t)((rand() % 256) - 128); ax[i] = (uint8_t)abs(xq[i]); }
    for (size_t i = 0; i < (size_t)VOCAB_SIZE * N_EMBD; i++) wte_q[i] = (int8_t)((rand() % 256) - 128);
    for (int v = 0; v < VOCAB_SIZE; v++) scale_w[v] = 0.001f;
    float scale_x = 0.01f;

    const int ITERS = 10;
    volatile float sink = 0;

    printf("=== LM head AVX-512 VNNI vs AVX2 (VOCAB=%d, N_EMBD=%d, %d iters) ===\n\n", VOCAB_SIZE, N_EMBD, ITERS);

    /* AVX2 单线程 */
    {
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            lm_head_avx2(logits, xq, ax, scale_x, wte_q, scale_w, 0, VOCAB_SIZE, N_EMBD);
            sink += logits[it % VOCAB_SIZE];
        }
        printf("AVX2 (1 thread):    %.2f ms/iter\n", (now_ms() - t0) / ITERS);
    }

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* AVX-512 VNNI 单线程 */
    {
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            lm_head_avx512_vnni(logits, xq, ax, scale_x, wte_q, scale_w, 0, VOCAB_SIZE, N_EMBD);
            sink += logits[it % VOCAB_SIZE];
        }
        printf("AVX-512 VNNI (1 thread): %.2f ms/iter\n", (now_ms() - t0) / ITERS);
    }

    /* 正确性验证 */
    lm_head_avx2(logits, xq, ax, scale_x, wte_q, scale_w, 0, 1000, N_EMBD);
    float logits2[1000];
    lm_head_avx512_vnni(logits2, xq, ax, scale_x, wte_q, scale_w, 0, 1000, N_EMBD);
    double max_err = 0;
    for (int v = 0; v < 1000; v++) {
        double err = fabs(logits[v] - logits2[v]);
        if (err > max_err) max_err = err;
    }
    printf("\n正确性 (AVX2 vs AVX-512): max_err = %.6e\n", max_err);

    /* AVX-512 VNNI 多线程 */
    printf("\n--- AVX-512 VNNI 多线程 ---\n");
    int nthreads[] = {1, 2, 4};
    for (int nt = 0; nt < 3; nt++) {
        int n = nthreads[nt];
        double t0 = now_ms();
        for (int it = 0; it < ITERS; it++) {
            #pragma omp parallel num_threads(n)
            {
                int tid = omp_get_thread_num();
                int v_per = (VOCAB_SIZE + n - 1) / n;
                int v_start = tid * v_per;
                int v_end = v_start + v_per;
                if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
                if (v_start < VOCAB_SIZE)
                    lm_head_avx512_vnni(logits, xq, ax, scale_x, wte_q, scale_w,
                                          v_start, v_end, N_EMBD);
            }
            sink += logits[it % VOCAB_SIZE];
        }
        printf("AVX-512 VNNI (%d threads): %.2f ms/iter\n", n, (now_ms() - t0) / ITERS);
    }
#else
    printf("AVX-512 VNNI not available\n");
#endif

    if (sink < -1e30) printf("impossible %f\n", (double)sink);
    free(xq); free(ax); free(wte_q); free(scale_w); free(logits);
    return 0;
}
