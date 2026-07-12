/* bench_lm_head_prefetch.c — 测试 LM head 不同 prefetch 距离
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_lmh_pf scripts/bench_lm_head_prefetch.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define N_EMBD 3584
#define VOCAB_SIZE 152064
#define XQ_MAX 18944

#include "runtime/lal_q8_kernel.h"
#include "runtime/lal_simd_optim.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* LM head with configurable prefetch distance (in cache lines = 64 bytes) */
static inline void lm_head_pf(float *logits,
                               const int8_t *xq, const uint8_t *ax,
                               float scale_x,
                               const int8_t *wte_q, const float *scale_w,
                               int v_start, int v_end, int n_embd,
                               int pf_dist_cl) {  /* prefetch distance in cache lines */
#if defined(__AVX2__)
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
            /* Prefetch pf_dist_cl cache lines ahead */
            if (pf_dist_cl > 0) {
                _mm_prefetch((const char*)(wv + i + pf_dist_cl * 64), _MM_HINT_T0);
            }
        }
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        int32_t dot = _mm_cvtsi128_si32(s);
        for (; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
#endif
}

int main(void) {
    int n_threads = 2;
    omp_set_num_threads(n_threads);

    int8_t *lm_head_q = _mm_malloc((size_t)VOCAB_SIZE * N_EMBD, 32);
    float *lm_head_s = _mm_malloc(VOCAB_SIZE * sizeof(float), 32);
    float *x = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *logits = _mm_malloc(VOCAB_SIZE * sizeof(float), 32);
    int8_t *xq = _mm_malloc(N_EMBD, 32);
    uint8_t *abs_xq = _mm_malloc(N_EMBD, 32);

    srand(42);
    for (size_t i = 0; i < (size_t)VOCAB_SIZE * N_EMBD; i++)
        lm_head_q[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < VOCAB_SIZE; i++) lm_head_s[i] = 0.01f;
    for (int i = 0; i < N_EMBD; i++) x[i] = (rand()/((float)RAND_MAX)-0.5f) * 0.3f;

    float scale_x = lal_quantize_x_int8(x, xq, N_EMBD);
    lal_compute_abs_xq(xq, abs_xq, N_EMBD);

    int n_iter = 5;
    double mem_read = (double)VOCAB_SIZE * N_EMBD + N_EMBD * 4;

    printf("=== LM Head Prefetch Distance Sweep (%d threads) ===\n", n_threads);
    printf("%-12s  %-10s  %-10s\n", "pf_dist", "time(ms)", "GB/s");

    int pf_dists[] = {0, 1, 2, 4, 8, 16, 32};
    int n_pf = sizeof(pf_dists)/sizeof(pf_dists[0]);

    for (int pi = 0; pi < n_pf; pi++) {
        int pf = pf_dists[pi];
        /* warmup */
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int n = omp_get_num_threads();
            int v_per = (VOCAB_SIZE + n - 1) / n;
            int v_start = tid * v_per;
            int v_end = v_start + v_per;
            if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
            if (v_start < VOCAB_SIZE)
                lm_head_pf(logits, xq, abs_xq, scale_x, lm_head_q, lm_head_s, v_start, v_end, N_EMBD, pf);
        }
        double t0 = now_ms();
        for (int it = 0; it < n_iter; it++) {
            #pragma omp parallel num_threads(n_threads)
            {
                int tid = omp_get_thread_num();
                int n = omp_get_num_threads();
                int v_per = (VOCAB_SIZE + n - 1) / n;
                int v_start = tid * v_per;
                int v_end = v_start + v_per;
                if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
                if (v_start < VOCAB_SIZE)
                    lm_head_pf(logits, xq, abs_xq, scale_x, lm_head_q, lm_head_s, v_start, v_end, N_EMBD, pf);
            }
        }
        double dt = (now_ms() - t0) / n_iter;
        printf("%-12d  %-10.2f  %-10.1f\n", pf, dt, mem_read/dt/1e6);
    }

    /* Also test 8-row version */
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        int n = omp_get_num_threads();
        int v_per = (VOCAB_SIZE + n - 1) / n;
        int v_start = tid * v_per;
        int v_end = v_start + v_per;
        if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
        if (v_start < VOCAB_SIZE)
            lal_lm_head_int8_range_abs_8row(logits, xq, abs_xq, scale_x, lm_head_q, lm_head_s, v_start, v_end, N_EMBD);
    }
    double t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int n = omp_get_num_threads();
            int v_per = (VOCAB_SIZE + n - 1) / n;
            int v_start = tid * v_per;
            int v_end = v_start + v_per;
            if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
            if (v_start < VOCAB_SIZE)
                lal_lm_head_int8_range_abs_8row(logits, xq, abs_xq, scale_x, lm_head_q, lm_head_s, v_start, v_end, N_EMBD);
        }
    }
    double dt = (now_ms() - t0) / n_iter;
    printf("%-12s  %-10.2f  %-10.1f\n", "8row+pf2", dt, mem_read/dt/1e6);

    _mm_free(lm_head_q); _mm_free(lm_head_s); _mm_free(x);
    _mm_free(logits); _mm_free(xq); _mm_free(abs_xq);
    return 0;
}
