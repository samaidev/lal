/* bench_lm_head.c — LM head int8 dot product 性能基准
 * Qwen2.5-7B: VOCAB=152064, N_EMBD=3584
 *
 * 测试:
 *   1. 当前实现 (baseline)
 *   2. 优化版: abs_xq 预计算一次 + 软件流水线 prefetch
 *
 * 构建: gcc -O3 -march=native -fopenmp -I. -o bench_lm_head scripts/bench_lm_head.c -lm -lgomp
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

/* ============ Baseline LM head (从 lal_q8_kernel.h 复制) ============ */
static inline void lm_head_baseline(float *logits,
                                     const int8_t *xq, float scale_x,
                                     const int8_t *wte_q, const float *scale_w,
                                     int v_start, int v_end, int n_embd) {
    static uint8_t ax[N_EMBD] __attribute__((aligned(32)));
    for (int i = 0; i < n_embd; i += 32) {
        __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
        __m256i abs_xq = _mm256_sign_epi8(xv, xv);
        _mm256_storeu_si256((__m256i*)(ax + i), abs_xq);
    }
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
        for (; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
}

/* ============ 优化版 1: abs_xq 作为参数传入 (不在函数内重算) ============ */
static inline void lm_head_opt_precomputed_abs(float *logits,
                                                const int8_t *xq, const uint8_t *ax,
                                                float scale_x,
                                                const int8_t *wte_q, const float *scale_w,
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
        for (; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
}

/* ============ 优化版 2: + 软件 prefetch (next vocab row) ============ */
static inline void lm_head_opt_prefetch(float *logits,
                                         const int8_t *xq, const uint8_t *ax,
                                         float scale_x,
                                         const int8_t *wte_q, const float *scale_w,
                                         int v_start, int v_end, int n_embd) {
    __m256i ones = _mm256_set1_epi16(1);
    for (int v = v_start; v < v_end; v++) {
        const int8_t *wv = wte_q + (size_t)v * n_embd;
        /* prefetch next vocab row (2 rows ahead, 64 bytes = 1 cache line) */
        if (v + 2 < v_end) {
            const int8_t *next_wv = wte_q + (size_t)(v + 2) * n_embd;
            _mm_prefetch((const char*)(next_wv),     _MM_HINT_T0);
            _mm_prefetch((const char*)(next_wv + 64), _MM_HINT_T0);
        }
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
        for (; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
        logits[v] = scale_x * scale_w[v] * (float)dot;
    }
}

/* ============ Timing ============ */
static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    /* 分配对齐内存 */
    int8_t *xq     = NULL; posix_memalign((void**)&xq,     32, N_EMBD);
    int8_t *wte_q  = NULL; posix_memalign((void**)&wte_q,  32, (size_t)VOCAB_SIZE * N_EMBD);
    float  *scale_w= malloc(VOCAB_SIZE * sizeof(float));
    float  *logits = malloc(VOCAB_SIZE * sizeof(float));
    uint8_t *ax    = NULL; posix_memalign((void**)&ax,     32, N_EMBD);

    /* 初始化随机数据 */
    srand(42);
    for (int i = 0; i < N_EMBD; i++) xq[i] = (int8_t)((rand() % 256) - 128);
    for (size_t i = 0; i < (size_t)VOCAB_SIZE * N_EMBD; i++) wte_q[i] = (int8_t)((rand() % 256) - 128);
    for (int v = 0; v < VOCAB_SIZE; v++) scale_w[v] = 0.001f * (1.0f + 0.5f * (rand() / (float)RAND_MAX));
    float scale_x = 0.01f;

    /* 预计算 abs_xq */
    for (int i = 0; i < N_EMBD; i += 32) {
        __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
        __m256i abs_xq = _mm256_sign_epi8(xv, xv);
        _mm256_storeu_si256((__m256i*)(ax + i), abs_xq);
    }

    const int ITERS = 20;
    volatile float sink = 0;

    printf("=== LM head Benchmark (VOCAB=%d, N_EMBD=%d, %d iters) ===\n\n", VOCAB_SIZE, N_EMBD, ITERS);

    /* Baseline (单线程) */
    {
        double t0 = now_us();
        for (int it = 0; it < ITERS; it++) {
            lm_head_baseline(logits, xq, scale_x, wte_q, scale_w, 0, VOCAB_SIZE, N_EMBD);
            sink += logits[it % VOCAB_SIZE];
        }
        double t = (now_us() - t0) / ITERS;
        printf("Baseline (单线程, abs重算):      %.2f ms/iter\n", t / 1000);
    }

    /* 优化版 1: abs 预计算 (单线程) */
    {
        double t0 = now_us();
        for (int it = 0; it < ITERS; it++) {
            lm_head_opt_precomputed_abs(logits, xq, ax, scale_x, wte_q, scale_w, 0, VOCAB_SIZE, N_EMBD);
            sink += logits[it % VOCAB_SIZE];
        }
        double t = (now_us() - t0) / ITERS;
        printf("Opt1 (abs预计算, 单线程):         %.2f ms/iter\n", t / 1000);
    }

    /* 优化版 2: abs预计算 + prefetch (单线程) */
    {
        double t0 = now_us();
        for (int it = 0; it < ITERS; it++) {
            lm_head_opt_prefetch(logits, xq, ax, scale_x, wte_q, scale_w, 0, VOCAB_SIZE, N_EMBD);
            sink += logits[it % VOCAB_SIZE];
        }
        double t = (now_us() - t0) / ITERS;
        printf("Opt2 (abs预计算+prefetch, 单线程): %.2f ms/iter\n", t / 1000);
    }

    /* 多线程测试 */
    printf("\n--- 多线程 (OpenMP) ---\n");
    int nthreads[] = {1, 2, 4};
    for (int nt = 0; nt < 3; nt++) {
        int n = nthreads[nt];
        double t0 = now_us();
        for (int it = 0; it < ITERS; it++) {
            #pragma omp parallel num_threads(n)
            {
                int tid = omp_get_thread_num();
                int v_per = (VOCAB_SIZE + n - 1) / n;
                int v_start = tid * v_per;
                int v_end = v_start + v_per;
                if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
                if (v_start < VOCAB_SIZE)
                    lm_head_opt_prefetch(logits, xq, ax, scale_x, wte_q, scale_w,
                                          v_start, v_end, N_EMBD);
            }
            sink += logits[it % VOCAB_SIZE];
        }
        double t = (now_us() - t0) / ITERS;
        printf("Opt2 (%d threads): %.2f ms/iter\n", n, t / 1000);
    }

    /* 正确性验证: opt1 vs baseline */
    lm_head_baseline(logits, xq, scale_x, wte_q, scale_w, 0, 1000, N_EMBD);
    float logits2[1000];
    lm_head_opt_precomputed_abs(logits2, xq, ax, scale_x, wte_q, scale_w, 0, 1000, N_EMBD);
    double max_err = 0;
    for (int v = 0; v < 1000; v++) {
        double err = fabs(logits[v] - logits2[v]);
        if (err > max_err) max_err = err;
    }
    printf("\n正确性验证 (baseline vs opt1): 最大误差 = %.6e (应该为 0)\n", max_err);

    if (sink < -1e30) printf("impossible %f\n", (double)sink);

    free(xq); free(wte_q); free(scale_w); free(logits); free(ax);
    return 0;
}
