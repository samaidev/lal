/* bench_lm_head.c — Benchmark: FP32 vs int8 LM head, single vs multi-thread
 *
 * This standalone benchmark isolates the LM head cost from the full forward pass.
 * It quantizes wte at startup, then times:
 *   1. FP32 LM head (full vocab, single thread)
 *   2. Int8 LM head (full vocab, single thread)
 *   3. Int8 LM head (full vocab, 2 threads)
 *
 * Build: cc -O3 -mavx2 -mfma -o scripts/bench_lm_head scripts/bench_lm_head.c -lm -lpthread
 * Run:   ./scripts/bench_lm_head
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#if defined(__x86_64__)
  #include <immintrin.h>
  typedef __m256 v8f;
  #define V8F_ZERO()    _mm256_setzero_ps()
  #define V8F_LOAD(p)   _mm256_loadu_ps(p)
  #define V8F_FMADD(a,b,c) _mm256_fmadd_ps((a),(b),(c))
  static inline float v8f_hsum(v8f v) {
      float t[8]; _mm256_storeu_ps(t,v);
      return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
  }
#else
  typedef struct { float v[8]; } v8f;
  static inline v8f V8F_ZERO(void) { v8f r; memset(r.v,0,32); return r; }
  static inline v8f V8F_LOAD(const float *p) { v8f r; memcpy(r.v,p,32); return r; }
  static inline v8f V8F_FMADD(v8f a,v8f b,v8f c) { v8f r; for(int i=0;i<8;i++) r.v[i]=a.v[i]*b.v[i]+c.v[i]; return r; }
  static inline float v8f_hsum(v8f v) { float s=0; for(int i=0;i<8;i++) s+=v.v[i]; return s; }
#endif

#define N_EMBD     896
#define VOCAB_SIZE 151936
#define N_ITER     20

/* Simulated activation (random values typical of a real hidden state) */
static float g_x[N_EMBD];

/* FP32 wte (just a small portion loaded from file) */
static float *g_wte;

/* Int8 quantized wte */
static int8_t *g_wte_q;
static float  *g_wte_scale;

static void init_wte_random(void) {
    /* For benchmarking purposes, use random weights */
    g_wte = malloc((size_t)VOCAB_SIZE * N_EMBD * sizeof(float));
    g_wte_q = malloc((size_t)VOCAB_SIZE * N_EMBD);
    g_wte_scale = malloc(VOCAB_SIZE * sizeof(float));

    srand(42);
    for (int v = 0; v < VOCAB_SIZE; v++) {
        float mx = 0;
        for (int i = 0; i < N_EMBD; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            g_wte[(size_t)v * N_EMBD + i] = val;
            float a = fabsf(val); if (a > mx) mx = a;
        }
        g_wte_scale[v] = mx / 127.0f; if (g_wte_scale[v] < 1e-8f) g_wte_scale[v] = 1e-8f;
        float inv = 1.0f / g_wte_scale[v];
        for (int i = 0; i < N_EMBD; i++) {
            int val = (int)lroundf(g_wte[(size_t)v * N_EMBD + i] * inv);
            if (val > 127) val = 127; if (val < -127) val = -127;
            g_wte_q[(size_t)v * N_EMBD + i] = (int8_t)val;
        }
    }

    /* Random activation */
    for (int i = 0; i < N_EMBD; i++)
        g_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

static float *g_logits;

/* === FP32 LM head === */
static void lm_head_fp32(float *logits, const float *x) {
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const float *row = g_wte + (size_t)v * N_EMBD;
        v8f acc = V8F_ZERO(); int i = 0;
        for (; i + 8 <= N_EMBD; i += 8)
            acc = V8F_FMADD(V8F_LOAD(row+i), V8F_LOAD(x+i), acc);
        float dot = v8f_hsum(acc);
        for (; i < N_EMBD; i++) dot += row[i] * x[i];
        logits[v] = dot;
    }
}

/* === Int8 LM head (single-threaded) === */
static float lm_head_quantize_x(const float *x, int8_t *xq, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > max_abs) max_abs = a; }
    float scale = max_abs / 127.0f; if (scale < 1e-8f) scale = 1e-8f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)(x[i] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        xq[i] = (int8_t)v;
    }
}

#if defined(__AVX2__)
static inline int hsum_epi32_avx(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}
#endif

typedef struct {
    const int8_t *xq; float scale_x; float *logits;
    int v_start, v_end;
} Job;

static void *int8_worker(void *arg) {
    Job *j = (Job *)arg;
    const int8_t *xq = j->xq;
    for (int v = j->v_start; v < j->v_end; v++) {
        const int8_t *wv = g_wte_q + (size_t)v * N_EMBD;
        int dot = 0;
#if defined(__AVX2__)
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= N_EMBD; i += 32) {
            __m128i xa = _mm_loadu_si128((const __m128i *)(xq + i));
            __m128i wa = _mm_loadu_si128((const __m128i *)(wv + i));
            __m256i x16 = _mm256_cvtepi8_epi16(xa);
            __m256i w16 = _mm256_cvtepi8_epi16(wa);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x16, w16));
            __m128i xb = _mm_loadu_si128((const __m128i *)(xq + i + 16));
            __m128i wb = _mm_loadu_si128((const __m128i *)(wv + i + 16));
            __m256i x16b = _mm256_cvtepi8_epi16(xb);
            __m256i w16b = _mm256_cvtepi8_epi16(wb);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x16b, w16b));
        }
        dot = hsum_epi32_avx(acc);
        for (; i < N_EMBD; i++) dot += (int)xq[i] * (int)wv[i];
#else
        for (int i = 0; i < N_EMBD; i++) dot += (int)xq[i] * (int)wv[i];
#endif
        j->logits[v] = j->scale_x * g_wte_scale[v] * (float)dot;
    }
    return NULL;
}

static void lm_head_int8_st(float *logits, const float *x) {
    static int8_t xq[N_EMBD];
    float scale_x = lm_head_quantize_x(x, xq, N_EMBD);
    Job j = {xq, scale_x, logits, 0, VOCAB_SIZE};
    int8_worker(&j);
}

static void lm_head_int8_mt(float *logits, const float *x, int n_threads) {
    static int8_t xq[N_EMBD];
    float scale_x = lm_head_quantize_x(x, xq, N_EMBD);
    pthread_t threads[8];
    pthread_attr_t attr;
    Job jobs[8];
    int nt = n_threads > 8 ? 8 : n_threads;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 256 * 1024);
    int chunk = (VOCAB_SIZE + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        jobs[t] = (Job){xq, scale_x, logits, t*chunk, (t+1)*chunk};
        if (jobs[t].v_end > VOCAB_SIZE) jobs[t].v_end = VOCAB_SIZE;
        if (jobs[t].v_end > jobs[t].v_start)
            pthread_create(&threads[t], &attr, int8_worker, &jobs[t]);
    }
    pthread_attr_destroy(&attr);
    for (int t = 0; t < nt; t++)
        if (jobs[t].v_end > jobs[t].v_start) pthread_join(threads[t], NULL);
}

static double bench(void (*fn)(float*, const float*), const char *label) {
    struct timespec t0, t1;
    /* Warmup */
    fn(g_logits, g_x);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N_ITER; i++) fn(g_logits, g_x);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    double per_call = dt / N_ITER * 1000; /* ms */
    printf("  %-30s  %6.2f ms/call  (%d iters in %.2fs)\n", label, per_call, N_ITER, dt);
    return per_call;
}

static double bench_mt(void (*fn)(float*, const float*, int), int nt, const char *label) {
    struct timespec t0, t1;
    fn(g_logits, g_x, nt);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N_ITER; i++) fn(g_logits, g_x, nt);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    double per_call = dt / N_ITER * 1000;
    printf("  %-30s  %6.2f ms/call  (%d iters in %.2fs)\n", label, per_call, N_ITER, dt);
    return per_call;
}

int main(void) {
    printf("=== LM Head Benchmark (Qwen2.5-0.5B: %d vocab x %d embd) ===\n\n",
           VOCAB_SIZE, N_EMBD);
    printf("  Memory per call: FP32=%.0f MB, Int8=%.1f MB\n\n",
           (double)VOCAB_SIZE*N_EMBD*4/1048576,
           (double)VOCAB_SIZE*N_EMBD/1048576);

    printf("[*] initializing random weights...\n");
    init_wte_random();
    g_logits = malloc(VOCAB_SIZE * sizeof(float));

    printf("[*] benchmarking (%d iterations each)...\n\n", N_ITER);

    double fp32_ms = bench(lm_head_fp32, "FP32 LM head (1 thread)");
    double i8_1t_ms = bench(lm_head_int8_st, "Int8 LM head (1 thread)");
    double i8_2t_ms = bench_mt(lm_head_int8_mt, 2, "Int8 LM head (2 threads)");

    printf("\n=== Summary ===\n");
    printf("  FP32 1T: %.2f ms  (baseline)\n", fp32_ms);
    printf("  Int8 1T: %.2f ms  (%.2fx speedup)\n", i8_1t_ms, fp32_ms/i8_1t_ms);
    printf("  Int8 2T: %.2f ms  (%.2fx speedup vs FP32)\n", i8_2t_ms, fp32_ms/i8_2t_ms);

    free(g_wte); free(g_wte_q); free(g_wte_scale); free(g_logits);
    return 0;
}