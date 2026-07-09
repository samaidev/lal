/* test_q8_mistral_style.c — Q8 SIMD with mistral.rs-inspired optimizations
 *
 * mistral.rs key techniques (from v0.9.0 report):
 * 1. Output accumulators held in registers (not spilled to memory)
 * 2. Stream weights at memory bandwidth (no per-position overhead)
 * 3. x loaded once, reused across outputs (register blocking)
 *
 * Applied to Q8 SIMD:
 * - Process 8 outputs simultaneously, 8 __m256i accumulators in registers
 * - Pre-compute w_sums at quantization time (eliminate per-call overhead)
 * - x quantized once, loaded once per 8-output block
 *
 * Compare:
 *   A) Current Q8 SIMD: 1 output at a time, w_sum computed per-call
 *   B) mistral.rs style: 8 outputs at a time, pre-computed w_sums
 *
 * Compile: gcc -O3 -mavx2 -mfma -o test_q8_mistral test_q8_mistral_style.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#define IN_DIM 768
#define OUT_DIM 768
#define N_TRIALS 500

/* A) Current: single output, per-call w_sum */
static void q8_simd_current(float *y, const int8_t *q8_T, const float *scale,
                            const float *x, int in_dim, int out_dim) {
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    uint8_t xq[4096];
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] / x_scale) + 128;
        xq[i] = (uint8_t)(v > 255 ? 255 : (v < 0 ? 0 : v));
    }
    __m256i ones = _mm256_set1_epi16(1);
    for (int j = 0; j < out_dim; j++) {
        const int8_t *w = q8_T + (size_t)j * in_dim;
        __m256i acc32 = _mm256_setzero_si256();
        for (int i = 0; i < in_dim; i += 32) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
            __m256i wv = _mm256_loadu_si256((__m256i*)(w + i));
            __m256i prod16 = _mm256_maddubs_epi16(xv, wv);
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(prod16, ones));
        }
        __m128i lo=_mm256_castsi256_si128(acc32), hi=_mm256_extracti128_si256(acc32,1);
        __m128i s=_mm_hadd_epi32(lo,hi); s=_mm_hadd_epi32(s,s); s=_mm_hadd_epi32(s,s);
        int32_t dot = _mm_cvtsi128_si32(s);
        /* Per-call w_sum — THIS IS THE OVERHEAD mistral.rs eliminates */
        int32_t w_sum = 0;
        for (int i = 0; i < in_dim; i++) w_sum += w[i];
        dot -= 128 * w_sum;
        y[j] = (float)dot * x_scale * scale[j];
    }
}

/* B) mistral.rs style: 8 outputs parallel, pre-computed w_sums, register accumulators */
static void q8_simd_mistral(float *y, const int8_t *q8_T, const float *scale,
                            const int32_t *w_sums, /* pre-computed! */
                            const float *x, int in_dim, int out_dim) {
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    uint8_t xq[4096];
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] / x_scale) + 128;
        xq[i] = (uint8_t)(v > 255 ? 255 : (v < 0 ? 0 : v));
    }
    __m256i ones = _mm256_set1_epi16(1);

    /* Process 8 outputs at a time: 8 accumulators stay in registers */
    int j = 0;
    for (; j + 8 <= out_dim; j += 8) {
        const int8_t *w0 = q8_T + (size_t)(j+0) * in_dim;
        const int8_t *w1 = q8_T + (size_t)(j+1) * in_dim;
        const int8_t *w2 = q8_T + (size_t)(j+2) * in_dim;
        const int8_t *w3 = q8_T + (size_t)(j+3) * in_dim;
        const int8_t *w4 = q8_T + (size_t)(j+4) * in_dim;
        const int8_t *w5 = q8_T + (size_t)(j+5) * in_dim;
        const int8_t *w6 = q8_T + (size_t)(j+6) * in_dim;
        const int8_t *w7 = q8_T + (size_t)(j+7) * in_dim;
        /* 8 accumulators — ALL in YMM registers, never spilled */
        __m256i a0=_mm256_setzero_si256(), a1=_mm256_setzero_si256();
        __m256i a2=_mm256_setzero_si256(), a3=_mm256_setzero_si256();
        __m256i a4=_mm256_setzero_si256(), a5=_mm256_setzero_si256();
        __m256i a6=_mm256_setzero_si256(), a7=_mm256_setzero_si256();

        for (int i = 0; i < in_dim; i += 32) {
            /* Load x ONCE — shared across all 8 outputs (register blocking) */
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
            /* 8 weight loads, 8 maddubs, 8 madd, 8 adds — all pipelined */
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((__m256i*)(w0+i))), ones));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((__m256i*)(w1+i))), ones));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((__m256i*)(w2+i))), ones));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((__m256i*)(w3+i))), ones));
            a4 = _mm256_add_epi32(a4, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((__m256i*)(w4+i))), ones));
            a5 = _mm256_add_epi32(a5, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((__m256i*)(w5+i))), ones));
            a6 = _mm256_add_epi32(a6, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((__m256i*)(w6+i))), ones));
            a7 = _mm256_add_epi32(a7, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((__m256i*)(w7+i))), ones));
        }
        /* Horizontal sum + zero-point removal (pre-computed w_sums — NO per-call loop!) */
        #define HSUM32(v) ({ __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1); \
            __m128i s=_mm_hadd_epi32(lo,hi); s=_mm_hadd_epi32(s,s); s=_mm_hadd_epi32(s,s); _mm_cvtsi128_si32(s); })
        y[j+0] = (float)(HSUM32(a0) - 128*w_sums[j+0]) * x_scale * scale[j+0];
        y[j+1] = (float)(HSUM32(a1) - 128*w_sums[j+1]) * x_scale * scale[j+1];
        y[j+2] = (float)(HSUM32(a2) - 128*w_sums[j+2]) * x_scale * scale[j+2];
        y[j+3] = (float)(HSUM32(a3) - 128*w_sums[j+3]) * x_scale * scale[j+3];
        y[j+4] = (float)(HSUM32(a4) - 128*w_sums[j+4]) * x_scale * scale[j+4];
        y[j+5] = (float)(HSUM32(a5) - 128*w_sums[j+5]) * x_scale * scale[j+5];
        y[j+6] = (float)(HSUM32(a6) - 128*w_sums[j+6]) * x_scale * scale[j+6];
        y[j+7] = (float)(HSUM32(a7) - 128*w_sums[j+7]) * x_scale * scale[j+7];
    }
    /* Tail: remaining outputs (single) */
    for (; j < out_dim; j++) {
        const int8_t *w = q8_T + (size_t)j * in_dim;
        __m256i acc32 = _mm256_setzero_si256();
        for (int i = 0; i < in_dim; i += 32) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i));
            __m256i wv = _mm256_loadu_si256((__m256i*)(w + i));
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, wv), ones));
        }
        __m128i lo=_mm256_castsi256_si128(acc32), hi=_mm256_extracti128_si256(acc32,1);
        __m128i s=_mm_hadd_epi32(lo,hi); s=_mm_hadd_epi32(s,s); s=_mm_hadd_epi32(s,s);
        y[j] = (float)(_mm_cvtsi128_si32(s) - 128*w_sums[j]) * x_scale * scale[j];
    }
}

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void) {
    srand(42);
    int8_t *q8_T = malloc((size_t)IN_DIM*OUT_DIM);
    float *scale = malloc(OUT_DIM*sizeof(float));
    int32_t *w_sums = malloc(OUT_DIM*sizeof(int32_t));
    float *x = malloc(IN_DIM*sizeof(float));
    float *y1 = malloc(OUT_DIM*sizeof(float));
    float *y2 = malloc(OUT_DIM*sizeof(float));

    for (int i=0;i<IN_DIM;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*4;
    for (int j=0;j<OUT_DIM;j++) { scale[j]=0.001f+0.001f*j; }
    for (size_t i=0;i<(size_t)IN_DIM*OUT_DIM;i++) q8_T[i]=(rand()%255)-127;
    /* Pre-compute w_sums (mistral.rs: do this at quantization time, not per-call) */
    for (int j=0;j<OUT_DIM;j++) {
        int32_t s=0;
        for (int i=0;i<IN_DIM;i++) s+=q8_T[(size_t)j*IN_DIM+i];
        w_sums[j]=s;
    }

    /* Correctness */
    q8_simd_current(y1, q8_T, scale, x, IN_DIM, OUT_DIM);
    q8_simd_mistral(y2, q8_T, scale, w_sums, x, IN_DIM, OUT_DIM);
    float max_d=0;
    for (int j=0;j<OUT_DIM;j++) max_d=fmaxf(max_d, fabsf(y1[j]-y2[j]));
    printf("Correctness: max diff = %.6f (should be ~0)\n", max_d);

    /* Benchmark */
    double t0=now_sec();
    for (int t=0;t<N_TRIALS;t++) q8_simd_current(y1,q8_T,scale,x,IN_DIM,OUT_DIM);
    double t1=now_sec()-t0;
    t0=now_sec();
    for (int t=0;t<N_TRIALS;t++) q8_simd_mistral(y2,q8_T,scale,w_sums,x,IN_DIM,OUT_DIM);
    double t2=now_sec()-t0;

    printf("\nBenchmark (%d trials, 768x768):\n", N_TRIALS);
    printf("  current (1-out, per-call w_sum): %.3f ms\n", t1/N_TRIALS*1000);
    printf("  mistral style (8-out, pre w_sum): %.3f ms  (%.2fx)\n", t2/N_TRIALS*1000, t1/t2);

    /* Estimated end-to-end */
    double ms = t2/N_TRIALS*1000;
    double tok = 1000.0 / (ms * 48 * 1.3);  /* 48 matmuls + 30% overhead */
    printf("\n  Estimated: %.1f ms/matmul → ~%.0f tok/s end-to-end\n", ms, tok);

    free(q8_T);free(scale);free(w_sums);free(x);free(y1);free(y2);
    return 0;
}
