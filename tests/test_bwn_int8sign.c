/* test_bwn_int8sign.c — BWN with int8 sign storage (4x less memory than float)
 *
 * Instead of w_sign_float (4 bytes per sign), use int8 (1 byte per sign).
 * sign = +1 → 1, sign = -1 → -1 (as int8)
 * Then: x[i] * sign_int8[i] via _mm256_cvtepi8_epi32 + FMA
 *
 * Memory: 27MB (vs 108MB float, vs 3.5MB wbits)
 * This is a middle ground: 4x less memory than float, still fast SIMD.
 *
 * Also test: direct wbits with _mm256_srai_epi32 trick for sign expansion
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#define IN_DIM 768
#define OUT_DIM 768
#define N_TRIALS 500

/* Baseline: w_sign_float (108MB) */
static void bwn_float(float *y, const float *w_sign, const float *x,
                      const float *alpha, const float *bias, int id, int od) {
    for (int j = 0; j < od; j++) {
        const float *w = w_sign + (size_t)j * id;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < id; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w+i), _mm256_loadu_ps(x+i), acc);
        __m128 lo=_mm256_castps256_ps128(acc), hi=_mm256_extractf128_ps(acc,1);
        __m128 s=_mm_hadd_ps(_mm_add_ps(lo,hi),_mm_hadd_ps(_mm_add_ps(lo,hi),_mm_add_ps(lo,hi)));
        y[j] = _mm_cvtss_f32(s) * alpha[j] + bias[j];
    }
}

/* int8 sign (27MB): expand int8 to float on the fly, then FMA
 * _mm256_cvtepi8_epi32: 8 x int8 → 8 x int32
 * _mm256_cvtepi32_ps: int32 → float
 * Then FMA as usual */
static void bwn_int8sign(float *y, const int8_t *w_sign_i8, const float *x,
                         const float *alpha, const float *bias, int id, int od) {
    for (int j = 0; j < od; j++) {
        const int8_t *w = w_sign_i8 + (size_t)j * id;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < id; i += 8) {
            /* Load 8 int8, expand to 8 int32, convert to float */
            __m128i w_i8 = _mm_loadl_epi64((const __m128i*)(w+i));  /* 8 int8 */
            __m256i w_i32 = _mm256_cvtepi8_epi32(w_i8);             /* 8 int32 */
            __m256 wf = _mm256_cvtepi32_ps(w_i32);                  /* 8 float (+/-1) */
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x+i), acc);
        }
        __m128 lo=_mm256_castps256_ps128(acc), hi=_mm256_extractf128_ps(acc,1);
        __m128 s=_mm_hadd_ps(_mm_add_ps(lo,hi),_mm_hadd_ps(_mm_add_ps(lo,hi),_mm_add_ps(lo,hi)));
        y[j] = _mm_cvtss_f32(s) * alpha[j] + bias[j];
    }
}

/* wbits with broadcast trick: process 32 outputs at once
 * For each group of 32 outputs, load x once, apply 32 different sign patterns.
 * This amortizes x load across 32 outputs (like OpenBLAS register blocking).
 * Sign from wbits: 32 outputs × 64 bits each, but we process 8 floats at a time.
 *
 * Key: for 8 floats x[0..7], each output j has 8 sign bits.
 * Expand to 8 x {0,1} → 8 x {0xFFFFFFFF, 0} mask.
 * Then: result = blendv(x, -x, mask) = x where sign=+1, -x where sign=-1.
 *
 * Efficient mask from 8 bits using _mm256_set_epi32 + shuffle:
 * Actually fastest: use lookup table 256 entries → 8 x int32 mask.
 * 8KB LUT (256 × 32 bytes), fits L1. */
static float lut_masks[256][8] __attribute__((aligned(32)));
static void init_lut(void) {
    for (int b = 0; b < 256; b++)
        for (int k = 0; k < 8; k++)
            lut_masks[b][k] = ((b >> k) & 1) ? 0.0f : -0.0f;  /* +1: 0.0 (no flip), -1: -0.0 (sign bit set) */
}

/* wbits + LUT sign flip: 3.5MB memory, uses XOR with -0.0 to flip signs */
static void bwn_wbits_lut(float *y, const uint64_t *wbits, int n_words,
                          const float *x, const float *alpha, const float *bias,
                          int id, int od) {
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    for (int j = 0; j < od; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        __m256 acc = _mm256_setzero_ps();
        for (int wi = 0; wi < n_words; wi++) {
            uint64_t w = wb[wi];
            int base = wi * 64;
            for (int bi = 0; bi < 8 && base+bi*8 < id; bi++) {
                int idx = base + bi * 8;
                if (idx + 8 > id) break;
                uint8_t byte = (w >> (bi*8)) & 0xFF;
                /* LUT gives 8 floats that are 0.0 or -0.0
                 * XOR x with -0.0 flips sign where mask is -0.0 */
                __m256 flip = _mm256_load_ps(lut_masks[byte]);
                __m256 xv = _mm256_loadu_ps(x + idx);
                __m256 x_signed = _mm256_xor_ps(xv, flip);
                acc = _mm256_add_ps(acc, x_signed);
            }
        }
        __m128 lo=_mm256_castps256_ps128(acc), hi=_mm256_extractf128_ps(acc,1);
        __m128 s=_mm_hadd_ps(_mm_add_ps(lo,hi),_mm_hadd_ps(_mm_add_ps(lo,hi),_mm_add_ps(lo,hi)));
        y[j] = _mm_cvtss_f32(s) * alpha[j] + bias[j];
    }
}

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void) {
    srand(42);
    init_lut();
    uint64_t *wbits = calloc((size_t)OUT_DIM*((IN_DIM+63)/64), sizeof(uint64_t));
    float *w_sf = malloc((size_t)OUT_DIM*IN_DIM*sizeof(float));
    int8_t *w_i8 = malloc((size_t)OUT_DIM*IN_DIM);
    float *x=malloc(IN_DIM*sizeof(float)), *alpha=malloc(OUT_DIM*sizeof(float)), *bias=malloc(OUT_DIM*sizeof(float));
    float *y1=malloc(OUT_DIM*sizeof(float)), *y2=malloc(OUT_DIM*sizeof(float)), *y3=malloc(OUT_DIM*sizeof(float));
    int n_words=(IN_DIM+63)/64;

    for (int i=0;i<IN_DIM;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*4;
    for (int j=0;j<OUT_DIM;j++) { alpha[j]=0.1f; bias[j]=0.01f*j; }
    for (size_t i=0;i<(size_t)OUT_DIM*((IN_DIM+63)/64);i++) wbits[i]=((uint64_t)rand()<<32)|rand();
    for (int j=0;j<OUT_DIM;j++) for (int i=0;i<IN_DIM;i++) {
        int bit = (wbits[(size_t)j*n_words+i/64]>>(i%64))&1;
        w_sf[(size_t)j*IN_DIM+i] = bit?1.0f:-1.0f;
        w_i8[(size_t)j*IN_DIM+i] = bit?1:-1;
    }

    bwn_float(y1,w_sf,x,alpha,bias,IN_DIM,OUT_DIM);
    bwn_int8sign(y2,w_i8,x,alpha,bias,IN_DIM,OUT_DIM);
    bwn_wbits_lut(y3,wbits,n_words,x,alpha,bias,IN_DIM,OUT_DIM);
    float d2=0,d3=0;
    for (int j=0;j<OUT_DIM;j++) { d2=fmaxf(d2,fabsf(y1[j]-y2[j])); d3=fmaxf(d3,fabsf(y1[j]-y3[j])); }
    printf("Correctness: int8sign=%.1e  wbits_lut=%.1e\n", d2, d3);

    double t0=now_sec();
    for(int t=0;t<N_TRIALS;t++) bwn_float(y1,w_sf,x,alpha,bias,IN_DIM,OUT_DIM);
    double tf=now_sec()-t0;
    t0=now_sec();
    for(int t=0;t<N_TRIALS;t++) bwn_int8sign(y2,w_i8,x,alpha,bias,IN_DIM,OUT_DIM);
    double ti=now_sec()-t0;
    t0=now_sec();
    for(int t=0;t<N_TRIALS;t++) bwn_wbits_lut(y3,wbits,n_words,x,alpha,bias,IN_DIM,OUT_DIM);
    double tw=now_sec()-t0;

    printf("\nBenchmark (%d trials):\n", N_TRIALS);
    printf("  sign_float (108MB):  %.3f ms\n", tf/N_TRIALS*1000);
    printf("  int8_sign  (27MB):   %.3f ms  (%.2fx)\n", ti/N_TRIALS*1000, tf/ti);
    printf("  wbits_lut  (3.5MB):  %.3f ms  (%.2fx)\n", tw/N_TRIALS*1000, tf/tw);
    printf("\nMemory per matrix: float 2304KB, int8 576KB, wbits 72KB\n");

    free(wbits);free(w_sf);free(w_i8);free(x);free(alpha);free(bias);free(y1);free(y2);free(y3);
    return 0;
}
