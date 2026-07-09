/* test_bwn_llamacpp_v2.c — Clean BWN: float XOR vs int16 madd (llama.cpp style) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#define IN_DIM 768
#define OUT_DIM 768
#define N_TRIALS 500

static float lut_masks[256][8] __attribute__((aligned(32)));

/* A) Current: float XOR sign-flip */
static void bwn_float_xor(float *y, const uint64_t *wbits, int n_words,
                          const float *x, const float *alpha, const float *bias,
                          int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        __m256 acc = _mm256_setzero_ps();
        for (int wi = 0; wi < n_words; wi++) {
            uint64_t w = wb[wi];
            for (int bi = 0; bi < 8; bi++) {
                int idx = wi*64 + bi*8;
                if (idx + 8 > in_dim) break;
                uint8_t byte = (w >> (bi*8)) & 0xFF;
                __m256 flip = _mm256_load_ps(lut_masks[byte]);
                __m256 xv = _mm256_loadu_ps(x + idx);
                acc = _mm256_add_ps(acc, _mm256_xor_ps(xv, flip));
            }
        }
        __m128 lo=_mm256_castps256_ps128(acc), hi=_mm256_extractf128_ps(acc,1);
        __m128 s=_mm_hadd_ps(lo,hi); s=_mm_hadd_ps(s,s); s=_mm_hadd_ps(s,s);
        y[j] = _mm_cvtss_f32(s) * alpha[j] + bias[j];
    }
}

/* B) llama.cpp-style: int16 madd (VPMADDWD)
 * x quantized to int16, sign(W) as int16, _mm256_madd_epi16 does int16×int16→int32 */
static void bwn_int16_madd(float *y, const uint64_t *wbits, int n_words,
                           const float *x, const float *alpha, const float *bias,
                           int in_dim, int out_dim) {
    int16_t xq[4096];
    float x_scale = 0;
    for (int i = 0; i < in_dim; i++) x_scale = fmaxf(x_scale, fabsf(x[i]));
    x_scale /= 32767.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] / x_scale);
        xq[i] = (int16_t)(v > 32767 ? 32767 : (v < -32767 ? -32767 : v));
    }
    /* Pre-build sign LUT as int16: 256 × 16 int16 (each byte → 8 signs, padded to 16) */
    static int16_t sign_lut[256][16] __attribute__((aligned(32)));
    static int init = 0;
    if (!init) {
        for (int b = 0; b < 256; b++)
            for (int k = 0; k < 8; k++) {
                sign_lut[b][k] = (b >> k) & 1 ? 1 : -1;
                sign_lut[b][k+8] = 0;  /* padding */
            }
        init = 1;
    }

    for (int j = 0; j < out_dim; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        __m256i acc = _mm256_setzero_si256();
        for (int wi = 0; wi < n_words; wi++) {
            uint64_t w = wb[wi];
            for (int bi = 0; bi < 4; bi++) {  /* 4 chunks of 16 */
                int idx = wi*64 + bi*16;
                if (idx + 16 > in_dim) break;
                /* Build 16 int16 sign from 16 bits */
                uint16_t bits16 = (w >> (bi*16)) & 0xFFFF;
                /* Use 2 bytes from LUT + combine */
                uint8_t b0 = bits16 & 0xFF;
                uint8_t b1 = (bits16 >> 8) & 0xFF;
                /* Load 8 signs from b0 into low, 8 from b1 into high */
                __m128i s0 = _mm_load_si128((__m128i*)sign_lut[b0]);
                __m128i s1 = _mm_load_si128((__m128i*)sign_lut[b1]);
                __m256i sw = _mm256_inserti128_si256(_mm256_castsi128_si256(s0), s1, 1);
                __m256i xq_vec = _mm256_loadu_si256((__m256i*)(xq + idx));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(sw, xq_vec));
            }
        }
        int32_t tmp[8] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)tmp, acc);
        int64_t sum = (int64_t)tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        y[j] = (float)sum * x_scale * alpha[j] + bias[j];
    }
}

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void) {
    srand(42);
    for (int b = 0; b < 256; b++)
        for (int k = 0; k < 8; k++)
            lut_masks[b][k] = ((b>>k)&1) ? 0.0f : -0.0f;

    uint64_t *wbits = calloc((size_t)OUT_DIM*((IN_DIM+63)/64), sizeof(uint64_t));
    float *x=malloc(IN_DIM*sizeof(float)), *alpha=malloc(OUT_DIM*sizeof(float)), *bias=malloc(OUT_DIM*sizeof(float));
    float *y1=malloc(OUT_DIM*sizeof(float)), *y2=malloc(OUT_DIM*sizeof(float));
    int n_words=(IN_DIM+63)/64;

    for (int i=0;i<IN_DIM;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*4;
    for (int j=0;j<OUT_DIM;j++) { alpha[j]=0.1f; bias[j]=0.01f*j; }
    for (size_t i=0;i<(size_t)OUT_DIM*((IN_DIM+63)/64);i++) wbits[i]=((uint64_t)rand()<<32)|rand();

    bwn_float_xor(y1,wbits,n_words,x,alpha,bias,IN_DIM,OUT_DIM);
    bwn_int16_madd(y2,wbits,n_words,x,alpha,bias,IN_DIM,OUT_DIM);
    float max_d=0, corr=0;
    float m1=0,m2=0;
    for (int j=0;j<OUT_DIM;j++) { max_d=fmaxf(max_d,fabsf(y1[j]-y2[j])); m1+=y1[j]; m2+=y2[j]; }
    m1/=OUT_DIM; m2/=OUT_DIM;
    float num=0,d1=0,d2=0;
    for (int j=0;j<OUT_DIM;j++) { num+=(y1[j]-m1)*(y2[j]-m2); d1+=(y1[j]-m1)*(y1[j]-m1); d2+=(y2[j]-m2)*(y2[j]-m2); }
    corr=num/sqrtf(d1*d2+1e-12f);
    printf("Correctness: max diff=%.4f, correlation=%.6f\n", max_d, corr);

    double t0=now_sec();
    for(int t=0;t<N_TRIALS;t++) bwn_float_xor(y1,wbits,n_words,x,alpha,bias,IN_DIM,OUT_DIM);
    double t1=now_sec()-t0;
    t0=now_sec();
    for(int t=0;t<N_TRIALS;t++) bwn_int16_madd(y2,wbits,n_words,x,alpha,bias,IN_DIM,OUT_DIM);
    double t2=now_sec()-t0;

    printf("\nBenchmark (%d trials, 768x768):\n", N_TRIALS);
    printf("  float_xor (current):    %.3f ms\n", t1/N_TRIALS*1000);
    printf("  int16_madd (llama.cpp): %.3f ms  (%.2fx)\n", t2/N_TRIALS*1000, t1/t2);

    free(wbits);free(x);free(alpha);free(bias);free(y1);free(y2);
    return 0;
}
