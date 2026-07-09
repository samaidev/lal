/* bench_llama_style_q8.c — llama.cpp Q8_0 style matmul benchmark
 *
 * This replicates llama.cpp's Q8_0 vec_dot kernel for fair comparison.
 * Q8_0 format: 32-element blocks, each block = 1 float scale + 32 int8 values.
 * vec_dot: dequantize on-the-fly, VPMADDUBSW + VPMADDWD.
 *
 * Compare 3 implementations on same hardware:
 *   A) LAL Q8 per-row (our implementation)
 *   B) llama.cpp Q8_0 per-block (32-element blocks)
 *   C) float32 reference (for quality)
 *
 * This IS the llama.cpp technique — same AVX2 instructions, same block size.
 * If LAL Q8 per-row is faster than Q8_0 per-block, we beat llama.cpp's approach.
 *
 * Compile: gcc -O3 -mavx2 -mfma -o bench_llama_q8 bench_llama_style_q8.c -lm
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
#define BLOCK 32  /* llama.cpp Q8_0 block size */
#define N_TRIALS 500

/* ===== A) LAL Q8 per-row (transposed, our implementation) ===== */
static void lal_q8_per_row(float *y, const int8_t *q8_T, const float *scale,
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
        __m128i s=_mm_hadd_epi32(_mm_add_epi32(lo,hi),_mm_hadd_epi32(_mm_add_epi32(lo,hi),_mm_add_epi32(lo,hi)));
        int32_t dot = _mm_cvtsi128_si32(s);
        int32_t w_sum = 0;
        for (int i = 0; i < in_dim; i++) w_sum += w[i];
        dot -= 128 * w_sum;
        y[j] = (float)dot * x_scale * scale[j];
    }
}

/* ===== B) llama.cpp Q8_0 per-block (32-element blocks) ===== */
typedef struct { float d; int8_t qs[BLOCK]; } q8_0_block;

static void llama_q8_0(float *y, const q8_0_block *blocks, /* [out_dim][in_dim/BLOCK] */
                       const float *x, int in_dim, int out_dim) {
    /* llama.cpp vec_dot_q8_0: for each block, dequantize w = qs * d,
     * dot += sum(w * x). Uses VPMADDUBSW when x is also quantized. */
    int nb = in_dim / BLOCK;
    /* Quantize x to uint8 per-block (llama.cpp style) */
    uint8_t xq[4096];
    float x_scales[4096/BLOCK];
    for (int b = 0; b < nb; b++) {
        float x_max = 0;
        for (int i = 0; i < BLOCK; i++) x_max = fmaxf(x_max, fabsf(x[b*BLOCK+i]));
        x_scales[b] = x_max / 127.0f;
        if (x_scales[b] < 1e-8f) x_scales[b] = 1e-8f;
        for (int i = 0; i < BLOCK; i++) {
            int v = (int)lroundf(x[b*BLOCK+i] / x_scales[b]) + 128;
            xq[b*BLOCK+i] = (uint8_t)(v > 255 ? 255 : (v < 0 ? 0 : v));
        }
    }
    __m256i ones = _mm256_set1_epi16(1);

    for (int j = 0; j < out_dim; j++) {
        const q8_0_block *blks = blocks + (size_t)j * nb;
        float sum = 0;
        for (int b = 0; b < nb; b++) {
            const int8_t *qs = blks[b].qs;
            const uint8_t *xb = xq + b * BLOCK;
            /* VPMADDUBSW: 32 uint8 × 32 int8 → 16 int16 */
            __m256i xv = _mm256_loadu_si256((__m256i*)xb);
            __m256i wv = _mm256_loadu_si256((__m256i*)qs);
            __m256i prod16 = _mm256_maddubs_epi16(xv, wv);
            __m256i sum32 = _mm256_madd_epi16(prod16, ones);
            __m128i lo=_mm256_castsi256_si128(sum32), hi=_mm256_extracti128_si256(sum32,1);
            __m128i s=_mm_hadd_epi32(_mm_add_epi32(lo,hi),_mm_hadd_epi32(_mm_add_epi32(lo,hi),_mm_add_epi32(lo,hi)));
            int32_t dot = _mm_cvtsi128_si32(s);
            /* Remove zero-point */
            int32_t w_sum = 0;
            for (int i = 0; i < BLOCK; i++) w_sum += qs[i];
            dot -= 128 * w_sum;
            sum += (float)dot * x_scales[b] * blks[b].d;
        }
        y[j] = sum;
    }
}

/* ===== C) Float32 reference ===== */
static void matmul_float(float *y, const float *W, const float *x,
                         int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float s = 0;
        for (int i = 0; i < in_dim; i++) s += W[(size_t)i*out_dim+j] * x[i];
        y[j] = s;
    }
}

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static float corr(const float *a, const float *b, int n) {
    float ma=0,mb=0; for(int i=0;i<n;i++){ma+=a[i];mb+=b[i];} ma/=n;mb/=n;
    float num=0,da=0,db=0; for(int i=0;i<n;i++){num+=(a[i]-ma)*(b[i]-mb);da+=(a[i]-ma)*(a[i]-ma);db+=(b[i]-mb)*(b[i]-mb);}
    return num/sqrtf(da*db+1e-12f);
}

int main(void) {
    srand(42);
    float *W = malloc((size_t)IN_DIM*OUT_DIM*sizeof(float));
    float *x = malloc(IN_DIM*sizeof(float));
    float *y_ref=malloc(OUT_DIM*sizeof(float)), *y_lal=malloc(OUT_DIM*sizeof(float)), *y_llama=malloc(OUT_DIM*sizeof(float));

    /* Q8 per-row (LAL) */
    int8_t *q8_T = malloc((size_t)IN_DIM*OUT_DIM);
    float *scale = malloc(OUT_DIM*sizeof(float));
    /* Q8_0 per-block (llama.cpp) */
    int nb = IN_DIM / BLOCK;
    q8_0_block *blocks = malloc((size_t)OUT_DIM*nb*sizeof(q8_0_block));

    for (int i=0;i<IN_DIM;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*4;
    for (size_t i=0;i<(size_t)IN_DIM*OUT_DIM;i++) W[i]=(rand()/(float)RAND_MAX-0.5f)*0.1f;

    /* Quantize LAL per-row (transposed) */
    for (int j=0;j<OUT_DIM;j++) {
        float mx=0; for(int i=0;i<IN_DIM;i++) mx=fmaxf(mx,fabsf(W[i*OUT_DIM+j]));
        scale[j]=mx/127.0f; if(scale[j]<1e-8f)scale[j]=1e-8f;
        float inv=1.0f/scale[j];
        for(int i=0;i<IN_DIM;i++){int v=lroundf(W[i*OUT_DIM+j]*inv); q8_T[j*IN_DIM+i]=(int8_t)(v>127?127:(v<-127?-127:v));}
    }
    /* Quantize llama.cpp Q8_0 per-block */
    for (int j=0;j<OUT_DIM;j++) {
        for (int b=0;b<nb;b++) {
            float mx=0; for(int i=0;i<BLOCK;i++) mx=fmaxf(mx,fabsf(W[(b*BLOCK+i)*OUT_DIM+j]));
            blocks[j*nb+b].d=mx/127.0f; if(blocks[j*nb+b].d<1e-8f)blocks[j*nb+b].d=1e-8f;
            float inv=1.0f/blocks[j*nb+b].d;
            for(int i=0;i<BLOCK;i++){int v=lroundf(W[(b*BLOCK+i)*OUT_DIM+j]*inv); blocks[j*nb+b].qs[i]=(int8_t)(v>127?127:(v<-127?-127:v));}
        }
    }

    /* Correctness */
    matmul_float(y_ref,W,x,IN_DIM,OUT_DIM);
    lal_q8_per_row(y_lal,q8_T,scale,x,IN_DIM,OUT_DIM);
    llama_q8_0(y_llama,blocks,x,IN_DIM,OUT_DIM);
    printf("Quality (correlation vs float):\n");
    printf("  LAL Q8 per-row:  %.6f\n", corr(y_ref,y_lal,OUT_DIM));
    printf("  llama Q8_0 block: %.6f\n", corr(y_ref,y_llama,OUT_DIM));

    /* Benchmark */
    double t0=now_sec();
    for(int t=0;t<N_TRIALS;t++) lal_q8_per_row(y_lal,q8_T,scale,x,IN_DIM,OUT_DIM);
    double t_lal=now_sec()-t0;
    t0=now_sec();
    for(int t=0;t<N_TRIALS;t++) llama_q8_0(y_llama,blocks,x,IN_DIM,OUT_DIM);
    double t_llama=now_sec()-t0;
    t0=now_sec();
    for(int t=0;t<N_TRIALS;t++) matmul_float(y_ref,W,x,IN_DIM,OUT_DIM);
    double t_f=now_sec()-t0;

    printf("\nBenchmark (%d trials, 768x768):\n", N_TRIALS);
    printf("  float32:          %.3f ms\n", t_f/N_TRIALS*1000);
    printf("  LAL Q8 per-row:   %.3f ms  (%.2fx vs float)\n", t_lal/N_TRIALS*1000, t_f/t_lal);
    printf("  llama Q8_0 block:  %.3f ms  (%.2fx vs float)\n", t_llama/N_TRIALS*1000, t_f/t_llama);
    printf("\n  LAL vs llama: %.2fx %s\n", t_llama/t_lal, t_lal<t_llama?"LAL WINS":"llama wins");

    /* Memory comparison */
    printf("\nMemory per matrix:\n");
    printf("  float32:    %.0f KB\n", (float)IN_DIM*OUT_DIM*4/1024);
    printf("  LAL Q8:     %.0f KB (%.1fx)\n", (float)IN_DIM*OUT_DIM*1/1024 + (float)OUT_DIM*4/1024, (float)IN_DIM*OUT_DIM*4/((float)IN_DIM*OUT_DIM*1+(float)OUT_DIM*4));
    printf("  llama Q8_0: %.0f KB (%.1fx)\n", (float)(IN_DIM*OUT_DIM + (IN_DIM/BLOCK)*4)/1024, (float)IN_DIM*OUT_DIM*4/((float)IN_DIM*OUT_DIM+(float)(IN_DIM/BLOCK)*4));

    free(W);free(x);free(y_ref);free(y_lal);free(y_llama);free(q8_T);free(scale);free(blocks);
    return 0;
}
