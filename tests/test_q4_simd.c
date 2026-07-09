/* test_q4_simd.c — Q4 per-row quantization with AVX2 SIMD
 *
 * Q4: 4-bit per-row quantization. Each weight stored as int4 [-7, 7].
 * Two weights packed per byte → 2x less memory than Q8.
 *
 * Quality: correlation 0.977 vs float (usable, not as good as Q8 0.99994)
 * Memory: 288KB/matrix (8x less than float, 2x less than Q8)
 *
 * SIMD: VPMADDUBSW can't do 4-bit directly. Two approaches:
 *   A) Unpack int4 → int8 on the fly, then VPMADDUBSW (like Q8)
 *   B) Unpack int4 → int16, then VPMADDWD
 *
 * Approach A: unpack 2 int4 from 1 byte, expand to 2 int8.
 * Use LUT: 256 entries × 16 int8 (each byte → 2 expanded int4 values, padded to 16)
 *
 * Compile: gcc -O3 -mavx2 -mfma -o test_q4_simd test_q4_simd.c -lm
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

/* Q4 LUT: 256 entries (1 byte = 2 int4) → 16 int8 (for VPMADDUBSW)
 * byte b: low nibble = first int4, high nibble = second int4
 * int4 value = nibble - 8 (range -8..7, but we use -7..7) */
static int8_t q4_lut[256][16] __attribute__((aligned(32)));

static void init_q4_lut(void) {
    for (int b = 0; b < 256; b++) {
        int lo = (b & 0xF) - 8;  /* low nibble: -8..7 */
        int hi = (b >> 4) - 8;   /* high nibble: -8..7 */
        /* Clamp to -7..7 (avoid -8 overflow in int8 multiply) */
        if (lo < -7) lo = -7; if (lo > 7) lo = 7;
        if (hi < -7) hi = -7; if (hi > 7) hi = 7;
        q4_lut[b][0] = (int8_t)lo;
        q4_lut[b][1] = (int8_t)hi;
        /* Rest zero-padded for 16-element vector */
        for (int k = 2; k < 16; k++) q4_lut[b][k] = 0;
    }
}

/* Quantize W [in, out] to q4_T [out, in] (transposed, 2 weights per byte) */
static void quantize_q4(const float *W, uint8_t *q4_T, float *scale,
                        int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float max_abs = 0;
        for (int i = 0; i < in_dim; i++)
            max_abs = fmaxf(max_abs, fabsf(W[(size_t)i * out_dim + j]));
        scale[j] = max_abs / 7.0f;  /* int4 range: -7..7 */
        if (scale[j] < 1e-8f) scale[j] = 1e-8f;
        float inv = 1.0f / scale[j];
        for (int i = 0; i < in_dim; i += 2) {
            int v0 = (int)lroundf(W[(size_t)i * out_dim + j] * inv);
            int v1 = (int)lroundf(W[(size_t)(i+1) * out_dim + j] * inv);
            if (v0 > 7) v0 = 7; if (v0 < -7) v0 = -7;
            if (v1 > 7) v1 = 7; if (v1 < -7) v1 = -7;
            /* Pack: low nibble = v0+8, high nibble = v1+8 */
            q4_T[(size_t)j * (in_dim/2) + i/2] = (uint8_t)((v0 + 8) | ((v1 + 8) << 4));
        }
    }
}

/* Q4 matmul SIMD: unpack int4 → int8 via LUT, then VPMADDUBSW like Q8 */
static void matmul_q4_simd(float *y, const uint8_t *q4_T, const float *scale,
                           const int32_t *w_sums, const float *x,
                           const float *b, int in_dim, int out_dim) {
    /* Quantize x to uint8 (same as Q8) */
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
    int half_in = in_dim / 2;

    for (int j = 0; j < out_dim; j++) {
        const uint8_t *w_packed = q4_T + (size_t)j * half_in;
        __m256i acc32 = _mm256_setzero_si256();
        /* Process 16 packed bytes = 32 int4 weights per iteration */
        for (int i = 0; i < half_in; i += 16) {
            /* Load 16 packed bytes, unpack each to 2 int8 via LUT */
            /* We need 32 int8 for VPMADDUBSW, but LUT gives 16 per byte.
             * Process 16 bytes → 16 LUT lookups → too many gather ops.
             * Better: unpack manually using bit operations. */
            /* Unpack 16 bytes → 32 int8 using shift+mask */
            __m128i packed = _mm_loadu_si128((__m128i*)(w_packed + i));
            /* Low nibbles: AND with 0x0F, subtract 8 */
            __m128i lo_nibbles = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
            __m128i hi_nibbles = _mm_srli_epi16(packed, 4);
            hi_nibbles = _mm_and_si128(hi_nibbles, _mm_set1_epi8(0x0F));
            /* Subtract 8 to get signed range -8..7 */
            lo_nibbles = _mm_sub_epi8(lo_nibbles, _mm_set1_epi8(8));
            hi_nibbles = _mm_sub_epi8(hi_nibbles, _mm_set1_epi8(8));
            /* Interleave lo/hi to get [w0, w1, w2, w3, ...] order */
            __m128i w_low = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);  /* 16 int8 */
            __m128i w_high = _mm_unpackhi_epi8(lo_nibbles, hi_nibbles); /* 16 int8 */
            __m256i wv = _mm256_inserti128_si256(_mm256_castsi128_si256(w_low), w_high, 1);

            /* Load 32 uint8 from xq */
            __m256i xv = _mm256_loadu_si256((__m256i*)(xq + i * 2));
            /* VPMADDUBSW: 32 uint8 × 32 int8 → 16 int16 */
            __m256i prod16 = _mm256_maddubs_epi16(xv, wv);
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(prod16, ones));
        }
        __m128i lo = _mm256_castsi256_si128(acc32);
        __m128i hi = _mm256_extracti128_si256(acc32, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        int32_t dot = _mm_cvtsi128_si32(s);
        dot -= 128 * w_sums[j];
        y[j] = (float)dot * x_scale * scale[j] + (b ? b[j] : 0);
    }
}

/* Float reference */
static void matmul_float(float *y, const float *W, const float *x, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float s = 0;
        for (int i = 0; i < in_dim; i++) s += W[(size_t)i*out_dim+j] * x[i];
        y[j] = s;
    }
}

static float corr(const float *a, const float *b, int n) {
    float ma=0,mb=0; for(int i=0;i<n;i++){ma+=a[i];mb+=b[i];} ma/=n;mb/=n;
    float num=0,da=0,db=0; for(int i=0;i<n;i++){num+=(a[i]-ma)*(b[i]-mb);da+=(a[i]-ma)*(a[i]-ma);db+=(b[i]-mb)*(b[i]-mb);}
    return num/sqrtf(da*db+1e-12f);
}

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void) {
    srand(42);
    init_q4_lut();
    float *W = malloc((size_t)IN_DIM*OUT_DIM*sizeof(float));
    float *x = malloc(IN_DIM*sizeof(float));
    float *scale = malloc(OUT_DIM*sizeof(float));
    int32_t *w_sums = malloc(OUT_DIM*sizeof(int32_t));
    uint8_t *q4_T = malloc((size_t)OUT_DIM*(IN_DIM/2));
    float *y_ref = malloc(OUT_DIM*sizeof(float));
    float *y_q4 = malloc(OUT_DIM*sizeof(float));

    /* GPT-2-like weights */
    for (size_t i = 0; i < (size_t)IN_DIM*OUT_DIM; i++) W[i] = (rand()/(float)RAND_MAX-0.5f)*0.1f;
    for (int i = 0; i < IN_DIM; i++) x[i] = (rand()/(float)RAND_MAX-0.5f)*4;

    /* Quantize Q4 */
    quantize_q4(W, q4_T, scale, IN_DIM, OUT_DIM);
    /* Pre-compute w_sums (sum of unpacked int4 values per output) */
    for (int j = 0; j < OUT_DIM; j++) {
        int32_t s = 0;
        for (int i = 0; i < IN_DIM/2; i++) {
            uint8_t b = q4_T[(size_t)j*(IN_DIM/2)+i];
            s += (b & 0xF) - 8; s += ((b >> 4) & 0xF) - 8;
        }
        w_sums[j] = s;
    }

    /* Correctness */
    matmul_float(y_ref, W, x, IN_DIM, OUT_DIM);
    matmul_q4_simd(y_q4, q4_T, scale, w_sums, x, NULL, IN_DIM, OUT_DIM);
    printf("Quality: Q4 correlation vs float = %.6f\n", corr(y_ref, y_q4, OUT_DIM));

    /* Benchmark */
    double t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) matmul_float(y_ref, W, x, IN_DIM, OUT_DIM);
    double tf = now_sec() - t0;
    t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) matmul_q4_simd(y_q4, q4_T, scale, w_sums, x, NULL, IN_DIM, OUT_DIM);
    double tq = now_sec() - t0;

    printf("\nBenchmark (%d trials, 768x768):\n", N_TRIALS);
    printf("  float32:  %.3f ms\n", tf/N_TRIALS*1000);
    printf("  Q4 SIMD:  %.3f ms  (%.2fx vs float)\n", tq/N_TRIALS*1000, tf/tq);

    printf("\nMemory per matrix:\n");
    printf("  float32: %.0f KB\n", (float)IN_DIM*OUT_DIM*4/1024);
    printf("  Q8:      %.0f KB\n", (float)IN_DIM*OUT_DIM*1/1024 + (float)OUT_DIM*4/1024);
    printf("  Q4:      %.0f KB (%.1fx less than float, %.1fx less than Q8)\n",
           (float)OUT_DIM*(IN_DIM/2)/1024 + (float)OUT_DIM*4/1024,
           (float)IN_DIM*OUT_DIM*4/((float)OUT_DIM*(IN_DIM/2)+(float)OUT_DIM*4),
           ((float)IN_DIM*OUT_DIM*1+(float)OUT_DIM*4)/((float)OUT_DIM*(IN_DIM/2)+(float)OUT_DIM*4));

    free(W);free(x);free(scale);free(w_sums);free(q4_T);free(y_ref);free(y_q4);
    return 0;
}
