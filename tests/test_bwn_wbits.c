/* test_bwn_wbits.c — BWN matmul directly from wbits (1-bit packed)
 *
 * Key insight: w_sign_float (108MB) wastes 32x memory vs wbits (3.5MB).
 * This version uses wbits directly with AVX2 sign-bit flip:
 *   - Read 64 sign bits from wbits (8 bytes)
 *   - For each group of 8 floats in x, extract 8 sign bits
 *   - Flip x's sign bits using XOR (sign=-1) or keep (sign=+1)
 *   - Sum the 8 floats (no multiply needed!)
 *
 * Compare:
 *   - w_sign_float: 108MB, float FMA, 20 tok/s
 *   - wbits direct: 3.5MB, XOR sign flip + sum, should be much faster
 *
 * Compile: gcc -O3 -mavx2 -mfma -o test_bwn_wbits test_bwn_wbits.c -lm
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

/* Reference: w_sign_float (current approach, 32x memory waste) */
static void bwn_sign_float(float *y, const float *w_sign, const float *x,
                           const float *alpha, const float *bias,
                           int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        const float *w = w_sign + (size_t)j * in_dim;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < in_dim; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w+i), _mm256_loadu_ps(x+i), acc);
        /* hsum */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        y[j] = _mm_cvtss_f32(s) * alpha[j] + bias[j];
    }
}

/* Optimized: wbits direct with AVX2 sign-bit flip
 * sign(W) = ±1, so x * sign(W) = x with sign flipped or not.
 * float sign bit is bit 31. XOR with 0x80000000 flips sign.
 *
 * For 8 floats x[0..7] and 8 sign bits from wbits:
 *   mask = _mm256_set1_epi32(0x80000000)  // sign bit mask
 *   sign_bits = extract 8 bits from wbits
 *   flip_mask = expand sign_bits to 8 x 32-bit (0 or 0x80000000)
 *   x_flipped = x XOR flip_mask  (flips sign where sign_bit=0, i.e. -1)
 *   dot += hsum(x_flipped)
 */
static void bwn_wbits_direct(float *y, const uint64_t *wbits, int n_words,
                             const float *x, const float *alpha, const float *bias,
                             int in_dim, int out_dim) {
    /* Sign convention: bit set = +1, bit clear = -1
     * For -1 (bit clear): flip x's sign → XOR with 0x80000000
     * For +1 (bit set): keep x → XOR with 0
     * So: flip_mask = (~sign_bits expanded) & 0x80000000
     * Or: if bit set, mask=0; if bit clear, mask=0x80000000 */
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    for (int j = 0; j < out_dim; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        __m256 acc = _mm256_setzero_ps();
        for (int wi = 0; wi < n_words; wi++) {
            uint64_t w = wb[wi];
            int base = wi * 64;
            /* Process 8 floats at a time (8 sign bits) */
            for (int bi = 0; bi < 8 && base + bi*8 < in_dim; bi++) {
                int idx = base + bi * 8;
                if (idx + 8 > in_dim) break;
                /* Extract 8 sign bits: bits bi*8..bi*8+7 of w */
                uint8_t byte = (w >> (bi * 8)) & 0xFF;
                /* Build flip mask: bit set (+1) → 0, bit clear (-1) → 0x80000000
                 * ~byte: bit set → 0, bit clear → 1
                 * We need each of 8 lanes to be 0 or 0x80000000 */
                __m256i flip = _mm256_set1_epi32(~byte & 0xFF);
                /* Extract bit k: (byte >> k) & 1, negate, expand to 32-bit
                 * Actually use _mm256_movemask inverse... simpler:
                 * Build 8 int32 from byte bits */
                int32_t vals[8];
                for (int k = 0; k < 8; k++) {
                    int bit = (byte >> k) & 1;
                    vals[k] = bit ? 0 : (int32_t)0x80000000;  /* +1: no flip, -1: flip */
                }
                __m256 flip_mask = _mm256_loadu_ps((float*)vals);
                __m256 xv = _mm256_loadu_ps(x + idx);
                __m256 x_signed = _mm256_xor_ps(xv, flip_mask);  /* flip sign where -1 */
                acc = _mm256_add_ps(acc, x_signed);  /* no multiply! just add */
            }
        }
        /* hsum */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        y[j] = _mm_cvtss_f32(s) * alpha[j] + bias[j];
    }
}

/* Even faster: use _mm256_blendv_ps for sign selection
 * blendv(a, b, mask) = mask ? b : a
 * If sign=-1: select -x (negate), if sign=+1: select x
 * neg_x = XOR x with sign_mask
 * result = blendv(x, neg_x, flip_mask)  where flip_mask = all-1s for -1 */
static void bwn_wbits_blendv(float *y, const uint64_t *wbits, int n_words,
                             const float *x, const float *alpha, const float *bias,
                             int in_dim, int out_dim) {
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    for (int j = 0; j < out_dim; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        __m256 acc = _mm256_setzero_ps();
        for (int wi = 0; wi < n_words; wi++) {
            uint64_t w = wb[wi];
            int base = wi * 64;
            for (int bi = 0; bi < 8 && base + bi*8 < in_dim; bi++) {
                int idx = base + bi * 8;
                if (idx + 8 > in_dim) break;
                uint8_t byte = (w >> (bi * 8)) & 0xFF;
                /* flip_mask: all-1s (0xFFFFFFFF) where sign=-1 (bit clear), 0 where +1 */
                __m256i flip = _mm256_set1_epi32(~(int32_t)byte);
                /* Need per-lane: bit k clear → all 1s, bit k set → all 0s */
                int32_t vals[8];
                for (int k = 0; k < 8; k++) {
                    int bit = (byte >> k) & 1;
                    vals[k] = bit ? 0 : -1;  /* +1: 0, -1: all-1s */
                }
                __m256 flip_mask = _mm256_loadu_ps((float*)vals);
                __m256 xv = _mm256_loadu_ps(x + idx);
                __m256 neg_x = _mm256_xor_ps(xv, sign_mask);
                __m256 x_signed = _mm256_blendv_ps(xv, neg_x, flip_mask);
                acc = _mm256_add_ps(acc, x_signed);
            }
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        y[j] = _mm_cvtss_f32(s) * alpha[j] + bias[j];
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    srand(42);
    uint64_t *wbits = calloc((size_t)OUT_DIM * ((IN_DIM+63)/64), sizeof(uint64_t));
    float *w_sign = malloc((size_t)OUT_DIM * IN_DIM * sizeof(float));
    float *x = malloc(IN_DIM * sizeof(float));
    float *alpha = malloc(OUT_DIM * sizeof(float));
    float *bias = malloc(OUT_DIM * sizeof(float));
    float *y1 = malloc(OUT_DIM * sizeof(float));
    float *y2 = malloc(OUT_DIM * sizeof(float));
    float *y3 = malloc(OUT_DIM * sizeof(float));
    int n_words = (IN_DIM + 63) / 64;

    for (int i = 0; i < IN_DIM; i++) x[i] = (rand()/(float)RAND_MAX - 0.5f) * 4;
    for (int j = 0; j < OUT_DIM; j++) { alpha[j] = 0.1f; bias[j] = 0.01f * j; }

    /* Pack sign(W): random bits */
    for (size_t i = 0; i < (size_t)OUT_DIM * ((IN_DIM+63)/64); i++) wbits[i] = ((uint64_t)rand() << 32) | rand();
    /* Build w_sign_float from wbits */
    for (int j = 0; j < OUT_DIM; j++) {
        for (int i = 0; i < IN_DIM; i++)
            w_sign[(size_t)j*IN_DIM + i] = (wbits[(size_t)j*n_words + i/64] >> (i%64)) & 1 ? 1.0f : -1.0f;
    }

    /* Correctness */
    bwn_sign_float(y1, w_sign, x, alpha, bias, IN_DIM, OUT_DIM);
    bwn_wbits_direct(y2, wbits, n_words, x, alpha, bias, IN_DIM, OUT_DIM);
    bwn_wbits_blendv(y3, wbits, n_words, x, alpha, bias, IN_DIM, OUT_DIM);
    float d2=0, d3=0;
    for (int j = 0; j < OUT_DIM; j++) { d2 = fmaxf(d2, fabsf(y1[j]-y2[j])); d3 = fmaxf(d3, fabsf(y1[j]-y3[j])); }
    printf("Correctness (max diff vs sign_float):\n");
    printf("  wbits_direct: %.2e\n", d2);
    printf("  wbits_blendv: %.2e\n", d3);

    /* Benchmark */
    double t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) bwn_sign_float(y1, w_sign, x, alpha, bias, IN_DIM, OUT_DIM);
    double t_sf = now_sec() - t0;

    t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) bwn_wbits_direct(y2, wbits, n_words, x, alpha, bias, IN_DIM, OUT_DIM);
    double t_wd = now_sec() - t0;

    t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) bwn_wbits_blendv(y3, wbits, n_words, x, alpha, bias, IN_DIM, OUT_DIM);
    double t_wb = now_sec() - t0;

    printf("\nBenchmark (%d trials, 768x768):\n", N_TRIALS);
    printf("  sign_float (108MB mem):  %.3f ms/trial\n", t_sf/N_TRIALS*1000);
    printf("  wbits_direct (3.5MB):    %.3f ms/trial\n", t_wd/N_TRIALS*1000);
    printf("  wbits_blendv (3.5MB):    %.3f ms/trial\n", t_wb/N_TRIALS*1000);
    printf("  speedup direct: %.2fx\n", t_sf/t_wd);
    printf("  speedup blendv: %.2fx\n", t_sf/t_wb);
    printf("\nMemory: wbits %.1fKB vs sign_float %.1fKB per matrix (32x less)\n",
           (double)OUT_DIM*n_words*8/1024, (double)OUT_DIM*IN_DIM*4/1024);

    free(wbits); free(w_sign); free(x); free(alpha); free(bias);
    free(y1); free(y2); free(y3);
    return 0;
}
