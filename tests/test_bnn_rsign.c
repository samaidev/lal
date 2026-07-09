/* test_bnn_rsign.c — BNN with RSign (learnable threshold) quality test
 *
 * Techniques from ReActNet / Unbalanced BNN:
 *   RSign: sign(x - threshold) instead of sign(x)
 *   threshold = per-channel mean (approximation of learnable shift)
 *
 * Hypothesis: LayerNorm output is not zero-centered, so sign(x) at threshold 0
 * loses information. Shifting to mean preserves more binary activation info.
 *
 * Test: compare BNN matmul output correlation with/without RSign.
 *
 * Compile: gcc -O3 -mavx2 -mfma -o test_bnn_rsign test_bnn_rsign.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#define IN_DIM 768
#define OUT_DIM 768

/* Reference float matmul: y = W^T @ x + b */
static void matmul_float(float *y, const float *W, const float *x, const float *b,
                         int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float s = b ? b[j] : 0;
        for (int i = 0; i < in_dim; i++) s += W[i * out_dim + j] * x[i];
        y[j] = s;
    }
}

/* BNN with sign(x) (threshold=0), standard XNOR+popcount */
static void bnn_zero(float *y, const uint64_t *wbits, int n_words,
                     const float *x, const float *alpha, const float *bias,
                     int in_dim, int out_dim) {
    uint64_t x_bits[16];
    for (int wi = 0; wi < n_words; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in_dim && x[idx] > 0.0f) word |= (1ULL << bi);
        }
        x_bits[wi] = word;
    }
    float mean_abs = 0;
    for (int i = 0; i < in_dim; i++) mean_abs += fabsf(x[i]);
    mean_abs /= in_dim;
    for (int j = 0; j < out_dim; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        int pc = 0;
        for (int wi = 0; wi < n_words; wi++)
            pc += __builtin_popcountll(~(x_bits[wi] ^ wb[wi]));
        y[j] = mean_abs * alpha[j] * (float)(2 * pc - in_dim) + bias[j];
    }
}

/* BNN with RSign: sign(x - mean) — shift threshold to per-channel mean */
static void bnn_rsign(float *y, const uint64_t *wbits, int n_words,
                      const float *x, const float *alpha, const float *bias,
                      int in_dim, int out_dim) {
    /* Compute mean as threshold (ReActNet RSign approximation) */
    float mean = 0;
    for (int i = 0; i < in_dim; i++) mean += x[i];
    mean /= in_dim;

    uint64_t x_bits[16];
    for (int wi = 0; wi < n_words; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in_dim && x[idx] > mean) word |= (1ULL << bi);  /* RSign */
        }
        x_bits[wi] = word;
    }
    /* K-norm: mean(|x|) still used for magnitude */
    float mean_abs = 0;
    for (int i = 0; i < in_dim; i++) mean_abs += fabsf(x[i]);
    mean_abs /= in_dim;
    for (int j = 0; j < out_dim; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        int pc = 0;
        for (int wi = 0; wi < n_words; wi++)
            pc += __builtin_popcountll(~(x_bits[wi] ^ wb[wi]));
        y[j] = mean_abs * alpha[j] * (float)(2 * pc - in_dim) + bias[j];
    }
}

/* BNN with per-channel threshold (median approximation) */
static void bnn_median(float *y, const uint64_t *wbits, int n_words,
                       const float *x, const float *alpha, const float *bias,
                       int in_dim, int out_dim) {
    /* Use median (50th percentile) as threshold — for symmetric dist = 0,
     * but for skewed dist captures center better than mean */
    float sorted[4096];
    memcpy(sorted, x, in_dim * sizeof(float));
    /* Simple sort (in_dim=768, OK) */
    for (int i = 0; i < in_dim - 1; i++)
        for (int j = i + 1; j < in_dim; j++)
            if (sorted[i] > sorted[j]) { float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    float threshold = sorted[in_dim / 2];

    uint64_t x_bits[16];
    for (int wi = 0; wi < n_words; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in_dim && x[idx] > threshold) word |= (1ULL << bi);
        }
        x_bits[wi] = word;
    }
    float mean_abs = 0;
    for (int i = 0; i < in_dim; i++) mean_abs += fabsf(x[i]);
    mean_abs /= in_dim;
    for (int j = 0; j < out_dim; j++) {
        const uint64_t *wb = wbits + (size_t)j * n_words;
        int pc = 0;
        for (int wi = 0; wi < n_words; wi++)
            pc += __builtin_popcountll(~(x_bits[wi] ^ wb[wi]));
        y[j] = mean_abs * alpha[j] * (float)(2 * pc - in_dim) + bias[j];
    }
}

static float correlation(const float *a, const float *b, int n) {
    float ma = 0, mb = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    float num = 0, da = 0, db = 0;
    for (int i = 0; i < n; i++) {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma) * (a[i] - ma);
        db += (b[i] - mb) * (b[i] - mb);
    }
    return num / sqrtf(da * db + 1e-12f);
}

int main(void) {
    srand(42);
    float *W = malloc(IN_DIM * OUT_DIM * sizeof(float));
    float *x = malloc(IN_DIM * sizeof(float));
    float *alpha = malloc(OUT_DIM * sizeof(float));
    float *bias = malloc(OUT_DIM * sizeof(float));
    uint64_t *wbits = malloc((size_t)OUT_DIM * ((IN_DIM+63)/64) * sizeof(uint64_t));
    float *y_ref = malloc(OUT_DIM * sizeof(float));
    float *y_zero = malloc(OUT_DIM * sizeof(float));
    float *y_rsign = malloc(OUT_DIM * sizeof(float));
    float *y_median = malloc(OUT_DIM * sizeof(float));
    int n_words = (IN_DIM + 63) / 64;

    /* Simulate LayerNorm output: mean~0 but with skew */
    for (int i = 0; i < IN_DIM; i++) {
        x[i] = (rand()/(float)RAND_MAX - 0.5f) * 4;
        /* Add skew: 60% positive, 40% negative */
        if (rand() % 10 < 6) x[i] += 0.5f;
    }
    for (int i = 0; i < IN_DIM * OUT_DIM; i++) W[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
    for (int j = 0; j < OUT_DIM; j++) { alpha[j] = 0.1f; bias[j] = 0.01f * j; }

    /* Pack sign(W) */
    for (int j = 0; j < OUT_DIM; j++) {
        const float *wj = W;
        for (int i = 0; i < IN_DIM; i++) {
            if (W[i * OUT_DIM + j] > 0)
                wbits[(size_t)j * n_words + i/64] |= (1ULL << (i % 64));
        }
    }

    matmul_float(y_ref, W, x, bias, IN_DIM, OUT_DIM);
    bnn_zero(y_zero, wbits, n_words, x, alpha, bias, IN_DIM, OUT_DIM);
    bnn_rsign(y_rsign, wbits, n_words, x, alpha, bias, IN_DIM, OUT_DIM);
    bnn_median(y_median, wbits, n_words, x, alpha, bias, IN_DIM, OUT_DIM);

    printf("=== BNN activation threshold comparison ===\n");
    printf("Input: LayerNorm-like, mean=%.4f, skew positive\n",
           ({ float m=0; for(int i=0;i<IN_DIM;i++)m+=x[i]; m/IN_DIM; }));
    printf("\nCorrelation with float reference (higher = better quality):\n");
    printf("  sign(x)       [threshold=0]:  %.6f\n", correlation(y_ref, y_zero, OUT_DIM));
    printf("  sign(x-mean)  [RSign]:        %.6f\n", correlation(y_ref, y_rsign, OUT_DIM));
    printf("  sign(x-median)               %.6f\n", correlation(y_ref, y_median, OUT_DIM));

    printf("\nMean abs error vs float:\n");
    float ez=0, er=0, em=0;
    for (int j = 0; j < OUT_DIM; j++) {
        ez += fabsf(y_ref[j] - y_zero[j]);
        er += fabsf(y_ref[j] - y_rsign[j]);
        em += fabsf(y_ref[j] - y_median[j]);
    }
    printf("  sign(x):     %.4f\n", ez/OUT_DIM);
    printf("  RSign:       %.4f\n", er/OUT_DIM);
    printf("  median:      %.4f\n", em/OUT_DIM);

    free(W); free(x); free(alpha); free(bias); free(wbits);
    free(y_ref); free(y_zero); free(y_rsign); free(y_median);
    return 0;
}
