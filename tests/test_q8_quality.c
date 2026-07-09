/* test_q8_quality.c — Q8 quantization quality test (llama.cpp Q8_0 inspired)
 *
 * Key insight: 1-bit (BWN) loses too much info. Q8 (8-bit) retains ~99%.
 * llama.cpp Q8_0: per-block (32 elements) scale + int8 values.
 *
 * Test: compare output correlation of:
 *   A) float (reference)
 *   B) BWN 1-bit (sign only)
 *   C) Q8 8-bit (per-row scale + int8)
 *   D) Q4 4-bit (per-row scale + int4)
 *
 * This tells us the quality ceiling of each quantization level.
 *
 * Compile: gcc -O3 -mavx2 -mfma -o test_q8_quality test_q8_quality.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>

#define IN_DIM 768
#define OUT_DIM 768
#define BLOCK 32  /* Q8_0 block size */

/* Reference float matmul: y = W^T @ x + b */
static void matmul_float(float *y, const float *W, const float *x, const float *b,
                         int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float s = b ? b[j] : 0;
        for (int i = 0; i < in_dim; i++) s += W[i * out_dim + j] * x[i];
        y[j] = s;
    }
}

/* BWN 1-bit: y = sum(sign(W)*x) * alpha + b, alpha = mean|W| */
static void matmul_bwn(float *y, const float *W, const float *x, const float *b,
                       int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float s = b ? b[j] : 0;
        float alpha = 0;
        for (int i = 0; i < in_dim; i++) {
            float w = W[i * out_dim + j];
            alpha += fabsf(w);
            s += (w > 0 ? 1.0f : -1.0f) * x[i];
        }
        alpha /= in_dim;
        y[j] = s * alpha + (b ? b[j] : 0);
    }
}

/* Q8 per-row: y = sum(q8_w[i] * x[i]) * scale + b
 * q8_w[i] = round(W[i] / scale), scale = max|W| / 127 */
static void matmul_q8_row(float *y, const float *W, const float *x, const float *b,
                          int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        /* Per-output-row quantization */
        float max_abs = 0;
        for (int i = 0; i < in_dim; i++) max_abs = fmaxf(max_abs, fabsf(W[i * out_dim + j]));
        float scale = max_abs / 127.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        float s = b ? b[j] : 0;
        for (int i = 0; i < in_dim; i++) {
            int q = (int)lroundf(W[i * out_dim + j] / scale);
            if (q > 127) q = 127; if (q < -127) q = -127;
            s += (float)q * x[i];
        }
        y[j] = s * scale + (b ? b[j] : 0);
    }
}

/* Q8 per-block (llama.cpp Q8_0 style): 32-element blocks */
static void matmul_q8_block(float *y, const float *W, const float *x, const float *b,
                            int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float s = b ? b[j] : 0;
        for (int bi = 0; bi < in_dim; bi += BLOCK) {
            int end = bi + BLOCK < in_dim ? bi + BLOCK : in_dim;
            float max_abs = 0;
            for (int i = bi; i < end; i++) max_abs = fmaxf(max_abs, fabsf(W[i * out_dim + j]));
            float scale = max_abs / 127.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            for (int i = bi; i < end; i++) {
                int q = (int)lroundf(W[i * out_dim + j] / scale);
                if (q > 127) q = 127; if (q < -127) q = -127;
                s += (float)q * x[i] * scale;
            }
        }
        y[j] = s + (b ? b[j] : 0);
    }
}

/* Q4 per-row: 4-bit quantization */
static void matmul_q4_row(float *y, const float *W, const float *x, const float *b,
                          int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float max_abs = 0;
        for (int i = 0; i < in_dim; i++) max_abs = fmaxf(max_abs, fabsf(W[i * out_dim + j]));
        float scale = max_abs / 7.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        float s = b ? b[j] : 0;
        for (int i = 0; i < in_dim; i++) {
            int q = (int)lroundf(W[i * out_dim + j] / scale);
            if (q > 7) q = 7; if (q < -7) q = -7;
            s += (float)q * x[i];
        }
        y[j] = s * scale + (b ? b[j] : 0);
    }
}

static float correlation(const float *a, const float *b, int n) {
    float ma=0,mb=0;
    for(int i=0;i<n;i++){ma+=a[i];mb+=b[i];}
    ma/=n;mb/=n;
    float num=0,da=0,db=0;
    for(int i=0;i<n;i++){num+=(a[i]-ma)*(b[i]-mb);da+=(a[i]-ma)*(a[i]-ma);db+=(b[i]-mb)*(b[i]-mb);}
    return num/sqrtf(da*db+1e-12f);
}

static float max_rel_err(const float *ref, const float *test, int n) {
    float maxe = 0;
    for (int i = 0; i < n; i++) {
        float r = fabsf(ref[i]);
        if (r < 1e-6f) r = 1e-6f;
        float e = fabsf(ref[i] - test[i]) / r;
        if (e > maxe) maxe = e;
    }
    return maxe;
}

int main(void) {
    srand(42);
    /* Use realistic GPT-2-like weights (normal distribution, small std) */
    float *W = malloc(IN_DIM * OUT_DIM * sizeof(float));
    float *x = malloc(IN_DIM * sizeof(float));
    float *b = malloc(OUT_DIM * sizeof(float));
    float *y_ref=malloc(OUT_DIM*sizeof(float)), *y_bwn=malloc(OUT_DIM*sizeof(float));
    float *y_q8r=malloc(OUT_DIM*sizeof(float)), *y_q8b=malloc(OUT_DIM*sizeof(float));
    float *y_q4=malloc(OUT_DIM*sizeof(float));

    /* GPT-2 weights: ~N(0, 0.02/sqrt(dim)), but some large outliers */
    for (int i = 0; i < IN_DIM * OUT_DIM; i++) {
        W[i] = (rand()/(float)RAND_MAX - 0.5f) * 0.1f;
        /* 5% outliers */
        if (rand() % 20 == 0) W[i] *= 10;
    }
    /* LayerNorm output: ~N(0, 1) */
    for (int i = 0; i < IN_DIM; i++) x[i] = (rand()/(float)RAND_MAX - 0.5f) * 4;
    for (int j = 0; j < OUT_DIM; j++) b[j] = 0.01f * j;

    matmul_float(y_ref, W, x, b, IN_DIM, OUT_DIM);
    matmul_bwn(y_bwn, W, x, b, IN_DIM, OUT_DIM);
    matmul_q8_row(y_q8r, W, x, b, IN_DIM, OUT_DIM);
    matmul_q8_block(y_q8b, W, x, b, IN_DIM, OUT_DIM);
    matmul_q4_row(y_q4, W, x, b, IN_DIM, OUT_DIM);

    printf("=== Quantization quality comparison (768x768, GPT-2-like weights) ===\n\n");
    printf("%-20s %12s %12s %12s\n", "Method", "Correlation", "Max Rel Err", "Bits/weight");
    printf("%-20s %12.6f %12.4f %12s\n", "BWN (1-bit)",
        correlation(y_ref,y_bwn,OUT_DIM), max_rel_err(y_ref,y_bwn,OUT_DIM), "1");
    printf("%-20s %12.6f %12.4f %12s\n", "Q4 per-row",
        correlation(y_ref,y_q4,OUT_DIM), max_rel_err(y_ref,y_q4,OUT_DIM), "4");
    printf("%-20s %12.6f %12.4f %12s\n", "Q8 per-block (Q8_0)",
        correlation(y_ref,y_q8b,OUT_DIM), max_rel_err(y_ref,y_q8b,OUT_DIM), "8.3");
    printf("%-20s %12.6f %12.4f %12s\n", "Q8 per-row",
        correlation(y_ref,y_q8r,OUT_DIM), max_rel_err(y_ref,y_q8r,OUT_DIM), "8");

    printf("\nOutput samples (first 5):\n");
    printf("  float:  %.4f %.4f %.4f %.4f %.4f\n", y_ref[0],y_ref[1],y_ref[2],y_ref[3],y_ref[4]);
    printf("  BWN:    %.4f %.4f %.4f %.4f %.4f\n", y_bwn[0],y_bwn[1],y_bwn[2],y_bwn[3],y_bwn[4]);
    printf("  Q8 row: %.4f %.4f %.4f %.4f %.4f\n", y_q8r[0],y_q8r[1],y_q8r[2],y_q8r[3],y_q8r[4]);
    printf("  Q8 blk: %.4f %.4f %.4f %.4f %.4f\n", y_q8b[0],y_q8b[1],y_q8b[2],y_q8b[3],y_q8b[4]);

    /* Memory comparison */
    printf("\nMemory per matrix (768x768):\n");
    printf("  float:  %.1f KB\n", (float)IN_DIM*OUT_DIM*4/1024);
    printf("  BWN:    %.1f KB (1 bit)\n", (float)OUT_DIM*((IN_DIM+63)/64)*8/1024);
    printf("  Q4:     %.1f KB (4 bit)\n", (float)IN_DIM*OUT_DIM*4/8/1024);
    printf("  Q8_0:   %.1f KB (8 bit + scale/32)\n", (float)(IN_DIM*OUT_DIM + (IN_DIM/BLOCK)*4)/1024);

    free(W);free(x);free(b);free(y_ref);free(y_bwn);free(y_q8r);free(y_q8b);free(y_q4);
    return 0;
}
