/* test_int8_quant.c — Prototype Int8 activation quantization for BWN speedup
 *
 * Tests:
 *   1. Symmetric quantization: float → int8 with per-tensor scale
 *   2. Sign(W) × x_int8 via conditional add/sub (no multiply)
 *   3. Accuracy: compare int8 matmul vs float matmul on random data
 *   4. Speed: benchmark int8 vs float
 *
 * Compile: gcc -O3 -mavx2 -mfma -o test_int8_quant test_int8_quant.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define IN_DIM 768
#define OUT_DIM 768
#define N_TRIALS 100

/* Symmetric quantization: x_int8 = round(x / scale), scale = max(|x|) / 127 */
static float quantize_activation(const float *x, int8_t *x_q, int n) {
    float max_abs = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)lroundf(x[i] * inv_scale);
        if (v > 127) v = 127;
        if (v < -127) v = -127;  /* -128 reserved for sparsity, use -127 */
        x_q[i] = (int8_t)v;
    }
    return scale;
}

/* Sign(W) as int8: +1 or -1 */
static void pack_sign_int8(const float *W, int8_t *w_sign, int n) {
    for (int i = 0; i < n; i++) w_sign[i] = W[i] >= 0 ? 1 : -1;
}

/* Int8 matmul: y = sum(w_sign[i] * x_q[i]) * scale * alpha
 * w_sign is ±1, so w_sign * x_q = (w_sign > 0) ? x_q : -x_q
 * This is conditional add/sub — no multiply needed */
static void matmul_int8(float *y, const int8_t *w_sign, const int8_t *x_q,
                        float x_scale, const float *alpha, const float *bias,
                        int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        const int8_t *ws = w_sign + (size_t)j * in_dim;
        int32_t acc = 0;
        for (int i = 0; i < in_dim; i++) {
            /* w_sign[i] is ±1, x_q[i] is int8.
             * w_sign * x_q = x_q if w_sign>0, -x_q if w_sign<0 */
            acc += ws[i] * x_q[i];  /* int8 * int8 -> int16, accumulate int32 */
        }
        y[j] = (float)acc * x_scale * alpha[j] + bias[j];
    }
}

/* Float reference matmul: y = sum(w_sign_float[i] * x[i]) * alpha + bias */
static void matmul_float(float *y, const float *w_sign_float, const float *x,
                         const float *alpha, const float *bias,
                         int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        const float *ws = w_sign_float + (size_t)j * in_dim;
        float acc = 0;
        for (int i = 0; i < in_dim; i++) acc += ws[i] * x[i];
        y[j] = acc * alpha[j] + bias[j];
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    srand(42);
    float *W = malloc(IN_DIM * OUT_DIM * sizeof(float));
    float *x = malloc(IN_DIM * sizeof(float));
    float *alpha = malloc(OUT_DIM * sizeof(float));
    float *bias = malloc(OUT_DIM * sizeof(float));
    float *w_sign_float = malloc(IN_DIM * OUT_DIM * sizeof(float));
    int8_t *w_sign_int8 = malloc(IN_DIM * OUT_DIM);
    int8_t *x_q = malloc(IN_DIM);
    float *y_float = malloc(OUT_DIM * sizeof(float));
    float *y_int8 = malloc(OUT_DIM * sizeof(float));

    /* Init: W random (sign matters, magnitude irrelevant for BWN) */
    for (int i = 0; i < IN_DIM * OUT_DIM; i++) W[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
    /* x: simulate LayerNorm output — normal-ish distribution around 0 */
    for (int i = 0; i < IN_DIM; i++) x[i] = (rand()/(float)RAND_MAX - 0.5f) * 6.0f;
    for (int j = 0; j < OUT_DIM; j++) {
        alpha[j] = 0.1f;
        bias[j] = 0.01f * j;
    }

    /* Pack sign(W) */
    pack_sign_int8(W, w_sign_int8, IN_DIM * OUT_DIM);
    for (int i = 0; i < IN_DIM * OUT_DIM; i++) w_sign_float[i] = W[i] >= 0 ? 1.0f : -1.0f;

    /* Test 1: Quantization accuracy */
    printf("=== Test 1: Activation quantization accuracy ===\n");
    float x_scale = quantize_activation(x, x_q, IN_DIM);
    float max_err = 0, mean_err = 0;
    for (int i = 0; i < IN_DIM; i++) {
        float x_deq = x_q[i] * x_scale;
        float err = fabsf(x[i] - x_deq);
        if (err > max_err) max_err = err;
        mean_err += err;
    }
    mean_err /= IN_DIM;
    printf("  scale = %.4f\n", x_scale);
    printf("  max quant error = %.4f\n", max_err);
    printf("  mean quant error = %.4f\n", mean_err);
    printf("  relative max error = %.2f%%\n", max_err / x_scale * 100);

    /* Test 2: Matmul accuracy */
    printf("\n=== Test 2: Matmul accuracy (int8 vs float) ===\n");
    matmul_float(y_float, w_sign_float, x, alpha, bias, IN_DIM, OUT_DIM);
    matmul_int8(y_int8, w_sign_int8, x_q, x_scale, alpha, bias, IN_DIM, OUT_DIM);

    float max_diff = 0, mean_diff = 0, corr = 0;
    float mean_float = 0, mean_int8 = 0;
    for (int j = 0; j < OUT_DIM; j++) {
        float d = fabsf(y_float[j] - y_int8[j]);
        if (d > max_diff) max_diff = d;
        mean_diff += d;
        mean_float += y_float[j];
        mean_int8 += y_int8[j];
    }
    mean_diff /= OUT_DIM;
    mean_float /= OUT_DIM;
    mean_int8 /= OUT_DIM;
    for (int j = 0; j < OUT_DIM; j++) {
        corr += (y_float[j] - mean_float) * (y_int8[j] - mean_int8);
    }
    float std_f = 0, std_i = 0;
    for (int j = 0; j < OUT_DIM; j++) {
        std_f += (y_float[j] - mean_float) * (y_float[j] - mean_float);
        std_i += (y_int8[j] - mean_int8) * (y_int8[j] - mean_int8);
    }
    corr = corr / sqrtf(std_f * std_i + 1e-12f);

    printf("  float output range: [%.2f, %.2f]\n",
           y_float[0], y_float[OUT_DIM-1]);
    printf("  max abs diff = %.4f\n", max_diff);
    printf("  mean abs diff = %.4f\n", mean_diff);
    printf("  correlation = %.6f\n", corr);

    /* Test 3: Speed benchmark */
    printf("\n=== Test 3: Speed benchmark (%d trials) ===\n", N_TRIALS);
    double t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) {
        matmul_float(y_float, w_sign_float, x, alpha, bias, IN_DIM, OUT_DIM);
    }
    double t_float = now_sec() - t0;

    t0 = now_sec();
    for (int t = 0; t < N_TRIALS; t++) {
        float s = quantize_activation(x, x_q, IN_DIM);
        matmul_int8(y_int8, w_sign_int8, x_q, s, alpha, bias, IN_DIM, OUT_DIM);
    }
    double t_int8 = now_sec() - t0;

    printf("  float matmul: %.2f ms/trial (%.1f MOps/s)\n",
           t_float/N_TRIALS*1000, IN_DIM*OUT_DIM*N_TRIALS/t_float/1e6);
    printf("  int8 matmul:  %.2f ms/trial (%.1f MOps/s)\n",
           t_int8/N_TRIALS*1000, IN_DIM*OUT_DIM*N_TRIALS/t_int8/1e6);
    printf("  speedup: %.2fx\n", t_float/t_int8);

    free(W); free(x); free(alpha); free(bias);
    free(w_sign_float); free(w_sign_int8); free(x_q);
    free(y_float); free(y_int8);
    return 0;
}
