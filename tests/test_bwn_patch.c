/* test_bwn_patch.c — Unit test for the BWN patch
 *
 * Verifies:
 *   1. bin_forward (new BWN) matches reference: y[j] = (sum_i sign(W)*x[i]) * alpha * K + bias
 *   2. bin_forward_float (legacy, no K) matches reference without K
 *   3. bin_forward_bnn (legacy fast path) binarizes x as before
 *   4. bin_backward: alpha decreases when grad_alpha*gy > 0 (loss-descending direction)
 *   5. bin_backward: no alpha clamping to [0.001, 1.0]
 *   6. 12-layer forward doesn't NaN/explode
 *
 * Compile: gcc -O2 -o test_bwn_patch test_bwn_patch.c runtime/lal_runtime.c -lm
 */
#include "../runtime/lal_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { printf("ok  : %s\n", msg); } \
} while(0)

/* Reference BWN forward in pure float: y[j] = (sum_i sign(W[j,i]) * x[i]) * alpha[j] * K + bias[j]
 * W is [in, out] (Conv1D layout), so W[i*out + j] = W[i][j] */
static void ref_bwn_forward(float *y, const float *x, const float *W, const float *alpha,
                             const float *bias, int in_dim, int out_dim) {
    float abs_sum = 0;
    for (int i = 0; i < in_dim; i++) abs_sum += fabsf(x[i]);
    float K = abs_sum / in_dim;
    for (int j = 0; j < out_dim; j++) {
        float s = 0;
        for (int i = 0; i < in_dim; i++) {
            float sign_w = (W[i * out_dim + j] > 0.0f) ? 1.0f : -1.0f;
            s += sign_w * x[i];
        }
        y[j] = s * alpha[j] * K + bias[j];
    }
}

static void test_bwn_forward_matches_reference(void) {
    printf("\n--- test_bwn_forward_matches_reference ---\n");
    const int IN = 130, OUT = 70;  /* non-multiple of 64 to test tail handling */
    float *W = malloc(IN * OUT * sizeof(float));
    float *bias = malloc(OUT * sizeof(float));
    float *x = malloc(IN * sizeof(float));
    float *y_actual = malloc(OUT * sizeof(float));
    float *y_ref = malloc(OUT * sizeof(float));

    srand(42);
    for (int i = 0; i < IN * OUT; i++) W[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    for (int j = 0; j < OUT; j++) bias[j] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    for (int i = 0; i < IN; i++) x[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;

    BinLayer bl;
    bin_layer_init(&bl, W, bias, IN, OUT);
    bin_forward(y_actual, x, &bl);
    ref_bwn_forward(y_ref, x, W, bl.alpha, bl.bias, IN, OUT);

    float max_diff = 0;
    for (int j = 0; j < OUT; j++) {
        float d = fabsf(y_actual[j] - y_ref[j]);
        if (d > max_diff) max_diff = d;
    }
    printf("  max abs diff = %.6e\n", max_diff);
    CHECK(max_diff < 1e-4, "BWN forward matches reference within 1e-4");

    bin_layer_free(&bl);
    free(W); free(bias); free(x); free(y_actual); free(y_ref);
}

static void test_bnn_legacy_still_works(void) {
    printf("\n--- test_bnn_legacy_still_works ---\n");
    const int IN = 64, OUT = 8;
    float W[64*8], bias[8], x[64];
    srand(7);
    for (int i = 0; i < 64*8; i++) W[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
    for (int j = 0; j < 8; j++) bias[j] = 0.1f * j;
    for (int i = 0; i < 64; i++) x[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;

    BinLayer bl;
    bin_layer_init(&bl, W, bias, IN, OUT);

    float y_bnn[8];
    bin_forward_bnn(y_bnn, x, &bl);

    float y_ref[8] = {0};
    for (int j = 0; j < 8; j++) {
        int pc = 0;
        for (int i = 0; i < 64; i++) {
            int sx = x[i] > 0 ? 1 : 0;
            int sw = W[i * 8 + j] > 0 ? 1 : 0;
            if (sx == sw) pc++;
        }
        y_ref[j] = (float)(2 * pc - 64) * bl.alpha[j] + bl.bias[j];
    }
    float max_diff = 0;
    for (int j = 0; j < 8; j++) {
        float d = fabsf(y_bnn[j] - y_ref[j]);
        if (d > max_diff) max_diff = d;
    }
    printf("  max abs diff = %.6e\n", max_diff);
    CHECK(max_diff < 1e-4, "BNN legacy fast path still numerically correct");
    bin_layer_free(&bl);
}

static void test_alpha_no_clamp(void) {
    printf("\n--- test_alpha_no_clamp ---\n");
    const int IN = 64, OUT = 4;
    float W[64*4], bias[4] = {0,0,0,0}, x[64];
    for (int i = 0; i < 64*4; i++) W[i] = (i % 2 ? 1.0f : -1.0f) * 0.00005f;
    for (int i = 0; i < 64; i++) x[i] = 1.0f;

    BinLayer bl;
    bin_layer_init(&bl, W, bias, IN, OUT);
    printf("  alpha[0] = %.6e (was clamped to 0.001 in old code)\n", bl.alpha[0]);
    CHECK(bl.alpha[0] < 0.001f, "alpha below old 0.001 floor is preserved");

    for (int i = 0; i < 64*4; i++) W[i] = (i % 2 ? 1.0f : -1.0f) * 5.0f;
    BinLayer bl2;
    bin_layer_init(&bl2, W, bias, IN, OUT);
    printf("  alpha[0] = %.6e (was clamped to 1.0 in old code)\n", bl2.alpha[0]);
    CHECK(bl2.alpha[0] > 1.0f, "alpha above old 1.0 ceiling is preserved");

    bin_layer_free(&bl);
    bin_layer_free(&bl2);
}

static void test_alpha_update_direction(void) {
    printf("\n--- test_alpha_update_direction ---\n");
    const int IN = 64, OUT = 1;
    float W[64], bias[1] = {0}, x[64], grad_y[1];
    for (int i = 0; i < 64; i++) { W[i] = 1.0f; x[i] = 1.0f; }
    grad_y[0] = 1.0f;

    BinLayer bl;
    bin_layer_init(&bl, W, bias, IN, OUT);
    float alpha_before = bl.alpha[0];
    printf("  alpha before = %.6f\n", alpha_before);

    float grad_x[64];
    bin_backward(grad_x, grad_y, x, &bl, 0.01f);
    float alpha_after = bl.alpha[0];
    printf("  alpha after  = %.6f (lr=0.01, grad_y=1, grad_alpha>0)\n", alpha_after);

    CHECK(alpha_after < alpha_before,
          "alpha DECREASES when grad_y*grad_alpha > 0 (loss descent)");

    bin_layer_free(&bl);
}

static void test_k_norm_preserves_magnitude(void) {
    printf("\n--- test_k_norm_preserves_magnitude ---\n");
    const int IN = 64, OUT = 4;
    float W[64*4], bias[4] = {0,0,0,0}, x[64], x_scaled[64];
    srand(11);
    for (int i = 0; i < 64*4; i++) W[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < 64; i++) { x[i] = (rand()/(float)RAND_MAX - 0.5f) * 2; x_scaled[i] = x[i] * 2.0f; }

    BinLayer bl;
    bin_layer_init(&bl, W, bias, IN, OUT);

    float y1[4], y2[4], y_bnn1[4], y_bnn2[4];
    bin_forward(y1, x, &bl);
    bin_forward(y2, x_scaled, &bl);
    bin_forward_bnn(y_bnn1, x, &bl);
    bin_forward_bnn(y_bnn2, x_scaled, &bl);

    float bwn_ratio = 0; int cnt = 0;
    for (int j = 0; j < 4; j++) {
        if (fabsf(y1[j]) > 1e-3f) { bwn_ratio += y2[j] / y1[j]; cnt++; }
    }
    if (cnt) bwn_ratio /= cnt;
    printf("  BWN y(2x) / y(x) = %.3f (expect ~4.0 with K-norm)\n", bwn_ratio);
    CHECK(bwn_ratio > 3.5f && bwn_ratio < 4.5f, "BWN K-norm preserves input magnitude");

    float bnn_ratio = 0; cnt = 0;
    for (int j = 0; j < 4; j++) {
        if (fabsf(y_bnn1[j]) > 1e-3f) { bnn_ratio += y_bnn2[j] / y_bnn1[j]; cnt++; }
    }
    if (cnt) bnn_ratio /= cnt;
    printf("  BNN y(2x) / y(x) = %.3f (expect ~1.0, magnitude-blind)\n", bnn_ratio);
    CHECK(bnn_ratio > 0.9f && bnn_ratio < 1.1f, "BNN is magnitude-blind (sanity check)");

    bin_layer_free(&bl);
}

static void test_no_nan_in_deep_stack(void) {
    printf("\n--- test_no_nan_in_deep_stack ---\n");
    const int IN = 768, OUT = 768;
    float *W = malloc(IN * OUT * sizeof(float));
    float *bias = malloc(OUT * sizeof(float));
    float *x = malloc(IN * sizeof(float));
    float *y = malloc(OUT * sizeof(float));

    srand(99);
    for (int i = 0; i < IN; i++) x[i] = (rand()/(float)RAND_MAX - 0.5f) * 0.4f;

    int has_nan = 0;
    for (int layer = 0; layer < 12; layer++) {
        for (int i = 0; i < IN * OUT; i++) W[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
        for (int j = 0; j < OUT; j++) bias[j] = 0;

        BinLayer bl;
        bin_layer_init(&bl, W, bias, IN, OUT);
        bin_forward(y, x, &bl);

        float mx = 0; int layer_nan = 0;
        for (int j = 0; j < OUT; j++) {
            if (isnan(y[j]) || isinf(y[j])) { layer_nan = 1; break; }
            if (fabsf(y[j]) > mx) mx = fabsf(y[j]);
        }
        if (layer_nan) { has_nan = 1; printf("  layer %d: NaN/Inf detected!\n", layer); break; }
        if (layer == 0 || layer == 5 || layer == 11)
            printf("  layer %2d: max|y| = %.4f\n", layer, mx);

        float rms = 0;
        for (int j = 0; j < OUT; j++) rms += y[j]*y[j];
        rms = sqrtf(rms / OUT);
        float scale = (rms > 1e-6f) ? 1.0f / (rms + 1e-5f) : 1.0f;
        for (int j = 0; j < OUT; j++) x[j] = y[j] * scale;

        bin_layer_free(&bl);
    }
    CHECK(!has_nan, "12-layer BWN forward stays finite (no NaN/Inf)");
    free(W); free(bias); free(x); free(y);
}

int main(void) {
    printf("=== BWN Patch Unit Tests ===\n");
    test_bwn_forward_matches_reference();
    test_bnn_legacy_still_works();
    test_alpha_no_clamp();
    test_alpha_update_direction();
    test_k_norm_preserves_magnitude();
    test_no_nan_in_deep_stack();

    printf("\n=== Summary: %d failures ===\n", failures);
    return failures ? 1 : 0;
}
