/* Quick test: verify binary matmul matches float matmul on a small example */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Simple binary matmul test */
int main() {
    /* 4x3 weight matrix (in_dim=4, out_dim=3) */
    float W[12] = {0.5, -0.3, 0.2,  0.1, -0.8, 0.6,  -0.4, 0.7, 0.3,  0.2, -0.1, 0.9};
    /* W is [in=4, out=3] row-major: W[i*out+j] */
    float b[3] = {0.1, 0.2, 0.3};
    float x[4] = {1.0, -2.0, 0.5, -0.3};

    /* Float matmul reference */
    float y_ref[3];
    for (int j = 0; j < 3; j++) {
        float s = b[j];
        for (int i = 0; i < 4; i++) s += x[i] * W[i*3+j];
        y_ref[j] = s;
    }
    printf("Float:    [%.4f, %.4f, %.4f]\n", y_ref[0], y_ref[1], y_ref[2]);

    /* Binary matmul */
    int in_dim = 4, out_dim = 3;
    int n_words = (in_dim + 63) / 64;  // 1

    /* Binarize W: sign + alpha */
    uint64_t wbits[3];  // per output
    float alpha[3];
    for (int j = 0; j < out_dim; j++) {
        float abs_sum = 0;
        uint64_t word = 0;
        for (int i = 0; i < in_dim; i++) {
            float w = W[i*out_dim+j];
            abs_sum += fabsf(w);
            if (w > 0) word |= (1ULL << i);
        }
        alpha[j] = abs_sum / in_dim;
        wbits[j] = word;
    }

    /* Binarize x */
    uint64_t xbits = 0;
    for (int i = 0; i < in_dim; i++) {
        if (x[i] > 0) xbits |= (1ULL << i);
    }

    /* Binary dot product */
    float y_bin[3];
    for (int j = 0; j < out_dim; j++) {
        int pc = __builtin_popcountll(~(xbits ^ wbits[j]));
        y_bin[j] = (float)(2 * pc - in_dim) * alpha[j] + b[j];
    }
    printf("Binary:   [%.4f, %.4f, %.4f]\n", y_bin[0], y_bin[1], y_bin[2]);

    printf("Alpha:    [%.4f, %.4f, %.4f]\n", alpha[0], alpha[1], alpha[2]);
    printf("Diff:     [%.4f, %.4f, %.4f]\n",
           y_ref[0]-y_bin[0], y_ref[1]-y_bin[1], y_ref[2]-y_bin[2]);

    /* The binary version is an APPROXIMATION. The fine-tuned model
     * compensates for this by adjusting the signs and alpha during training. */
    return 0;
}
