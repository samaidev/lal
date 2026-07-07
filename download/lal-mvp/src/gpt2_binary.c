/* gpt2_binary.c — Binary (XNOR+popcount) weight layers for GPT-2.
 *
 * Loads all 48 weight matrices from gpt2_weights.bin at runtime, binarizes
 * them to {-1,+1} (1 bit each), and provides binary matmul functions.
 *
 * This is the same XNOR+popcount algorithm as LAL's --binarize mode,
 * but implemented directly in C for the full GPT-2 model.
 *
 * Binary matmul: y[m] = (2*popcount(XNOR(x_bits, w_bits)) - n) * alpha + bias
 * where w_bits is packed 64 per uint64, alpha = mean(|w|) per output.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define N_LAYER 12
#define N_EMBD  768

/* Per-layer binary weight storage */
typedef struct {
    uint64_t *wbits;    /* packed sign bits: [out_dim * n_words] */
    float    *alpha;    /* [out_dim] scale factors */
    float    *bias;     /* [out_dim] biases */
    int       in_dim;
    int       out_dim;
    int       n_words;  /* (in_dim + 63) / 64 */
} BinLayer;

/* 48 binary layers: [N_LAYER][4] = attn_qkv, attn_proj, mlp_fc, mlp_proj */
static BinLayer g_bin[N_LAYER][4];

/* Weight file loader (same GPW2 format) */
static float *load_tensor(const char *path, const char *key, int *out_ndim, int *out_shape) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4]; fread(magic, 1, 4, f);
    int n_tensors; fread(&n_tensors, 4, 1, f);
    for (int i = 0; i < n_tensors; i++) {
        int klen; fread(&klen, 4, 1, f);
        char kbuf[256]; fread(kbuf, 1, klen, f); kbuf[klen] = '\0';
        int ndim; fread(&ndim, 4, 1, f);
        int shape[4], n = 1;
        for (int d = 0; d < ndim; d++) { fread(&shape[d], 4, 1, f); n *= shape[d]; }
        float *data = malloc(n * sizeof(float));
        fread(data, 4, n, f);
        if (strcmp(kbuf, key) == 0) {
            fclose(f);
            *out_ndim = ndim;
            memcpy(out_shape, shape, sizeof(int)*4);
            return data;
        }
        free(data);
    }
    fclose(f);
    return NULL;
}

/* Binarize a weight matrix [in_dim, out_dim] (row-major) into BinLayer */
static void binarize(BinLayer *bl, const float *W, const float *bias,
                     int in_dim, int out_dim) {
    bl->in_dim = in_dim;
    bl->out_dim = out_dim;
    bl->n_words = (in_dim + 63) / 64;
    bl->wbits = calloc(out_dim * bl->n_words, sizeof(uint64_t));
    bl->alpha = calloc(out_dim, sizeof(float));
    bl->bias = bias ? malloc(out_dim * sizeof(float)) : calloc(out_dim, sizeof(float));

    for (int j = 0; j < out_dim; j++) {
        /* alpha = mean(|w[j,:]|) */
        float abs_sum = 0;
        for (int i = 0; i < in_dim; i++) abs_sum += fabsf(W[i * out_dim + j]);
        bl->alpha[j] = abs_sum / in_dim;
        if (bias) bl->bias[j] = bias[j];

        /* Pack signs: bit=1 if w>0 */
        for (int wi = 0; wi < bl->n_words; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int idx = wi * 64 + bi;
                if (idx < in_dim && W[idx * out_dim + j] > 0.0f)
                    word |= (1ULL << bi);
            }
            bl->wbits[j * bl->n_words + wi] = word;
        }
    }
}

/* Load and binarize all 48 layers */
void gpt2_binary_init(const char *weight_path) {
    printf("[*] binarizing 48 weight matrices...\n");
    char key[64];
    const char *suffixes[4] = {"attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"};
    int in_dims[4] = {768, 768, 768, 3072};

    for (int layer = 0; layer < N_LAYER; layer++) {
        for (int m = 0; m < 4; m++) {
            sprintf(key, "h.%d.%s.weight", layer, suffixes[m]);
            int ndim, shape[4];
            float *W = load_tensor(weight_path, key, &ndim, shape);
            if (!W) { fprintf(stderr, "missing %s\n", key); exit(1); }

            /* Load bias */
            sprintf(key, "h.%d.%s.bias", layer, suffixes[m]);
            float *b = load_tensor(weight_path, key, &ndim, shape);

            binarize(&g_bin[layer][m], W, b, in_dims[m], shape[1]);
            free(W); free(b);
        }
    }
    printf("[*] binarization complete (32x compression)\n");
}

/* Binary matmul: y[out_dim] = binarize(x) · binarize(W) + bias
 * x: [in_dim] float, W: pre-binarized, y: [out_dim] float output */
void binary_matmul(float *y, const float *x, const BinLayer *bl) {
    int in_dim = bl->in_dim;
    int out_dim = bl->out_dim;
    int n_words = bl->n_words;

    /* Binarize input x into packed bits */
    uint64_t xbits[32]; /* max 2048/64 = 32 words */
    for (int wi = 0; wi < n_words; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in_dim && x[idx] > 0.0f)
                word |= (1ULL << bi);
        }
        xbits[wi] = word;
    }

    /* For each output: XNOR + popcount */
    for (int j = 0; j < out_dim; j++) {
        int pc = 0;
        const uint64_t *wbits = &bl->wbits[j * n_words];
        for (int wi = 0; wi < n_words; wi++) {
            uint64_t xnor = ~(xbits[wi] ^ wbits[wi]);
            pc += __builtin_popcountll(xnor);
        }
        /* dot = (2*pc - N) * alpha + bias */
        y[j] = (float)(2 * pc - in_dim) * bl->alpha[j] + bl->bias[j];
    }
}

/* Convenience functions for each layer's 4 matrices */
void binary_attn_qkv(int layer, float *out, const float *x) {
    binary_matmul(out, x, &g_bin[layer][0]);
}
void binary_attn_proj(int layer, float *out, const float *x) {
    binary_matmul(out, x, &g_bin[layer][1]);
}
void binary_mlp_fc(int layer, float *out, const float *x) {
    binary_matmul(out, x, &g_bin[layer][2]);
}
void binary_mlp_proj(int layer, float *out, const float *x) {
    binary_matmul(out, x, &g_bin[layer][3]);
}
