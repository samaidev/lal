/* lal_runtime.c — LAL Universal Runtime implementation
 *
 * Model-agnostic binary neural network operations.
 * Used by models/gpt2.c, models/bert.c, etc.
 */
#include "lal_runtime.h"

/* ========================================================================
 * Binary Weight Layer
 * ======================================================================== */
void bin_layer_init(BinLayer *bl, const float *W, const float *bias,
                    int in_dim, int out_dim) {
    bl->in_dim = in_dim;
    bl->out_dim = out_dim;
    bl->n_words = (in_dim + 63) / 64;
    bl->n_words_T = (out_dim + 63) / 64;
    bl->wbits = calloc(out_dim * bl->n_words, sizeof(uint64_t));
    bl->wbits_T = calloc(in_dim * bl->n_words_T, sizeof(uint64_t));
    bl->alpha = calloc(out_dim, sizeof(float));
    bl->bias = bias ? malloc(out_dim * sizeof(float)) : calloc(out_dim, sizeof(float));

    /* Row-major: pack sign(w[j][i]) per output j */
    for (int j = 0; j < out_dim; j++) {
        float abs_sum = 0;
        for (int i = 0; i < in_dim; i++) abs_sum += fabsf(W[i * out_dim + j]);
        bl->alpha[j] = abs_sum / in_dim;
        if (bias) bl->bias[j] = bias[j];
        for (int wi = 0; wi < bl->n_words; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int idx = wi * 64 + bi;
                if (idx < in_dim && W[idx * out_dim + j] > 0.0f) word |= (1ULL << bi);
            }
            bl->wbits[j * bl->n_words + wi] = word;
        }
    }

    /* Col-major (transposed): pack sign(w[j][i]) per input i */
    for (int i = 0; i < in_dim; i++) {
        for (int wi = 0; wi < bl->n_words_T; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int j = wi * 64 + bi;
                if (j < out_dim && W[i * out_dim + j] > 0.0f) word |= (1ULL << bi);
            }
            bl->wbits_T[i * bl->n_words_T + wi] = word;
        }
    }
}

void bin_layer_free(BinLayer *bl) {
    free(bl->wbits); free(bl->wbits_T); free(bl->alpha); free(bl->bias);
    bl->wbits = NULL; bl->wbits_T = NULL; bl->alpha = NULL; bl->bias = NULL;
}

/* ========================================================================
 * Binary Forward: XNOR + popcount
 * ======================================================================== */
void bin_forward(float *y, const float *x, const BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;
    /* Binarize input */
    uint64_t xbits[64];
    for (int wi = 0; wi < nw; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in && x[idx] > 0.0f) word |= (1ULL << bi);
        }
        xbits[wi] = word;
    }
    /* XNOR + popcount per output */
    for (int j = 0; j < out; j++) {
        int pc = 0;
        const uint64_t *wb = &bl->wbits[j * nw];
        for (int wi = 0; wi < nw; wi++)
            pc += __builtin_popcountll(~(xbits[wi] ^ wb[wi]));
        y[j] = (float)(2 * pc - in) * bl->alpha[j] + bl->bias[j];
    }
}

void bin_forward_float(float *y, const float *x, const BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;
    for (int j = 0; j < out; j++) {
        float s = bl->bias[j];
        const uint64_t *wb = &bl->wbits[j * nw];
        float a = bl->alpha[j];
        for (int wi = 0; wi < nw; wi++) {
            uint64_t w = wb[wi];
            for (int bi = 0; bi < 64; bi++) {
                int idx = wi * 64 + bi;
                if (idx >= in) break;
                s += x[idx] * ((w >> bi) & 1 ? 1.0f : -1.0f) * a;
            }
        }
        y[j] = s;
    }
}

/* ========================================================================
 * Binary Backward: popcount for grad_x, popcount for alpha update
 * ======================================================================== */
void bin_backward(float *grad_x, const float *grad_y, const float *x,
                  BinLayer *bl, float lr) {
    int in = bl->in_dim, out = bl->out_dim;
    int nw_T = bl->n_words_T;

    /* Part 1: grad_x via XNOR+popcount using transposed weights */
    uint64_t gybits[64];
    for (int wi = 0; wi < nw_T; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int j = wi * 64 + bi;
            if (j < out && grad_y[j] > 0.0f) word |= (1ULL << bi);
        }
        gybits[wi] = word;
    }
    float mean_abs_gy = 0, mean_alpha = 0;
    for (int j = 0; j < out; j++) mean_abs_gy += fabsf(grad_y[j]);
    mean_abs_gy /= out;
    for (int j = 0; j < out; j++) mean_alpha += bl->alpha[j];
    mean_alpha /= out;
    for (int i = 0; i < in; i++) {
        int pc = 0;
        const uint64_t *wbT = &bl->wbits_T[i * nw_T];
        for (int wi = 0; wi < nw_T; wi++)
            pc += __builtin_popcountll(~(gybits[wi] ^ wbT[wi]));
        grad_x[i] = (float)(2 * pc - out) * mean_alpha * mean_abs_gy;
    }

    /* Part 2: alpha + bias update via popcount (reuse x_bits) */
    float mean_abs_x = 0;
    for (int i = 0; i < in; i++) mean_abs_x += fabsf(x[i]);
    mean_abs_x /= in;
    uint64_t xbits[64];
    for (int wi = 0; wi < bl->n_words; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in && x[idx] > 0.0f) word |= (1ULL << bi);
        }
        xbits[wi] = word;
    }
    for (int j = 0; j < out; j++) {
        float gy = grad_y[j];
        if (fabsf(gy) < 1e-6f) continue;
        int pc = 0;
        const uint64_t *wb = &bl->wbits[j * bl->n_words];
        for (int wi = 0; wi < bl->n_words; wi++)
            pc += __builtin_popcountll(~(xbits[wi] ^ wb[wi]));
        float grad_alpha = (float)(2 * pc - in) * mean_abs_x;
        bl->alpha[j] += lr * grad_alpha * gy / in;
        if (bl->alpha[j] < 0.001f) bl->alpha[j] = 0.001f;
        if (bl->alpha[j] > 1.0f) bl->alpha[j] = 1.0f;
        bl->bias[j] -= lr * gy;
    }
}

/* ========================================================================
 * Standard Neural Network Operations
 * ======================================================================== */
void layer_norm(float *out, const float *x, const float *w, const float *b, int n) {
    float mean = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float is = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * is * w[i] + b[i];
}

void layer_norm_backward(float *grad_x, const float *grad_y, const float *x,
                         const float *w, float mean, float std_inv, int n) {
    float sum_grad = 0;
    for (int i = 0; i < n; i++) sum_grad += grad_y[i] * w[i] * (x[i] - mean);
    float common = std_inv / n * sum_grad;
    float scale = (1.0f - 1.0f / n);
    for (int i = 0; i < n; i++) grad_x[i] = grad_y[i] * w[i] * std_inv * scale - common;
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

float gelu_grad(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    float t = tanhf(inner);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * 0.7978845608f * (1.0f + 0.134145f * x * x);
}

void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

float cross_entropy_sampled(const float *hidden, const float *wte,
                            int target, int vocab_size, int n_embd,
                            int n_samples, unsigned int *seed) {
    float tl = 0;
    for (int i = 0; i < n_embd; i++) tl += hidden[i] * wte[target * n_embd + i];
    float mx = tl;
    float neg[256];
    for (int k = 0; k < n_samples && k < 256; k++) {
        int v = rand_r(seed) % vocab_size;
        float s = 0;
        for (int i = 0; i < n_embd; i++) s += hidden[i] * wte[v * n_embd + i];
        neg[k] = s;
        if (s > mx) mx = s;
    }
    float se = expf(tl - mx);
    for (int k = 0; k < n_samples && k < 256; k++) se += expf(neg[k] - mx);
    return -logf(expf(tl - mx) / se + 1e-7f);
}

void cross_entropy_grad(float *grad_hidden, const float *hidden, const float *wte,
                        int target, int vocab_size, int n_embd,
                        int n_samples, unsigned int *seed) {
    float tl = 0;
    for (int i = 0; i < n_embd; i++) tl += hidden[i] * wte[target * n_embd + i];
    float mx = tl;
    for (int k = 0; k < n_samples; k++) {
        int v = rand_r(seed) % vocab_size;
        float s = 0;
        for (int i = 0; i < n_embd; i++) s += hidden[i] * wte[v * n_embd + i];
        if (s > mx) mx = s;
    }
    float se = expf(tl - mx);
    for (int k = 0; k < n_samples; k++) se += 1.0f; /* approx */
    float prob = expf(tl - mx) / se;
    float grad_scale = 0.001f;
    for (int i = 0; i < n_embd; i++)
        grad_hidden[i] = (1.0f - prob) * wte[target * n_embd + i] * grad_scale;
}

void clip_array(float *x, int n, float clip_val) {
    for (int i = 0; i < n; i++) {
        if (x[i] > clip_val) x[i] = clip_val;
        if (x[i] < -clip_val) x[i] = -clip_val;
    }
}

void compute_mean_std(const float *x, int n, float *mean, float *std_inv) {
    float m = 0;
    for (int i = 0; i < n; i++) m += x[i];
    m /= n;
    float var = 0;
    for (int i = 0; i < n; i++) { float d = x[i] - m; var += d * d; }
    var /= n;
    *mean = m;
    *std_inv = 1.0f / sqrtf(var + 1e-5f);
}

/* ========================================================================
 * Tensor File Loading (GPW2 format)
 * ======================================================================== */
Tensor *tensor_load_all(const char *path, int *n_tensors) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "GPW2", 4) != 0) { fprintf(stderr, "bad magic\n"); fclose(f); return NULL; }
    fread(n_tensors, 4, 1, f);
    Tensor *t = calloc(*n_tensors, sizeof(Tensor));
    for (int i = 0; i < *n_tensors; i++) {
        int klen;
        fread(&klen, 4, 1, f);
        fread(t[i].key, 1, klen, f);
        t[i].key[klen] = '\0';
        fread(&t[i].ndim, 4, 1, f);
        int n = 1;
        for (int d = 0; d < t[i].ndim; d++) {
            fread(&t[i].shape[d], 4, 1, f);
            n *= t[i].shape[d];
        }
        t[i].data = malloc(n * sizeof(float));
        fread(t[i].data, 4, n, f);
    }
    fclose(f);
    return t;
}

float *tensor_get(Tensor *tensors, int n, const char *key) {
    for (int i = 0; i < n; i++)
        if (strcmp(tensors[i].key, key) == 0) return tensors[i].data;
    fprintf(stderr, "tensor not found: %s\n", key);
    return NULL;
}

void tensor_free_all(Tensor *tensors, int n) {
    for (int i = 0; i < n; i++) free(tensors[i].data);
    free(tensors);
}
