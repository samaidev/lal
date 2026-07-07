/* lal_runtime.h — LAL Universal Runtime
 *
 * A model-agnostic C library for binary neural network inference + training.
 * Any transformer model (GPT-2, BERT, LLaMA, ...) can build on top of this.
 *
 * Core primitives:
 *   - BinaryLinear: binarized weight layer (sign + alpha + bias)
 *   - bin_forward / bin_backward: XNOR+popcount matmul (forward + backward)
 *   - layer_norm / gelu / softmax: standard neural network ops
 *   - ModelConfig: generic model configuration (dim, layers, heads, vocab)
 *
 * Architecture:
 *   lal_runtime.h/c     — this file (model-agnostic)
 *   models/gpt2.c       — GPT-2 specific (model definition + forward + train)
 *   models/bert.c       — BERT (future)
 *   models/llama.c      — LLaMA (future)
 */
#ifndef LAL_RUNTIME_H
#define LAL_RUNTIME_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Model Configuration (generic — any transformer)
 * ======================================================================== */
typedef struct {
    int n_layer;    /* number of transformer layers */
    int n_embd;     /* embedding dimension */
    int n_head;     /* number of attention heads */
    int n_ctx;      /* max context length */
    int vocab_size; /* vocabulary size */
    int mlp_dim;    /* MLP hidden dimension (usually 4 * n_embd) */
} ModelConfig;

/* ========================================================================
 * Binary Weight Layer
 * ======================================================================== */
typedef struct {
    uint64_t *wbits;     /* [out_dim * n_words] — row-major packed sign bits */
    uint64_t *wbits_T;   /* [in_dim * n_words_T] — col-major (transposed) */
    float    *alpha;     /* [out_dim] — per-output scale factor */
    float    *bias;      /* [out_dim] */
    int       in_dim;
    int       out_dim;
    int       n_words;   /* (in_dim + 63) / 64 */
    int       n_words_T; /* (out_dim + 63) / 64 */
} BinLayer;

/* Create a binary layer from float weights */
void bin_layer_init(BinLayer *bl, const float *W, const float *bias,
                    int in_dim, int out_dim);

/* Free a binary layer */
void bin_layer_free(BinLayer *bl);

/* ========================================================================
 * Binary Forward Matmul (XNOR + popcount)
 * y[out_dim] = (2 * popcount(XNOR(x_bits, w_bits)) - in_dim) * alpha + bias
 * ======================================================================== */
void bin_forward(float *y, const float *x, const BinLayer *bl);

/* Float-input binary forward (more accurate, for training) */
void bin_forward_float(float *y, const float *x, const BinLayer *bl);

/* ========================================================================
 * Binary Backward (XNOR + popcount for grad_x, popcount for alpha update)
 * grad_x[in_dim] = backward through binary matmul
 * Also updates alpha and bias in-place
 * ======================================================================== */
void bin_backward(float *grad_x, const float *grad_y, const float *x,
                  BinLayer *bl, float lr);

/* ========================================================================
 * Standard Neural Network Operations
 * ======================================================================== */

/* LayerNorm: out = (x - mean) / std * w + b */
void layer_norm(float *out, const float *x, const float *w, const float *b, int n);

/* LayerNorm backward (with cached mean/std) */
void layer_norm_backward(float *grad_x, const float *grad_y, const float *x,
                         const float *w, float mean, float std_inv, int n);

/* GELU activation (tanh approximation) */
float gelu(float x);

/* GELU gradient */
float gelu_grad(float x);

/* Softmax in-place */
void softmax(float *x, int n);

/* Cross-entropy loss with sampled softmax (for efficient training) */
float cross_entropy_sampled(const float *hidden, const float *wte,
                            int target, int vocab_size, int n_embd,
                            int n_samples, unsigned int *seed);

/* Gradient of cross-entropy w.r.t. hidden (sampled) */
void cross_entropy_grad(float *grad_hidden, const float *hidden, const float *wte,
                        int target, int vocab_size, int n_embd,
                        int n_samples, unsigned int *seed);

/* ========================================================================
 * Tensor File Loading (GPW2 format — model-agnostic)
 * ======================================================================== */
typedef struct {
    char key[128];
    int ndim;
    int shape[4];
    float *data;
} Tensor;

/* Load all tensors from a GPW2 file into memory */
Tensor *tensor_load_all(const char *path, int *n_tensors);

/* Find a tensor by key */
float *tensor_get(Tensor *tensors, int n, const char *key);

/* Free all tensors */
void tensor_free_all(Tensor *tensors, int n);

/* ========================================================================
 * Utility
 * ======================================================================== */
/* Clip values to [-clip, +clip] */
void clip_array(float *x, int n, float clip_val);

/* Compute mean and inverse-std for LayerNorm caching */
void compute_mean_std(const float *x, int n, float *mean, float *std_inv);

#endif /* LAL_RUNTIME_H */
