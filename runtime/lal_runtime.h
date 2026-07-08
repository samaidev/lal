/* lal_runtime.h — LAL Universal Runtime
 *
 * A model-agnostic C library for binary neural network inference + training.
 * Any transformer model (GPT-2, BERT, LLaMA, Qwen, ...) can build on top.
 *
 * Three API levels:
 *   1. Operator level: bin_forward, layer_norm, gelu, softmax (building blocks)
 *   2. Layer level: transformer_layer_forward/backward (one call = one layer)
 *   3. Model level: model_load, model_forward, model_train (full model)
 *
 * Architecture:
 *   lal_runtime.h/c     — this file (model-agnostic, all 3 levels)
 *   models/gpt2.c       — GPT-2 (just config + weight keys, ~30 lines)
 *   models/qwen.c       — Qwen (just config + weight keys, ~30 lines)
 */
#ifndef LAL_RUNTIME_H
#define LAL_RUNTIME_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Level 0: Types & Enums
 * ======================================================================== */

typedef enum {
    NORM_LAYER = 0,   /* GPT-2 style: LayerNorm */
    NORM_RMS    = 1,  /* LLaMA/Qwen style: RMSNorm */
} NormType;

typedef enum {
    ATTN_LEARNED = 0, /* GPT-2: learned positional embeddings */
    ATTN_ROPE    = 1, /* LLaMA/Qwen: Rotary Position Embedding */
} AttnType;

typedef enum {
    ACT_GELU   = 0,   /* GPT-2: GELU */
    ACT_SWIGLU = 1,   /* LLaMA/Qwen: SwiGLU (gate * SiLU(up)) */
    ACT_SILU   = 2,   /* SiLU/Swish */
} ActType;

/* ========================================================================
 * Level 0: Model Configuration
 * ======================================================================== */
typedef struct {
    int n_layer;
    int n_embd;
    int n_head;
    int n_ctx;
    int vocab_size;
    int mlp_dim;       /* MLP hidden dim (GPT-2: 4*embd, LLaMA: ~2.7*embd) */
    NormType norm_type;
    AttnType attn_type;
    ActType  act_type;
    float residual_scale;  /* 0.5 for stability, 1.0 for standard */
    int   qkv_merged;      /* 1 = GPT-2 (c_attn = merged QKV), 0 = LLaMA (separate Q/K/V/O) */
} ModelConfig;

/* ========================================================================
 * Level 1: Binary Weight Layer (operator level)
 * ======================================================================== */
typedef struct {
    uint64_t *wbits;
    uint64_t *wbits_T;
    float    *alpha;
    float    *bias;
    float    *w_float;   /* STE: full-precision weights for gradient update */
    float    *w_core;    /* Logic-guided: CORE weights kept as float (not binarized) */
    uint8_t  *logic_mask; /* Per-output: 0=CORE(float), 1=BINARY(sign+alpha), 2=PRUNE(zero) */
    int       n_core;     /* Count of CORE outputs */
    int       n_prune;    /* Count of PRUNE outputs */
    int       in_dim, out_dim, n_words, n_words_T;
} BinLayer;

void bin_layer_init(BinLayer *bl, const float *W, const float *bias,
                    int in_dim, int out_dim);

/* Logic-guided binarization: initialize with a per-output logic_mask.
 * mask[j]: 0=CORE (keep float in w_core), 1=BINARY (sign+alpha), 2=PRUNE (zero).
 * This implements PHONE's "logic extraction at binarization time":
 * core logic weights stay float, noise is pruned, rest is binarized. */
void bin_layer_init_logic(BinLayer *bl, const float *W, const float *bias,
                          int in_dim, int out_dim, const uint8_t *logic_mask);
void bin_layer_free(BinLayer *bl);

/* bin_forward (BWN, default): x stays float, only W is binarized.
 * Matches Python STE training (tools/train_binary_gpt2.py) so train/inference
 * distributions are aligned. Applies XNOR-Net K-norm scaling
 *   K = ||x||_1 / in_dim
 * to preserve input magnitude information.
 *
 * y[j] = (sum_i sign(W[j,i]) * x[i]) * alpha[j] * K + bias[j] */
void bin_forward(float *y, const float *x, const BinLayer *bl);

/* bin_forward_bnn (legacy fast path): binarizes BOTH x and W via XNOR+popcount.
 * ~64x faster than BWN on long vectors but loses input magnitude → quality
 * collapse after a few layers. Kept as opt-in for max-speed-low-quality mode.
 * Use ONLY when you can prove BNN quality is acceptable for your task. */
void bin_forward_bnn(float *y, const float *x, const BinLayer *bl);

/* bin_forward_float: same as BWN but without K-norm (legacy interface).
 * Kept for backward compat with callers that don't want K scaling. */
void bin_forward_float(float *y, const float *x, const BinLayer *bl);

void bin_backward(float *grad_x, const float *grad_y, const float *x,
                  BinLayer *bl, float lr);

/* STE (Straight-Through Estimator) backward pass.
 * Updates w_float using gradient, then re-binarizes wbits from sign(w_float).
 * This is the key to recovering accuracy lost by binarization:
 *   - Forward uses sign(w) (binary)
 *   - Backward treats sign() as identity, so gradient flows to w_float
 *   - After update, wbits = sign(w_float) is recomputed
 * Call this instead of bin_backward for STE fine-tuning. */
void bin_backward_ste(float *grad_x, const float *grad_y, const float *x,
                      BinLayer *bl, float lr);

/* Global flag: set to 1 to use STE backward in trans_layer_backward.
 * Models can set this before calling model_backward(). */
extern int g_use_ste;
extern int g_use_logic_binarization;  /* norm-based auto logic mask in model_load */

/* Global flag: set to 1 to use the legacy BNN fast path (bin_forward_bnn) in
 * trans_layer_forward instead of the BWN default. Off by default — BNN causes
 * train/inference mismatch and quality collapse. Only enable if you have a
 * very tight latency budget AND can verify quality is still acceptable. */
extern int g_use_bnn_fast_path;

/* Global flag: set to 1 to use real causal multi-head self-attention with
 * KV cache in trans_layer_forward (replaces the degenerate V-copy).
 * Off by default for backward compatibility. When on, Model.k_cache and
 * Model.v_cache must be allocated — call model_kv_cache_alloc() or set the
 * flag before model_load(). */
extern int g_use_real_attention;

/* ========================================================================
 * Level 1: Standard NN Operations (operator level)
 * ======================================================================== */
void layer_norm(float *out, const float *x, const float *w, const float *b, int n);
void rms_norm(float *out, const float *x, const float *w, int n);
void layer_norm_backward(float *grad_x, const float *grad_y, const float *x,
                         const float *w, float mean, float std_inv, int n);
void rms_norm_backward(float *grad_x, const float *grad_y, const float *x,
                       const float *w, int n);
float gelu(float x);
float gelu_grad(float x);
float silu(float x);
float silu_grad(float x);
void softmax(float *x, int n);

/* Universal normalization dispatch (calls LayerNorm or RMSNorm based on type) */
void norm_forward(float *out, const float *x, const float *w, const float *b,
                  NormType type, int n);
void norm_backward(float *grad_x, const float *grad_y, const float *x,
                   const float *w, const float *cached, NormType type, int n);

/* Universal activation dispatch */
float act_forward(float x, ActType type);
float act_grad(float x, ActType type);

/* Cross-entropy (sampled softmax for efficient training) */
float cross_entropy_sampled(const float *hidden, const float *wte,
                            int target, int vocab_size, int n_embd,
                            int n_samples, unsigned int *seed);
void cross_entropy_grad(float *grad_hidden, const float *hidden, const float *wte,
                        int target, int vocab_size, int n_embd,
                        int n_samples, unsigned int *seed);

/* Tensor I/O (GPW2 format) */
typedef struct { char key[128]; int ndim; int shape[4]; float *data; } Tensor;
Tensor *tensor_load_all(const char *path, int *n_tensors);
float *tensor_get(Tensor *tensors, int n, const char *key);
void tensor_free_all(Tensor *tensors, int n);

/* Utility */
void clip_array(float *x, int n, float clip_val);
void compute_mean_std(const float *x, int n, float *mean, float *std_inv);

/* RoPE (Rotary Position Embedding) — for LLaMA/Qwen */
void apply_rope(float *q, float *k, int seq_len, int n_head, int head_dim, int n_embd);

/* ========================================================================
 * Level 2: Transformer Layer (building block — one call = one layer)
 * ======================================================================== */

/* A transformer layer's binary weights */
typedef struct {
    BinLayer attn_q;      /* Q projection (or merged QKV if qkv_merged) */
    BinLayer attn_k;      /* K projection (unused if qkv_merged) */
    BinLayer attn_v;      /* V projection (unused if qkv_merged) */
    BinLayer attn_o;      /* output projection */
    BinLayer mlp_gate;    /* MLP gate (SwiGLU) or c_fc (GELU) */
    BinLayer mlp_up;      /* MLP up (SwiGLU only, unused for GELU) */
    BinLayer mlp_down;    /* MLP down projection */
    float *norm1_w, *norm1_b;  /* first norm weight/bias */
    float *norm2_w, *norm2_b;  /* second norm weight/bias */
    /* KV cache pointers (set by model_load when g_use_real_attention is on).
     * Each points to Model-owned [n_ctx * n_embd] float array.
     * NULL in legacy V-copy mode. */
    float *_kv_k;
    float *_kv_v;
} TransLayer;

/* Per-layer activation cache (for backward) */
typedef struct {
    float *x_pre_norm1;   /* x before first norm */
    float *norm1_out;     /* after first norm */
    float norm1_cache[4]; /* cached mean, std_inv (LN) or just w (RMS) */
    float *q, *k, *v;     /* attention projections */
    float *attn_out;      /* after attention */
    float *proj_out;      /* after output projection */
    float *x_pre_norm2;   /* x before second norm */
    float *norm2_out;     /* after second norm */
    float norm2_cache[4];
    float *mlp_hidden;    /* MLP hidden state (after activation) */
    float *mlp_out;       /* MLP output */
} TransAct;

/* Initialize a transformer layer from tensors (model-agnostic) */
void trans_layer_init(TransLayer *tl, Tensor *tensors, int n_tensors,
                      ModelConfig *cfg, int layer_idx,
                      const char *qkv_key, const char *q_key, const char *k_key,
                      const char *v_key, const char *o_key,
                      const char *gate_key, const char *up_key, const char *down_key,
                      const char *norm1_w_key, const char *norm1_b_key,
                      const char *norm2_w_key, const char *norm2_b_key);

/* Free a transformer layer */
void trans_layer_free(TransLayer *tl, ModelConfig *cfg);

/* Forward: one transformer layer (x modified in-place, activations cached) */
void trans_layer_forward(float *x, TransLayer *tl, TransAct *act,
                         ModelConfig *cfg, int seq_pos);

/* Backward: one transformer layer (updates weights, computes grad_x) */
void trans_layer_backward(float *grad_x, TransLayer *tl, TransAct *act,
                          ModelConfig *cfg, float lr);

/* Allocate/free activation cache */
TransAct *trans_act_alloc(ModelConfig *cfg);
void trans_act_free(TransAct *acts, int n_layer);

/* ========================================================================
 * Level 2: Causal Multi-Head Self-Attention (KV cache)
 * ========================================================================
 * Replaces the degenerate `memcpy(attn_out, v, n)` in trans_layer_forward.
 * Mirrors tools/server/gpt2_server.c:real_attention semantics:
 *   - Stores K, V at current position into per-layer cache
 *   - Multi-head QK^T dot product with 1/sqrt(head_dim) scaling
 *   - Causal mask (only attend to positions 0..seq_pos)
 *   - Softmax with max subtraction (numerical stability)
 *   - Weighted sum of V
 *
 * KV cache is owned by Model (k_cache/v_cache arrays, [n_layer][n_ctx*n_embd]).
 * Allocated in model_load when g_use_real_attention is on, or via
 * model_kv_cache_alloc(). Freed in model_free.
 *
 * Backward: NOT YET IMPLEMENTED. trans_layer_backward still does pass-through
 *   (memcpy g_qkv+2n, g_attn). This means gradient flow through attention
 *   is incorrect — Q, K gradients are zero. Sufficient for inference but
 *   training with attention enabled will not learn Q/K properly. TODO. */
void attention_forward(float *attn_out, const float *qkv,
                       int n_embd, int n_head,
                       int seq_pos,
                       float *k_cache_layer, float *v_cache_layer);

/* Forward decl — model_kv_cache_alloc/free defined after Model struct below */
struct Model;
void model_kv_cache_alloc(struct Model *m);
void model_kv_cache_free(struct Model *m);

/* ========================================================================
 * Level 3: Full Model (highest level — just config + weight keys)
 * ======================================================================== */

typedef struct Model {
    ModelConfig cfg;
    Tensor *tensors;
    int n_tensors;
    TransLayer *layers;
    TransAct *acts;
    float *wte, *wpe;       /* token + position embeddings */
    float *ln_f_w, *ln_f_b; /* final norm */
    float *final_ln;         /* cached final norm output */
    float *x_before_final;   /* cached for backward */
    float final_mean, final_std_inv;
    /* KV cache for real causal attention — [n_layer] pointers, each
     * [n_ctx * n_embd] floats. NULL when g_use_real_attention is 0. */
    float **k_cache;
    float **v_cache;
} Model;
/* Load a model from a GPW2 weight file.
 * key_prefix: "h." for GPT-2, "model.layers." for LLaMA/Qwen
 * This is the ONLY function a new model needs to customize. */
void model_load(Model *m, const char *weight_path, ModelConfig cfg,
                const char *layer_prefix,  /* e.g. "h.%d." or "model.layers.%d." */
                int qkv_merged);            /* 1=GPT-2, 0=LLaMA/Qwen */

/* Forward pass (returns loss) */
float model_forward(Model *m, const int *tokens, int n_tokens);

/* Backward pass (updates all weights) */
void model_backward(Model *m, const int *tokens, int n_tokens, float lr);

/* Free model */
void model_free(Model *m);

#endif /* LAL_RUNTIME_H */
