/* lal_runtime.c — LAL Universal Runtime implementation
 *
 * Three API levels:
 *   Level 1: operators (bin_forward, norm, gelu, etc.)
 *   Level 2: transformer layer (trans_layer_forward/backward)
 *   Level 3: full model (model_load/forward/backward)
 *
 * Models only need Level 3 — just config + weight key patterns.
 */
#include "lal_runtime.h"

/* ========================================================================
 * Level 1 additions: RMSNorm, SiLU, dispatch functions, RoPE
 * ======================================================================== */

void rms_norm(float *out, const float *x, const float *w, int n) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * ms * w[i];
}

void rms_norm_backward(float *grad_x, const float *grad_y, const float *x,
                       const float *w, int n) {
    /* Simplified: pass gradient through (approximate) */
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + 1e-5f);
    for (int i = 0; i < n; i++) grad_x[i] = grad_y[i] * w[i] * ms;
}

float silu(float x) { return x / (1.0f + expf(-x)); }
float silu_grad(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s + x * s * (1.0f - s);
}

void norm_forward(float *out, const float *x, const float *w, const float *b,
                  NormType type, int n) {
    if (type == NORM_RMS) rms_norm(out, x, w, n);
    else layer_norm(out, x, w, b, n);
}

void norm_backward(float *grad_x, const float *grad_y, const float *x,
                   const float *w, const float *cached, NormType type, int n) {
    if (type == NORM_RMS) rms_norm_backward(grad_x, grad_y, x, w, n);
    else layer_norm_backward(grad_x, grad_y, x, w, cached[0], cached[1], n);
}

float act_forward(float x, ActType type) {
    switch (type) {
        case ACT_GELU:   return gelu(x);
        case ACT_SWIGLU: return silu(x);  /* gate * silu(up), caller handles gate */
        case ACT_SILU:   return silu(x);
        default:         return x;
    }
}

float act_grad(float x, ActType type) {
    switch (type) {
        case ACT_GELU:   return gelu_grad(x);
        case ACT_SWIGLU: return silu_grad(x);
        case ACT_SILU:   return silu_grad(x);
        default:         return 1.0f;
    }
}

void apply_rope(float *q, float *k, int seq_len, int n_head, int head_dim, int n_embd) {
    /* Simplified RoPE: rotate pairs by position-dependent angle */
    for (int h = 0; h < n_head; h++) {
        float *qh = q + h * head_dim;
        float *kh = k + h * head_dim;
        for (int d = 0; d < head_dim / 2; d++) {
            float angle = (float)seq_len / powf(10000.0f, (float)(2 * d) / head_dim);
            float c = cosf(angle), s = sinf(angle);
            float q0 = qh[d], q1 = qh[d + head_dim / 2];
            float k0 = kh[d], k1 = kh[d + head_dim / 2];
            qh[d] = q0 * c - q1 * s;
            qh[d + head_dim / 2] = q0 * s + q1 * c;
            kh[d] = k0 * c - k1 * s;
            kh[d + head_dim / 2] = k0 * s + k1 * c;
        }
    }
}

/* ========================================================================
 * Level 2: Transformer Layer (building block)
 * ======================================================================== */

void trans_layer_init(TransLayer *tl, Tensor *tensors, int n_tensors,
                      ModelConfig *cfg, int layer_idx,
                      const char *qkv_key, const char *q_key, const char *k_key,
                      const char *v_key, const char *o_key,
                      const char *gate_key, const char *up_key, const char *down_key,
                      const char *norm1_w_key, const char *norm1_b_key,
                      const char *norm2_w_key, const char *norm2_b_key) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    tl->_kv_k = NULL;
    tl->_kv_v = NULL;
    char full_key[256];

    if (cfg->qkv_merged) {
        /* GPT-2: merged QKV [n → 3n] */
        sprintf(full_key, qkv_key, layer_idx);
        float *W = tensor_get(tensors, n_tensors, full_key);
        sprintf(full_key, "%s.bias", full_key);
        /* Remove ".weight" suffix for bias — actually qkv_key already has .weight */
        /* The key format is like "h.%d.attn.c_attn.weight" */
        char bias_key[256];
        strncpy(bias_key, full_key, sizeof(bias_key));
        /* Replace ".weight" with ".bias" */
        char *dot = strstr(bias_key, ".weight");
        if (dot) { *dot = 0; strcat(bias_key, ".bias"); }
        float *b = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->attn_q, W, b, n, 3 * n);
    } else {
        /* LLaMA/Qwen: separate Q, K, V, O */
        sprintf(full_key, q_key, layer_idx);
        char bias_key[256];
        float *Wq = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        char *dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bq = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->attn_q, Wq, bq, n, n);

        sprintf(full_key, k_key, layer_idx);
        float *Wk = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bk = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->attn_k, Wk, bk, n, n);

        sprintf(full_key, v_key, layer_idx);
        float *Wv = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bv = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->attn_v, Wv, bv, n, n);
    }

    /* Output projection */
    sprintf(full_key, o_key, layer_idx);
    float *Wo = tensor_get(tensors, n_tensors, full_key);
    char bias_key[256]; strncpy(bias_key, full_key, sizeof(bias_key));
    char *dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
    float *bo = tensor_get(tensors, n_tensors, bias_key);
    bin_layer_init(&tl->attn_o, Wo, bo, n, n);

    /* MLP */
    if (cfg->act_type == ACT_SWIGLU) {
        sprintf(full_key, gate_key, layer_idx);
        float *Wg = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bg = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->mlp_gate, Wg, bg, n, m);

        sprintf(full_key, up_key, layer_idx);
        float *Wu = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bu = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->mlp_up, Wu, bu, n, m);
    } else {
        /* GELU: single c_fc */
        sprintf(full_key, gate_key, layer_idx);
        float *Wg = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bg = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->mlp_gate, Wg, bg, n, m);
    }

    sprintf(full_key, down_key, layer_idx);
    float *Wd = tensor_get(tensors, n_tensors, full_key);
    strncpy(bias_key, full_key, sizeof(bias_key));
    dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
    float *bd = tensor_get(tensors, n_tensors, bias_key);
    bin_layer_init(&tl->mlp_down, Wd, bd, m, n);

    /* Norm weights */
    sprintf(full_key, norm1_w_key, layer_idx);
    tl->norm1_w = tensor_get(tensors, n_tensors, full_key);
    sprintf(full_key, norm1_b_key, layer_idx);
    tl->norm1_b = tensor_get(tensors, n_tensors, full_key);
    sprintf(full_key, norm2_w_key, layer_idx);
    tl->norm2_w = tensor_get(tensors, n_tensors, full_key);
    sprintf(full_key, norm2_b_key, layer_idx);
    tl->norm2_b = tensor_get(tensors, n_tensors, full_key);
}

void trans_layer_free(TransLayer *tl, ModelConfig *cfg) {
    bin_layer_free(&tl->attn_q);
    if (!cfg->qkv_merged) { bin_layer_free(&tl->attn_k); bin_layer_free(&tl->attn_v); }
    bin_layer_free(&tl->attn_o);
    bin_layer_free(&tl->mlp_gate);
    if (cfg->act_type == ACT_SWIGLU) bin_layer_free(&tl->mlp_up);
    bin_layer_free(&tl->mlp_down);
}

void trans_layer_forward(float *x, TransLayer *tl, TransAct *act,
                         ModelConfig *cfg, int seq_pos) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    float rs = cfg->residual_scale;
    act->seq_pos = seq_pos;  /* cached for attention backward */

    /* Save x before norm1 */
    memcpy(act->x_pre_norm1, x, n * sizeof(float));
    norm_forward(act->norm1_out, x, tl->norm1_w, tl->norm1_b, cfg->norm_type, n);
    compute_mean_std(act->x_pre_norm1, n, &act->norm1_cache[0], &act->norm1_cache[1]);

    /* Attention */
    if (cfg->qkv_merged) {
        if (g_use_bnn_fast_path) bin_forward_bnn(act->q, act->norm1_out, &tl->attn_q);
        else                     bin_forward     (act->q, act->norm1_out, &tl->attn_q);
        act->k = act->q + n;
        act->v = act->q + 2 * n;
    } else {
        if (g_use_bnn_fast_path) {
            bin_forward_bnn(act->q, act->norm1_out, &tl->attn_q);
            bin_forward_bnn(act->k, act->norm1_out, &tl->attn_k);
            bin_forward_bnn(act->v, act->norm1_out, &tl->attn_v);
        } else {
            bin_forward(act->q, act->norm1_out, &tl->attn_q);
            bin_forward(act->k, act->norm1_out, &tl->attn_k);
            bin_forward(act->v, act->norm1_out, &tl->attn_v);
        }
    }

    if (cfg->attn_type == ATTN_ROPE)
        apply_rope(act->q, act->k, seq_pos, cfg->n_head, n / cfg->n_head, n);

    /* Simplified attention: V copy (full attention in future) */
    memcpy(act->attn_out, act->v, n * sizeof(float));
    /* Attention: real causal multi-head (KV cache) if flag is on and cache
     * is allocated, else legacy V-copy (degenerate, no token mixing).
     * [FIX 致命2] The V-copy was a placeholder — see attention_forward(). */
    if (g_use_real_attention && tl->_kv_k && tl->_kv_v) {
        attention_forward(act->attn_out, act->q, n, cfg->n_head, seq_pos,
                          tl->_kv_k, tl->_kv_v);
    } else {
        /* Legacy V-copy (degenerate, no QK mixing). */
        memcpy(act->attn_out, act->v, n * sizeof(float));
    }
    if (g_use_bnn_fast_path) bin_forward_bnn(act->proj_out, act->attn_out, &tl->attn_o);
    else                     bin_forward     (act->proj_out, act->attn_out, &tl->attn_o);
    for (int i = 0; i < n; i++) x[i] += rs * act->proj_out[i];
    clip_array(x, n, 10.0f);

    /* MLP */
    memcpy(act->x_pre_norm2, x, n * sizeof(float));
    norm_forward(act->norm2_out, x, tl->norm2_w, tl->norm2_b, cfg->norm_type, n);
    compute_mean_std(act->x_pre_norm2, n, &act->norm2_cache[0], &act->norm2_cache[1]);

    if (cfg->act_type == ACT_SWIGLU) {
        /* SwiGLU: hidden = silu(gate(ln2)) * up(ln2) */
        float *gate = malloc(m * sizeof(float));
        float *up = malloc(m * sizeof(float));
        if (g_use_bnn_fast_path) {
            bin_forward_bnn(gate, act->norm2_out, &tl->mlp_gate);
            bin_forward_bnn(up,   act->norm2_out, &tl->mlp_up);
        } else {
            bin_forward(gate, act->norm2_out, &tl->mlp_gate);
            bin_forward(up,   act->norm2_out, &tl->mlp_up);
        }
        for (int i = 0; i < m; i++) act->mlp_hidden[i] = silu(gate[i]) * up[i];
        free(gate); free(up);
    } else {
        /* GELU: hidden = gelu(c_fc(ln2)) */
        if (g_use_bnn_fast_path) bin_forward_bnn(act->mlp_hidden, act->norm2_out, &tl->mlp_gate);
        else                     bin_forward     (act->mlp_hidden, act->norm2_out, &tl->mlp_gate);
        for (int i = 0; i < m; i++) act->mlp_hidden[i] = gelu(act->mlp_hidden[i]);
    }

    if (g_use_bnn_fast_path) bin_forward_bnn(act->mlp_out, act->mlp_hidden, &tl->mlp_down);
    else                     bin_forward     (act->mlp_out, act->mlp_hidden, &tl->mlp_down);
    for (int i = 0; i < n; i++) x[i] += rs * act->mlp_out[i];
    clip_array(x, n, 10.0f);
}

/* Global flag: use STE backward (updates w_float + repacks wbits) */
int g_use_ste = 0;
int g_use_logic_binarization = 0;  /* norm-based auto logic mask in model_load */

/* Auto-generate per-output logic mask based on weight norms.
 * W is [in, out] (GPT-2 Conv1D format). We compute per-output column norms.
 * top 20% → CORE (0), bottom 10% → PRUNE (2), middle 70% → BINARY (1).
 * mask: [out_dim] bytes, 0=CORE, 1=BINARY, 2=PRUNE. */
static void compute_norm_mask(const float *W, int in_dim, int out_dim, uint8_t *mask) {
    /* Compute per-output norms (W is [in, out] row-major) */
    float *norms = malloc(out_dim * sizeof(float));
    for (int j = 0; j < out_dim; j++) {
        float s = 0;
        for (int i = 0; i < in_dim; i++) {
            float w = W[i * out_dim + j];
            s += w * w;
        }
        norms[j] = sqrtf(s);
    }
    /* Find thresholds via partial sort (simple: sort a copy) */
    float *sorted = malloc(out_dim * sizeof(float));
    memcpy(sorted, norms, out_dim * sizeof(float));
    /* Simple insertion sort (out_dim ≤ 3072, OK) */
    for (int i = 1; i < out_dim; i++) {
        float v = sorted[i]; int k = i - 1;
        while (k >= 0 && sorted[k] > v) { sorted[k+1] = sorted[k]; k--; }
        sorted[k+1] = v;
    }
    float core_threshold = sorted[(int)(out_dim * 0.8)];   /* top 20% */
    float prune_threshold = sorted[(int)(out_dim * 0.1)];  /* bottom 10% */

    for (int j = 0; j < out_dim; j++) {
        if (norms[j] >= core_threshold) mask[j] = 0;       /* CORE */
        else if (norms[j] <= prune_threshold) mask[j] = 2; /* PRUNE */
        else mask[j] = 1;                                   /* BINARY */
    }
    free(norms); free(sorted);
}

/* Global flag: use legacy BNN fast path (binarizes x too). Off by default —
 * BNN causes train/inference mismatch. Enable only for max-speed-low-quality. */
int g_use_bnn_fast_path = 0;

/* Global flag: use real causal multi-head self-attention with KV cache.
 * Off by default — backward compat with V-copy. When on, trans_layer_forward
 * calls attention_forward() instead of memcpy(act->attn_out, act->v, n). */
int g_use_real_attention = 0;

void trans_layer_backward(float *grad_x, TransLayer *tl, TransAct *act,
                          ModelConfig *cfg, float lr) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    float rs = cfg->residual_scale;
    static float g_mlp[4096], g_hidden[4096], g_norm2[4096], g_proj[4096];
    static float g_attn[4096], g_qkv[4096*3], g_norm1[4096], g_pre[4096];

    /* Choose backward function based on STE flag */
    #define BIN_BW(gx, gy, x, bl, lr) \
        (g_use_ste ? bin_backward_ste(gx, gy, x, bl, lr) : bin_backward(gx, gy, x, bl, lr))

    /* MLP backward */
    for (int i = 0; i < n; i++) g_mlp[i] = grad_x[i] * rs;
    BIN_BW(g_hidden, g_mlp, act->mlp_hidden, &tl->mlp_down, lr);
    if (cfg->act_type == ACT_SWIGLU) {
        for (int i = 0; i < m; i++) g_hidden[i] *= silu_grad(act->mlp_hidden[i]);
        BIN_BW(g_norm2, g_hidden, act->norm2_out, &tl->mlp_gate, lr);
    } else {
        for (int i = 0; i < m; i++) g_hidden[i] *= gelu_grad(act->mlp_hidden[i]);
        BIN_BW(g_norm2, g_hidden, act->norm2_out, &tl->mlp_gate, lr);
    }
    norm_backward(g_pre, g_norm2, act->x_pre_norm2, tl->norm2_w,
                  act->norm2_cache, cfg->norm_type, n);
    for (int i = 0; i < n; i++) grad_x[i] += g_pre[i] * rs;

    /* Attention backward */
    for (int i = 0; i < n; i++) g_proj[i] = grad_x[i] * rs;
    BIN_BW(g_attn, g_proj, act->attn_out, &tl->attn_o, lr);
    if (g_use_real_attention && tl->_kv_k && tl->_kv_v) {
        /* Real attention: dQ/dK/dV at the current position. act->q is the
         * contiguous [Q|K|V] buffer (k=q+n, v=q+2n), valid for both merged
         * and separate QKV layouts. */
        attention_backward(g_qkv, g_attn, act->q, n, cfg->n_head,
                           act->seq_pos, tl->_kv_k, tl->_kv_v);
    } else {
        /* Legacy V-copy: only V receives gradient (Q, K grad = 0). */
        memset(g_qkv, 0, 3 * n * sizeof(float));
        memcpy(g_qkv + 2 * n, g_attn, n * sizeof(float));
    }
    if (cfg->qkv_merged) {
        /* GPT-2: attn_q is the merged [in→3n] projection. */
        BIN_BW(g_norm1, g_qkv, act->norm1_out, &tl->attn_q, lr);
    } else {
        /* LLaMA/Qwen: separate Q/K/V projections — backprop each. */
        static float g_n1k[4096], g_n1v[4096];
        BIN_BW(g_norm1,  g_qkv,       act->norm1_out, &tl->attn_q, lr);
        BIN_BW(g_n1k,    g_qkv + n,   act->norm1_out, &tl->attn_k, lr);
        BIN_BW(g_n1v,    g_qkv + 2*n, act->norm1_out, &tl->attn_v, lr);
        for (int i = 0; i < n; i++) g_norm1[i] += g_n1k[i] + g_n1v[i];
    }
    norm_backward(g_pre, g_norm1, act->x_pre_norm1, tl->norm1_w,
                  act->norm1_cache, cfg->norm_type, n);
    for (int i = 0; i < n; i++) grad_x[i] += g_pre[i] * rs;

    float gnorm = 0;
    for (int i = 0; i < n; i++) gnorm += grad_x[i] * grad_x[i];
    gnorm = sqrtf(gnorm);
    if (gnorm > 1.0f) { float clip = 1.0f / gnorm; for (int i = 0; i < n; i++) grad_x[i] *= clip; }
}

TransAct *trans_act_alloc(ModelConfig *cfg) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    TransAct *acts = malloc(cfg->n_layer * sizeof(TransAct));
    for (int l = 0; l < cfg->n_layer; l++) {
        acts[l].x_pre_norm1 = malloc(n * sizeof(float));
        acts[l].norm1_out = malloc(n * sizeof(float));
        acts[l].q = malloc(3 * n * sizeof(float));
        /* k/v alias into the contiguous Q|K|V buffer so both merged (GPT-2)
         * and separate (LLaMA/Qwen) paths share one [3n] layout. Previously
         * k/v were left NULL for the separate path → segfault. */
        acts[l].k = acts[l].q + n;
        acts[l].v = acts[l].q + 2 * n;
        acts[l].attn_out = malloc(n * sizeof(float));
        acts[l].proj_out = malloc(n * sizeof(float));
        acts[l].x_pre_norm2 = malloc(n * sizeof(float));
        acts[l].norm2_out = malloc(n * sizeof(float));
        acts[l].mlp_hidden = malloc(m * sizeof(float));
        acts[l].mlp_out = malloc(n * sizeof(float));
    }
    return acts;
}

void trans_act_free(TransAct *acts, int n_layer) {
    for (int l = 0; l < n_layer; l++) {
        free(acts[l].x_pre_norm1); free(acts[l].norm1_out);
        free(acts[l].q); free(acts[l].attn_out); free(acts[l].proj_out);
        free(acts[l].x_pre_norm2); free(acts[l].norm2_out);
        free(acts[l].mlp_hidden); free(acts[l].mlp_out);
    }
    free(acts);
}

/* ========================================================================
 * Level 3: Full Model
 * ======================================================================== */

/* ----- Causal Multi-Head Self-Attention (KV cache) -----
 * Replaces the degenerate V-copy in trans_layer_forward.
 * Mirrors tools/server/gpt2_server.c:real_attention (scalar version).
 *
 * Layout:
 *   qkv:        [3 * n_embd]  — Q | K | V concatenated, single token
 *   k_cache_layer / v_cache_layer: [n_ctx * n_embd] — filled position-by-position
 *   attn_out:   [n_embd]      — output, weighted sum of V across heads
 *
 * Causal: position seq_pos attends only to positions 0..seq_pos (inclusive).
 * Multi-head: n_head heads, head_dim = n_embd / n_head (must divide evenly).
 */
void attention_forward(float *attn_out, const float *qkv,
                       int n_embd, int n_head,
                       int seq_pos,
                       float *k_cache_layer, float *v_cache_layer) {
    int head_dim = n_embd / n_head;
    float scale = 1.0f / sqrtf((float)head_dim);

    const float *Q = qkv;                  /* [n_embd] */
    const float *K_new = qkv + n_embd;
    const float *V_new = qkv + 2 * n_embd;

    /* Store current K, V into cache at position seq_pos */
    memcpy(k_cache_layer + (size_t)seq_pos * n_embd, K_new, n_embd * sizeof(float));
    memcpy(v_cache_layer + (size_t)seq_pos * n_embd, V_new, n_embd * sizeof(float));

    /* Stack scratch — max n_ctx per head. 1024 is GPT-2 default; if n_ctx
     * grows beyond this, switch to malloc. */
    float scores[1024];
    float attn_weights[1024];

    for (int h = 0; h < n_head; h++) {
        const float *Q_h = Q + h * head_dim;
        /* scores[j] = Q_h · K[j, h] * scale, j = 0..seq_pos (causal) */
        float max_score = -1e30f;
        int n_attend = seq_pos + 1;  /* positions 0..seq_pos inclusive */
        if (n_attend > 1024) n_attend = 1024;  /* clip to scratch size */
        for (int j = 0; j < n_attend; j++) {
            const float *K_jh = k_cache_layer + (size_t)j * n_embd + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += Q_h[d] * K_jh[d];
            dot *= scale;
            scores[j] = dot;
            if (dot > max_score) max_score = dot;
        }
        /* Softmax with max subtraction (numerical stability) */
        float sum_exp = 0.0f;
        for (int j = 0; j < n_attend; j++) {
            float e = expf(scores[j] - max_score);
            attn_weights[j] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (int j = 0; j < n_attend; j++) attn_weights[j] *= inv_sum;
        /* Weighted sum: out_h[d] = sum_j attn_weights[j] * V[j, h, d] */
        float *out_h = attn_out + h * head_dim;
        for (int d = 0; d < head_dim; d++) out_h[d] = 0.0f;
        for (int j = 0; j < n_attend; j++) {
            float w = attn_weights[j];
            const float *V_jh = v_cache_layer + (size_t)j * n_embd + h * head_dim;
            for (int d = 0; d < head_dim; d++) out_h[d] += w * V_jh[d];
        }
    }
}

/* ----- Attention backward (dQ/dK/dV) -----
 * Computes gradients for the current token's Q, K, V. Cached K/V at positions
 * 0..seq_pos-1 are treated as constants (they are context, not learned here —
 * only the current token's QKV projection receives gradient, matching the
 * single-position activation cache used by model_forward/backward).
 *
 * Per head h (head_dim d, scale = 1/sqrt(head_dim)):
 *   forward: scores[j]=Q·K_j*scale; w=softmax(scores); out=sum_j w[j]*V_j
 *   backward:
 *     g_w[j]      = <g_out, V_j>                       (grad w.r.t. weight j)
 *     g_scores[j] = w[j] * (g_w[j] - <g_w, w>)         (softmax bwd)
 *     g_Q[d]     += sum_j g_scores[j] * K_j[d] * scale
 *     g_K_cur[d] += g_scores[seq_pos] * Q[d] * scale   (current K only)
 *     g_V_cur[d] += w[seq_pos] * g_out[d]              (current V only)
 */
void attention_backward(float *grad_qkv, const float *grad_attn_out,
                        const float *qkv, int n_embd, int n_head, int seq_pos,
                        const float *k_cache_layer, const float *v_cache_layer) {
    int head_dim = n_embd / n_head;
    float scale = 1.0f / sqrtf((float)head_dim);
    int n_attend = seq_pos + 1;  /* positions 0..seq_pos inclusive */
    if (n_attend > 1024) n_attend = 1024;  /* clip to scratch size */

    const float *Q = qkv;
    float *gQ = grad_qkv;
    float *gK = grad_qkv + n_embd;      /* grad K at current position */
    float *gV = grad_qkv + 2 * n_embd;  /* grad V at current position */
    memset(grad_qkv, 0, 3 * n_embd * sizeof(float));

    /* If the current position fell outside the (clipped) attended window,
     * there is no self-attention gradient to propagate. */
    int have_self = (seq_pos >= 0 && seq_pos < n_attend);

    float scores[1024], w[1024], g_w[1024];

    for (int h = 0; h < n_head; h++) {
        const float *Q_h = Q + h * head_dim;
        const float *g_out_h = grad_attn_out + h * head_dim;

        /* Recompute scores + softmax weights (K is in the cache). */
        float max_score = -1e30f;
        for (int j = 0; j < n_attend; j++) {
            const float *K_jh = k_cache_layer + (size_t)j * n_embd + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += Q_h[d] * K_jh[d];
            dot *= scale;
            scores[j] = dot;
            if (dot > max_score) max_score = dot;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < n_attend; j++) {
            float e = expf(scores[j] - max_score);
            w[j] = e; sum_exp += e;
        }
        float inv = 1.0f / (sum_exp + 1e-12f);
        for (int j = 0; j < n_attend; j++) w[j] *= inv;

        /* g_w[j] = <g_out, V_j>; dot_gw_w = <g_w, w> */
        float dot_gw_w = 0.0f;
        for (int j = 0; j < n_attend; j++) {
            const float *V_jh = v_cache_layer + (size_t)j * n_embd + h * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) s += g_out_h[d] * V_jh[d];
            g_w[j] = s;
            dot_gw_w += w[j] * s;
        }
        /* g_scores[j] = w[j] * (g_w[j] - dot_gw_w)  (now reuse g_w buffer) */
        for (int j = 0; j < n_attend; j++) g_w[j] = w[j] * (g_w[j] - dot_gw_w);

        /* g_Q[d] += sum_j g_scores[j] * K_j[d] * scale */
        float *gQ_h = gQ + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float s = 0.0f;
            for (int j = 0; j < n_attend; j++) {
                const float *K_jh = k_cache_layer + (size_t)j * n_embd + h * head_dim;
                s += g_w[j] * K_jh[d];
            }
            gQ_h[d] += s * scale;
        }

        if (have_self) {
            /* g_K_cur[d] += g_scores[seq_pos] * Q[d] * scale  (current K only) */
            float gs_cur = g_w[seq_pos] * scale;
            float *gK_h = gK + h * head_dim;
            for (int d = 0; d < head_dim; d++) gK_h[d] += gs_cur * Q_h[d];
            /* g_V_cur[d] += w[seq_pos] * g_out[d]  (current V only) */
            float w_cur = w[seq_pos];
            float *gV_h = gV + h * head_dim;
            for (int d = 0; d < head_dim; d++) gV_h[d] += w_cur * g_out_h[d];
        }
    }
}

void model_kv_cache_alloc(Model *m) {
    if (m->k_cache) return;  /* idempotent */
    int n_layer = m->cfg.n_layer;
    size_t per_layer = (size_t)m->cfg.n_ctx * m->cfg.n_embd * sizeof(float);
    m->k_cache = calloc(n_layer, sizeof(float *));
    m->v_cache = calloc(n_layer, sizeof(float *));
    for (int l = 0; l < n_layer; l++) {
        m->k_cache[l] = calloc(1, per_layer);
        m->v_cache[l] = calloc(1, per_layer);
        /* Wire into TransLayer so trans_layer_forward can find them */
        if (m->layers) {
            m->layers[l]._kv_k = m->k_cache[l];
            m->layers[l]._kv_v = m->v_cache[l];
        }
    }
}

void model_kv_cache_free(Model *m) {
    if (!m->k_cache) return;
    for (int l = 0; l < m->cfg.n_layer; l++) {
        free(m->k_cache[l]);
        free(m->v_cache[l]);
    }
    free(m->k_cache);
    free(m->v_cache);
    m->k_cache = NULL;
    m->v_cache = NULL;
}

void model_load(Model *m, const char *weight_path, ModelConfig cfg,
                const char *layer_prefix, int qkv_merged) {
    m->cfg = cfg;
    m->cfg.qkv_merged = qkv_merged;
    m->tensors = tensor_load_all(weight_path, &m->n_tensors);
    if (!m->tensors) { fprintf(stderr, "failed to load %s\n", weight_path); exit(1); }
    printf("[*] loaded %d tensors\n", m->n_tensors);

    m->wte = tensor_get(m->tensors, m->n_tensors, "wte.weight");
    m->wpe = (cfg.attn_type == ATTN_LEARNED)
        ? tensor_get(m->tensors, m->n_tensors, "wpe.weight") : NULL;
    m->ln_f_w = tensor_get(m->tensors, m->n_tensors, "ln_f.weight");
    m->ln_f_b = tensor_get(m->tensors, m->n_tensors, "ln_f.bias");

    printf("[*] binarizing %d layers%s...\n", cfg.n_layer,
           g_use_logic_binarization ? " (logic-guided)" : "");
    m->layers = malloc(cfg.n_layer * sizeof(TransLayer));
    m->acts = trans_act_alloc(&cfg);

    /* Build keys and binarize each layer */
    char key[256], bk[256];
    for (int l = 0; l < cfg.n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        int n = cfg.n_embd, mm = cfg.mlp_dim;

        /* Helper: bin_layer_init or bin_layer_init_logic depending on flag */
        #define BIN_INIT(bl, W, b, in, out) do { \
            if (g_use_logic_binarization) { \
                uint8_t *mask = malloc(out); \
                compute_norm_mask(W, in, out, mask); \
                bin_layer_init_logic(bl, W, b, in, out, mask); \
                free(mask); \
            } else { \
                bin_layer_init(bl, W, b, in, out); \
            } \
        } while(0)

        if (qkv_merged) {
            sprintf(key, "h.%d.attn.c_attn.weight", l);
            char bk[256]; strncpy(bk, key, sizeof(bk));
            char *dot = strstr(bk, ".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->attn_q, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, 3*n);
        } else {
            sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l);
            char bk[256]; strncpy(bk, key, sizeof(bk));
            char *dot = strstr(bk, ".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->attn_q, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, n);
            sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->attn_k, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, n);
            sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->attn_v, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, n);
        }

        sprintf(key, qkv_merged ? "h.%d.attn.c_proj.weight" : "model.layers.%d.self_attn.o_proj.weight", l);
        char bk[256]; strncpy(bk, key, sizeof(bk));
        char *dot = strstr(bk, ".weight"); if(dot){*dot=0;strcat(bk,".bias");}
        BIN_INIT(&tl->attn_o, tensor_get(m->tensors, m->n_tensors, key),
                 tensor_get(m->tensors, m->n_tensors, bk), n, n);

        if (cfg.act_type == ACT_SWIGLU) {
            sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->mlp_gate, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, mm);
            sprintf(key, "model.layers.%d.mlp.up_proj.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->mlp_up, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, mm);
        } else {
            sprintf(key, "h.%d.mlp.c_fc.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->mlp_gate, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, mm);
        }

        sprintf(key, qkv_merged ? "h.%d.mlp.c_proj.weight" : "model.layers.%d.mlp.down_proj.weight", l);
        strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
        BIN_INIT(&tl->mlp_down, tensor_get(m->tensors, m->n_tensors, key),
                 tensor_get(m->tensors, m->n_tensors, bk), mm, n);
        #undef BIN_INIT

        /* Norm weights */
        if (qkv_merged) {
            sprintf(key, "h.%d.ln_1.weight", l); tl->norm1_w = tensor_get(m->tensors, m->n_tensors, key);
            sprintf(key, "h.%d.ln_1.bias", l); tl->norm1_b = tensor_get(m->tensors, m->n_tensors, key);
            sprintf(key, "h.%d.ln_2.weight", l); tl->norm2_w = tensor_get(m->tensors, m->n_tensors, key);
            sprintf(key, "h.%d.ln_2.bias", l); tl->norm2_b = tensor_get(m->tensors, m->n_tensors, key);
        } else {
            sprintf(key, "model.layers.%d.input_layernorm.weight", l); tl->norm1_w = tensor_get(m->tensors, m->n_tensors, key);
            tl->norm1_b = NULL;
            sprintf(key, "model.layers.%d.post_attention_layernorm.weight", l); tl->norm2_w = tensor_get(m->tensors, m->n_tensors, key);
            tl->norm2_b = NULL;
        }
    }
    printf("[*] done\n");

    m->final_ln = malloc(cfg.n_embd * sizeof(float));
    m->x_before_final = malloc(cfg.n_embd * sizeof(float));
    m->k_cache = NULL;
    m->v_cache = NULL;
    /* Auto-allocate KV cache if real attention is requested at load time.
     * Callers can also call model_kv_cache_alloc() later to enable it. */
    if (g_use_real_attention) model_kv_cache_alloc(m);
}

float model_forward(Model *m, const int *tokens, int n_tokens) {
    int n = m->cfg.n_embd;
    int nL = m->cfg.n_layer;
    static float x[4096];
    int t = n_tokens - 1;  /* predict tokens[t+1] from context tokens[0..t] */

    if (g_use_real_attention && m->k_cache) {
        /* Real attention needs the KV cache for positions 0..t to hold THIS
         * sequence's keys/values. The cache persists across training steps
         * (different sentences), so (re)fill 0..t-1 as context before running
         * the target position t. Context positions are run through all layers
         * with a scratch activation buffer (not stored); only their K/V land
         * in the cache and are treated as constants during backward. */
        static float xc[4096];
        static TransAct *scratch = NULL;
        if (!scratch) scratch = trans_act_alloc(&m->cfg);
        for (int p = 0; p < t; p++) {
            for (int i = 0; i < n; i++) {
                xc[i] = m->wte[tokens[p] * n + i];
                if (m->wpe) xc[i] += m->wpe[p * n + i];
            }
            for (int l = 0; l < nL; l++)
                trans_layer_forward(xc, &m->layers[l], &scratch[l], &m->cfg, p);
        }
    }

    for (int i = 0; i < n; i++) {
        x[i] = m->wte[tokens[t] * n + i];
        if (m->wpe) x[i] += m->wpe[t * n + i];
    }

    for (int l = 0; l < nL; l++)
        trans_layer_forward(x, &m->layers[l], &m->acts[l], &m->cfg, t);

    memcpy(m->x_before_final, x, n * sizeof(float));
    norm_forward(m->final_ln, x, m->ln_f_w, m->ln_f_b, m->cfg.norm_type, n);
    compute_mean_std(m->x_before_final, n, &m->final_mean, &m->final_std_inv);

    int target = tokens[n_tokens];
    unsigned int seed = 42;
    return cross_entropy_sampled(m->final_ln, m->wte, target, m->cfg.vocab_size, n, 100, &seed);
}

void model_backward(Model *m, const int *tokens, int n_tokens, float lr) {
    int n = m->cfg.n_embd;
    int target = tokens[n_tokens];
    static float gh[4096];
    unsigned int seed = 42;
    cross_entropy_grad(gh, m->final_ln, m->wte, target, m->cfg.vocab_size, n, 100, &seed);
    float gnorm = 0;
    for (int i = 0; i < n; i++) gnorm += gh[i] * gh[i];
    gnorm = sqrtf(gnorm);
    if (gnorm > 0.1f) { float clip = 0.1f / gnorm; for (int i = 0; i < n; i++) gh[i] *= clip; }

    static float g_pre[4096];
    norm_backward(g_pre, gh, m->x_before_final, m->ln_f_w,
                  (float[]){m->final_mean, m->final_std_inv}, m->cfg.norm_type, n);
    memcpy(gh, g_pre, n * sizeof(float));

    for (int l = m->cfg.n_layer - 1; l >= 0; l--)
        trans_layer_backward(gh, &m->layers[l], &m->acts[l], &m->cfg, lr);
}

void model_free(Model *m) {
    for (int l = 0; l < m->cfg.n_layer; l++)
        trans_layer_free(&m->layers[l], &m->cfg);
    free(m->layers);
    trans_act_free(m->acts, m->cfg.n_layer);
    free(m->final_ln);
    free(m->x_before_final);
    model_kv_cache_free(m);
    tensor_free_all(m->tensors, m->n_tensors);
}

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
    bl->w_float = malloc((size_t)in_dim * out_dim * sizeof(float));  /* STE */

    /* Copy float weights for STE updates — TRANSPOSE to [out, in] layout!
     * W is [in, out] row-major (GPT-2 Conv1D format). We store w_float as
     * [out, in] so that w_float[j*in + i] is contiguous per output j.
     * This makes repack/alpha/update loops all contiguous → SIMD-friendly. */
    for (int j = 0; j < out_dim; j++)
        for (int i = 0; i < in_dim; i++)
            bl->w_float[j * in_dim + i] = W[i * out_dim + j];

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

/* Logic-guided binarization: initialize with per-output logic_mask.
 * mask[j]: 0=CORE (keep float), 1=BINARY (sign+alpha), 2=PRUNE (zero).
 *
 * This implements PHONE's "logic extraction at binarization time":
 * - CORE outputs: weights stored as float in w_core, NOT binarized
 * - BINARY outputs: sign(w) packed into wbits, alpha = mean(|w|)
 * - PRUNE outputs: wbits all zero, alpha=0, bias=0 (effectively removed)
 *
 * The forward pass (bin_forward) checks logic_mask per output:
 * - CORE: y[j] = x @ w_core[j] (float matmul, no binarization)
 * - BINARY: y[j] = alpha * (2*popcount - N) + bias (XNOR+popcount)
 * - PRUNE: y[j] = 0 (skipped entirely)
 */
void bin_layer_init_logic(BinLayer *bl, const float *W, const float *bias,
                          int in_dim, int out_dim, const uint8_t *logic_mask) {
    bl->in_dim = in_dim;
    bl->out_dim = out_dim;
    bl->n_words = (in_dim + 63) / 64;
    bl->n_words_T = (out_dim + 63) / 64;
    bl->wbits = calloc(out_dim * bl->n_words, sizeof(uint64_t));
    bl->wbits_T = calloc(in_dim * bl->n_words_T, sizeof(uint64_t));
    bl->alpha = calloc(out_dim, sizeof(float));
    bl->bias = bias ? malloc(out_dim * sizeof(float)) : calloc(out_dim, sizeof(float));
    bl->w_float = malloc((size_t)out_dim * in_dim * sizeof(float));
    bl->w_core = NULL;
    bl->logic_mask = NULL;
    bl->n_core = 0;
    bl->n_prune = 0;

    if (!logic_mask) {
        /* No logic mask → use standard binarization (all BINARY) */
        bin_layer_init(bl, W, bias, in_dim, out_dim);
        return;
    }

    /* Copy logic mask + count categories */
    bl->logic_mask = malloc(out_dim);
    memcpy(bl->logic_mask, logic_mask, out_dim);
    for (int j = 0; j < out_dim; j++) {
        if (logic_mask[j] == 0) bl->n_core++;
        else if (logic_mask[j] == 2) bl->n_prune++;
    }

    /* Allocate w_core for CORE outputs (float weights, [n_core, in_dim]) */
    if (bl->n_core > 0) {
        bl->w_core = malloc((size_t)bl->n_core * in_dim * sizeof(float));
    }

    /* Process each output based on its logic category */
    int core_idx = 0;
    for (int j = 0; j < out_dim; j++) {
        const float *wj = &W[j * in_dim];  /* W is [out, in] (transposed) */

        switch (logic_mask[j]) {
        case 0: /* CORE: keep float */
            memcpy(&bl->w_core[core_idx * in_dim], wj, in_dim * sizeof(float));
            bl->alpha[j] = 0.0f;  /* not used for CORE */
            if (bias) bl->bias[j] = bias[j];
            /* wbits for CORE: all zero (not used, but keep for indexing) */
            core_idx++;
            break;

        case 1: /* BINARY: sign(w) + alpha */
            {
                float abs_sum = 0;
                for (int i = 0; i < in_dim; i++) abs_sum += fabsf(wj[i]);
                bl->alpha[j] = abs_sum / in_dim;
                if (bias) bl->bias[j] = bias[j];
                for (int wi = 0; wi < bl->n_words; wi++) {
                    uint64_t word = 0;
                    for (int bi = 0; bi < 64; bi++) {
                        int idx = wi * 64 + bi;
                        if (idx < in_dim && wj[idx] > 0.0f) word |= (1ULL << bi);
                    }
                    bl->wbits[j * bl->n_words + wi] = word;
                }
            }
            break;

        case 2: /* PRUNE: zero out */
            bl->alpha[j] = 0.0f;
            bl->bias[j] = 0.0f;
            /* wbits already zero from calloc */
            break;
        }

        /* Copy to w_float (transposed [out, in] for STE compatibility) */
        memcpy(&bl->w_float[j * in_dim], wj, in_dim * sizeof(float));
    }

    /* Build wbits_T (transposed) only for BINARY outputs */
    for (int i = 0; i < in_dim; i++) {
        for (int wi = 0; wi < bl->n_words_T; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int j = wi * 64 + bi;
                if (j < out_dim && logic_mask[j] == 1 && W[j * in_dim + i] > 0.0f)
                    word |= (1ULL << bi);
            }
            bl->wbits_T[i * bl->n_words_T + wi] = word;
        }
    }
}

void bin_layer_free(BinLayer *bl) {
    free(bl->wbits); free(bl->wbits_T); free(bl->alpha); free(bl->bias);
    free(bl->w_float); free(bl->w_core); free(bl->logic_mask);
    bl->wbits = NULL; bl->wbits_T = NULL; bl->alpha = NULL; bl->bias = NULL;
    bl->w_float = NULL; bl->w_core = NULL; bl->logic_mask = NULL;
}

/* Re-pack wbits and wbits_T from sign(w_float).
 * w_float is [out, in] (transposed from Conv1D's [in, out] for contiguous
 * per-output access). All loops here are now contiguous → auto-vectorizable.
 *
 * Key optimization: wbits[j] packs sign(w_float[j*in + 0..in-1]) which is
 * contiguous memory. The compiler auto-vectorizes the 8x unrolled comparison
 * into SIMD compare + movemask-style bit extraction. */
static void bin_layer_repack(BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim;

    /* Pack wbits[j][wi] from sign(w_float[j*in + i]) — CONTIGUOUS in i! */
    for (int j = 0; j < out; j++) {
        const float *wf = &bl->w_float[j * in];  /* contiguous [in] */
        for (int wi = 0; wi < bl->n_words; wi++) {
            uint64_t word = 0;
            int base = wi * 64;
            for (int grp = 0; grp < 8; grp++) {
                int idx = base + grp * 8;
                if (idx + 7 < in) {
                    /* 8 contiguous floats — compiler auto-vectorizes to SIMD */
                    if (wf[idx+0] > 0.0f) word |= (1ULL << (grp*8 + 0));
                    if (wf[idx+1] > 0.0f) word |= (1ULL << (grp*8 + 1));
                    if (wf[idx+2] > 0.0f) word |= (1ULL << (grp*8 + 2));
                    if (wf[idx+3] > 0.0f) word |= (1ULL << (grp*8 + 3));
                    if (wf[idx+4] > 0.0f) word |= (1ULL << (grp*8 + 4));
                    if (wf[idx+5] > 0.0f) word |= (1ULL << (grp*8 + 5));
                    if (wf[idx+6] > 0.0f) word |= (1ULL << (grp*8 + 6));
                    if (wf[idx+7] > 0.0f) word |= (1ULL << (grp*8 + 7));
                } else {
                    for (int bi = 0; bi < 8; bi++) {
                        int i = idx + bi;
                        if (i < in && wf[i] > 0.0f)
                            word |= (1ULL << (grp * 8 + bi));
                    }
                }
            }
            bl->wbits[j * bl->n_words + wi] = word;
        }
    }

    /* Skip wbits_T repack in STE mode — grad_x is now computed from w_float
     * directly (float arithmetic), so wbits_T is never read during STE training.
     * This saves the strided wbits_T repack loop (the slowest part). */
    for (int j = 0; j < out; j++) {
        const float *wf = &bl->w_float[j * in];  /* contiguous [in] */
        float abs_sum = 0;
        for (int i = 0; i + 7 < in; i += 8) {
            abs_sum += fabsf(wf[i+0]) + fabsf(wf[i+1]) + fabsf(wf[i+2]) + fabsf(wf[i+3]);
            abs_sum += fabsf(wf[i+4]) + fabsf(wf[i+5]) + fabsf(wf[i+6]) + fabsf(wf[i+7]);
        }
        for (int i = (in / 8) * 8; i < in; i++)
            abs_sum += fabsf(wf[i]);
        bl->alpha[j] = abs_sum / in;
    }
}

/* ========================================================================
 * Binary Forward — BWN (default, matches Python STE training)
 * ========================================================================
 * x stays float. Only W is binarized (sign(W) * alpha).
 * Adds XNOR-Net K-norm: K = ||x||_1 / in_dim, preserves input magnitude.
 *
 *   y[j] = (sum_i sign(W[j,i]) * x[i]) * alpha[j] * K + bias[j]
 *
 * This is the mathematically correct BWN forward. The old bin_forward was
 * BNN (binarized x too) which diverged from training and caused quality
 * collapse. BNN is retained as bin_forward_bnn() for opt-in fast mode.
 * ======================================================================== */
void bin_forward(float *y, const float *x, const BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;

    /* Logic-guided: if logic_mask exists, dispatch per-output */
    if (bl->logic_mask) {
        /* K-norm for BINARY outputs */
        float abs_sum = 0.0f;
        for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
        float K = abs_sum / in;

        static float sign_lut[256][8];
        static int lut_init = 0;
        if (!lut_init) {
            for (int b = 0; b < 256; b++)
                for (int i = 0; i < 8; i++)
                    sign_lut[b][i] = (b >> i) & 1 ? 1.0f : -1.0f;
            lut_init = 1;
        }

        int core_idx = 0;
        for (int j = 0; j < out; j++) {
            switch (bl->logic_mask[j]) {
            case 0: { /* CORE: float matmul */
                const float *wc = &bl->w_core[core_idx * in];
                float s = 0.0f;
                for (int i = 0; i + 7 < in; i += 8) {
                    s += x[i+0]*wc[i+0] + x[i+1]*wc[i+1] + x[i+2]*wc[i+2] + x[i+3]*wc[i+3];
                    s += x[i+4]*wc[i+4] + x[i+5]*wc[i+5] + x[i+6]*wc[i+6] + x[i+7]*wc[i+7];
                }
                for (int i = (in/8)*8; i < in; i++) s += x[i] * wc[i];
                y[j] = s + bl->bias[j];
                core_idx++;
                break;
            }
            case 1: { /* BINARY: sign(w) * alpha * K + bias */
                const uint64_t *wb = &bl->wbits[j * nw];
                float s = 0.0f;
                for (int wi = 0; wi < nw; wi++) {
                    uint64_t w = wb[wi];
                    int base = wi * 64;
                    for (int bi = 0; bi < 8; bi++) {
                        int idx = base + bi * 8;
                        uint8_t byte = (uint8_t)((w >> (bi * 8)) & 0xFF);
                        const float *sw = sign_lut[byte];
                        if (idx + 7 < in) {
                            s += x[idx+0]*sw[0] + x[idx+1]*sw[1] + x[idx+2]*sw[2] + x[idx+3]*sw[3];
                            s += x[idx+4]*sw[4] + x[idx+5]*sw[5] + x[idx+6]*sw[6] + x[idx+7]*sw[7];
                        } else {
                            for (int k = 0; k < 8; k++) {
                                int i = idx + k;
                                if (i < in) s += x[i] * sw[k];
                            }
                        }
                    }
                }
                y[j] = s * bl->alpha[j] * K + bl->bias[j];
                break;
            }
            default: /* PRUNE: zero */
                y[j] = 0.0f;
                break;
            }
        }
        return;
    }

    /* Standard BWN path (no logic_mask) */
    float abs_sum = 0.0f;
    for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
    float K = abs_sum / in;

    static float sign_lut[256][8];
    static int lut_init = 0;
    if (!lut_init) {
        for (int b = 0; b < 256; b++)
            for (int i = 0; i < 8; i++)
                sign_lut[b][i] = (b >> i) & 1 ? 1.0f : -1.0f;
        lut_init = 1;
    }

    for (int j = 0; j < out; j++) {
        const uint64_t *wb = &bl->wbits[j * nw];
        float s = 0.0f;
        for (int wi = 0; wi < nw; wi++) {
            uint64_t w = wb[wi];
            int base = wi * 64;
            /* Process 8 bytes (8×8=64 bits) per word, 8 floats at a time */
            for (int bi = 0; bi < 8; bi++) {
                int idx = base + bi * 8;
                uint8_t byte = (uint8_t)((w >> (bi * 8)) & 0xFF);
                const float *sw = sign_lut[byte];
                if (idx + 7 < in) {
                    /* 8x unrolled dot product — auto-vectorizes to SIMD */
                    s += x[idx+0] * sw[0];
                    s += x[idx+1] * sw[1];
                    s += x[idx+2] * sw[2];
                    s += x[idx+3] * sw[3];
                    s += x[idx+4] * sw[4];
                    s += x[idx+5] * sw[5];
                    s += x[idx+6] * sw[6];
                    s += x[idx+7] * sw[7];
                } else {
                    /* Tail: handle remaining elements (< 8) */
                    for (int k = 0; k < 8; k++) {
                        int i = idx + k;
                        if (i < in) s += x[i] * sw[k];
                    }
                }
            }
        }
        y[j] = s * bl->alpha[j] * K + bl->bias[j];
    }
}

/* BNN fast path: XNOR + popcount, binarizes BOTH x and W.
 * ~64x faster than BWN. With K-norm input scaling (XNOR-Net, Rastegari 2016),
 * the output magnitude is restored: y = (2*pc-in) * alpha * K + bias, where
 * K = mean(|x|). Without K, outputs have wrong magnitude → garbled generation. */
void bin_forward_bnn(float *y, const float *x, const BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;

    /* Compute input scale K = mean(|x|) — restores magnitude lost by sign(x).
     * O(in) cost is negligible vs the O(in*out) XNOR+popcount matmul. */
    float abs_sum = 0.0f;
    for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
    float K = abs_sum / in;

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
    /* XNOR + popcount per output, scaled by alpha * K */
    for (int j = 0; j < out; j++) {
        int pc = 0;
        const uint64_t *wb = &bl->wbits[j * nw];
        for (int wi = 0; wi < nw; wi++)
            pc += __builtin_popcountll(~(xbits[wi] ^ wb[wi]));
        y[j] = (float)(2 * pc - in) * bl->alpha[j] * K + bl->bias[j];
    }
}

/* Legacy bin_forward_float: BWN without K-norm. Kept for callers that
 * explicitly don't want input magnitude scaling. */
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

    /* Logic-guided: if logic_mask exists, dispatch per-output.
     * Without this, the non-logic path uses mean_alpha = sum(alpha)/out,
     * but PRUNE (alpha=0) and CORE (alpha=0) dilute mean_alpha → wrong
     * grad_x → NaN divergence. This was the root cause of ai_4116f587's
     * step-100 NaN in --logic + --real-attention testing.
     *   CORE: grad_x += grad_y * w_core (proper float gradient)
     *   BINARY: grad_x += grad_y * sign(wbits) * alpha (original logic)
     *   PRUNE: skip (zero gradient, output is zeroed in forward) */
    if (bl->logic_mask) {
        for (int i = 0; i < in; i++) grad_x[i] = 0.0f;
        int core_idx = 0;
        for (int j = 0; j < out; j++) {
            float gy = grad_y[j];
            if (bl->logic_mask[j] == 0) {
                /* CORE: float gradient through w_core */
                if (fabsf(gy) >= 1e-8f) {
                    const float *wc = &bl->w_core[core_idx * in];
                    for (int i = 0; i + 7 < in; i += 8) {
                        grad_x[i+0] += gy * wc[i+0];
                        grad_x[i+1] += gy * wc[i+1];
                        grad_x[i+2] += gy * wc[i+2];
                        grad_x[i+3] += gy * wc[i+3];
                        grad_x[i+4] += gy * wc[i+4];
                        grad_x[i+5] += gy * wc[i+5];
                        grad_x[i+6] += gy * wc[i+6];
                        grad_x[i+7] += gy * wc[i+7];
                    }
                    for (int i = (in/8)*8; i < in; i++) grad_x[i] += gy * wc[i];
                }
                core_idx++;
                bl->bias[j] -= lr * gy;
            } else if (bl->logic_mask[j] == 1) {
                /* BINARY: gradient through sign(wbits) * alpha */
                if (fabsf(gy) >= 1e-8f) {
                    const uint64_t *wb = &bl->wbits[j * bl->n_words];
                    float scale = gy * bl->alpha[j];
                    for (int wi = 0; wi < bl->n_words; wi++) {
                        uint64_t w = wb[wi];
                        int base = wi * 64;
                        for (int bi = 0; bi < 8; bi++) {
                            int idx = base + bi * 8;
                            if (idx + 7 < in) {
                                grad_x[idx+0] += scale * ((w >> (bi*8+0)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+1] += scale * ((w >> (bi*8+1)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+2] += scale * ((w >> (bi*8+2)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+3] += scale * ((w >> (bi*8+3)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+4] += scale * ((w >> (bi*8+4)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+5] += scale * ((w >> (bi*8+5)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+6] += scale * ((w >> (bi*8+6)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+7] += scale * ((w >> (bi*8+7)) & 1 ? 1.0f : -1.0f);
                            } else {
                                for (int k = 0; k < 8; k++) {
                                    int i = idx + k;
                                    if (i < in) grad_x[i] += scale * ((w >> (bi*8+k)) & 1 ? 1.0f : -1.0f);
                                }
                            }
                        }
                    }
                }
                bl->bias[j] -= lr * gy;
            }
            /* PRUNE (case 2): no gradient, skip entirely */
        }
        return;
    }

    /* Non-logic path: original bin_backward */

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

    /* Part 2: alpha + bias update via popcount (reuse x_bits)
     *
     * [FIX 严重4] alpha update direction: was '+=', now '-'= to match bias.
     *   Old code: bl->alpha[j] += lr * grad_alpha * gy / in;   (WRONG: ascends loss)
     *   New code: bl->alpha[j] -= lr * grad_alpha * gy;         (correct: descends loss)
     * Also dropped spurious '/in' that shrank alpha's effective LR by in_dim.
     * [FIX 严重6] Removed alpha clamp to [0.001, 1.0] — it prevented alpha from
     *   converging to its natural magnitude and forced a fake floor. */
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
        bl->alpha[j] -= lr * grad_alpha * gy;   /* FIXED: direction + no /in */
        /* Removed: if (bl->alpha[j] < 0.001f) bl->alpha[j] = 0.001f;
         *         if (bl->alpha[j] > 1.0f)   bl->alpha[j] = 1.0f; */
        if (bl->alpha[j] < 0.0f) bl->alpha[j] = 0.0f;  /* only non-negativity */
        bl->bias[j] -= lr * gy;
    }
}

/* STE (Straight-Through Estimator) backward pass.
 *
 * Key difference from bin_backward: this updates w_float (the full-precision
 * weights) using the gradient, treating sign() as identity. After the update,
 * wbits is re-packed from sign(w_float) via bin_layer_repack().
 *
 * This allows the binary weights to actually change during training, which
 * is impossible with bin_backward (it only updates alpha and bias).
 *
 * STE gradient: d(loss)/d(w_float) = d(loss)/d(sign(w)) * d(sign(w))/d(w)
 *                                  = grad_y * x * 1  (STE: sign'(w) = 1)
 * So: w_float[i,j] -= lr * grad_y[j] * x[i]
 *
 * Memory note: w_float is [in_dim, out_dim] row-major, same as input W.
 * This adds ~in*out*4 bytes per layer (e.g. 768*2304*4 = 7MB for c_attn).
 * Total for 12 layers × 4 matrices ≈ 339 MB extra during training.
 * For inference, w_float can be freed (set to NULL after training). */
void bin_backward_ste(float *grad_x, const float *grad_y, const float *x,
                      BinLayer *bl, float lr) {
    int in = bl->in_dim, out = bl->out_dim;

    /* Part 1: grad_x computation.
     * In STE mode, skip wbits_T repack entirely — compute grad_x directly
     * from w_float using float arithmetic. This avoids the strided wbits_T
     * repack (50% of repack cost) at the expense of float mul-adds.
     *
     * grad_x[i] = sum_j grad_y[j] * sign(w_float[j*in+i]) * alpha[j]
     *
     * w_float is [out, in], so w_float[j*in+i] has i contiguous per j.
     * But we need i fixed, j varying — that's strided. So we compute
     * per-i by accumulating over j. With [out,in] layout, w_float[j*in+i]
     * for fixed i has stride=in. This is still strided but avoids repack.
     *
     * Alternative: compute grad_x = sign(w_float)^T @ (grad_y * alpha)
     * which is a matrix-vector product. We can do it per-output j and
     * accumulate into grad_x (since w_float[j*in+i] is contiguous in i). */
    if (bl->w_float) {
        /* Zero grad_x first */
        for (int i = 0; i < in; i++) grad_x[i] = 0.0f;
        /* For each output j: grad_x += grad_y[j] * alpha[j] * sign(w_float[j*in+i])
         * w_float[j*in + 0..in-1] is contiguous → SIMD-friendly! */
        for (int j = 0; j < out; j++) {
            float gy = grad_y[j];
            if (fabsf(gy) < 1e-8f) continue;
            float scale = gy * bl->alpha[j];
            const float *wf = &bl->w_float[j * in];  /* contiguous [in] */
            for (int i = 0; i + 7 < in; i += 8) {
                grad_x[i+0] += scale * (wf[i+0] > 0.0f ? 1.0f : -1.0f);
                grad_x[i+1] += scale * (wf[i+1] > 0.0f ? 1.0f : -1.0f);
                grad_x[i+2] += scale * (wf[i+2] > 0.0f ? 1.0f : -1.0f);
                grad_x[i+3] += scale * (wf[i+3] > 0.0f ? 1.0f : -1.0f);
                grad_x[i+4] += scale * (wf[i+4] > 0.0f ? 1.0f : -1.0f);
                grad_x[i+5] += scale * (wf[i+5] > 0.0f ? 1.0f : -1.0f);
                grad_x[i+6] += scale * (wf[i+6] > 0.0f ? 1.0f : -1.0f);
                grad_x[i+7] += scale * (wf[i+7] > 0.0f ? 1.0f : -1.0f);
            }
            for (int i = (in / 8) * 8; i < in; i++)
                grad_x[i] += scale * (wf[i] > 0.0f ? 1.0f : -1.0f);
        }
    } else {
        /* No w_float — use popcount on existing wbits_T (original path) */
        int nw_T = bl->n_words_T;
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
    }

    /* Part 2: STE update — w_float[j*in + i] -= lr * grad_y[j] * x[i]
     * w_float is [out, in] (transposed), so w_float[j*in + i] is CONTIGUOUS in i!
     * This means the inner loop over i is a contiguous SAXPY: w_float[j] -= scale * x
     * The compiler auto-vectorizes this to SIMD FMA (8 floats per iteration). */
    if (bl->w_float) {
        for (int j = 0; j < out; j++) {
            float gy = grad_y[j];
            if (fabsf(gy) < 1e-8f) continue;
            float scale = lr * gy;
            float *wf = &bl->w_float[j * in];  /* contiguous [in] */
            /* SAXPY: wf[i] -= scale * x[i], 8x unrolled for SIMD */
            for (int i = 0; i + 7 < in; i += 8) {
                wf[i+0] -= scale * x[i+0];
                wf[i+1] -= scale * x[i+1];
                wf[i+2] -= scale * x[i+2];
                wf[i+3] -= scale * x[i+3];
                wf[i+4] -= scale * x[i+4];
                wf[i+5] -= scale * x[i+5];
                wf[i+6] -= scale * x[i+6];
                wf[i+7] -= scale * x[i+7];
            }
            for (int i = (in / 8) * 8; i < in; i++)
                wf[i] -= scale * x[i];
            /* Update bias */
            bl->bias[j] -= lr * gy;
        }
        /* Re-pack binary weights from updated w_float */
        bin_layer_repack(bl);
    } else {
        /* No w_float — fall back to alpha-only update */
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
            bl->alpha[j] -= lr * grad_alpha * gy;   /* FIXED: direction + no /in */
            if (bl->alpha[j] < 0.0f) bl->alpha[j] = 0.0f;
            bl->bias[j] -= lr * gy;
        }
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
