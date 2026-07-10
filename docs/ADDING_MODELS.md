# Adding a New Model to LAL

This document describes the standard recipe for adding a new LLM architecture
to LAL. The process has been distilled from GPT-2 124M and Qwen2.5-0.5B
implementations, and leverages 5 reusable header files in `runtime/`.

## Reusable Components

| Header | Purpose | Key Functions |
|--------|---------|---------------|
| `runtime/lal_q8_kernel.h` | Q8 matmul kernels (sign-trick, 8-output parallel) | `lal_matmul_q8_signtrick()`, `lal_quantize_q8_per_row()`, `lal_lm_head_int8_range()` |
| `runtime/lal_sampling.h` | Token sampling (temperature, top-k, rep_penalty) | `lal_sample_token()`, `lal_recent_push()` |
| `runtime/lal_dequant.h` | SIMD int8→float32 dequantization | `lal_dequant_row_f32()`, `lal_dequant_add_f32()` |
| `runtime/lal_tokenizer.h` | BPE byte-level decode (Ġ→space, Ċ→newline) | `lal_decode_bpe_token()` |
| `runtime/lal_weight_utils.h` | Free float weights after Q8 quantization | `lal_free_float_tensors()`, `lal_free_layer_weights()` |

## Recipe (8 Steps)

### 1. Define Model Architecture Constants

```c
#define N_EMBD       896    // hidden size
#define N_LAYER      24     // number of transformer layers
#define N_HEAD       14     // attention heads
#define N_KV_HEAD    2      // KV heads (for GQA; = N_HEAD if no GQA)
#define HEAD_DIM     64     // dimension per head
#define MLP_DIM      4864   // MLP intermediate size
#define VOCAB_SIZE   151936 // vocabulary size
#define N_CTX        2048   // max context length
```

### 2. Define Layer Struct

```c
typedef struct {
    float *norm1_w, *norm2_w;        // RMSNorm/LayerNorm weights
    int8_t *q8_q, *q8_k, *q8_v, *q8_o;  // Q8 quantized attention weights
    float  *s_q, *s_k, *s_v, *s_o;       // per-row scales
    int8_t *q8_gate, *q8_up, *q8_down;   // Q8 quantized MLP weights
    float  *s_gate, *s_up, *s_down;
    float  *q_bias, *k_bias, *v_bias;    // attention biases (if any)
    float  *gate_bias, *up_bias, *down_bias;
} ModelLayer;
```

### 3. Include Reusable Headers

```c
#define XQ_MAX 4864  // = max(in_dim) across all matmuls
#include "runtime/lal_q8_kernel.h"
#include "runtime/lal_sampling.h"
#include "runtime/lal_dequant.h"
#include "runtime/lal_tokenizer.h"
#include "runtime/lal_weight_utils.h"
```

### 4. Weight Loading + Q8 Quantization

```c
static void load_weights(const char *path) {
    int n_tensors;
    Tensor *tensors = tensor_load_all(path, &n_tensors);

    g_wte = tensor_get(tensors, n_tensors, "model.embed_tokens.weight");
    g_norm_f_w = tensor_get(tensors, n_tensors, "model.norm.weight");

    for (int l = 0; l < N_LAYER; l++) {
        // Get float weights by key
        float *Wq = tensor_get(tensors, n_tensors, "model.layers.%d.self_attn.q_proj.weight", l);
        // Allocate Q8 buffers
        L->q8_q = malloc(N_EMBD * N_EMBD);
        L->s_q = malloc(N_EMBD * sizeof(float));
        // Quantize using shared helper
        lal_quantize_q8_per_row(Wq, L->q8_q, L->s_q, N_EMBD, N_EMBD);
        // ... repeat for k, v, o, gate, up, down
    }

    // Quantize wte for int8 LM head
    g_wte_q = malloc(VOCAB_SIZE * N_EMBD);
    g_wte_scale = malloc(VOCAB_SIZE * sizeof(float));
    for (int v = 0; v < VOCAB_SIZE; v++)
        g_wte_scale[v] = lal_quantize_x_int8(g_wte + v*N_EMBD, g_wte_q + v*N_EMBD, N_EMBD);

    // FREE float weights (recover RSS)
    for (int l = 0; l < N_LAYER; l++) {
        char k[256];
        const char *keys[8];
        sprintf(k, "model.layers.%d.self_attn.q_proj.weight", l); keys[0] = strdup(k);
        // ... add k, v, o, gate, up, down
        keys[7] = NULL;
        lal_free_float_tensors(tensors, n_tensors, keys);
    }
    // Free float wte
    const char *wte_key[] = {"model.embed_tokens.weight", NULL};
    lal_free_float_tensors(tensors, n_tensors, wte_key);
    g_wte = NULL;  // mark as freed
}
```

### 5. Forward Pass

```c
static int forward(int tok, int pos) {
    // Embedding lookup (SIMD dequant if g_wte was freed)
    if (g_wte) {
        memcpy(g_x, g_wte + tok * N_EMBD, N_EMBD * sizeof(float));
    } else {
        lal_dequant_row_f32(g_wte_q + tok * N_EMBD, g_x, g_wte_scale[tok], N_EMBD);
    }

    for (int l = 0; l < N_LAYER; l++) {
        // RMSNorm
        rms_norm(g_ln, g_x, L->norm1_w, N_EMBD);
        // Q8 matmul (sign-trick, 8-output parallel)
        lal_matmul_q8_signtrick(g_q, L->q8_q, L->s_q, g_ln, L->q_bias, N_EMBD, N_EMBD);
        // ... RoPE, attention, o_proj, residual
        // ... RMSNorm, MLP (fused SwiGLU or separate matmuls), residual
    }

    // Final norm
    rms_norm(g_ln, g_x, g_norm_f_w, N_EMBD);

    // LM head (int8 with rerank)
    static int8_t xq[N_EMBD];
    float scale_x = lal_quantize_x_int8(g_ln, xq, N_EMBD);
    lal_lm_head_int8_range(g_logits, xq, scale_x, g_wte_q, g_wte_scale, 0, VOCAB_SIZE, N_EMBD);

    // Sample
    int next = lal_sample_token(g_logits, VOCAB_SIZE, g_temperature, g_top_k, g_rep_penalty, g_recent, g_n_recent);
    return next;
}
```

### 6. Tokenizer

Load `tokenizer.json` (HuggingFace format). For decode, use:
```c
lal_decode_bpe_token(g_vocab_str[id], out, maxlen);
```

### 7. Weight Conversion (safetensors → GPW2)

Write a Python script (or reuse `scripts/convert_safetensors_to_gpw2.py`):
- Handle BF16 → float32 (left-shift 16 bits)
- Handle F32 directly
- Skip attn.bias / attn.masked_bias buffers
- Output GPW2 format: magic(4B) + n_tensors(4B) + per-tensor(key_len + key + ndim + shape + data)

### 8. Build + Test

```makefile
newmodel-server: prebuilt/newmodel_server
prebuilt/newmodel_server: tools/server/newmodel_server.c runtime/lal_runtime.c
	$(CC) $(CFLAGS) -o $@ $< runtime/lal_runtime.c -lm -lpthread
```

## Checklist

- [ ] Architecture constants defined
- [ ] Layer struct defined
- [ ] Headers included (lal_q8_kernel, lal_sampling, lal_dequant, lal_tokenizer, lal_weight_utils)
- [ ] Weight loading with Q8 quantization
- [ ] Free float weights after quantization
- [ ] Forward pass uses `lal_matmul_q8_signtrick` for all linear layers
- [ ] Embedding lookup uses `lal_dequant_row_f32` when g_wte is NULL
- [ ] LM head uses `lal_lm_head_int8_range`
- [ ] Sampling uses `lal_sample_token` (not argmax)
- [ ] Tokenizer decode uses `lal_decode_bpe_token`
- [ ] Weight converter script written
- [ ] Build succeeds with 0 errors
- [ ] Benchmark: verify tok/s, RSS, output quality
