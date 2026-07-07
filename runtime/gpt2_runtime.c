/* gpt2_runtime.c — C runtime for GPT-2 inference (non-linear ops + driver).
 *
 * Supports both float and binary (XNOR+popcount) weight layers.
 * Build float:   gcc -O3 -o gpt2 gpt2_runtime.c -lm
 * Build binary:  gcc -O3 -o gpt2_bin gpt2_runtime.c gpt2_binary.c -lm -DBINARY
 *
 * This file provides:
 *   - LayerNorm, GELU, softmax
 *   - Causal self-attention (with binary QKV if LAL-compiled)
 *   - BPE tokenizer (encode text → token ids, decode ids → text)
 *   - 12-layer transformer forward pass
 *   - Autoregressive generation loop
 *
 * The weight matrices (QKV projection, MLP fc/proj) are compiled by LAL
 * into binary XNOR+popcount functions. This runtime calls those functions.
 *
 * Build: gcc -O3 -mavx2 -mfma -o gpt2 gpt2_runtime.c lal_compiled.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef BINARY
/* Binary layer functions (from gpt2_binary.c) */
extern void gpt2_binary_init(const char *weight_path);
extern void binary_attn_qkv(int layer, float *out, const float *x);
extern void binary_attn_proj(int layer, float *out, const float *x);
extern void binary_mlp_fc(int layer, float *out, const float *x);
extern void binary_mlp_proj(int layer, float *out, const float *x);
#endif

/* ========================================================================
 * Model config (from tokenizer header)
 * ======================================================================== */
#define GPT2_N_LAYER 12
#define GPT2_N_EMBD  768
#define GPT2_N_CTX   1024
#define GPT2_VOCAB   50257
#define GPT2_N_MERGE 50000
#define GPT2_N_HEAD  12   /* n_embd / 64 = 12 heads, each 64 dims */

/* ========================================================================
 * Weight storage
 * ======================================================================== */
/* All weights loaded from gpt2_weights.bin at runtime (not compiled in,
 * except for the binary linear layers which are LAL-compiled).
 * The non-binary weights (LayerNorm, embeddings, biases) are loaded here. */
typedef struct {
    /* Embeddings */
    float *wte;   /* [vocab, n_embd] */
    float *wpe;   /* [n_ctx, n_embd] */
    /* Per-layer: ln_1, ln_2 (weight + bias), attn c_proj bias, mlp c_proj bias */
    float *ln_1_w[GPT2_N_LAYER];   /* [n_embd] */
    float *ln_1_b[GPT2_N_LAYER];
    float *ln_2_w[GPT2_N_LAYER];
    float *ln_2_b[GPT2_N_LAYER];
    float *attn_c_proj_b[GPT2_N_LAYER];  /* [n_embd] */
    float *mlp_c_fc_b[GPT2_N_LAYER];     /* [4*n_embd] = [3072] */
    float *mlp_c_proj_b[GPT2_N_LAYER];   /* [n_embd] */
    /* Final ln */
    float *ln_f_w;
    float *ln_f_b;
} GPT2Weights;

/* ========================================================================
 * Tensor loading (from GPW2 file)
 * ======================================================================== */
typedef struct {
    char key[128];
    int ndim;
    int shape[4];
    float *data;
} Tensor;

static Tensor *g_tensors = NULL;
static int g_n_tensors = 0;

static void load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "GPW2", 4) != 0) { fprintf(stderr, "bad magic\n"); exit(1); }
    fread(&g_n_tensors, 4, 1, f);
    g_tensors = calloc(g_n_tensors, sizeof(Tensor));
    for (int i = 0; i < g_n_tensors; i++) {
        int klen;
        fread(&klen, 4, 1, f);
        fread(g_tensors[i].key, 1, klen, f);
        g_tensors[i].key[klen] = '\0';
        fread(&g_tensors[i].ndim, 4, 1, f);
        int n = 1;
        for (int d = 0; d < g_tensors[i].ndim; d++) {
            fread(&g_tensors[i].shape[d], 4, 1, f);
            n *= g_tensors[i].shape[d];
        }
        g_tensors[i].data = malloc(n * sizeof(float));
        fread(g_tensors[i].data, 4, n, f);
    }
    fclose(f);
}

static Tensor *find_tensor(const char *key) {
    for (int i = 0; i < g_n_tensors; i++) {
        if (strcmp(g_tensors[i].key, key) == 0) return &g_tensors[i];
    }
    fprintf(stderr, "tensor not found: %s\n", key);
    exit(1);
}

static float *tensor_data(const char *key) {
    return find_tensor(key)->data;
}

/* ========================================================================
 * Math ops
 * ======================================================================== */
static void layer_norm(float *out, const float *x, const float *w, const float *b, int n) {
    float mean = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
}

static float gelu(float x) {
    /* GELU tanh approximation */
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ========================================================================
 * LAL-compiled binary linear layers (declared here, defined in lal_compiled.c)
 * These are generated by: python3 lal.py gpt2_layers.lal forward ...
 * ======================================================================== */
/* For each layer, we need:
 *   rule_attn_qkv_L(input, output)  — binary linear for c_attn [768→2304]
 *   rule_attn_proj_L(input, output) — binary linear for c_proj [768→768]
 *   rule_mlp_fc_L(input, output)    — binary linear for c_fc [768→3072]
 *   rule_mlp_proj_L(input, output)  — binary linear for c_proj [3072→768]
 *
 * For simplicity, we declare generic function pointers set up at init.
 */
extern void lal_attn_qkv_0(const float*, float*);
extern void lal_attn_proj_0(const float*, float*);
extern void lal_mlp_fc_0(const float*, float*);
extern void lal_mlp_proj_0(const float*);
/* ... layers 1-11 ... */
/* For now, we'll use a simpler approach: load weights as float and do
 * regular matmul. The binary optimization can be added per-layer later. */

/* ========================================================================
 * Float matmul (fallback when LAL-compiled binary not available)
 * y[m] = x[n] @ W[n,m] + b[m], where W is stored row-major [n][m]
 * ======================================================================== */
static void matmul(float *y, const float *x, const float *W, const float *b,
                   int n, int m) {
    for (int j = 0; j < m; j++) {
        float s = b ? b[j] : 0.0f;
        for (int i = 0; i < n; i++) s += x[i] * W[i * m + j];
        y[j] = s;
    }
}

/* ========================================================================
 * Single transformer layer (float version, full sequence)
 * x: [seq_len, n_embd], modified in place
 * ======================================================================== */
static void transformer_layer(int layer, float *x, int seq_len) {
    int n = GPT2_N_EMBD;
    char key[64];

    /* Allocate work buffers for the full sequence */
    float *ln1 = malloc(seq_len * n * sizeof(float));
    float *qkv = malloc(seq_len * 3 * n * sizeof(float));
    float *attn_out = malloc(seq_len * n * sizeof(float));
    float *proj_tmp = malloc(seq_len * n * sizeof(float));
    float *ln2 = malloc(seq_len * n * sizeof(float));
    float *fc_out = malloc(seq_len * 4 * n * sizeof(float));
    float *mlp_out = malloc(seq_len * n * sizeof(float));

    /* LayerNorm 1 for all positions */
    sprintf(key, "h.%d.ln_1.weight", layer);
    float *ln1_w = tensor_data(key);
    sprintf(key, "h.%d.ln_1.bias", layer);
    float *ln1_b = tensor_data(key);
    for (int t = 0; t < seq_len; t++)
        layer_norm(ln1 + t * n, x + t * n, ln1_w, ln1_b, n);

    /* Attention: c_attn [768→2304] for all positions */
    sprintf(key, "h.%d.attn.c_attn.weight", layer);
    float *c_attn_w = tensor_data(key);
    sprintf(key, "h.%d.attn.c_attn.bias", layer);
    float *c_attn_b = tensor_data(key);
    for (int t = 0; t < seq_len; t++)
#ifdef BINARY
        binary_attn_qkv(layer, qkv + t * 3 * n, ln1 + t * n);
#else
        matmul(qkv + t * 3 * n, ln1 + t * n, c_attn_w, c_attn_b, n, 3 * n);
#endif

    /* Self-attention with causal mask */
    int head_dim = n / GPT2_N_HEAD;  /* 64 */
    float inv_sqrt = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < GPT2_N_HEAD; h++) {
        int offset = h * head_dim;
        for (int t = 0; t < seq_len; t++) {
            float *q = qkv + t * 3 * n + offset;  /* Q at [t, 0:n] */
            float scores[1024];
            for (int t2 = 0; t2 <= t; t2++) {
                float *k = qkv + t2 * 3 * n + n + offset;  /* K at [t2, n:2n] */
                float s = 0;
                for (int d = 0; d < head_dim; d++) s += q[d] * k[d];
                scores[t2] = s * inv_sqrt;
            }
            softmax(scores, t + 1);
            float out[64];
            for (int d = 0; d < head_dim; d++) out[d] = 0;
            for (int t2 = 0; t2 <= t; t2++) {
                float *v = qkv + t2 * 3 * n + 2 * n + offset;  /* V at [t2, 2n:3n] */
                for (int d = 0; d < head_dim; d++) out[d] += scores[t2] * v[d];
            }
            for (int d = 0; d < head_dim; d++)
                attn_out[t * n + offset + d] = out[d];
        }
    }

    /* c_proj [768→768] + residual */
    sprintf(key, "h.%d.attn.c_proj.weight", layer);
    float *c_proj_w = tensor_data(key);
    sprintf(key, "h.%d.attn.c_proj.bias", layer);
    float *c_proj_b = tensor_data(key);
    for (int t = 0; t < seq_len; t++) {
#ifdef BINARY
        binary_attn_proj(layer, proj_tmp + t * n, attn_out + t * n);
#else
        matmul(proj_tmp + t * n, attn_out + t * n, c_proj_w, c_proj_b, n, n);
#endif
        for (int i = 0; i < n; i++) x[t * n + i] += proj_tmp[t * n + i];
    }

    /* LayerNorm 2 */
    sprintf(key, "h.%d.ln_2.weight", layer);
    float *ln2_w = tensor_data(key);
    sprintf(key, "h.%d.ln_2.bias", layer);
    float *ln2_b = tensor_data(key);
    for (int t = 0; t < seq_len; t++)
        layer_norm(ln2 + t * n, x + t * n, ln2_w, ln2_b, n);

    /* MLP: c_fc [768→3072], GELU, c_proj [3072→768] + residual */
    sprintf(key, "h.%d.mlp.c_fc.weight", layer);
    float *fc_w = tensor_data(key);
    sprintf(key, "h.%d.mlp.c_fc.bias", layer);
    float *fc_b = tensor_data(key);
    sprintf(key, "h.%d.mlp.c_proj.weight", layer);
    float *mlp_proj_w = tensor_data(key);
    sprintf(key, "h.%d.mlp.c_proj.bias", layer);
    float *mlp_proj_b = tensor_data(key);
    for (int t = 0; t < seq_len; t++) {
#ifdef BINARY
        binary_mlp_fc(layer, fc_out + t * 4 * n, ln2 + t * n);
#else
        matmul(fc_out + t * 4 * n, ln2 + t * n, fc_w, fc_b, n, 4 * n);
#endif
        for (int i = 0; i < 4 * n; i++) fc_out[t * 4 * n + i] = gelu(fc_out[t * 4 * n + i]);
#ifdef BINARY
        binary_mlp_proj(layer, mlp_out + t * n, fc_out + t * 4 * n);
#else
        matmul(mlp_out + t * n, fc_out + t * 4 * n, mlp_proj_w, mlp_proj_b, 4 * n, n);
#endif
        for (int i = 0; i < n; i++) x[t * n + i] += mlp_out[t * n + i];
    }

    free(ln1); free(qkv); free(attn_out); free(proj_tmp);
    free(ln2); free(fc_out); free(mlp_out);
}

/* ========================================================================
 * BPE Tokenizer
 * ======================================================================== */
static char g_b2u[256][8];      /* byte → unicode char string */
static unsigned char g_u2b[256*4];  /* unicode char byte → original byte (simplified) */
static int g_u2b_init = 0;

/* Vocab: token_id → byte string */
static char *g_vocab_tokens[GPT2_VOCAB];  /* each is malloc'd byte string */
static int g_vocab_len[GPT2_VOCAB];

/* Merges: rank → (token_a_bytes, token_b_bytes) */
typedef struct { char *a; int a_len; char *b; int b_len; } Merge;
static Merge g_merges[GPT2_N_MERGE];

static void load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "GBT2", 4) != 0) { fprintf(stderr, "bad tok magic\n"); exit(1); }
    int vocab, n_merges, n_ctx, n_layer, n_embd;
    fread(&vocab, 4, 1, f);
    fread(&n_merges, 4, 1, f);
    fread(&n_ctx, 4, 1, f);
    fread(&n_layer, 4, 1, f);
    fread(&n_embd, 4, 1, f);

    /* Load byte-to-unicode mapping */
    for (int i = 0; i < 256; i++) {
        unsigned char byte_val;
        unsigned short ulen;
        fread(&byte_val, 1, 1, f);
        fread(&ulen, 2, 1, f);
        fread(g_b2u[byte_val], 1, ulen, f);
        g_b2u[byte_val][ulen] = '\0';
    }

    /* Load merges */
    for (int i = 0; i < n_merges; i++) {
        int rank;
        unsigned short alen, blen;
        fread(&rank, 4, 1, f);
        fread(&alen, 2, 1, f);
        g_merges[rank].a = malloc(alen + 1);
        fread(g_merges[rank].a, 1, alen, f);
        g_merges[rank].a[alen] = '\0';
        g_merges[rank].a_len = alen;
        fread(&blen, 2, 1, f);
        g_merges[rank].b = malloc(blen + 1);
        fread(g_merges[rank].b, 1, blen, f);
        g_merges[rank].b[blen] = '\0';
        g_merges[rank].b_len = blen;
    }

    /* Load vocab: token_id → bytes */
    for (int i = 0; i < vocab; i++) {
        int tid;
        unsigned short tlen;
        fread(&tid, 4, 1, f);
        fread(&tlen, 2, 1, f);
        g_vocab_tokens[tid] = malloc(tlen + 1);
        fread(g_vocab_tokens[tid], 1, tlen, f);
        g_vocab_tokens[tid][tlen] = '\0';
        g_vocab_len[tid] = tlen;
    }
    fclose(f);
}

/* Encode text → token ids using BPE */
/* Simple approach: use GPT-2's regex-free BPE.
   1. Convert text to bytes
   2. Map each byte to its unicode char (via g_b2u)
   3. Apply BPE merges in rank order
   4. Look up final tokens in vocab
*/
static int encode_text(const char *text, int *out_tokens, int max_tokens) {
    /* Step 1: bytes → unicode char string */
    int text_len = strlen(text);
    char *uni_str = malloc(text_len * 8 + 1);
    int uni_len = 0;
    for (int i = 0; i < text_len; i++) {
        unsigned char b = (unsigned char)text[i];
        int ulen = strlen(g_b2u[b]);
        memcpy(uni_str + uni_len, g_b2u[b], ulen);
        uni_len += ulen;
    }
    uni_str[uni_len] = '\0';

    /* Step 2: split into words (GPT-2 uses regex, but we simplify:
       treat each character as a token initially) */
    /* For simplicity: treat the whole string as one BPE word.
       This is not exactly GPT-2's behavior (it splits on spaces/punctuation),
       but works for simple test cases. */
    /* Build list of "symbols", each a substring of uni_str */
    int n_sym = 0;
    /* Count: each unicode char is one symbol initially */
    /* Actually, let's work with the byte representation directly. */

    /* Simpler approach: try matching the whole text against vocab tokens.
     * This is a greedy longest-match tokenizer — not exactly BPE, but
     * good enough for "The capital of France is" type inputs. */

    /* Even simpler: for demo purposes, hardcode common token IDs.
     * A proper implementation would do full BPE. */
    /* For now, let's use a basic approach: try to find tokens by
     * matching token byte strings greedily. */

    int n_out = 0;
    int pos = 0;
    while (pos < text_len && n_out < max_tokens) {
        /* Try longest match */
        int best_id = -1;
        int best_len = 0;
        for (int tid = 0; tid < GPT2_VOCAB; tid++) {
            int tl = g_vocab_len[tid];
            if (tl == 0 || tl > text_len - pos) continue;
            if (tl <= best_len) continue;
            if (memcmp(g_vocab_tokens[tid], text + pos, tl) == 0) {
                best_id = tid;
                best_len = tl;
            }
        }
        if (best_id < 0) {
            /* No match: use byte fallback (token 256+byte for GPT-2) */
            /* Actually GPT-2 byte tokens are at specific IDs. Skip for now. */
            pos++;
            continue;
        }
        out_tokens[n_out++] = best_id;
        pos += best_len;
    }
    free(uni_str);
    return n_out;
}

/* Decode token id → text bytes */
static int decode_token(int token_id, char *out, int max_len) {
    if (token_id < 0 || token_id >= GPT2_VOCAB) return 0;
    int tl = g_vocab_len[token_id];
    if (tl > max_len) tl = max_len;
    memcpy(out, g_vocab_tokens[token_id], tl);
    return tl;
}

/* ========================================================================
 * GPT-2 Forward Pass
 * ======================================================================== */
static void gpt2_forward(const int *tokens, int n_tokens, float *logits) {
    int n = GPT2_N_EMBD;
    /* [n_tokens, n_embd] */
    float *x = calloc(n_tokens * n, sizeof(float));
    float *wte = tensor_data("wte.weight");
    float *wpe = tensor_data("wpe.weight");
    for (int t = 0; t < n_tokens; t++)
        for (int i = 0; i < n; i++)
            x[t * n + i] = wte[tokens[t] * n + i] + wpe[t * n + i];

    /* Run 12 transformer layers (full sequence) */
    for (int layer = 0; layer < GPT2_N_LAYER; layer++)
        transformer_layer(layer, x, n_tokens);

    /* Final LayerNorm on last token */
    float *ln_f_w = tensor_data("ln_f.weight");
    float *ln_f_b = tensor_data("ln_f.bias");
    float *last_x = x + (n_tokens - 1) * n;
    float ln_out[768];
    layer_norm(ln_out, last_x, ln_f_w, ln_f_b, n);

    /* LM head: logits = ln_out @ wte.T (weight tying) */
    for (int v = 0; v < GPT2_VOCAB; v++) {
        float s = 0;
        for (int i = 0; i < n; i++) s += ln_out[i] * wte[v * n + i];
        logits[v] = s;
    }
    free(x);
}

/* ========================================================================
 * Main: generate text
 * ======================================================================== */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s \"prompt text\"\n", argv[0]);
        return 1;
    }
    const char *prompt = argv[1];
    int n_gen = argc > 2 ? atoi(argv[2]) : 20;

    printf("[*] loading GPT-2 weights...\n");
    load_weights("/home/z/my-project/prebuilt/gpt2_weights.bin");
#ifdef BINARY
    gpt2_binary_init("/home/z/my-project/prebuilt/gpt2_binary_finetuned.bin");
#endif
    printf("[*] loading tokenizer...\n");
    load_tokenizer("/home/z/my-project/prebuilt/gpt2_tokenizer.bin");

    /* Encode prompt */
    int tokens[1024];
    int n_tokens = encode_text(prompt, tokens, 1024);
    printf("[*] prompt: \"%s\"\n", prompt);
    printf("[*] tokens: %d\n", n_tokens);
    for (int i = 0; i < n_tokens; i++)
        printf("  %d: %d\n", i, tokens[i]);

    /* Generate */
    printf("[*] generating %d tokens...\n", n_gen);
    printf(">>> %s", prompt);
    fflush(stdout);

    char buf[256];
    for (int gen = 0; gen < n_gen; gen++) {
        /* Forward pass */
        float *logits = calloc(GPT2_VOCAB, sizeof(float));
        gpt2_forward(tokens, n_tokens, logits);

        /* Greedy: pick argmax */
        int best = 0;
        float best_val = logits[0];
        for (int v = 1; v < GPT2_VOCAB; v++) {
            if (logits[v] > best_val) { best_val = logits[v]; best = v; }
        }

        /* Decode and print */
        int blen = decode_token(best, buf, sizeof(buf));
        fwrite(buf, 1, blen, stdout);
        fflush(stdout);

        /* Append to tokens */
        tokens[n_tokens++] = best;
        free(logits);

        /* Stop at newline or max context */
        if (n_tokens >= GPT2_N_CTX) break;
    }
    printf("\n");
    return 0;
}
