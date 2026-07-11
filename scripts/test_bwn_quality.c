/* test_bwn_quality.c — test if BWN (sign(w) + alpha + K-norm) produces
 * acceptable output on Qwen2.5-7B without STE retraining.
 *
 * Approach:
 *   1. Load Q8 weights (existing GPQ8 file)
 *   2. For each layer's MLP matrices, binarize: sign(w) + alpha = mean(|w_row|)
 *   3. Run forward with BWN for MLP, Q8 for attention
 *   4. Compare output quality with full Q8
 *
 * BWN forward: y[j] = sum_i sign(w[j,i]) * x[i] * alpha[j] + bias[j]
 * With K-norm (XNOR-Net): y[j] = sum_i sign(w[j,i]) * x[i] * alpha[j] * K + bias[j]
 *   where K = mean(|x|)
 *
 * Bandwidth: sign(w) packed as 1 bit/elem → 8x less than Q8 (1 byte/elem)
 * For MLP (gate+up+down = 3 × 18944 × 3584 = 204M params):
 *   Q8: 204 MB/layer
 *   BWN packed: 25.5 MB/layer (8x less!)
 *
 * If quality is acceptable, this gives ~3x speedup on MLP (the bottleneck).
 */
#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define N_EMBD       3584
#define N_LAYER      28
#define N_HEAD       28
#define N_KV_HEAD    4
#define HEAD_DIM     128
#define MLP_DIM      18944
#define VOCAB_SIZE   152064
#define N_CTX        4096
#define ROPE_THETA   1000000.0f
#define RMS_EPS      1e-6f
#define KV_DIM       (N_KV_HEAD * HEAD_DIM)
#define Q_DIM        (N_HEAD * HEAD_DIM)
#define N_Q_PER_KV   (N_HEAD / N_KV_HEAD)

#define XQ_MAX 18944
#include "runtime/lal_runtime.h"
#include "runtime/lal_q8_kernel.h"
#include "runtime/lal_sampling.h"
#include "runtime/lal_dequant.h"
#include "runtime/lal_tokenizer.h"

/* === BWN layer (binary weight, packed bits) === */
typedef struct {
    uint64_t *wbits;    /* [out_dim, in_dim/64] packed sign bits */
    float    *alpha;    /* [out_dim] per-row scale = mean(|w|) */
    float    *bias;     /* [out_dim] or NULL */
    int in_dim, out_dim, n_words;
} BwnLayer;

/* Binarize a Q8 weight matrix to BWN: sign(w) packed + alpha = mean(|w_row|)
 * q8_T is [out_dim, in_dim] int8, scale is [out_dim] float.
 * Output: wbits (packed sign bits), alpha (mean|w| * scale per row). */
static void binarize_q8_to_bwn(const int8_t *q8_T, const float *scale,
                                BwnLayer *bl, int in_dim, int out_dim) {
    bl->in_dim = in_dim;
    bl->out_dim = out_dim;
    bl->n_words = (in_dim + 63) / 64;
    bl->wbits = memalign(32, (size_t)out_dim * bl->n_words * sizeof(uint64_t));
    bl->alpha = memalign(32, out_dim * sizeof(float));
    for (int j = 0; j < out_dim; j++) {
        const int8_t *row = q8_T + (size_t)j * in_dim;
        uint64_t *wb = bl->wbits + (size_t)j * bl->n_words;
        /* alpha = mean(|w|) * scale (true weight magnitude) */
        float abs_sum = 0;
        for (int i = 0; i < in_dim; i++) abs_sum += fabsf((float)row[i]);
        bl->alpha[j] = (abs_sum / in_dim) * scale[j];
        /* Pack sign bits: 1 = positive, 0 = negative */
        for (int wi = 0; wi < bl->n_words; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int idx = wi * 64 + bi;
                if (idx < in_dim && row[idx] >= 0) word |= (1ULL << bi);
            }
            wb[wi] = word;
        }
    }
}

/* BWN matmul using XNOR+popcount (LAL's core optimization).
 * y[j] = (2*popcount(XNOR(sign(x), wbits[j])) - n_active) * x_scale * alpha[j] * K + bias[j]
 *
 * For float x (BWN mode, not BNN): we can't use XNOR directly.
 * Instead: sign(w) as ±1, dot with float x, scale by alpha*K.
 * But that's slow. Use int8 x quantization + sign(w) int8:
 *   y[j] = sum(sign(w[i]) * x_q[i]) * x_scale * alpha[j] * K + bias[j]
 * sign(w) = ±1, so sign(w)*x_q = conditional add/sub.
 *
 * For packed-bit speed: use XNOR+popcount with binarized x (BNN mode).
 * y[j] = (2*pc - in_dim) * x_scale * alpha[j] * K + bias[j]
 * This binarizes x too → BNN. Quality may suffer.
 *
 * We test BNN mode here (fastest, lowest quality). */
static void bwn_matmul_bnn(float *y, const float *x, const BwnLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;
    /* K-norm: mean(|x|) */
    float abs_sum = 0;
    for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
    float K = abs_sum / in;
    /* x_scale: for magnitude recovery with binarized x */
    float x_scale = K;  /* BNN: x_bin = sign(x), scale by mean|x| */

    /* Binarize x */
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
        const uint64_t *wb = bl->wbits + (size_t)j * nw;
        int pc = 0;
        for (int wi = 0; wi < nw; wi++)
            pc += __builtin_popcountll(~(xbits[wi] ^ wb[wi]));
        y[j] = (float)(2 * pc - in) * x_scale * bl->alpha[j] * K + (bl->bias ? bl->bias[j] : 0);
    }
}

/* BWN matmul using AVX2 XNOR+popcount (LAL's core fast path).
 * This is BNN mode: binarize BOTH x and w to ±1, use XNOR+popcount.
 * y[j] = (2*popcount(XNOR(sign(x), wbits[j])) - in_dim) * x_scale * alpha[j] + bias[j]
 * where x_scale = mean(|x|) (K-norm, restores magnitude).
 *
 * AVX2: _mm256_popcnt not available, but __builtin_popcountll is fast on x86_64.
 * We process 4 uint64 words per iteration for ILP. */
static void bwn_matmul_bnn_avx2(float *y, const float *x, const BwnLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;
    /* K-norm: mean(|x|) = x_scale */
    float abs_sum = 0;
    for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
    float K = abs_sum / in;

    /* Binarize x to packed bits */
    static uint64_t xbits[64] __attribute__((aligned(32)));
    for (int wi = 0; wi < nw; wi++) {
        uint64_t word = 0;
        int base = wi * 64;
        for (int bi = 0; bi < 64 && base + bi < in; bi++)
            if (x[base + bi] > 0.0f) word |= (1ULL << bi);
        xbits[wi] = word;
    }

    /* XNOR + popcount, 4 words at a time for ILP */
    for (int j = 0; j < out; j++) {
        const uint64_t *wb = bl->wbits + (size_t)j * nw;
        int pc = 0;
        int wi = 0;
        for (; wi + 4 <= nw; wi += 4) {
            pc += __builtin_popcountll(~(xbits[wi] ^ wb[wi]));
            pc += __builtin_popcountll(~(xbits[wi+1] ^ wb[wi+1]));
            pc += __builtin_popcountll(~(xbits[wi+2] ^ wb[wi+2]));
            pc += __builtin_popcountll(~(xbits[wi+3] ^ wb[wi+3]));
        }
        for (; wi < nw; wi++)
            pc += __builtin_popcountll(~(xbits[wi] ^ wb[wi]));
        y[j] = (float)(2 * pc - in) * K * bl->alpha[j] + (bl->bias ? bl->bias[j] : 0);
    }
}

/* === GPQ8 loading (same as qwen7b_server.c) === */
typedef struct {
    char key[128]; int ndim, shape[4]; int qtype; uint64_t data_len;
    void *data; int n_scale; float *scale;
} GPQ8Tensor;
static GPQ8Tensor *g_gp_tensors; static int g_gp_n;
static void *g_mmap_base; static size_t g_mmap_size; static int g_mmap_fd;
static GPQ8Tensor *gp_find(const char *key) {
    for (int i = 0; i < g_gp_n; i++)
        if (strcmp(g_gp_tensors[i].key, key) == 0) return &g_gp_tensors[i];
    return NULL;
}
static void load_gpq8(const char *path) {
    g_mmap_fd = open(path, O_RDONLY);
    struct stat st; fstat(g_mmap_fd, &st); g_mmap_size = st.st_size;
    g_mmap_base = mmap(NULL, g_mmap_size, PROT_READ, MAP_PRIVATE, g_mmap_fd, 0);
    const unsigned char *p = g_mmap_base;
    p += 4; g_gp_n = *(const int *)p; p += 4;
    g_gp_tensors = calloc(g_gp_n, sizeof(GPQ8Tensor));
    for (int i = 0; i < g_gp_n; i++) {
        GPQ8Tensor *t = &g_gp_tensors[i];
        int klen = *(const int *)p; p += 4;
        memcpy(t->key, p, klen); t->key[klen] = 0; p += klen;
        t->ndim = *(const int *)p; p += 4;
        for (int d = 0; d < t->ndim; d++) { t->shape[d] = *(const int *)p; p += 4; }
        t->qtype = *p; p += 1;
        t->data_len = *(const uint64_t *)p; p += 8;
        t->data = (void *)p; p += t->data_len;
        t->n_scale = *(const int *)p; p += 4;
        t->scale = (t->n_scale > 0) ? (float *)p : NULL;
        if (t->n_scale > 0) p += t->n_scale * 4;
    }
}
static void get_q8(const char *key, int8_t **q, float **s) {
    GPQ8Tensor *t = gp_find(key);
    *q = (int8_t*)t->data; *s = t->scale;
}
static float *get_f32(const char *key) {
    GPQ8Tensor *t = gp_find(key);
    return (float*)t->data;
}

/* === Layer struct (hybrid: Q8 attention + BWN MLP) === */
typedef struct {
    float *norm1_w, *norm2_w;
    int8_t *q8_q; float *s_q;
    int8_t *q8_k; float *s_k;
    int8_t *q8_v; float *s_v;
    int8_t *q8_o; float *s_o;
    BwnLayer bwn_gate, bwn_up, bwn_down;  /* MLP uses BWN (if layer in BWN range) */
    int8_t *q8_gate; float *s_gate;       /* MLP uses Q8 (if layer outside BWN range) */
    int8_t *q8_up;   float *s_up;
    int8_t *q8_down; float *s_down;
    int use_bwn;  /* 1 = BWN for this layer's MLP, 0 = Q8 */
    float *q_bias, *k_bias, *v_bias;
} Layer;
static Layer g_layers[N_LAYER];

static float *g_wte, *g_norm_f_w;
static int8_t *g_lm_head_q; static float *g_lm_head_s;
static float *g_x, *g_ln, *g_q, *g_k, *g_v, *g_attn_out, *g_proj;
static float *g_mlp_out, *g_logits;
static int8_t *g_xq_cache;
static float **kv_k, **kv_v;
static int g_n_threads = 1;

static void my_rms_norm(float *out, const float *x, const float *w, int n) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * ms * w[i];
}

static float g_rope_cos[N_CTX][HEAD_DIM/2], g_rope_sin[N_CTX][HEAD_DIM/2];
static void rope_init(void) {
    for (int p = 0; p < N_CTX; p++)
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float theta = (float)p / powf(ROPE_THETA, (float)(2*d) / HEAD_DIM);
            g_rope_cos[p][d] = cosf(theta); g_rope_sin[p][d] = sinf(theta);
        }
}
static void rope_apply(float *q, float *k, int pos) {
    for (int h = 0; h < N_HEAD; h++) {
        float *qh = q + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float c = g_rope_cos[pos][d], s = g_rope_sin[pos][d];
            float q0 = qh[d], q1 = qh[d + HEAD_DIM/2];
            qh[d] = q0*c - q1*s; qh[d+HEAD_DIM/2] = q0*s + q1*c;
        }
    }
    for (int h = 0; h < N_KV_HEAD; h++) {
        float *kh = k + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float c = g_rope_cos[pos][d], s = g_rope_sin[pos][d];
            float k0 = kh[d], k1 = kh[d + HEAD_DIM/2];
            kh[d] = k0*c - k1*s; kh[d+HEAD_DIM/2] = k0*s + k1*c;
        }
    }
}

static void gqa_attn(float *out, const float *Q, const float *Kn, const float *Vn, int layer, int pos) {
    float *kc = kv_k[layer], *vc = kv_v[layer];
    memcpy(kc + pos * KV_DIM, Kn, KV_DIM * sizeof(float));
    memcpy(vc + pos * KV_DIM, Vn, KV_DIM * sizeof(float));
    float inv = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h = 0; h < N_HEAD; h++) {
        const float *qh = Q + h * HEAD_DIM;
        int kvh = h / N_Q_PER_KV;
        float mx = -1e30f; static float scores[N_CTX];
        for (int t = 0; t <= pos; t++) {
            const float *kt = kc + t * KV_DIM + kvh * HEAD_DIM;
            float dot = 0;
            for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * kt[d];
            scores[t] = dot * inv;
            if (scores[t] > mx) mx = scores[t];
        }
        float sum = 0;
        for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
        float *oh = out + h * HEAD_DIM;
        memset(oh, 0, HEAD_DIM * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float w = scores[t] / sum;
            const float *vt = vc + t * KV_DIM + kvh * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) oh[d] += w * vt[d];
        }
    }
}

static int forward(int tok, int pos) {
    if (tok < 0 || tok >= VOCAB_SIZE) tok = 0;
    memcpy(g_x, g_wte + (size_t)tok * N_EMBD, N_EMBD * sizeof(float));
    for (int l = 0; l < N_LAYER; l++) {
        Layer *L = &g_layers[l];
        my_rms_norm(g_ln, g_x, L->norm1_w, N_EMBD);
        lal_matmul_q8_signtrick(g_q, L->q8_q, L->s_q, g_ln, L->q_bias, N_EMBD, Q_DIM);
        lal_matmul_q8_signtrick(g_k, L->q8_k, L->s_k, g_ln, L->k_bias, N_EMBD, KV_DIM);
        lal_matmul_q8_signtrick(g_v, L->q8_v, L->s_v, g_ln, L->v_bias, N_EMBD, KV_DIM);
        rope_apply(g_q, g_k, pos);
        gqa_attn(g_attn_out, g_q, g_k, g_v, l, pos);
        lal_matmul_q8_signtrick(g_proj, L->q8_o, L->s_o, g_attn_out, NULL, Q_DIM, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) g_x[i] += g_proj[i];
        my_rms_norm(g_ln, g_x, L->norm2_w, N_EMBD);
        /* MLP: BWN (binary weight) for layers in BWN range, Q8 otherwise */
        static float gate_buf[MLP_DIM], up_buf[MLP_DIM], act_buf[MLP_DIM];
        if (L->use_bwn) {
            bwn_matmul_bnn_avx2(gate_buf, g_ln, &L->bwn_gate);
            bwn_matmul_bnn_avx2(up_buf,   g_ln, &L->bwn_up);
        } else {
            lal_matmul_q8_signtrick(gate_buf, L->q8_gate, L->s_gate, g_ln, NULL, N_EMBD, MLP_DIM);
            lal_matmul_q8_signtrick(up_buf,   L->q8_up,   L->s_up,   g_ln, NULL, N_EMBD, MLP_DIM);
        }
        for (int i = 0; i < MLP_DIM; i++)
            act_buf[i] = (gate_buf[i] / (1.0f + expf(-gate_buf[i]))) * up_buf[i];
        if (L->use_bwn)
            bwn_matmul_bnn_avx2(g_mlp_out, act_buf, &L->bwn_down);
        else
            lal_matmul_q8_signtrick(g_mlp_out, L->q8_down, L->s_down, act_buf, NULL, MLP_DIM, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) g_x[i] += g_mlp_out[i];
    }
    my_rms_norm(g_ln, g_x, g_norm_f_w, N_EMBD);
    float sx = lal_quantize_x_int8(g_ln, g_xq_cache, N_EMBD);
    lal_lm_head_int8_range(g_logits, g_xq_cache, sx, g_lm_head_q, g_lm_head_s, 0, VOCAB_SIZE, N_EMBD);
    int next = lal_sample_token(g_logits, VOCAB_SIZE, 0.8f, 40, 1.1f, NULL, 0);
    return next;
}

/* Tokenizer (minimal, from qwen7b_server.c) */
typedef struct { char key[512]; int id; } TEntry;
#define TOK_HASH_BITS 18
#define TOK_HASH_SIZE (1 << TOK_HASH_BITS)
static TEntry g_htab[TOK_HASH_SIZE];
static char **g_vocab_str; static int g_vocab_str_n;
static unsigned tok_hash(const char *s) {
    unsigned h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h & (TOK_HASH_SIZE - 1);
}
static void tins(const char *key, int id) {
    unsigned h = tok_hash(key);
    while (g_htab[h].key[0]) h = (h + 1) & (TOK_HASH_SIZE - 1);
    strncpy(g_htab[h].key, key, 511); g_htab[h].key[511] = 0; g_htab[h].id = id;
}
static int tok_find(const char *key) {
    unsigned h = tok_hash(key);
    while (g_htab[h].key[0]) {
        if (strcmp(g_htab[h].key, key) == 0) return g_htab[h].id;
        h = (h + 1) & (TOK_HASH_SIZE - 1);
    }
    return -1;
}
static void load_tokenizer(const char *dir) {
    char path[1024]; snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    FILE *f = fopen(path, "r"); if (!f) { fprintf(stderr, "[!] no tokenizer\n"); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz+1); fread(buf, 1, sz, f); buf[sz] = 0; fclose(f);
    g_vocab_str = calloc(VOCAB_SIZE + 200, sizeof(char*));
    memset(g_htab, 0, sizeof(g_htab));
    char *p = strstr(buf, "\"vocab\"");
    p = strchr(p+6, '{'); p++;
    while (*p && *p != '}') {
        while (*p && (*p==' '||*p=='\n'||*p=='\r'||*p=='\t'||*p==',')) p++;
        if (*p == '}') break;
        if (*p != '"') { p++; continue; }
        p++;
        char key[512]; int klen = 0;
        while (*p && *p != '"' && klen < 511) {
            if (*p == '\\') { p++; key[klen++] = *p ? *p : '?'; }
            else key[klen++] = *p;
            p++;
        }
        key[klen] = 0;
        if (*p == '"') p++;
        while (*p && (*p==' '||*p==':')) p++;
        int id = 0;
        while (*p >= '0' && *p <= '9') { id = id*10 + (*p-'0'); p++; }
        tins(key, id);
        if (id < VOCAB_SIZE + 200) g_vocab_str[id] = strdup(key);
        if (id+1 > g_vocab_str_n) g_vocab_str_n = id+1;
    }
    free(buf);
}

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    const char *weights = "prebuilt/qwen7b_weights.bin";
    const char *tokdir = "prebuilt/qwen7b_tokenizer";
    const char *prompt = "What is the capital of France?";
    int n_gen = 20;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--weights") && i+1<argc) weights=argv[++i];
        else if (!strcmp(argv[i],"--tokenizer") && i+1<argc) tokdir=argv[++i];
        else if (!strcmp(argv[i],"--prompt") && i+1<argc) prompt=argv[++i];
        else if (!strcmp(argv[i],"--n") && i+1<argc) n_gen=atoi(argv[++i]);
    }
    printf("=== Qwen2.5-7B Hybrid: Q8 attention + BWN MLP (test quality) ===\n");
    load_gpq8(weights);
    g_norm_f_w = get_f32("model.norm.weight");
    g_wte = get_f32("model.embed_tokens.weight");
    char key[256];
    /* First pass: wire up attention + load Q8 MLP pointers for all layers */
    for (int l = 0; l < N_LAYER; l++) {
        Layer *L = &g_layers[l];
        sprintf(key, "model.layers.%d.input_layernorm.weight", l); L->norm1_w = get_f32(key);
        sprintf(key, "model.layers.%d.post_attention_layernorm.weight", l); L->norm2_w = get_f32(key);
        sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l); get_q8(key, &L->q8_q, &L->s_q);
        sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l); get_q8(key, &L->q8_k, &L->s_k);
        sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l); get_q8(key, &L->q8_v, &L->s_v);
        sprintf(key, "model.layers.%d.self_attn.o_proj.weight", l); get_q8(key, &L->q8_o, &L->s_o);
        sprintf(key, "model.layers.%d.self_attn.q_proj.bias", l); L->q_bias = get_f32(key);
        sprintf(key, "model.layers.%d.self_attn.k_proj.bias", l); L->k_bias = get_f32(key);
        sprintf(key, "model.layers.%d.self_attn.v_proj.bias", l); L->v_bias = get_f32(key);
        sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l); get_q8(key, &L->q8_gate, &L->s_gate);
        sprintf(key, "model.layers.%d.mlp.up_proj.weight", l); get_q8(key, &L->q8_up, &L->s_up);
        sprintf(key, "model.layers.%d.mlp.down_proj.weight", l); get_q8(key, &L->q8_down, &L->s_down);
    }
    /* Parse --bwn-start/--bwn-end: BWN for middle layers, Q8 for first/last */
    int bwn_start = 2, bwn_end = N_LAYER - 2;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--bwn-start") && i+1<argc) bwn_start=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--bwn-end") && i+1<argc) bwn_end=atoi(argv[++i]);
    }
    /* Binarize MLP for layers in [bwn_start, bwn_end) */
    for (int l = bwn_start; l < bwn_end && l < N_LAYER; l++) {
        Layer *L = &g_layers[l];
        L->use_bwn = 1;
        binarize_q8_to_bwn(L->q8_gate, L->s_gate, &L->bwn_gate, N_EMBD, MLP_DIM);
        binarize_q8_to_bwn(L->q8_up,   L->s_up,   &L->bwn_up,   N_EMBD, MLP_DIM);
        binarize_q8_to_bwn(L->q8_down, L->s_down, &L->bwn_down, MLP_DIM, N_EMBD);
        if ((l+1) % 7 == 0 || l == bwn_end - 1) {
            printf("[*] binarized MLP for layers %d-%d\n", bwn_start, l);
            fflush(stdout);
        }
    }
    printf("[*] BWN MLP for layers %d-%d (%d layers), Q8 for layers 0-%d and %d-%d\n",
           bwn_start, bwn_end-1, bwn_end - bwn_start, bwn_start-1, bwn_end, N_LAYER-1);
    /* int8 LM head */
    float *lm_head_f = get_f32("lm_head.weight");
    g_lm_head_q = memalign(32, (size_t)VOCAB_SIZE * N_EMBD);
    g_lm_head_s = memalign(32, VOCAB_SIZE * sizeof(float));
    lal_quantize_q8_per_row(lm_head_f, g_lm_head_q, g_lm_head_s, N_EMBD, VOCAB_SIZE);
    g_xq_cache = memalign(32, N_EMBD);
    g_x = memalign(32, N_EMBD * sizeof(float));
    g_ln = memalign(32, N_EMBD * sizeof(float));
    g_q = memalign(32, Q_DIM * sizeof(float));
    g_k = memalign(32, KV_DIM * sizeof(float));
    g_v = memalign(32, KV_DIM * sizeof(float));
    g_attn_out = memalign(32, Q_DIM * sizeof(float));
    g_proj = memalign(32, N_EMBD * sizeof(float));
    g_mlp_out = memalign(32, N_EMBD * sizeof(float));
    g_logits = memalign(32, VOCAB_SIZE * sizeof(float));
    kv_k = malloc(N_LAYER * sizeof(float*));
    kv_v = malloc(N_LAYER * sizeof(float*));
    for (int l = 0; l < N_LAYER; l++) {
        kv_k[l] = memalign(32, N_CTX * KV_DIM * sizeof(float));
        kv_v[l] = memalign(32, N_CTX * KV_DIM * sizeof(float));
    }
    rope_init();
    load_tokenizer(tokdir);

    printf("\n[*] prompt: \"%s\"  n_gen=%d\n\n", prompt, n_gen);

    /* Simple single-token prompt encoding (just use first vocab id for testing) */
    /* For quality test, encode prompt as space-prefixed words */
    int pids[64]; int n_prompt = 0;
    {
        char word[256]; int wlen = 0; int first = 1;
        for (const char *p = prompt; ; p++) {
            if (*p == ' ' || *p == '\0') {
                if (wlen > 0) {
                    word[wlen] = 0;
                    char prefixed[257];
                    snprintf(prefixed, sizeof(prefixed), "%s%s", first ? "" : " ", word);
                    int id = tok_find(prefixed);
                    if (id < 0) id = tok_find(word);
                    if (id >= 0 && n_prompt < 64) pids[n_prompt++] = id;
                    wlen = 0; first = 0;
                }
                if (*p == '\0') break;
            } else { if (wlen < 255) word[wlen++] = *p; }
        }
    }
    /* Add chat template: im_start + user + \n + prompt + im_end + \n + im_start + assistant + \n */
    int chat_pids[80]; int n_chat = 0;
    int im_start = 151644, im_end = 151643;
    int nl_id = tok_find("\xC4\x8A");  /* Ċ = newline */
    chat_pids[n_chat++] = im_start;
    int u_id = tok_find("user"); if (u_id >= 0) chat_pids[n_chat++] = u_id;
    if (nl_id >= 0) chat_pids[n_chat++] = nl_id;
    for (int i = 0; i < n_prompt; i++) chat_pids[n_chat++] = pids[i];
    chat_pids[n_chat++] = im_end;
    if (nl_id >= 0) chat_pids[n_chat++] = nl_id;
    chat_pids[n_chat++] = im_start;
    int a_id = tok_find("assistant"); if (a_id >= 0) chat_pids[n_chat++] = a_id;
    if (nl_id >= 0) chat_pids[n_chat++] = nl_id;

    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int pos = 0, next = -1;
    for (int i = 0; i < n_chat; i++) { next = forward(chat_pids[i], pos); pos++; }
    int gen_count = 0;
    for (int g = 0; g < n_gen && pos < N_CTX; g++) {
        char ts[256];
        if (g_vocab_str && next >= 0 && next < g_vocab_str_n && g_vocab_str[next])
            lal_decode_bpe_token(g_vocab_str[next], ts, (int)sizeof(ts));
        else ts[0] = 0;
        printf("%s", ts); fflush(stdout);
        if (next == im_end) break;
        next = forward(next, pos); pos++; gen_count++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("\n\n[*] %d tokens in %.2fs (%.2f tok/s)\n", gen_count, dt, gen_count/(dt+1e-9));
    printf("[*] MLP bandwidth: BWN packed = %.0f MB vs Q8 = %.0f MB per layer\n",
           3.0*MLP_DIM*N_EMBD/8/1048576, 3.0*MLP_DIM*N_EMBD/1048576);
    return 0;
}
