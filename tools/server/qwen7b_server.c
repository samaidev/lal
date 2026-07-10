/* qwen7b_server.c — Qwen2.5-7B-Instruct inference server (LAL, Q8, GQA)
 *
 * Architecture: 28 layers, 3584 hidden, 28Q/4KV heads, 128 head_dim,
 *               18944 MLP (SwiGLU), 152064 vocab, 32768 max ctx.
 *
 * Key innovation: loads pre-quantized GPQ8 file (7 GB) instead of
 * float32 GPW2 (30 GB). No runtime Q8 quantization for layer weights —
 * they're already Q8 in the file. Only embed_tokens (F32 in file) is
 * quantized at startup for the int8 LM head.
 *
 * Build: make qwen7b-server
 * Run:   ./prebuilt/qwen7b_server --weights prebuilt/qwen7b_weights.bin \
 *          --tokenizer prebuilt/qwen7b_tokenizer --prompt "Hello" --n 30
 */
#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* === Architecture constants === */
#define N_EMBD       3584
#define N_LAYER      28
#define N_HEAD       28
#define N_KV_HEAD    4
#define HEAD_DIM     128
#define N_Q_PER_KV   (N_HEAD / N_KV_HEAD)  /* 7 */
#define MLP_DIM      18944
#define VOCAB_SIZE   152064
#define N_CTX        4096   /* reduced from 32768 to save KV cache memory */
#define ROPE_THETA   1000000.0f
#define RMS_EPS      1e-6f
#define KV_DIM       (N_KV_HEAD * HEAD_DIM)  /* 512 */
#define Q_DIM        (N_HEAD * HEAD_DIM)     /* 3584 */

/* === SIMD macros === */
#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  #define LAL_HAVE_AVX2 1
  typedef __m256 v8f;
  #define V8F_ZERO()  _mm256_setzero_ps()
  #define V8F_LOAD(p) _mm256_loadu_ps(p)
  #define V8F_FMADD(a,b,c) _mm256_fmadd_ps((a),(b),(c))
  static inline float v8f_hsum(v8f v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
  }
#endif

/* === Reusable SDK headers === */
#define XQ_MAX 18944  /* max in_dim = MLP_DIM */
#include "runtime/lal_runtime.h"
#include "runtime/lal_q8_kernel.h"
#include "runtime/lal_sampling.h"
#include "runtime/lal_dequant.h"
#include "runtime/lal_tokenizer.h"

/* === Layer struct === */
typedef struct {
    float *norm1_w, *norm2_w;
    int8_t *q8_q;  float *s_q;   /* [Q_DIM, N_EMBD] = [3584, 3584] */
    int8_t *q8_k;  float *s_k;   /* [KV_DIM, N_EMBD] = [512, 3584] */
    int8_t *q8_v;  float *s_v;   /* [KV_DIM, N_EMBD] = [512, 3584] */
    int8_t *q8_o;  float *s_o;   /* [N_EMBD, Q_DIM] = [3584, 3584] */
    int8_t *q8_gate; float *s_gate; /* [MLP_DIM, N_EMBD] */
    int8_t *q8_up;   float *s_up;   /* [MLP_DIM, N_EMBD] */
    int8_t *q8_down; float *s_down; /* [N_EMBD, MLP_DIM] */
    float *q_bias, *k_bias, *v_bias;
} Layer;

static Layer g_layers[N_LAYER];

/* === Global state === */
static float *g_wte;          /* [VOCAB, N_EMBD] float (from file) */
static float *g_norm_f_w;
static int8_t *g_wte_q;       /* [VOCAB, N_EMBD] int8 (quantized at startup) */
static float *g_wte_scale;    /* [VOCAB] */
static float *g_x, *g_ln, *g_q, *g_k, *g_v, *g_attn_out, *g_proj;
static float *g_gate, *g_up, *g_gate_up, *g_mlp_out;
static float *g_logits;
static float **kv_k, **kv_v;  /* [N_LAYER][N_CTX * KV_DIM] */
static int g_n_threads = 1;
static float g_temperature = 0.8f;
static int g_top_k = 40;
static float g_rep_penalty = 1.1f;
static int g_recent[256], g_n_recent = 0;

/* === GPQ8 tensor (loaded from file) === */
typedef struct {
    char key[128];
    int ndim, shape[4];
    int qtype;      /* 0=F32, 1=Q8 */
    uint64_t data_len;
    void *data;     /* int8_t* if Q8, float* if F32 */
    int n_scale;
    float *scale;   /* per-row scale if Q8, NULL if F32 */
} GPQ8Tensor;

static GPQ8Tensor *g_gp_tensors;
static int g_gp_n;

/* Find tensor by key */
static GPQ8Tensor *gp_find(const char *key) {
    for (int i = 0; i < g_gp_n; i++)
        if (strcmp(g_gp_tensors[i].key, key) == 0) return &g_gp_tensors[i];
    fprintf(stderr, "[!] tensor not found: %s\n", key);
    return NULL;
}

/* Load GPQ8 file */
static void *g_mmap_base;
static size_t g_mmap_size;
static int g_mmap_fd;

static void load_gpq8(const char *path) {
    printf("[*] mmap-loading %s ...\n", path); fflush(stdout);
    g_mmap_fd = open(path, O_RDONLY);
    if (g_mmap_fd < 0) { fprintf(stderr, "[!] cannot open %s\n", path); exit(1); }
    struct stat st;
    if (fstat(g_mmap_fd, &st) < 0) { fprintf(stderr, "[!] fstat\n"); exit(1); }
    g_mmap_size = st.st_size;
    g_mmap_base = mmap(NULL, g_mmap_size, PROT_READ, MAP_PRIVATE, g_mmap_fd, 0);
    if (g_mmap_base == MAP_FAILED) { fprintf(stderr, "[!] mmap failed\n"); exit(1); }
    const unsigned char *p = (const unsigned char *)g_mmap_base;
    if (memcmp(p, "GPQ8", 4) != 0) { fprintf(stderr, "[!] bad magic\n"); exit(1); }
    p += 4;
    g_gp_n = *(const int *)p; p += 4;
    printf("[*] %d tensors (%.1f GB mmap'd)\n", g_gp_n, (double)g_mmap_size / 1073741824); fflush(stdout);
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
    printf("[*] all tensors mapped\n"); fflush(stdout);
}

/* Get Q8 data pointer + scale for a weight tensor */
static void get_q8(const char *key, int8_t **q, float **s) {
    GPQ8Tensor *t = gp_find(key);
    if (!t || t->qtype != 1) { fprintf(stderr, "[!] %s not Q8\n", key); exit(1); }
    *q = (int8_t*)t->data;
    *s = t->scale;
}

/* Get F32 data pointer */
static float *get_f32(const char *key) {
    GPQ8Tensor *t = gp_find(key);
    if (!t || t->qtype != 0) { fprintf(stderr, "[!] %s not F32\n", key); exit(1); }
    return (float*)t->data;
}

/* === RMSNorm === */
static void qwen7b_rms_norm(float *out, const float *x, const float *w, int n) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * ms * w[i];
}

/* === RoPE === */
static float g_rope_cos[N_CTX][HEAD_DIM/2];
static float g_rope_sin[N_CTX][HEAD_DIM/2];
static void rope_init(void) {
    for (int p = 0; p < N_CTX; p++)
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float theta = (float)p / powf(ROPE_THETA, (float)(2*d) / HEAD_DIM);
            g_rope_cos[p][d] = cosf(theta);
            g_rope_sin[p][d] = sinf(theta);
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

/* === GQA Attention === */
static void gqa_attn(float *out, const float *Q, const float *Kn, const float *Vn,
                     int layer, int pos) {
    float *k_cache = kv_k[layer];
    float *v_cache = kv_v[layer];
    /* Store K, V into cache */
    memcpy(k_cache + pos * KV_DIM, Kn, KV_DIM * sizeof(float));
    memcpy(v_cache + pos * KV_DIM, Vn, KV_DIM * sizeof(float));

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h = 0; h < N_HEAD; h++) {
        const float *qh = Q + h * HEAD_DIM;
        int kvh = h / N_Q_PER_KV;
        float max_score = -1e30f;
        static float scores[N_CTX];
        for (int t = 0; t <= pos; t++) {
            const float *kt = k_cache + t * KV_DIM + kvh * HEAD_DIM;
            float dot = 0;
            for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * kt[d];
            scores[t] = dot * inv_sqrt;
            if (scores[t] > max_score) max_score = scores[t];
        }
        float sum = 0;
        for (int t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        float *oh = out + h * HEAD_DIM;
        memset(oh, 0, HEAD_DIM * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float w = scores[t] / sum;
            const float *vt = v_cache + t * KV_DIM + kvh * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) oh[d] += w * vt[d];
        }
    }
}

/* === Fused SwiGLU MLP === */
static void fused_swiglu(const int8_t *q_gate, const float *s_gate,
                         const int8_t *q_up, const float *s_up,
                         const int8_t *q_down, const float *s_down,
                         const float *x, float *out, int in_dim, int hid, int out_dim) {
    static float gate_buf[MLP_DIM], up_buf[MLP_DIM], act_buf[MLP_DIM];
    /* gate = q_gate @ x */
    lal_matmul_q8_signtrick(gate_buf, q_gate, s_gate, x, NULL, in_dim, hid);
    /* up = q_up @ x */
    lal_matmul_q8_signtrick(up_buf, q_up, s_up, x, NULL, in_dim, hid);
    /* SiLU(gate) * up */
    for (int i = 0; i < hid; i++)
        act_buf[i] = (gate_buf[i] / (1.0f + expf(-gate_buf[i]))) * up_buf[i];
    /* down = q_down @ act */
    lal_matmul_q8_signtrick(out, q_down, s_down, act_buf, NULL, hid, out_dim);
}

/* === Forward pass === */
static int forward(int tok, int pos) {
    /* Embedding lookup */

    for (int l = 0; l < N_LAYER; l++) {
        Layer *L = &g_layers[l];
        /* Pre-attn RMSNorm */
        qwen7b_rms_norm(g_ln, g_x, L->norm1_w, N_EMBD);
        /* Q/K/V projections (Q8 from GPQ8 file, no runtime quantization) */
        lal_matmul_q8_signtrick(g_q, L->q8_q, L->s_q, g_ln, L->q_bias, N_EMBD, Q_DIM);
        lal_matmul_q8_signtrick(g_k, L->q8_k, L->s_k, g_ln, L->k_bias, N_EMBD, KV_DIM);
        lal_matmul_q8_signtrick(g_v, L->q8_v, L->s_v, g_ln, L->v_bias, N_EMBD, KV_DIM);
        /* RoPE */
        rope_apply(g_q, g_k, pos);
        /* GQA Attention */
        gqa_attn(g_attn_out, g_q, g_k, g_v, l, pos);
        /* O proj + residual */
        lal_matmul_q8_signtrick(g_proj, L->q8_o, L->s_o, g_attn_out, NULL, Q_DIM, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) g_x[i] += g_proj[i];
        /* Pre-MLP RMSNorm */
        qwen7b_rms_norm(g_ln, g_x, L->norm2_w, N_EMBD);
        /* Fused SwiGLU MLP + residual */
        fused_swiglu(L->q8_gate, L->s_gate, L->q8_up, L->s_up,
                     L->q8_down, L->s_down, g_ln, g_mlp_out, N_EMBD, MLP_DIM, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) g_x[i] += g_mlp_out[i];
    }

    /* Final RMSNorm */
    qwen7b_rms_norm(g_ln, g_x, g_norm_f_w, N_EMBD);

    /* LM head: F32 dot product (logits = wte @ x) */
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const float *row = g_wte + (size_t)v * N_EMBD;
        v8f acc = V8F_ZERO(); int i = 0;
        for (; i + 8 <= N_EMBD; i += 8)
            acc = V8F_FMADD(V8F_LOAD(g_ln+i), V8F_LOAD(row+i), acc);
        float dot = v8f_hsum(acc);
        for (; i < N_EMBD; i++) dot += g_ln[i] * row[i];
        g_logits[v] = dot;
    }

    /* Sample */
    int next = lal_sample_token(g_logits, VOCAB_SIZE, g_temperature, g_top_k, g_rep_penalty, g_recent, g_n_recent);
    if (g_n_recent < 256) g_recent[g_n_recent++] = next;
    else { memmove(g_recent, g_recent+1, 255*sizeof(int)); g_recent[255] = next; }
    return next;
}

/* === Tokenizer (BPE, HuggingFace tokenizer.json) === */
/* (Copied from qwen_server.c — same logic) */
typedef struct { char key[512]; int id; } TEntry;
#define TOK_HASH_BITS 18
#define TOK_HASH_SIZE (1 << TOK_HASH_BITS)
static TEntry g_htab[TOK_HASH_SIZE];
static char **g_vocab_str;
static int g_vocab_str_n;

static unsigned tok_hash(const char *s) {
    unsigned h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h & (TOK_HASH_SIZE - 1);
}
static void tins(const char *key, int id) {
    unsigned h = tok_hash(key);
    while (g_htab[h].key[0]) h = (h + 1) & (TOK_HASH_SIZE - 1);
    strncpy(g_htab[h].key, key, 511); g_htab[h].key[511] = 0;
    g_htab[h].id = id;
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
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[!] cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz+1); fread(buf, 1, sz, f); buf[sz] = 0; fclose(f);
    int mx = VOCAB_SIZE + 200;
    g_vocab_str = calloc(mx, sizeof(char*));
    memset(g_htab, 0, sizeof(g_htab));
    char *p = strstr(buf, "\"vocab\"");
    if (!p) { fprintf(stderr, "[!] vocab not found\n"); free(buf); return; }
    p = strchr(p+6, '{'); if (!p) return; p++;
    int cnt = 0;
    while (*p && *p != '}' && cnt < mx) {
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
        if (id < mx) g_vocab_str[id] = strdup(key);
        if (id+1 > g_vocab_str_n) g_vocab_str_n = id+1;
        cnt++;
    }
    printf("[*] tokenizer: %d tokens\n", cnt);
    free(buf);
}

/* Simple text encoder: word-level + space token (Qwen BPE simplified) */
static int *encode_text(const char *text, int *n_out) {
    int *ids = malloc(strlen(text) * 4 * sizeof(int));
    int n = 0;
    /* Try encoding as space-prefixed words */
    char word[512];
    int wlen = 0;
    int first = 1;
    for (const char *p = text; ; p++) {
        if (*p == ' ' || *p == '\0' || *p == '\n') {
            if (wlen > 0) {
                word[wlen] = 0;
                char prefixed[513];
                snprintf(prefixed, sizeof(prefixed), "%s%s", first ? "" : " ", word);
                int id = tok_find(prefixed);
                if (id >= 0) ids[n++] = id;
                else {
                    /* Try without prefix */
                    id = tok_find(word);
                    if (id >= 0) ids[n++] = id;
                    else {
                        /* Fall back: encode each char */
                        for (int i = 0; i < wlen; i++) {
                            char c[2] = {word[i], 0};
                            int cid = tok_find(c);
                            if (cid >= 0) ids[n++] = cid;
                        }
                    }
                }
                wlen = 0;
                first = 0;
            }
            if (*p == ' ') { /* skip */ }
            else if (*p == '\0') break;
        } else {
            word[wlen++] = *p;
        }
    }
    *n_out = n;
    return ids;
}

/* === Main === */
int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    const char *weights = "prebuilt/qwen7b_weights.bin";
    const char *tokdir = "prebuilt/qwen7b_tokenizer";
    const char *prompt = "The meaning of life is";
    int n_gen = 30;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--weights") && i+1<argc) weights=argv[++i];
        else if (!strcmp(argv[i],"--tokenizer") && i+1<argc) tokdir=argv[++i];
        else if (!strcmp(argv[i],"--prompt") && i+1<argc) prompt=argv[++i];
        else if (!strcmp(argv[i],"--n") && i+1<argc) n_gen=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--threads") && i+1<argc) g_n_threads=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--temp") && i+1<argc) g_temperature=atof(argv[++i]);
        else if (!strcmp(argv[i],"--top-k") && i+1<argc) g_top_k=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--rep-penalty") && i+1<argc) g_rep_penalty=atof(argv[++i]);
    }

    printf("=== Qwen2.5-7B-Instruct (LAL, Q8, GQA, GPQ8) ===\n");
    printf("[*] %d layers, %d hidden, %dQ/%dKV heads, %d head_dim, %d MLP, %d vocab\n",
           N_LAYER, N_EMBD, N_HEAD, N_KV_HEAD, HEAD_DIM, MLP_DIM, VOCAB_SIZE);

    /* Load GPQ8 weights */
    load_gpq8(weights);

    /* Wire up layer pointers (Q8 data is already in GPQ8 tensors) */
    g_norm_f_w = get_f32("model.norm.weight");
    g_wte = get_f32("model.embed_tokens.weight");
    char key[256];
    for (int l = 0; l < N_LAYER; l++) {
        Layer *L = &g_layers[l];
        sprintf(key, "model.layers.%d.input_layernorm.weight", l); L->norm1_w = get_f32(key);
        sprintf(key, "model.layers.%d.post_attention_layernorm.weight", l); L->norm2_w = get_f32(key);
        sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l); get_q8(key, &L->q8_q, &L->s_q);
        sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l); get_q8(key, &L->q8_k, &L->s_k);
        sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l); get_q8(key, &L->q8_v, &L->s_v);
        sprintf(key, "model.layers.%d.self_attn.o_proj.weight", l); get_q8(key, &L->q8_o, &L->s_o);
        sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l); get_q8(key, &L->q8_gate, &L->s_gate);
        sprintf(key, "model.layers.%d.mlp.up_proj.weight", l); get_q8(key, &L->q8_up, &L->s_up);
        sprintf(key, "model.layers.%d.mlp.down_proj.weight", l); get_q8(key, &L->q8_down, &L->s_down);
        sprintf(key, "model.layers.%d.self_attn.q_proj.bias", l); L->q_bias = get_f32(key);
        sprintf(key, "model.layers.%d.self_attn.k_proj.bias", l); L->k_bias = get_f32(key);
        sprintf(key, "model.layers.%d.self_attn.v_proj.bias", l); L->v_bias = get_f32(key);
    }

/* F32 LM head (no int8 quantization) */
    printf("[*] using F32 LM head\n"); fflush(stdout);

        /* Allocate working buffers */
    g_x = memalign(32, N_EMBD * sizeof(float));
    g_ln = memalign(32, N_EMBD * sizeof(float));
    g_q = memalign(32, Q_DIM * sizeof(float));
    g_k = memalign(32, KV_DIM * sizeof(float));
    g_v = memalign(32, KV_DIM * sizeof(float));
    g_attn_out = memalign(32, Q_DIM * sizeof(float));
    g_proj = memalign(32, N_EMBD * sizeof(float));
    g_gate = memalign(32, MLP_DIM * sizeof(float));
    g_up = memalign(32, MLP_DIM * sizeof(float));
    g_gate_up = memalign(32, MLP_DIM * sizeof(float));
    g_mlp_out = memalign(32, N_EMBD * sizeof(float));
    g_logits = memalign(32, VOCAB_SIZE * sizeof(float));

    /* KV cache */
    kv_k = malloc(N_LAYER * sizeof(float*));
    kv_v = malloc(N_LAYER * sizeof(float*));
    for (int l = 0; l < N_LAYER; l++) {
        kv_k[l] = memalign(32, N_CTX * KV_DIM * sizeof(float));
        kv_v[l] = memalign(32, N_CTX * KV_DIM * sizeof(float));
    }
    printf("[*] KV cache: %d layers x %d ctx x %d kv_dim x 4B = %.0f MB\n",
           N_LAYER, N_CTX, KV_DIM, (double)N_LAYER*N_CTX*KV_DIM*4*2/1048576);

    /* RoPE */
    rope_init();

    /* Tokenizer */
    load_tokenizer(tokdir);

    /* Generate */
    printf("\n[*] prompt: \"%s\" (temp=%.2f top_k=%d rep_penalty=%.2f)\n",
           prompt, g_temperature, g_top_k, g_rep_penalty);
    printf("[*] generating %d tokens (threads=%d)...\n\n", n_gen, g_n_threads);

    int n_prompt;
    int *pids = encode_text(prompt, &n_prompt);
    printf("[*] prompt: %d tokens\n", n_prompt);

    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int pos = 0, next = -1;
    for (int i = 0; i < n_prompt; i++) { next = forward(pids[i], pos); pos++; if (pos >= N_CTX) break; }
    int gen_count = 0; char out_buf[65536] = {0}; int opos = 0;
    for (int g = 0; g < n_gen && pos < N_CTX; g++) {
        char ts[256];
        if (g_vocab_str && next >= 0 && next < g_vocab_str_n && g_vocab_str[next])
            lal_decode_bpe_token(g_vocab_str[next], ts, (int)sizeof(ts));
        else ts[0] = 0;
        int slen = strlen(ts);
        if (opos + slen < (int)sizeof(out_buf) - 1) { memcpy(out_buf+opos, ts, slen); opos += slen; }
        if (next == 151643) break; /* EOS */
        printf("%s", ts); fflush(stdout);
        next = forward(next, pos); pos++; gen_count++;
    }
    out_buf[opos] = 0;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("\n\n[*] %d tokens in %.2fs (%.1f tok/s)\n", gen_count, dt, gen_count/(dt+1e-9));
    printf("[*] output: %s\n", out_buf);
    printf("\n[*] done. Q8 + int8 LM head + GPQ8 + %d thread(s).\n", g_n_threads);
    free(pids);
    return 0;
}
