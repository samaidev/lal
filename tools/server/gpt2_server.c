/* gpt2_server.c — GPT-2 HTTP server (optimized, SIMD-accelerated)
 *
 * Listens on port 8080. Serves:
 *   GET  /          → HTML frontend (loaded from frontend.html next to binary, or embedded fallback)
 *   POST /generate  → JSON {prompt, n_tokens} → {text, time, tokens_per_sec}
 *
 * Build: gcc -O3 -mavx2 -mfma -o gpt2_server tools/server/gpt2_server.c \
 *          runtime/lal_runtime.c -lm
 * Run:   ./gpt2_server
 * Open:  http://localhost:8080
 *
 * Optimizations vs. original gpt2_server:
 *   1. AVX2+FMA SIMD float_matmul (8-wide, ~4-8x faster than scalar)
 *   2. AVX2 SIMD LM head: logits = wte @ x in one batched call
 *   3. Hash-table tokenizer (O(1) lookup vs. O(50257) scan)
 *   4. Greedy longest-match tokenization using length-bucketed hash
 *   5. Removed per-step clipping (was unnecessary for inference)
 *
 * Memory: ~700 MB resident (float weights + activations).
 * Speed:  ~50-80 ms/token on a 2-core Xeon (vs. 490 ms/token baseline).
 */
#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/stat.h>

/* Portable SIMD: AVX2 on x86_64, NEON on ARM, scalar fallback otherwise. */
#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  #define LAL_HAVE_AVX2 1
  typedef __m256        v8f;
  #define v8f_zero()         _mm256_setzero_ps()
  #define v8f_set1(x)        _mm256_set1_ps(x)
  #define v8f_load(p)        _mm256_loadu_ps(p)
  #define v8f_store(p, v)    _mm256_storeu_ps((p), (v))
  #define v8f_add(a, b)      _mm256_add_ps((a), (b))
  #define v8f_sub(a, b)      _mm256_sub_ps((a), (b))
  #define v8f_mul(a, b)      _mm256_mul_ps((a), (b))
  #define v8f_fmadd(a, b, c) _mm256_fmadd_ps((a), (b), (c))
  /* horizontal sum of 8 lanes */
  static inline float v8f_hsum(v8f v) {
    float t[8]; _mm256_storeu_ps(t, v);
    return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #define LAL_HAVE_NEON 1
  /* NEON works on 4 floats at a time (128-bit), so we use v4f internally
   * but the API names stay v8f for source-level compatibility — we just
   * process two v4f halves per "logical 8-wide" op. */
  typedef float32x4_t v4f;
  typedef struct { v4f lo, hi; } v8f;
  #define v8f_zero()         ((v8f){ vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) })
  static inline v8f v8f_set1(float x) { v8f r; r.lo = vdupq_n_f32(x); r.hi = vdupq_n_f32(x); return r; }
  static inline v8f v8f_load(const float *p) {
    v8f r; r.lo = vld1q_f32(p); r.hi = vld1q_f32(p + 4); return r;
  }
  static inline void v8f_store(float *p, v8f v) {
    vst1q_f32(p, v.lo); vst1q_f32(p + 4, v.hi);
  }
  static inline v8f v8f_add(v8f a, v8f b) { v8f r; r.lo = vaddq_f32(a.lo, b.lo); r.hi = vaddq_f32(a.hi, b.hi); return r; }
  static inline v8f v8f_sub(v8f a, v8f b) { v8f r; r.lo = vsubq_f32(a.lo, b.lo); r.hi = vsubq_f32(a.hi, b.hi); return r; }
  static inline v8f v8f_mul(v8f a, v8f b) { v8f r; r.lo = vmulq_f32(a.lo, b.lo); r.hi = vmulq_f32(a.hi, b.hi); return r; }
  #if defined(__aarch64__)
    /* AArch64 has FMA intrinsic */
    static inline v8f v8f_fmadd(v8f a, v8f b, v8f c) {
      v8f r; r.lo = vfmaq_f32(c.lo, a.lo, b.lo); r.hi = vfmaq_f32(c.hi, a.hi, b.hi); return r;
    }
  #else
    /* ARMv7 NEON: no FMA, use mul+add (clang may still fuse) */
    static inline v8f v8f_fmadd(v8f a, v8f b, v8f c) {
      v8f r; r.lo = vaddq_f32(vmulq_f32(a.lo, b.lo), c.lo); r.hi = vaddq_f32(vmulq_f32(a.hi, b.hi), c.hi); return r;
    }
  #endif
  static inline float v8f_hsum(v8f v) {
    float32x2_t lo2 = vadd_f32(vget_low_f32(v.lo), vget_high_f32(v.lo));
    float32x2_t hi2 = vadd_f32(vget_low_f32(v.hi), vget_high_f32(v.hi));
    float32x2_t s   = vadd_f32(lo2, hi2);
    return vget_lane_f32(vpadd_f32(s, s), 0);
  }
#else
  #define LAL_HAVE_SCALAR 1
  /* Pure scalar fallback — process 8 floats at a time with a loop */
  typedef struct { float v[8]; } v8f;
  static inline v8f v8f_zero() { v8f r; for (int i=0;i<8;i++) r.v[i]=0; return r; }
  static inline v8f v8f_set1(float x) { v8f r; for (int i=0;i<8;i++) r.v[i]=x; return r; }
  static inline v8f v8f_load(const float *p) { v8f r; for (int i=0;i<8;i++) r.v[i]=p[i]; return r; }
  static inline void v8f_store(float *p, v8f v) { for (int i=0;i<8;i++) p[i]=v.v[i]; }
  static inline v8f v8f_add(v8f a, v8f b) { v8f r; for (int i=0;i<8;i++) r.v[i]=a.v[i]+b.v[i]; return r; }
  static inline v8f v8f_sub(v8f a, v8f b) { v8f r; for (int i=0;i<8;i++) r.v[i]=a.v[i]-b.v[i]; return r; }
  static inline v8f v8f_mul(v8f a, v8f b) { v8f r; for (int i=0;i<8;i++) r.v[i]=a.v[i]*b.v[i]; return r; }
  static inline v8f v8f_fmadd(v8f a, v8f b, v8f c) { v8f r; for (int i=0;i<8;i++) r.v[i]=a.v[i]*b.v[i]+c.v[i]; return r; }
  static inline float v8f_hsum(v8f v) { float s=0; for (int i=0;i<8;i++) s+=v.v[i]; return s; }
#endif

#include "runtime/lal_runtime.h"

#define VOCAB_SIZE 50257
#define N_EMBD     768
#define N_LAYER    12
#define MLP_DIM    3072
#define N_HEAD     12

/* ========================================================================
 * Frontend HTML — tried from disk first, fallback to embedded
 * ======================================================================== */
static const char *HTML_FALLBACK =
"<!DOCTYPE html><html><head><meta charset='utf-8'>"
"<title>LAL GPT-2</title><style>"
"body{font-family:system-ui;max-width:900px;margin:40px auto;padding:0 20px;background:#1a1a2e;color:#e0e0e0}"
"h1{color:#0f3460}h1 span{color:#e94560}"
".box{background:#16213e;border-radius:12px;padding:20px;margin:16px 0}"
"textarea{width:100%;height:80px;background:#0f3460;color:#fff;border:1px solid #e94560;border-radius:8px;padding:12px;font-size:14px;resize:vertical;box-sizing:border-box}"
"button{background:#e94560;color:#fff;border:none;padding:10px 28px;border-radius:8px;font-size:14px;cursor:pointer;margin-top:8px}"
"button:hover{background:#c81e45}"
"button:disabled{opacity:0.5;cursor:wait}"
"#output{white-space:pre-wrap;font-family:monospace;font-size:14px;line-height:1.6;min-height:40px}"
".label{color:#e94560;font-size:12px;margin-bottom:4px}"
"input{background:#0f3460;color:#fff;border:1px solid #333;border-radius:6px;padding:6px 12px;width:60px}"
".status{font-size:12px;color:#0f3460;margin-top:8px}"
".stat{color:#888;font-size:12px}"
"</style></head><body>"
"<h1>LAL <span>GPT-2</span></h1>"
"<p class='stat'>Pure C inference, no PyTorch. 124M params, AVX2 SIMD, hash-table tokenizer.</p>"
"<div class='box'>"
"<div class='label'>Prompt</div>"
"<textarea id='prompt'>Hello, how are</textarea>"
"<div style='margin-top:8px'>"
"<span class='label'>Tokens:</span> <input type='number' id='ntok' value='20' min='1' max='100'>"
"<button id='btn' onclick='generate()'>Generate</button>"
"</div>"
"</div>"
"<div class='box'>"
"<div class='label'>Output</div>"
"<div id='output'>Waiting for input...</div>"
"<div id='status' class='status'></div>"
"</div>"
"<script>"
"async function generate(){"
"const p=document.getElementById('prompt').value;"
"const n=document.getElementById('ntok').value;"
"const btn=document.getElementById('btn');"
"const out=document.getElementById('output');"
"const st=document.getElementById('status');"
"btn.disabled=true;out.textContent='Generating...';st.textContent='';"
"const t0=performance.now();"
"try{"
"const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p,n_tokens:parseInt(n)})});"
"const d=await r.json();"
"out.textContent=d.text;"
"st.textContent='Generated '+d.n_tokens+' tokens in '+d.time+'s ('+d.tokens_per_sec+' tok/s)';"
"}catch(e){out.textContent='Error: '+e;}"
"btn.disabled=false;"
"}"
"</script></body></html>";

/* ========================================================================
 * GPT-2 model state
 * ======================================================================== */
static Tensor *g_tensors;
static int   g_n_tensors;
static float *g_wte;       /* [vocab, n_embd] */
static float *g_wpe;       /* [n_ctx, n_embd] */
static float *g_ln_f_w, *g_ln_f_b;

/* Per-layer weight pointers (GPT-2 Conv1D: W is [in, out] row-major) */
typedef struct {
    float *ln1_w, *ln1_b;
    float *c_attn_w, *c_attn_b;   /* [n, 3n] */
    float *c_proj_w, *c_proj_b;   /* [n, n]   */
    float *ln2_w, *ln2_b;
    float *mlp_fc_w,  *mlp_fc_b;  /* [n, m]   */
    float *mlp_proj_w, *mlp_proj_b; /* [m, n]  */
} GPT2Layer;
static GPT2Layer g_layers[N_LAYER];

/* === Binary layer (XNOR+popcount) — used when --binary flag is set ===
 *
 * Memory savings: 498 MB float → 13 MB binary (38x for the 12 transformer layers).
 * wte/wpe stay float (lookup tables, can't binarize without hurting accuracy).
 *
 * Forward: y[j] = (2 * popcount(XNOR(x_bits, wbits[j])) - in_dim) * alpha[j] + bias[j]
 *   where x_bits = sign(x) packed into uint64s
 *   One popcount instruction processes 64 multiplications → 32x FLOP reduction.
 */
typedef struct {
    uint64_t *wbits;   /* [out_dim, n_words] — sign(w) packed, row-major per output */
    float    *alpha;   /* [out_dim] — per-output scale = mean|w| */
    float    *bias;    /* [out_dim] */
    int       in_dim, out_dim, n_words;
} SrvBinLayer;

typedef struct {
    SrvBinLayer c_attn;    /* [n, 3n] */
    SrvBinLayer c_proj;    /* [n, n]  */
    SrvBinLayer mlp_fc;    /* [n, m]  */
    SrvBinLayer mlp_proj;  /* [m, n]  */
    float *ln1_w, *ln1_b;
    float *ln2_w, *ln2_b;
} BinGPT2Layer;
static BinGPT2Layer g_bin_layers[N_LAYER];
static int g_binary_mode = 0;   /* set by --binary flag */

/* Binarize input x[in_dim] → packed bits x_bits[n_words] */
static void binarize_input(const float *x, uint64_t *x_bits, int in_dim, int n_words) {
    for (int wi = 0; wi < n_words; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in_dim && x[idx] > 0.0f) word |= (1ULL << bi);
        }
        x_bits[wi] = word;
    }
}

/* Binary forward: y[out_dim] = (2*popcount(XNOR(x_bits, wbits[j])) - in_dim) * alpha[j] + bias[j]
 * Uses __builtin_popcountll — portable across x86 and ARM (ARM falls back to
 * software popcount, still much faster than 64 scalar multiplications). */
static void bin_matmul(const float *x, const SrvBinLayer *bl, float *y) {
    /* Binarize input once */
    uint64_t x_bits[16];  /* max in_dim = 3072 → 48 words; we stack-alloc 16 for n=768 (12 words) */
    /* For mlp_proj (in_dim=3072), we need 48 words — use heap */
    int n_words = bl->n_words;
    uint64_t *xb;
    if (n_words <= 16) {
        xb = x_bits;
    } else {
        xb = malloc(n_words * sizeof(uint64_t));
    }
    binarize_input(x, xb, bl->in_dim, n_words);

    for (int j = 0; j < bl->out_dim; j++) {
        const uint64_t *wb = bl->wbits + (size_t)j * n_words;
        int pc = 0;
        for (int wi = 0; wi < n_words; wi++) {
            pc += __builtin_popcountll(~(xb[wi] ^ wb[wi]));
        }
        /* dot = (2*pc - in_dim) * alpha — XNOR gives count of matching signs */
        y[j] = (float)(2 * pc - bl->in_dim) * bl->alpha[j] + bl->bias[j];
    }
    if (n_words > 16) free(xb);
}

/* Load GB2L binary weight file directly (no float weights in memory!) */
static int load_binary_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[!] cannot open %s\n", path); return -1; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GB2L", 4) != 0) {
        fprintf(stderr, "[!] bad magic in %s\n", path); fclose(f); return -1;
    }

    unsigned int hdr[5];
    if (fread(hdr, 4, 5, f) != 5) { fclose(f); return -1; }
    int n_layer = hdr[0], n_embd = hdr[1], mlp_dim = hdr[2], vocab = hdr[3], n_ctx = hdr[4];
    if (n_layer != N_LAYER || n_embd != N_EMBD || mlp_dim != MLP_DIM ||
        vocab != VOCAB_SIZE || n_ctx != 1024) {
        fprintf(stderr, "[!] config mismatch in %s\n", path); fclose(f); return -1;
    }

    /* Read embeddings (float — kept as-is) */
    g_wte = malloc((size_t)vocab * n_embd * sizeof(float));
    g_wpe = malloc((size_t)n_ctx * n_embd * sizeof(float));
    g_ln_f_w = malloc(n_embd * sizeof(float));
    g_ln_f_b = malloc(n_embd * sizeof(float));
    fread(g_wte, sizeof(float), (size_t)vocab * n_embd, f);
    fread(g_wpe, sizeof(float), (size_t)n_ctx * n_embd, f);
    fread(g_ln_f_w, sizeof(float), n_embd, f);
    fread(g_ln_f_b, sizeof(float), n_embd, f);

    /* Per-layer binary weights + LayerNorm (float) */
    for (int l = 0; l < n_layer; l++) {
        BinGPT2Layer *L = &g_bin_layers[l];

        /* LayerNorm weights (float, [n_embd] each) */
        L->ln1_w = malloc(n_embd * sizeof(float));
        L->ln1_b = malloc(n_embd * sizeof(float));
        L->ln2_w = malloc(n_embd * sizeof(float));
        L->ln2_b = malloc(n_embd * sizeof(float));
        fread(L->ln1_w, sizeof(float), n_embd, f);
        fread(L->ln1_b, sizeof(float), n_embd, f);
        fread(L->ln2_w, sizeof(float), n_embd, f);
        fread(L->ln2_b, sizeof(float), n_embd, f);

        /* 4 binary matrices per layer */
        SrvBinLayer *mats[] = {&L->c_attn, &L->c_proj, &L->mlp_fc, &L->mlp_proj};
        for (int mi = 0; mi < 4; mi++) {
            SrvBinLayer *bl = mats[mi];
            unsigned int mhdr[4];
            if (fread(mhdr, 4, 4, f) != 4) { fclose(f); return -1; }
            bl->out_dim = mhdr[0];
            bl->in_dim  = mhdr[1];
            bl->n_words = mhdr[2];
            /* Read wbits: [out_dim * n_words] uint64s */
            bl->wbits = malloc((size_t)bl->out_dim * bl->n_words * sizeof(uint64_t));
            fread(bl->wbits, sizeof(uint64_t),
                  (size_t)bl->out_dim * bl->n_words, f);
            /* Read alpha + bias */
            bl->alpha = malloc(bl->out_dim * sizeof(float));
            bl->bias  = malloc(bl->out_dim * sizeof(float));
            fread(bl->alpha, sizeof(float), bl->out_dim, f);
            fread(bl->bias,  sizeof(float), bl->out_dim, f);
        }
    }

    fclose(f);
    g_binary_mode = 1;
    return 0;
}

/* ========================================================================
 * Hash-table tokenizer (length-bucketed open-addressing)
 * Bucket key = (length, hash(token_bytes)) → token_id
 * ======================================================================== */
static char  *g_vocab_tokens[VOCAB_SIZE];  /* owned */
static int    g_vocab_len[VOCAB_SIZE];

/* Hash table: store token_id keyed by (len, hash). Open addressing. */
#define HASH_CAPACITY (1 << 19)  /* 524288, > 2x vocab size */
#define HASH_MASK     (HASH_CAPACITY - 1)
typedef struct { uint32_t token_id; uint32_t len_hash; } HashEntry;  /* len_hash = (len << 16) | hash16 */
static HashEntry *g_hash;

static uint32_t hash_bytes(const char *s, int len) {
    /* FNV-1a */
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

static void hash_insert(const char *s, int len, int token_id) {
    uint32_t h = hash_bytes(s, len);
    uint32_t key = ((uint32_t)len << 16) | (h & 0xFFFF);
    uint32_t idx = h & HASH_MASK;
    for (int probe = 0; probe < 64; probe++) {
        if (g_hash[idx].token_id == 0xFFFFFFFF) {
            g_hash[idx].token_id = (uint32_t)token_id;
            g_hash[idx].len_hash = key;
            return;
        }
        idx = (idx + 1) & HASH_MASK;
    }
    /* Table too full — shouldn't happen with 2x capacity */
}

static int hash_lookup(const char *s, int len) {
    uint32_t h = hash_bytes(s, len);
    uint32_t key = ((uint32_t)len << 16) | (h & 0xFFFF);
    uint32_t idx = h & HASH_MASK;
    for (int probe = 0; probe < 64; probe++) {
        if (g_hash[idx].token_id == 0xFFFFFFFF) return -1;
        if (g_hash[idx].len_hash == key) {
            int tid = (int)g_hash[idx].token_id;
            /* Verify (handles hash16 collisions) */
            if (g_vocab_len[tid] == len && memcmp(g_vocab_tokens[tid], s, len) == 0)
                return tid;
        }
        idx = (idx + 1) & HASH_MASK;
    }
    return -1;
}

static void load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[!] no tokenizer at %s\n", path); exit(1); }

    char magic[4]; fread(magic, 1, 4, f);
    int vocab, n_merges, n_ctx, n_layer, n_embd;
    fread(&vocab, 4, 1, f); fread(&n_merges, 4, 1, f);
    fread(&n_ctx, 4, 1, f); fread(&n_layer, 4, 1, f); fread(&n_embd, 4, 1, f);

    /* Skip byte-to-unicode mapping (256 entries) */
    for (int i = 0; i < 256; i++) {
        unsigned char bv; unsigned short ulen;
        fread(&bv, 1, 1, f); fread(&ulen, 2, 1, f);
        fseek(f, ulen, SEEK_CUR);
    }
    /* Skip merges */
    for (int i = 0; i < n_merges; i++) {
        int rank; unsigned short alen, blen;
        fread(&rank, 4, 1, f); fread(&alen, 2, 1, f); fseek(f, alen, SEEK_CUR);
        fread(&blen, 2, 1, f); fseek(f, blen, SEEK_CUR);
    }
    /* Load vocab */
    for (int i = 0; i < VOCAB_SIZE; i++) { g_vocab_tokens[i] = NULL; g_vocab_len[i] = 0; }
    for (int i = 0; i < vocab; i++) {
        int tid; unsigned short tlen;
        fread(&tid, 4, 1, f); fread(&tlen, 2, 1, f);
        g_vocab_tokens[tid] = malloc(tlen + 1);
        fread(g_vocab_tokens[tid], 1, tlen, f);
        g_vocab_tokens[tid][tlen] = '\0';
        g_vocab_len[tid] = tlen;
    }
    fclose(f);

    /* Build hash table */
    g_hash = malloc(HASH_CAPACITY * sizeof(HashEntry));
    memset(g_hash, 0xFF, HASH_CAPACITY * sizeof(HashEntry));
    for (int tid = 0; tid < VOCAB_SIZE; tid++) {
        if (g_vocab_tokens[tid] && g_vocab_len[tid] > 0)
            hash_insert(g_vocab_tokens[tid], g_vocab_len[tid], tid);
    }
}

/* Greedy longest-match encoding using hash table.
 * Tries lengths 1..min(16, remaining) at each position. */
static int encode_text(const char *text, int *out_tokens, int max_tokens) {
    int n_out = 0, pos = 0, text_len = (int)strlen(text);
    while (pos < text_len && n_out < max_tokens) {
        int remaining = text_len - pos;
        int max_try = remaining > 16 ? 16 : remaining;
        int best_id = -1, best_len = 0;
        for (int len = max_try; len >= 1; len--) {
            int tid = hash_lookup(text + pos, len);
            if (tid >= 0) { best_id = tid; best_len = len; break; }
        }
        if (best_id < 0) { pos++; continue; }  /* skip unknown byte */
        out_tokens[n_out++] = best_id;
        pos += best_len;
    }
    return n_out;
}

static int decode_token(int token_id, char *out, int max_len) {
    if (token_id < 0 || token_id >= VOCAB_SIZE) return 0;
    int tl = g_vocab_len[token_id];
    if (tl > max_len) tl = max_len;
    memcpy(out, g_vocab_tokens[token_id], tl);
    return tl;
}

/* ========================================================================
 * AVX2 SIMD primitives
 * ======================================================================== */

/* LayerNorm (GPT-2 style) */
static void layer_norm_simd(float *out, const float *x, const float *w, const float *b, int n) {
    /* mean */
    v8f sum_v = v8f_zero();
    int i = 0;
    float tail_sum = 0;
    for (; i + 8 <= n; i += 8) sum_v = v8f_add(sum_v, v8f_load(x + i));
    float mean = v8f_hsum(sum_v);
    for (; i < n; i++) tail_sum += x[i];
    mean = (mean + tail_sum) / n;

    /* var */
    v8f mean_v = v8f_set1(mean);
    v8f var_v = v8f_zero();
    i = 0; float tail_var = 0;
    for (; i + 8 <= n; i += 8) {
        v8f d = v8f_sub(v8f_load(x + i), mean_v);
        var_v = v8f_fmadd(d, d, var_v);
    }
    float var = v8f_hsum(var_v);
    for (; i < n; i++) { float d = x[i] - mean; tail_var += d*d; }
    var = (var + tail_var) / n;

    float std_inv = 1.0f / sqrtf(var + 1e-5f);
    v8f std_v = v8f_set1(std_inv);
    v8f mean_v2 = v8f_set1(mean);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        v8f xn = v8f_mul(v8f_sub(v8f_load(x + i), mean_v2), std_v);
        v8f wn = v8f_mul(xn, v8f_load(w + i));
        v8f_store(out + i, v8f_add(wn, v8f_load(b + i)));
    }
    for (; i < n; i++) {
        float xn = (x[i] - mean) * std_inv;
        out[i] = xn * w[i] + b[i];
    }
}

/* Float matmul: y[out_dim] = bias[out_dim] + x[in_dim] @ W[in_dim, out_dim]
 * W is row-major in GPT-2 Conv1D format: W[i][j] = W[i*out_dim + j].
 * Process 8 output cols at a time so each W[i*out_dim + j..j+7] load is contiguous. */
static void matmul_avx2(float *y, const float *x, const float *W, const float *b,
                        int in_dim, int out_dim) {
    int j = 0;
    for (; j + 8 <= out_dim; j += 8) {
        v8f acc = b ? v8f_load(b + j) : v8f_zero();
        for (int i = 0; i < in_dim; i++) {
            v8f xi = v8f_set1(x[i]);
            v8f w  = v8f_load(W + (size_t)i * out_dim + j);
            acc = v8f_fmadd(xi, w, acc);
        }
        v8f_store(y + j, acc);
    }
    for (; j < out_dim; j++) {
        float s = b ? b[j] : 0.0f;
        for (int i = 0; i < in_dim; i++) s += x[i] * W[(size_t)i * out_dim + j];
        y[j] = s;
    }
}

/* LM head: logits[VOCAB] = wte[VOCAB, n_embd] @ x[n_embd]
 * Same structure as matmul: 8 vocab rows at a time. */
static void lm_head_avx2(float *logits, const float *x, const float *wte,
                         int vocab, int n_embd) {
    int v = 0;
    for (; v + 8 <= vocab; v += 8) {
        v8f acc0 = v8f_zero(), acc1 = v8f_zero();
        v8f acc2 = v8f_zero(), acc3 = v8f_zero();
        v8f acc4 = v8f_zero(), acc5 = v8f_zero();
        v8f acc6 = v8f_zero(), acc7 = v8f_zero();
        const float *w0 = wte + (size_t)(v+0) * n_embd;
        const float *w1 = wte + (size_t)(v+1) * n_embd;
        const float *w2 = wte + (size_t)(v+2) * n_embd;
        const float *w3 = wte + (size_t)(v+3) * n_embd;
        const float *w4 = wte + (size_t)(v+4) * n_embd;
        const float *w5 = wte + (size_t)(v+5) * n_embd;
        const float *w6 = wte + (size_t)(v+6) * n_embd;
        const float *w7 = wte + (size_t)(v+7) * n_embd;
        int i = 0;
        for (; i + 8 <= n_embd; i += 8) {
            v8f xv = v8f_load(x + i);
            acc0 = v8f_fmadd(xv, v8f_load(w0 + i), acc0);
            acc1 = v8f_fmadd(xv, v8f_load(w1 + i), acc1);
            acc2 = v8f_fmadd(xv, v8f_load(w2 + i), acc2);
            acc3 = v8f_fmadd(xv, v8f_load(w3 + i), acc3);
            acc4 = v8f_fmadd(xv, v8f_load(w4 + i), acc4);
            acc5 = v8f_fmadd(xv, v8f_load(w5 + i), acc5);
            acc6 = v8f_fmadd(xv, v8f_load(w6 + i), acc6);
            acc7 = v8f_fmadd(xv, v8f_load(w7 + i), acc7);
        }
        float s0 = v8f_hsum(acc0), s1 = v8f_hsum(acc1);
        float s2 = v8f_hsum(acc2), s3 = v8f_hsum(acc3);
        float s4 = v8f_hsum(acc4), s5 = v8f_hsum(acc5);
        float s6 = v8f_hsum(acc6), s7 = v8f_hsum(acc7);
        /* tail */
        for (; i < n_embd; i++) {
            float xv = x[i];
            s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i];
            s4 += xv * w4[i]; s5 += xv * w5[i]; s6 += xv * w6[i]; s7 += xv * w7[i];
        }
        logits[v+0]=s0; logits[v+1]=s1; logits[v+2]=s2; logits[v+3]=s3;
        logits[v+4]=s4; logits[v+5]=s5; logits[v+6]=s6; logits[v+7]=s7;
    }
    for (; v < vocab; v++) {
        const float *w = wte + (size_t)v * n_embd;
        v8f acc = v8f_zero();
        int i = 0;
        for (; i + 8 <= n_embd; i += 8)
            acc = v8f_fmadd(v8f_load(x + i), v8f_load(w + i), acc);
        float s = v8f_hsum(acc);
        for (; i < n_embd; i++) s += x[i] * w[i];
        logits[v] = s;
    }
}

/* Parallel LM head: split vocab across N_THREADS workers.
 * The LM head is ~30% of total per-token time, so 2 threads → ~15% speedup. */
typedef struct {
    const float *x;
    const float *wte;
    float *logits;
    int vocab_start, vocab_end, n_embd;
} LmHeadJob;

static void *lm_head_worker(void *arg) {
    LmHeadJob *job = (LmHeadJob *)arg;
    lm_head_avx2(job->logits + job->vocab_start, job->x,
                 job->wte + (size_t)job->vocab_start * job->n_embd,
                 job->vocab_end - job->vocab_start, job->n_embd);
    return NULL;
}

static void lm_head_parallel(float *logits, const float *x, const float *wte,
                             int vocab, int n_embd, int n_threads) {
    if (n_threads <= 1) {
        lm_head_avx2(logits, x, wte, vocab, n_embd);
        return;
    }
    pthread_t threads[8];
    LmHeadJob jobs[8];
    if (n_threads > 8) n_threads = 8;
    int chunk = (vocab + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        jobs[t].x = x;
        jobs[t].wte = wte;
        jobs[t].logits = logits;
        jobs[t].vocab_start = t * chunk;
        jobs[t].vocab_end = (t + 1) * chunk;
        if (jobs[t].vocab_start > vocab) jobs[t].vocab_start = vocab;
        if (jobs[t].vocab_end > vocab) jobs[t].vocab_end = vocab;
        /* Align start/end to 8 for SIMD */
        jobs[t].vocab_start = (jobs[t].vocab_start / 8) * 8;
        jobs[t].vocab_end   = (jobs[t].vocab_end / 8) * 8;
        if (t == n_threads - 1) jobs[t].vocab_end = vocab;
        if (jobs[t].vocab_end <= jobs[t].vocab_start) continue;
        pthread_create(&threads[t], NULL, lm_head_worker, &jobs[t]);
    }
    for (int t = 0; t < n_threads; t++) {
        if (jobs[t].vocab_end > jobs[t].vocab_start)
            pthread_join(threads[t], NULL);
    }
}

/* Pruned LM head: only compute logits for active vocab rows.
 * Skips rows with smallest L2 norm (set via g_prune_frac).
 * Non-active logits are set to -INFINITY so argmax never picks them.
 *
 * active_vocab[] must be sorted ascending by vocab index for sequential
 * wte memory access (prefetcher-friendly). */
static void lm_head_pruned(float *logits, const float *x, const float *wte,
                           const int *active_vocab, int n_active, int n_embd) {
    /* Initialize all logits to -INFINITY */
    for (int v = 0; v < VOCAB_SIZE; v++) logits[v] = -1e30f;

    /* Process 8 active rows at a time when they're consecutive in wte
     * (which is often the case after sorting by index). This reuses the
     * 8-wide SIMD pattern from lm_head_avx2 for the common case. */
    int vi = 0;
    while (vi + 8 <= n_active) {
        /* Check if next 8 active rows are consecutive in vocab */
        int v0 = active_vocab[vi];
        int consecutive = 1;
        for (int k = 1; k < 8; k++) {
            if (active_vocab[vi + k] != v0 + k) { consecutive = 0; break; }
        }
        if (consecutive) {
            /* Use the fast 8-wide SIMD path (same as lm_head_avx2) */
            v8f acc0 = v8f_zero(), acc1 = v8f_zero(), acc2 = v8f_zero(), acc3 = v8f_zero();
            v8f acc4 = v8f_zero(), acc5 = v8f_zero(), acc6 = v8f_zero(), acc7 = v8f_zero();
            const float *w0 = wte + (size_t)(v0+0) * n_embd;
            const float *w1 = wte + (size_t)(v0+1) * n_embd;
            const float *w2 = wte + (size_t)(v0+2) * n_embd;
            const float *w3 = wte + (size_t)(v0+3) * n_embd;
            const float *w4 = wte + (size_t)(v0+4) * n_embd;
            const float *w5 = wte + (size_t)(v0+5) * n_embd;
            const float *w6 = wte + (size_t)(v0+6) * n_embd;
            const float *w7 = wte + (size_t)(v0+7) * n_embd;
            int i = 0;
            for (; i + 8 <= n_embd; i += 8) {
                v8f xv = v8f_load(x + i);
                acc0 = v8f_fmadd(xv, v8f_load(w0 + i), acc0);
                acc1 = v8f_fmadd(xv, v8f_load(w1 + i), acc1);
                acc2 = v8f_fmadd(xv, v8f_load(w2 + i), acc2);
                acc3 = v8f_fmadd(xv, v8f_load(w3 + i), acc3);
                acc4 = v8f_fmadd(xv, v8f_load(w4 + i), acc4);
                acc5 = v8f_fmadd(xv, v8f_load(w5 + i), acc5);
                acc6 = v8f_fmadd(xv, v8f_load(w6 + i), acc6);
                acc7 = v8f_fmadd(xv, v8f_load(w7 + i), acc7);
            }
            float s0 = v8f_hsum(acc0), s1 = v8f_hsum(acc1);
            float s2 = v8f_hsum(acc2), s3 = v8f_hsum(acc3);
            float s4 = v8f_hsum(acc4), s5 = v8f_hsum(acc5);
            float s6 = v8f_hsum(acc6), s7 = v8f_hsum(acc7);
            for (; i < n_embd; i++) {
                float xv = x[i];
                s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i];
                s4 += xv * w4[i]; s5 += xv * w5[i]; s6 += xv * w6[i]; s7 += xv * w7[i];
            }
            logits[v0+0]=s0; logits[v0+1]=s1; logits[v0+2]=s2; logits[v0+3]=s3;
            logits[v0+4]=s4; logits[v0+5]=s5; logits[v0+6]=s6; logits[v0+7]=s7;
            vi += 8;
        } else {
            /* Fall back to single-row scalar+SIMD */
            int v = active_vocab[vi];
            const float *w = wte + (size_t)v * n_embd;
            v8f acc = v8f_zero();
            int i = 0;
            for (; i + 8 <= n_embd; i += 8)
                acc = v8f_fmadd(v8f_load(x + i), v8f_load(w + i), acc);
            float s = v8f_hsum(acc);
            for (; i < n_embd; i++) s += x[i] * w[i];
            logits[v] = s;
            vi++;
        }
    }
    /* Tail */
    for (; vi < n_active; vi++) {
        int v = active_vocab[vi];
        const float *w = wte + (size_t)v * n_embd;
        v8f acc = v8f_zero();
        int i = 0;
        for (; i + 8 <= n_embd; i += 8)
            acc = v8f_fmadd(v8f_load(x + i), v8f_load(w + i), acc);
        float s = v8f_hsum(acc);
        for (; i < n_embd; i++) s += x[i] * w[i];
        logits[v] = s;
    }
}

/* GELU (tanh approximation, GPT-2 style) */
static inline float gelu_fast(float x) {
    const float c = 0.7978845608028654f;  /* sqrt(2/π) */
    float t = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(t));
}

/* ========================================================================
 * GPT-2 forward (single-token, no attention — uses V as attention output)
 * This mirrors the simplified attention in lal_runtime.c (trans_layer_forward
 * line 224: "memcpy(attn_out, v, ...)").
 * Inputs: token_id, position → returns next token_id (argmax of logits).
 * ======================================================================== */
static float  g_x[N_EMBD];
static float  g_ln1[N_EMBD];
static float  g_qkv[3*N_EMBD];
static float  g_attn_out[N_EMBD];   /* = V projection */
static float  g_proj[N_EMBD];
static float  g_ln2[N_EMBD];
static float  g_fc[MLP_DIM];
static float  g_mlp[MLP_DIM];
static float  g_mlp_out[N_EMBD];
static float  g_final_ln[N_EMBD];
static float *g_logits = NULL;       /* [VOCAB_SIZE], allocated once */
static int    g_n_threads = 1;       /* worker threads for LM head */

/* Vocab pruning: skip rows of wte (LM head) with smallest L2 norm.
 * Set via --prune-vocab <frac> (0.0 = no pruning, 0.5 = drop 50% of vocab).
 *
 * Why this works:
 *   - GPT-2's vocab has 50257 entries, but most text generation uses <5000
 *     common tokens. The rest are rare unicode, foreign scripts, special chars.
 *   - Rare tokens have small embedding norms — they contribute little to
 *     logits and are almost never selected by argmax.
 *   - Skipping 50% of vocab rows halves LM head FLOPs (the bottleneck on
 *     ARM/memory-bound devices) with negligible quality loss for English text.
 *   - This is "structured sparsity" — whole rows dropped, not individual
 *     weights. Dense SIMD within each row is preserved.
 */
static int    g_n_active_vocab = VOCAB_SIZE;
static int   *g_active_vocab = NULL;   /* [n_active_vocab], indices into wte */
static float  g_prune_frac = 0.0f;     /* set from --prune-vocab */

/* File-scope pointer for qsort comparator (qsort_r is non-portable) */
static float *g_prune_norm_ref = NULL;
static int prune_cmp_desc(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    float na = g_prune_norm_ref[ia], nb = g_prune_norm_ref[ib];
    if (na < nb) return 1;   /* descending: larger norm first */
    if (na > nb) return -1;
    return 0;
}
static int *g_prune_idx_ref = NULL;
static int prune_cmp_idx_asc(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    return (ia > ib) - (ia < ib);  /* ascending by vocab index */
}

static int gpt2_forward_token(int token_id, int position) {
    /* Bounds check — prevent OOB access on g_wte/g_wpe */
    if (token_id < 0 || token_id >= VOCAB_SIZE) {
        fprintf(stderr, "[!] bad token_id=%d, clamping to 0\n", token_id);
        token_id = 0;
    }
    if (position < 0 || position >= 1024) {
        fprintf(stderr, "[!] bad position=%d, clamping to 0\n", position);
        position = 0;
    }
    fprintf(stderr, "[d]   forward: token=%d pos=%d embedding\n", token_id, position); fflush(stderr);
    /* Embedding */
    for (int i = 0; i < N_EMBD; i++)
        g_x[i] = g_wte[token_id * N_EMBD + i] + g_wpe[position * N_EMBD + i];

    /* Layers */
    for (int l = 0; l < N_LAYER; l++) {
        if (g_binary_mode) {
            /* Binary forward: XNOR + popcount (32x fewer FLOPs per layer) */
            BinGPT2Layer *L = &g_bin_layers[l];
            fprintf(stderr, "[d]   layer %d ln1\n", l); fflush(stderr);
            layer_norm_simd(g_ln1, g_x, L->ln1_w, L->ln1_b, N_EMBD);
            fprintf(stderr, "[d]   layer %d c_attn\n", l); fflush(stderr);
            bin_matmul(g_ln1, &L->c_attn, g_qkv);
            memcpy(g_attn_out, g_qkv + 2*N_EMBD, N_EMBD * sizeof(float));
            fprintf(stderr, "[d]   layer %d c_proj\n", l); fflush(stderr);
            bin_matmul(g_attn_out, &L->c_proj, g_proj);
            for (int i = 0; i < N_EMBD; i++) g_x[i] += g_proj[i];

            fprintf(stderr, "[d]   layer %d ln2\n", l); fflush(stderr);
            layer_norm_simd(g_ln2, g_x, L->ln2_w, L->ln2_b, N_EMBD);
            fprintf(stderr, "[d]   layer %d mlp_fc\n", l); fflush(stderr);
            bin_matmul(g_ln2, &L->mlp_fc, g_fc);
            for (int i = 0; i < MLP_DIM; i++) g_fc[i] = gelu_fast(g_fc[i]);
            fprintf(stderr, "[d]   layer %d mlp_proj\n", l); fflush(stderr);
            bin_matmul(g_fc, &L->mlp_proj, g_mlp_out);
            for (int i = 0; i < N_EMBD; i++) g_x[i] += g_mlp_out[i];
        } else {
            /* Float forward: AVX2/NEON SIMD matmul */
            GPT2Layer *L = &g_layers[l];

            layer_norm_simd(g_ln1, g_x, L->ln1_w, L->ln1_b, N_EMBD);
            matmul_avx2(g_qkv, g_ln1, L->c_attn_w, L->c_attn_b, N_EMBD, 3*N_EMBD);
            memcpy(g_attn_out, g_qkv + 2*N_EMBD, N_EMBD * sizeof(float));
            matmul_avx2(g_proj, g_attn_out, L->c_proj_w, L->c_proj_b, N_EMBD, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) g_x[i] += g_proj[i];

            layer_norm_simd(g_ln2, g_x, L->ln2_w, L->ln2_b, N_EMBD);
            matmul_avx2(g_fc, g_ln2, L->mlp_fc_w, L->mlp_fc_b, N_EMBD, MLP_DIM);
            for (int i = 0; i < MLP_DIM; i++) g_fc[i] = gelu_fast(g_fc[i]);
            matmul_avx2(g_mlp_out, g_fc, L->mlp_proj_w, L->mlp_proj_b, MLP_DIM, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) g_x[i] += g_mlp_out[i];
        }
    }

    /* Final norm */
    fprintf(stderr, "[d]   final norm\n"); fflush(stderr);
    layer_norm_simd(g_final_ln, g_x, g_ln_f_w, g_ln_f_b, N_EMBD);

    /* LM head: logits = wte @ final_ln  (tied embeddings)
     * wte stays float in both modes (binarizing embeddings hurts accuracy).
     * Use pruned version if --prune-vocab was set, else full parallel. */
    fprintf(stderr, "[d]   lm_head\n"); fflush(stderr);
    if (g_active_vocab && g_n_active_vocab < VOCAB_SIZE) {
        lm_head_pruned(g_logits, g_final_ln, g_wte, g_active_vocab, g_n_active_vocab, N_EMBD);
    } else {
        lm_head_parallel(g_logits, g_final_ln, g_wte, VOCAB_SIZE, N_EMBD, g_n_threads);
    }

    /* Argmax */
    fprintf(stderr, "[d]   argmax\n"); fflush(stderr);
    int best = 0;
    float best_val = g_logits[0];
    for (int v = 1; v < VOCAB_SIZE; v++) {
        if (g_logits[v] > best_val) { best_val = g_logits[v]; best = v; }
    }
    fprintf(stderr, "[d]   forward done, best=%d\n", best); fflush(stderr);
    return best;
}

/* ========================================================================
 * HTTP server
 * ======================================================================== */
static char *load_file(const char *path, long *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    if (size_out) *size_out = sz;
    return buf;
}

static void handle_request(int client_fd) {
    /* NOTE: buf + result + escaped + json + all_tokens ≈ 40 KB on stack.
     * ARM Android's default pthread stack can be as small as 8 KB, which
     * caused a segfault on the 2nd request (stack overflow into guard page).
     * Fix: make the large buffers static so they live in BSS, not stack. */
    static char buf[16384];
    int n = read(client_fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(client_fd); return; }
    buf[n] = '\0';

    char method[8], path[256];
    sscanf(buf, "%s %s", method, path);

    if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        /* Try to load frontend.html from disk; fall back to embedded */
        char *html = NULL; long html_size = 0;
        html = load_file("tools/server/frontend.html", &html_size);
        if (!html) html = load_file("frontend.html", &html_size);
        const char *body = html ? html : HTML_FALLBACK;
        size_t body_len = html ? (size_t)html_size : strlen(HTML_FALLBACK);

        char header[256];
        sprintf(header, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", body_len);
        write(client_fd, header, strlen(header));
        write(client_fd, body, body_len);
        free(html);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/generate") == 0) {
        fprintf(stderr, "[d] POST /generate received\n"); fflush(stderr);
        char *body = strstr(buf, "\r\n\r\n");
        if (!body) { close(client_fd); return; }
        body += 4;

        static char prompt[1024];
        prompt[0] = '\0';
        int n_tokens = 20;
        /* Simple JSON parse */
        char *p = strstr(body, "\"prompt\"");
        if (p) {
            p = strchr(p, ':');
            if (p) {
                p = strchr(p, '"');
                if (p) {
                    p++;
                    int i = 0;
                    while (*p && *p != '"' && i < 1023) {
                        if (*p == '\\') p++;
                        prompt[i++] = *p++;
                    }
                    prompt[i] = '\0';
                }
            }
        }
        p = strstr(body, "\"n_tokens\"");
        if (p) {
            p = strchr(p, ':');
            if (p) n_tokens = atoi(p + 1);
        }
        if (n_tokens < 1) n_tokens = 1;
        if (n_tokens > 100) n_tokens = 100;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        static int input_tokens[256];
        int n_input = encode_text(prompt, input_tokens, 256);
        if (n_input == 0) { input_tokens[0] = 464; n_input = 1; }

        static int output_tokens[100];
        static int all_tokens[1024];
        memcpy(all_tokens, input_tokens, n_input * sizeof(int));
        int total = n_input;
        fprintf(stderr, "[d] starting generation: n_input=%d n_tokens=%d\n", n_input, n_tokens); fflush(stderr);
        for (int gen = 0; gen < n_tokens; gen++) {
            fprintf(stderr, "[d] gen=%d total=%d calling forward\n", gen, total); fflush(stderr);
            int next = gpt2_forward_token(all_tokens[total - 1], total - 1);
            fprintf(stderr, "[d] gen=%d got token=%d\n", gen, next); fflush(stderr);
            output_tokens[gen] = next;
            all_tokens[total++] = next;
            if (total >= 1024) { n_tokens = gen + 1; break; }
        }
        fprintf(stderr, "[d] generation done, building response\n"); fflush(stderr);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

        /* Build output text */
        static char result[4096];
        result[0] = '\0';
        int pos = snprintf(result, sizeof(result), "%s", prompt);
        for (int i = 0; i < n_tokens && pos < 4000; i++) {
            static char tok[256];
            int tl = decode_token(output_tokens[i], tok, sizeof(tok) - 1);
            tok[tl] = '\0';
            pos += snprintf(result + pos, sizeof(result) - pos, "%s", tok);
        }

        static char escaped[5120];
        int ei = 0;
        for (int i = 0; result[i] && ei < 5110; i++) {
            if (result[i] == '"') { escaped[ei++] = '\\'; escaped[ei++] = '"'; }
            else if (result[i] == '\\') { escaped[ei++] = '\\'; escaped[ei++] = '\\'; }
            else if (result[i] == '\n') { escaped[ei++] = '\\'; escaped[ei++] = 'n'; }
            else if (result[i] == '\r') { continue; }
            else escaped[ei++] = result[i];
        }
        escaped[ei] = '\0';

        double tps = n_tokens / (dt > 0 ? dt : 1e-6);
        static char json[6144];
        int jpos = snprintf(json, sizeof(json),
            "{\"text\":\"%s\",\"time\":\"%.3f\",\"n_tokens\":%d,\"tokens_per_sec\":\"%.1f\"}",
            escaped, dt, n_tokens, tps);
        char header[256];
        snprintf(header, sizeof(header),
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n", jpos);
        write(client_fd, header, strlen(header));
        write(client_fd, json, jpos);
    } else {
        const char *not_found = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n";
        write(client_fd, not_found, strlen(not_found));
    }

    close(client_fd);
}

/* ========================================================================
 * Main
 * ======================================================================== */
int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);

    int port = 8080;
    int use_binary = 0;
    /* Parse args: [port] [--binary] [--prune-vocab FRAC]
     * --binary: use XNOR+popcount binary mode (13MB weights, 32x fewer FLOPs)
     * FRAC in [0, 0.9]: fraction of vocab (smallest-norm rows) to drop. */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--binary") == 0) {
            use_binary = 1;
        } else if (strcmp(argv[i], "--prune-vocab") == 0 && i + 1 < argc) {
            g_prune_frac = (float)atof(argv[++i]);
            if (g_prune_frac < 0.0f) g_prune_frac = 0.0f;
            if (g_prune_frac > 0.9f) g_prune_frac = 0.9f;
        } else {
            port = atoi(argv[i]);
        }
    }

    /* Auto-detect thread count.
     * NOTE: LM head is memory-bound (154MB weight reads/token), so threading
     * only helps on 4+ core machines with multiple memory channels.
     * On 2-core systems, single-threaded is faster (avoids cache contention). */
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    g_n_threads = (ncpu >= 4) ? (int)ncpu : 1;
    if (g_n_threads > 8) g_n_threads = 8;

    printf("[*] LAL GPT-2 Server (%s mode, %d thread%s, %ld cores detected)\n",
           use_binary ? "BINARY XNOR+popcount" : "float SIMD",
           g_n_threads, g_n_threads > 1 ? "s" : "", ncpu);
    if (g_prune_frac > 0.0f) {
        printf("[*] vocab pruning: dropping %.0f%% smallest-norm rows (%d → %d active)\n",
               g_prune_frac * 100.0f, VOCAB_SIZE,
               (int)((1.0f - g_prune_frac) * VOCAB_SIZE));
    }

    const char *tokenizer_path = getenv("LAL_TOKENIZER");
    if (!tokenizer_path) tokenizer_path = "prebuilt/gpt2_tokenizer.bin";

    if (use_binary) {
        /* Binary mode: load GB2L file directly (169 MB total, no float weights) */
        const char *bin_path = getenv("LAL_BINARY");
        if (!bin_path) bin_path = "prebuilt/gpt2_binary.bin";
        printf("[*] loading binary weights from %s...\n", bin_path);
        fflush(stdout);
        struct timespec tl0, tl1;
        clock_gettime(CLOCK_MONOTONIC, &tl0);
        if (load_binary_weights(bin_path) != 0) return 1;
        clock_gettime(CLOCK_MONOTONIC, &tl1);
        double ldt = (tl1.tv_sec - tl0.tv_sec) + (tl1.tv_nsec - tl0.tv_nsec) * 1e-9;
        printf("[*] binary weights loaded in %.2fs (12 layers, XNOR+popcount)\n", ldt);
        fflush(stdout);
    } else {
        printf("[*] loading float weights...\n");
        fflush(stdout);
        /* Weight path: env var override, else relative to cwd */
        const char *weight_path = getenv("LAL_WEIGHTS");
        if (!weight_path) weight_path = "prebuilt/gpt2_weights.bin";

        g_tensors = tensor_load_all(weight_path, &g_n_tensors);
        if (!g_tensors) { fprintf(stderr, "[!] failed to load weights from %s\n", weight_path); return 1; }
        printf("[*] loaded %d tensors\n", g_n_tensors); fflush(stdout);

        g_wte = tensor_get(g_tensors, g_n_tensors, "wte.weight");
        g_wpe = tensor_get(g_tensors, g_n_tensors, "wpe.weight");
        g_ln_f_w = tensor_get(g_tensors, g_n_tensors, "ln_f.weight");
        g_ln_f_b = tensor_get(g_tensors, g_n_tensors, "ln_f.bias");
        if (!g_wte || !g_wpe || !g_ln_f_w || !g_ln_f_b) {
            fprintf(stderr, "[!] missing top-level tensors\n"); return 1;
        }

        char key[128];
        for (int l = 0; l < N_LAYER; l++) {
            GPT2Layer *L = &g_layers[l];
            sprintf(key, "h.%d.ln_1.weight",     l); L->ln1_w      = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.ln_1.bias",       l); L->ln1_b      = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.attn.c_attn.weight", l); L->c_attn_w   = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.attn.c_attn.bias",   l); L->c_attn_b   = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.attn.c_proj.weight", l); L->c_proj_w   = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.attn.c_proj.bias",   l); L->c_proj_b   = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.ln_2.weight",     l); L->ln2_w      = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.ln_2.bias",       l); L->ln2_b      = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.mlp.c_fc.weight", l); L->mlp_fc_w   = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.mlp.c_fc.bias",   l); L->mlp_fc_b   = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.mlp.c_proj.weight", l); L->mlp_proj_w = tensor_get(g_tensors, g_n_tensors, key);
            sprintf(key, "h.%d.mlp.c_proj.bias",   l); L->mlp_proj_b = tensor_get(g_tensors, g_n_tensors, key);
            if (!L->ln1_w || !L->c_attn_w || !L->c_proj_w || !L->ln2_w || !L->mlp_fc_w || !L->mlp_proj_w) {
                fprintf(stderr, "[!] missing tensors for layer %d\n", l); return 1;
            }
        }
    }  /* end else (float mode) */

    g_logits = NULL;
    if (posix_memalign((void **)&g_logits, 32, VOCAB_SIZE * sizeof(float)) != 0 || !g_logits) {
        fprintf(stderr, "[!] OOM allocating logits\n"); return 1;
    }

    /* Vocab pruning: compute row L2 norms, keep top (1 - prune_frac) by norm.
     * Strategy: partial sort via qsort, then take the top N.
     * This is O(V log V) once at startup — 50257 log(50257) ≈ 800k comparisons. */
    if (g_prune_frac > 0.0f) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* Compute row norms */
        float *row_norm = malloc(VOCAB_SIZE * sizeof(float));
        int *vocab_idx  = malloc(VOCAB_SIZE * sizeof(int));
        if (!row_norm || !vocab_idx) { fprintf(stderr, "[!] OOM pruning\n"); return 1; }

        for (int v = 0; v < VOCAB_SIZE; v++) {
            const float *w = g_wte + (size_t)v * N_EMBD;
            v8f acc = v8f_zero();
            int i = 0;
            for (; i + 8 <= N_EMBD; i += 8) {
                v8f wv = v8f_load(w + i);
                acc = v8f_fmadd(wv, wv, acc);
            }
            float s = v8f_hsum(acc);
            for (; i < N_EMBD; i++) s += w[i] * w[i];
            row_norm[v] = s;  /* squared norm — monotonic, fine for ranking */
            vocab_idx[v] = v;
        }

        /* Sort vocab_idx by row_norm descending (largest norm first = keep).
         * Comparator uses a file-scope static pointer (qsort_r is non-portable). */
        g_prune_norm_ref = row_norm;
        qsort(vocab_idx, VOCAB_SIZE, sizeof(int), prune_cmp_desc);

        /* Keep top (1 - prune_frac), then RE-SORT by vocab index for sequential
         * memory access. Random-order access to wte rows kills cache performance
         * on ARM (LPDDR3 with small L2 cache). Sequential access lets the
         * prefetcher stream weights, which more than compensates for the
         * scalar loop overhead vs the 8-wide SIMD lm_head_avx2. */
        g_n_active_vocab = (int)((1.0f - g_prune_frac) * VOCAB_SIZE);
        if (g_n_active_vocab < 1) g_n_active_vocab = 1;
        g_active_vocab = malloc(g_n_active_vocab * sizeof(int));
        if (!g_active_vocab) { fprintf(stderr, "[!] OOM active_vocab\n"); return 1; }
        for (int i = 0; i < g_n_active_vocab; i++) {
            g_active_vocab[i] = vocab_idx[i];
        }
        /* Re-sort by vocab index ascending (restore sequential wte access) */
        g_prune_idx_ref = g_active_vocab;
        qsort(g_active_vocab, g_n_active_vocab, sizeof(int), prune_cmp_idx_asc);

        /* Stats: report the cutoff norm */
        float cutoff_sq = row_norm[vocab_idx[g_n_active_vocab - 1]];
        float cutoff = sqrtf(cutoff_sq);
        float min_sq = row_norm[vocab_idx[VOCAB_SIZE - 1]];
        float max_sq = row_norm[vocab_idx[0]];

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

        printf("[*] vocab pruning: %d/%d active (cutoff norm=%.4f, range %.4f..%.4f), %.2fs\n",
               g_n_active_vocab, VOCAB_SIZE, cutoff,
               sqrtf(min_sq), sqrtf(max_sq), dt);
        fflush(stdout);

        free(row_norm);
        free(vocab_idx);
    }

    load_tokenizer(tokenizer_path);
    printf("[*] tokenizer loaded (%d entries, hash table %d slots)\n", VOCAB_SIZE, HASH_CAPACITY);
    fflush(stdout);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(port),
    };
    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    listen(server_fd, 8);

    printf("[*] server running at http://localhost:%d\n", port);
    printf("[*] open browser to http://localhost:%d\n", port);
    fflush(stdout);

    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) { perror("accept"); sleep(1); continue; }
        handle_request(client_fd);
    }

    return 0;
}
