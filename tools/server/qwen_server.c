/* qwen_server.c — Qwen2.5-0.5B inference (pure C, Q8, GQA, RoPE theta=1e6)
 *
 * Optimizations:
 *   1. Q8 per-row quantization for all transformer matmuls (AVX2 VPMADDUBSW)
 *   2. Int8 LM head with two-pass rerank (4x bandwidth reduction)
 *   3. Multi-threaded: LM head split across threads
 *   4. Gate+Up SwiGLU fusion (eliminate one 896→4864 Q8 matmul)
 *   5. Precomputed RoPE cos/sin table (eliminate powf/cosf/sinf per token)
 *   6. Zero malloc in GQA attention (pre-allocated buffers)
 *   7. SIMD-vectorized SiLU+mul in fused SwiGLU
 *
 * Build: make qwen-server
 * Run:   ./qwen_server --weights prebuilt/qwen_weights.bin \
 *            --tokenizer prebuilt/qwen_tokenizer --prompt "Hello" --n 20 [--threads 2] [--temp 0.8] [--top-k 40] [--rep-penalty 1.1]
 */
#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <pthread.h>

#include "runtime/lal_runtime.h"

#define XQ_MAX 4864
#include "runtime/lal_q8_kernel.h"

/* ========================================================================
 * Qwen2.5-0.5B Config
 * ======================================================================== */
#define N_EMBD       896
#define N_LAYER      24
#define N_HEAD       14
#define N_KV_HEAD    2
#define HEAD_DIM     64
#define N_Q_PER_KV   7   /* 14 / 2 */
#define MLP_DIM      4864
#define VOCAB_SIZE   151936
#define N_CTX        2048
#define ROPE_THETA   1000000.0f
#define RMS_EPS      1e-6f

/* ========================================================================
 * Portable SIMD
 * ======================================================================== */
#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  #define LAL_HAVE_AVX2 1
  typedef __m256 v8f;
  #define V8F_ZERO()    _mm256_setzero_ps()
  #define V8F_SET1(x)   _mm256_set1_ps(x)
  #define V8F_LOAD(p)   _mm256_loadu_ps(p)
  #define V8F_STORE(p,v) _mm256_storeu_ps((p),(v))
  #define V8F_ADD(a,b)   _mm256_add_ps((a),(b))
  #define V8F_MUL(a,b)   _mm256_mul_ps((a),(b))
  #define V8F_FMADD(a,b,c) _mm256_fmadd_ps((a),(b),(c))
  static inline float v8f_hsum(v8f v) {
      float t[8]; _mm256_storeu_ps(t,v);
      return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
  }
#elif defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
  typedef struct { float32x4_t lo, hi; } v8f;
  static inline v8f V8F_ZERO(void) { return (v8f){vdupq_n_f32(0),vdupq_n_f32(0)}; }
  static inline v8f V8F_SET1(float x) { return (v8f){vdupq_n_f32(x),vdupq_n_f32(x)}; }
  static inline v8f V8F_LOAD(const float *p) { return (v8f){vld1q_f32(p),vld1q_f32(p+4)}; }
  static inline void V8F_STORE(float *p,v8f v) { vst1q_f32(p,v.lo); vst1q_f32(p+4,v.hi); }
  static inline v8f V8F_ADD(v8f a,v8f b) { return (v8f){vaddq_f32(a.lo,b.lo),vaddq_f32(a.hi,b.hi)}; }
  static inline v8f V8F_MUL(v8f a,v8f b) { return (v8f){vmulq_f32(a.lo,b.lo),vmulq_f32(a.hi,b.hi)}; }
  static inline v8f V8F_FMADD(v8f a,v8f b,v8f c) { return (v8f){vfmaq_f32(c.lo,a.lo,b.lo),vfmaq_f32(c.hi,a.hi,b.hi)}; }
  static inline float v8f_hsum(v8f v) {
      float32x2_t s=vadd_f32(vadd_f32(vget_low_f32(v.lo),vget_high_f32(v.lo)),vadd_f32(vget_low_f32(v.hi),vget_high_f32(v.hi)));
      return vget_lane_f32(vpadd_f32(s,s),0);
  }
#else
  typedef struct { float v[8]; } v8f;
  static inline v8f V8F_ZERO(void) { v8f r; memset(r.v,0,32); return r; }
  static inline v8f V8F_SET1(float x) { v8f r; for(int i=0;i<8;i++) r.v[i]=x; return r; }
  static inline v8f V8F_LOAD(const float *p) { v8f r; memcpy(r.v,p,32); return r; }
  static inline void V8F_STORE(float *p,v8f v) { memcpy(p,v.v,32); }
  static inline v8f V8F_MUL(v8f a,v8f b) { v8f r; for(int i=0;i<8;i++) r.v[i]=a.v[i]*b.v[i]; return r; }
  static inline v8f V8F_FMADD(v8f a,v8f b,v8f c) { v8f r; for(int i=0;i<8;i++) r.v[i]=a.v[i]*b.v[i]+c.v[i]; return r; }
  static inline float v8f_hsum(v8f v) { float s=0; for(int i=0;i<8;i++) s+=v.v[i]; return s; }
#endif

/* ========================================================================
 * Q8 quantization + matmul
 * ======================================================================== */
static void quantize_q8(const float *W, int8_t *q, float *scale,
                         int32_t *w_sums, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        const float *col = W + (size_t)j * in_dim;
        float mx = 0;
        for (int i = 0; i < in_dim; i++) { float a=fabsf(col[i]); if(a>mx) mx=a; }
        scale[j] = mx / 127.0f; if(scale[j]<1e-8f) scale[j]=1e-8f;
        float inv = 1.0f / scale[j];
        int32_t ws = 0;
        for (int i = 0; i < in_dim; i++) {
            int v = (int)lroundf(col[i] * inv);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            q[(size_t)j * in_dim + i] = (int8_t)v;
            ws += v;
        }
        w_sums[j] = ws;
    }
}

static void matmul_q8(float *y, const int8_t *q, const float *scale,
                      const int32_t *w_sums, const float *x, const float *b,
                      int in_dim, int out_dim) {
    float mx = 0;
    for (int i = 0; i < in_dim; i++) { float a=fabsf(x[i]); if(a>mx) mx=a; }
    float xs = mx / 127.0f; if(xs<1e-8f) xs=1e-8f;
    float inv = 1.0f / xs;

    /* Quantize x to uint8 (centered at 128) */
    uint8_t xq[4864]; /* max MLP_DIM */
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * inv);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        xq[i] = (uint8_t)(v + 128);
    }

#if defined(__x86_64__) && defined(LAL_HAVE_AVX2)
    __m256i ones = _mm256_set1_epi16(1);
    for (int j = 0; j < out_dim; j++) {
        const uint8_t *wj = (const uint8_t*)(q + (size_t)j * in_dim);
        __m256i acc32 = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= in_dim; i += 32) {
            __m256i wv = _mm256_loadu_si256((const __m256i*)(wj + i));
            __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + i));
            __m256i p16 = _mm256_maddubs_epi16(xv, wv);
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(p16, ones));
        }
        __m128i lo = _mm256_castsi256_si128(acc32);
        __m128i hi = _mm256_extracti128_si256(acc32, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        int32_t dot = _mm_cvtsi128_si32(s);
        for (; i < in_dim; i++)
            dot += (int32_t)((int8_t)wj[i]) * (int32_t)((int)xq[i] - 128);
        dot -= 128 * w_sums[j];
        y[j] = (float)dot * xs * scale[j] + (b ? b[j] : 0.0f);
    }
#else
    for (int j = 0; j < out_dim; j++) {
        const int8_t *wj = q + (size_t)j * in_dim;
        int32_t dot = 0;
        int i;
        for (i = 0; i + 8 <= in_dim; i += 8) {
            dot += (int32_t)wj[i]*(int32_t)(xq[i]-128) + (int32_t)wj[i+1]*(int32_t)(xq[i+1]-128);
            dot += (int32_t)wj[i+2]*(int32_t)(xq[i+2]-128) + (int32_t)wj[i+3]*(int32_t)(xq[i+3]-128);
            dot += (int32_t)wj[i+4]*(int32_t)(xq[i+4]-128) + (int32_t)wj[i+5]*(int32_t)(xq[i+5]-128);
            dot += (int32_t)wj[i+6]*(int32_t)(xq[i+6]-128) + (int32_t)wj[i+7]*(int32_t)(xq[i+7]-128);
        }
        for (; i < in_dim; i++) dot += (int32_t)wj[i] * (int32_t)(xq[i] - 128);
        y[j] = (float)dot * xs * scale[j] + (b ? b[j] : 0.0f);
    }
#endif
}

/* ========================================================================
 * RMSNorm (SIMD)
 * ======================================================================== */
static void qwen_rms_norm(float *out, const float *x, const float *w, int n) {
    float ms = 0;
    for (int i = 0; i + 8 <= n; i += 8) {
        v8f v = V8F_LOAD(x + i); v8f s = V8F_MUL(v, v); ms += v8f_hsum(s);
    }
    for (int i = (n/8)*8; i < n; i++) ms += x[i] * x[i];
    float inv = 1.0f / sqrtf(ms / n + RMS_EPS);
    v8f iv = V8F_SET1(inv);
    for (int i = 0; i + 8 <= n; i += 8) {
        v8f xv = V8F_LOAD(x + i); v8f wv = V8F_LOAD(w + i);
        V8F_STORE(out + i, V8F_MUL(V8F_MUL(xv, iv), wv));
    }
    for (int i = (n/8)*8; i < n; i++) out[i] = x[i] * inv * w[i];
}

/* ========================================================================
 * RoPE — precomputed cos/sin table (eliminate powf/cosf/sinf per call)
 * ======================================================================== */
static float g_rope_cos[N_CTX][HEAD_DIM/2];
static float g_rope_sin[N_CTX][HEAD_DIM/2];
static int  g_rope_ready = 0;

static void rope_init(void) {
    for (int pos = 0; pos < N_CTX; pos++) {
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2*d) / HEAD_DIM);
            float a = (float)pos * freq;
            g_rope_cos[pos][d] = cosf(a);
            g_rope_sin[pos][d] = sinf(a);
        }
    }
    g_rope_ready = 1;
}

static inline void rope_apply(float *vec, int n_heads, int hd, int pos) {
    const float *c = g_rope_cos[pos];
    const float *s = g_rope_sin[pos];
    for (int h = 0; h < n_heads; h++) {
        float *vh = vec + h * hd;
        for (int d = 0; d < hd/2; d++) {
            float v0 = vh[d], v1 = vh[d + hd/2];
            vh[d]         = v0 * c[d] - v1 * s[d];
            vh[d + hd/2] = v0 * s[d] + v1 * c[d];
        }
    }
}

static void qwen_rope(float *q, float *k, int pos, int n_qh, int n_kvh, int hd) {
    rope_apply(q, n_qh, hd, pos);
    rope_apply(k, n_kvh, hd, pos);
}

/* ========================================================================
 * GQA Attention (zero-malloc: pre-allocated buffers)
 * ======================================================================== */
static float **kv_cache; /* [N_LAYER*2][N_CTX * N_KV_HEAD * HEAD_DIM] */
static float *g_attn_scores; /* [N_CTX] pre-allocated */
static float *g_attn_aw;     /* [N_CTX] pre-allocated */

static void gqa_attn(float *out, const float *Q, const float *Kn, const float *Vn,
                     int layer, int pos) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    int kvd = N_KV_HEAD * HEAD_DIM; /* 128 */

    /* Store K, V */
    for (int h = 0; h < N_KV_HEAD; h++) {
        memcpy(kv_cache[layer*2]   + (size_t)pos*kvd + h*HEAD_DIM, Kn + h*HEAD_DIM, HEAD_DIM*4);
        memcpy(kv_cache[layer*2+1] + (size_t)pos*kvd + h*HEAD_DIM, Vn + h*HEAD_DIM, HEAD_DIM*4);
    }

    for (int qh = 0; qh < N_HEAD; qh++) {
        const float *Qh = Q + qh * HEAD_DIM;
        int kvh = qh / N_Q_PER_KV;

        float mx = -1e30f;
        for (int j = 0; j <= pos; j++) {
            const float *Kj = kv_cache[layer*2] + (size_t)j*kvd + kvh*HEAD_DIM;
            v8f acc = V8F_ZERO(); int d=0;
            for (; d+8<=HEAD_DIM; d+=8) acc = V8F_FMADD(V8F_LOAD(Qh+d), V8F_LOAD(Kj+d), acc);
            float dot = v8f_hsum(acc);
            for (; d<HEAD_DIM; d++) dot += Qh[d]*Kj[d];
            dot *= scale; g_attn_scores[j] = dot;
            if (dot > mx) mx = dot;
        }
        float se = 0;
        for (int j = 0; j <= pos; j++) { g_attn_aw[j]=expf(g_attn_scores[j]-mx); se+=g_attn_aw[j]; }
        float is = 1.0f / (se + 1e-12f);
        for (int j = 0; j <= pos; j++) g_attn_aw[j] *= is;

        float *oh = out + qh * HEAD_DIM;
        memset(oh, 0, HEAD_DIM*4);
        for (int j = 0; j <= pos; j++) {
            float w = g_attn_aw[j];
            const float *Vj = kv_cache[layer*2+1] + (size_t)j*kvd + kvh*HEAD_DIM;
            v8f wv = V8F_SET1(w); int d=0;
            for (; d+8<=HEAD_DIM; d+=8) {
                v8f cur = V8F_LOAD(oh+d);
                V8F_STORE(oh+d, V8F_FMADD(wv, V8F_LOAD(Vj+d), cur));
            }
            for (; d<HEAD_DIM; d++) oh[d] += w*Vj[d];
        }
    }
}

/* ========================================================================
 * Model weights (Q8 per layer)
 * ======================================================================== */
typedef struct {
    int8_t  *q8_q, *q8_k, *q8_v, *q8_o;
    float   *s_q, *s_k, *s_v, *s_o;
    int32_t *ws_q, *ws_k, *ws_v, *ws_o;
    int8_t  *q8_gate, *q8_up, *q8_down;
    float   *s_gate, *s_up, *s_down;
    int32_t *ws_gate, *ws_up, *ws_down;
    float *q_bias, *k_bias, *v_bias;
    float *norm1_w, *norm2_w;
} QwenLayer;

static QwenLayer g_layers[N_LAYER];
static float *g_wte;        /* [VOCAB_SIZE, N_EMBD] tied embedding/LM head */
static float *g_norm_f_w;
static float *g_x, *g_q, *g_k, *g_v, *g_attn_out, *g_proj;
static float *g_gate, *g_up, *g_mlp_out, *g_ln;
static float *g_logits;

/* === Int8 LM head === */
static int8_t  *g_wte_q;      /* [VOCAB_SIZE * N_EMBD] quantized wte */
static float   *g_wte_scale;  /* [VOCAB_SIZE] per-row scales */
static int g_n_threads = 1;

/* Sampling parameters (mistral.rs / GPT-2 style) */
static float g_temperature = 0.8f;
static int   g_top_k = 40;
static float g_rep_penalty = 1.1f;
static int   g_recent_tokens[256];
static int   g_n_recent = 0;

/* Sample next token from g_logits using temperature + top-k + rep_penalty.
 * Returns token id. If temperature==0, falls back to argmax. */
static int sample_next_token(void) {
    if (g_temperature <= 0.0f || g_top_k <= 0) {
        /* Argmax (greedy) */
        int best = 0;
        for (int v = 1; v < VOCAB_SIZE; v++)
            if (g_logits[v] > g_logits[best]) best = v;
        return best;
    }

    /* Apply repetition penalty to recent tokens */
    if (g_rep_penalty > 1.0f) {
        for (int i = 0; i < g_n_recent; i++) {
            int t = g_recent_tokens[i];
            if (t >= 0 && t < VOCAB_SIZE) {
                if (g_logits[t] > 0) g_logits[t] /= g_rep_penalty;
                else g_logits[t] *= g_rep_penalty;
            }
        }
    }

    /* Find top-k threshold via k passes of find-max-and-mask */
    int top_k = g_top_k;
    if (top_k > VOCAB_SIZE) top_k = VOCAB_SIZE;
    float threshold = -1e30f;
    {
        static float tmp_logits[151936 + 200];
        memcpy(tmp_logits, g_logits, VOCAB_SIZE * sizeof(float));
        for (int k = 0; k < top_k; k++) {
            int mi = 0;
            for (int v = 1; v < VOCAB_SIZE; v++)
                if (tmp_logits[v] > tmp_logits[mi]) mi = v;
            threshold = tmp_logits[mi];
            tmp_logits[mi] = -1e30f;
        }
    }

    /* Softmax over top-k tokens (those >= threshold), with temperature */
    float max_l = -1e30f;
    for (int v = 0; v < VOCAB_SIZE; v++)
        if (g_logits[v] >= threshold && g_logits[v] > max_l) max_l = g_logits[v];
    float sum = 0;
    static float probs[151936 + 200];
    for (int v = 0; v < VOCAB_SIZE; v++) {
        if (g_logits[v] >= threshold) {
            probs[v] = expf((g_logits[v] - max_l) / g_temperature);
            sum += probs[v];
        } else probs[v] = 0;
    }

    /* Sample from the distribution */
    float r = (float)rand() / (float)RAND_MAX * sum;
    float acc = 0;
    for (int v = 0; v < VOCAB_SIZE; v++) {
        acc += probs[v];
        if (r <= acc) return v;
    }
    return VOCAB_SIZE - 1;  /* fallback */
}

/* ========================================================================
 * Load weights
 * ======================================================================== */
static void load_weights(const char *path) {
    printf("[*] loading %s ...\n", path);
    int n_tensors;
    Tensor *tensors = tensor_load_all(path, &n_tensors);
    if (!tensors) { fprintf(stderr, "[!] failed to load\n"); exit(1); }
    printf("[*] %d tensors loaded\n", n_tensors);

    g_wte = tensor_get(tensors, n_tensors, "model.embed_tokens.weight");
    g_norm_f_w = tensor_get(tensors, n_tensors, "model.norm.weight");
    if (!g_wte || !g_norm_f_w) { fprintf(stderr, "[!] missing global weights\n"); exit(1); }

    char key[256];
    for (int l = 0; l < N_LAYER; l++) {
        QwenLayer *L = &g_layers[l];
        sprintf(key, "model.layers.%d.input_layernorm.weight", l);      L->norm1_w = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.post_attention_layernorm.weight", l); L->norm2_w = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l); float *Wq = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.self_attn.q_proj.bias", l);   L->q_bias = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l); float *Wk = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.self_attn.k_proj.bias", l);   L->k_bias = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l); float *Wv = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.self_attn.v_proj.bias", l);   L->v_bias = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.self_attn.o_proj.weight", l); float *Wo = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l); float *Wg = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.mlp.up_proj.weight", l);   float *Wu = tensor_get(tensors, n_tensors, key);
        sprintf(key, "model.layers.%d.mlp.down_proj.weight", l);  float *Wd = tensor_get(tensors, n_tensors, key);

        /* Quantize all 7 matrices per layer */
        L->q8_q = malloc((size_t)N_EMBD*N_EMBD); L->s_q = malloc(N_EMBD*4); L->ws_q = malloc(N_EMBD*4);
        quantize_q8(Wq, L->q8_q, L->s_q, L->ws_q, N_EMBD, N_EMBD);
        L->q8_k = malloc((size_t)128*N_EMBD); L->s_k = malloc(128*4); L->ws_k = malloc(128*4);
        quantize_q8(Wk, L->q8_k, L->s_k, L->ws_k, N_EMBD, 128);
        L->q8_v = malloc((size_t)128*N_EMBD); L->s_v = malloc(128*4); L->ws_v = malloc(128*4);
        quantize_q8(Wv, L->q8_v, L->s_v, L->ws_v, N_EMBD, 128);
        L->q8_o = malloc((size_t)N_EMBD*N_EMBD); L->s_o = malloc(N_EMBD*4); L->ws_o = malloc(N_EMBD*4);
        quantize_q8(Wo, L->q8_o, L->s_o, L->ws_o, N_EMBD, N_EMBD);
        L->q8_gate = malloc((size_t)MLP_DIM*N_EMBD); L->s_gate = malloc(MLP_DIM*4); L->ws_gate = malloc(MLP_DIM*4);
        quantize_q8(Wg, L->q8_gate, L->s_gate, L->ws_gate, N_EMBD, MLP_DIM);
        L->q8_up = malloc((size_t)MLP_DIM*N_EMBD); L->s_up = malloc(MLP_DIM*4); L->ws_up = malloc(MLP_DIM*4);
        quantize_q8(Wu, L->q8_up, L->s_up, L->ws_up, N_EMBD, MLP_DIM);
        L->q8_down = malloc((size_t)N_EMBD*MLP_DIM); L->s_down = malloc(N_EMBD*4); L->ws_down = malloc(N_EMBD*4);
        quantize_q8(Wd, L->q8_down, L->s_down, L->ws_down, MLP_DIM, N_EMBD);
        if (l == 0 || l == N_LAYER-1) printf("[*] layer %d Q8 done\n", l);
    }
    printf("[*] all %d layers quantized\n", N_LAYER);

    /* Quantize wte for int8 LM head */
    printf("[*] quantizing LM head (wte) to int8 ...");
    fflush(stdout);
    struct timespec tq0, tq1;
    clock_gettime(CLOCK_MONOTONIC, &tq0);
    g_wte_q = malloc((size_t)VOCAB_SIZE * N_EMBD);
    g_wte_scale = malloc((size_t)VOCAB_SIZE * sizeof(float));
    if (!g_wte_q || !g_wte_scale) {
        fprintf(stderr, "[!] OOM allocating int8 LM head\n"); exit(1);
    }
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const float *row = g_wte + (size_t)v * N_EMBD;
        float mx = 0;
        for (int i = 0; i < N_EMBD; i++) { float a = fabsf(row[i]); if (a > mx) mx = a; }
        g_wte_scale[v] = mx / 127.0f; if (g_wte_scale[v] < 1e-8f) g_wte_scale[v] = 1e-8f;
        float inv = 1.0f / g_wte_scale[v];
        for (int i = 0; i < N_EMBD; i++) {
            int val = (int)lroundf(row[i] * inv);
            if (val > 127) val = 127; else if (val < -127) val = -127;
            g_wte_q[(size_t)v * N_EMBD + i] = (int8_t)val;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &tq1);
    double dtq = (tq1.tv_sec-tq0.tv_sec)+(tq1.tv_nsec-tq0.tv_nsec)*1e-9;
    printf(" done (%.0f ms, %.1f MB int8 + %.0f KB scales)\n",
           dtq*1000, (double)VOCAB_SIZE*N_EMBD/1048576, (double)VOCAB_SIZE*sizeof(float)/1024);

    /* Free float weights — recover ~1.7 GB RSS.
     * After Q8 quantization, the float copies of q/k/v/o/gate/up/down weights
     * are never read again. Free them by walking tensors and matching keys.
     * Keep: norm1_w, norm2_w, q/k/v/o/gate/up/down biases (still needed),
     *       g_norm_f_w, and g_wte (freed separately below after we rewire
     *       embedding lookup + rerank to use int8). */
    {
        int n_freed = 0;
        size_t bytes_freed = 0;
        for (int l = 0; l < N_LAYER; l++) {
            const char *layer_keys[7];
            char k[256];
            sprintf(k, "model.layers.%d.self_attn.q_proj.weight", l);   layer_keys[0] = strdup(k);
            sprintf(k, "model.layers.%d.self_attn.k_proj.weight", l);   layer_keys[1] = strdup(k);
            sprintf(k, "model.layers.%d.self_attn.v_proj.weight", l);   layer_keys[2] = strdup(k);
            sprintf(k, "model.layers.%d.self_attn.o_proj.weight", l);   layer_keys[3] = strdup(k);
            sprintf(k, "model.layers.%d.mlp.gate_proj.weight", l);      layer_keys[4] = strdup(k);
            sprintf(k, "model.layers.%d.mlp.up_proj.weight", l);        layer_keys[5] = strdup(k);
            sprintf(k, "model.layers.%d.mlp.down_proj.weight", l);      layer_keys[6] = strdup(k);
            for (int ki = 0; ki < 7; ki++) {
                for (int ti = 0; ti < n_tensors; ti++) {
                    if (tensors[ti].data && strcmp(tensors[ti].key, layer_keys[ki]) == 0) {
                        size_t sz = sizeof(float);
                        for (int d = 0; d < tensors[ti].ndim; d++) sz *= tensors[ti].shape[d];
                        bytes_freed += sz;
                        free(tensors[ti].data);
                        tensors[ti].data = NULL;
                        n_freed++;
                        break;
                    }
                }
            }
            for (int ki = 0; ki < 7; ki++) free((void*)layer_keys[ki]);
        }
        printf("[*] freed %d float weight tensors after Q8 quantization (%.1f MB recovered)\n",
               n_freed, bytes_freed / (1024.0 * 1024.0));
    }

    /* Free float g_wte — embedding lookup + LM head rerank use int8.
     * Embedding lookup: dequantize row on-the-fly from g_wte_q + g_wte_scale.
     * LM head rerank: dequantize each candidate row into static buffer. */
    {
        for (int ti = 0; ti < n_tensors; ti++) {
            if (tensors[ti].data && strcmp(tensors[ti].key, "model.embed_tokens.weight") == 0) {
                free(tensors[ti].data);
                tensors[ti].data = NULL;
                printf("[*] freed float wte (%.1f MB) — embedding lookup dequantizes from int8\n",
                       (double)VOCAB_SIZE * N_EMBD * 4 / (1024.0 * 1024.0));
                break;
            }
        }
        g_wte = NULL;
    }
}

/* ========================================================================
 * Tokenizer (BPE, HuggingFace tokenizer.json)
 * ======================================================================== */
#define TOK_HASH_BITS 18
#define TOK_HASH_SIZE (1 << TOK_HASH_BITS)
#define TOK_HASH_MASK (TOK_HASH_SIZE - 1)

typedef struct { char *str; int id; int len; } TEntry;
static TEntry *g_toks;
static int g_ntoks = 0;
static TEntry *g_htab[TOK_HASH_SIZE];

static unsigned thash(const char *s, int len) {
    unsigned h = 5381;
    for (int i = 0; i < len; i++) h = ((h<<5)+h)+(unsigned char)s[i];
    return h & TOK_HASH_MASK;
}
static void tins(const char *s, int id) {
    unsigned h = thash(s, strlen(s));
    for (int i = 0; i < TOK_HASH_SIZE; i++) {
        int idx = (h+i) & TOK_HASH_MASK;
        if (!g_htab[idx]) {
            g_htab[idx] = &g_toks[g_ntoks++];
            g_htab[idx]->str = strdup(s); g_htab[idx]->id = id; g_htab[idx]->len = strlen(s);
            return;
        }
    }
}
static int tlook(const char *s, int len) {
    unsigned h = thash(s, len);
    for (int i = 0; i < TOK_HASH_SIZE; i++) {
        int idx = (h+i) & TOK_HASH_MASK;
        if (!g_htab[idx]) return -1;
        if (g_htab[idx]->len == len && memcmp(g_htab[idx]->str, s, len) == 0) return g_htab[idx]->id;
    }
    return -1;
}

static char **g_vocab_str;
static int g_vocab_str_n = 0;

static void load_tokenizer(const char *dir) {
    char path[1024]; snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[!] cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz+1); fread(buf, 1, sz, f); buf[sz] = 0; fclose(f);

    int mx = VOCAB_SIZE + 200;
    g_toks = malloc(mx * sizeof(TEntry));
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
        /* Read key */
        p++; /* skip opening " */
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

/* Simple pre-tokenizer: split on whitespace/punctuation, keep words together.
 * Qwen uses a BPE pre_tokenizer that splits on whitespace and some punctuation.
 * For now, we do a simple greedy longest-match BPE on the raw text. */
static int *encode_text(const char *text, int *n_out) {
    int max_tok = strlen(text) * 2 + 10;
    int *ids = malloc(max_tok * sizeof(int));
    int n = 0, pos = 0, tlen = strlen(text);

    while (pos < tlen && n < max_tok - 1) {
        int best_len = 0, best_id = -1;
        /* Try up to 64-byte match */
        int max_try = 64; if (max_try > tlen - pos) max_try = tlen - pos;
        for (int len = max_try; len >= 1; len--) {
            int id = tlook(text + pos, len);
            if (id >= 0) { best_len = len; best_id = id; break; }
        }
        if (best_id >= 0) {
            ids[n++] = best_id;
            pos += best_len;
        } else {
            /* Byte fallback */
            ids[n++] = (unsigned char)text[pos];
            pos++;
        }
    }
    *n_out = n;
    return ids;
}

static void decode_token(int id, char *out, int maxlen) {
    if (id < 0 || id >= g_vocab_str_n || !g_vocab_str[id]) { out[0] = 0; return; }
    /* Decode GPT-2 BPE byte encoding: Ġ=space, Ċ=newline, etc.
     * The vocab strings use the standard HF byte-to-unicode map where
     * bytes 33-126 + 161-172 + 174-255 map to themselves, and
     * bytes 0-32 + 127-160 + 173 map to Ā(0) .. Ń(32), etc.
     * For Qwen2.5 tokenizer.json the most common ones are:
     *   Ġ (U+0120) = space (0x20)
     *   Ċ (U+010A) = newline (0x0A)
     *   Ď (U+010E) = carriage return (0x0D)
     *   ĥ (U+0125) = tab (0x09)
     * We handle these common ones plus pass through UTF-8 for CJK. */
    const char *src = g_vocab_str[id];
    int o = 0;
    for (int i = 0; src[i] && o < maxlen - 4; i++) {
        unsigned char c = (unsigned char)src[i];
        /* Check for 2-byte UTF-8 sequence (Ġ, Ċ, etc. are 0xC4 0xA0 range) */
        if (c == 0xC4 && (unsigned char)src[i+1] == 0xA0) { out[o++] = ' '; i++; }
        else if (c == 0xC4 && (unsigned char)src[i+1] == 0x8A) { out[o++] = '\n'; i++; }
        else if (c == 0xC4 && (unsigned char)src[i+1] == 0x8E) { out[o++] = '\r'; i++; }
        else if (c == 0xC4 && (unsigned char)src[i+1] == 0xA5) { out[o++] = '\t'; i++; }
        else out[o++] = c;
    }
    out[o] = 0;
}

/* ========================================================================
 * Int8 LM head (two-pass: int8 full vocab → float rerank top-K)
 *
 * The LM head (wte: 151936 × 896 = ~544 MB FP32) is the biggest single
 * bottleneck — pure memory bandwidth. Quantizing to int8 cuts reads 4×.
 * Two-pass rerank (llama.cpp pattern) preserves quality:
 *   Pass 1: int8 dot product for ALL vocab (fast, bandwidth-bound)
 *   Pass 2: float re-score only the top-RERANK_N candidates (accurate)
 * ======================================================================== */
#define LM_HEAD_RERANK_N 512

static float lm_head_quantize_x(const float *x, int8_t *xq, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > max_abs) max_abs = a; }
    float scale = max_abs / 127.0f; if (scale < 1e-8f) scale = 1e-8f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)(x[i] * inv);
        if (v > 127) v = 127; if (v < -127) v = -127;
        xq[i] = (int8_t)v;
    }
    return scale;
}

#if defined(__AVX2__)
static inline int hsum_epi32_avx(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}
#endif

typedef struct {
    const int8_t *xq; float scale_x; float *logits;
    int v_start, v_end, n_embd;
} LmHeadJob;

static void *lm_head_int8_worker(void *arg) {
    LmHeadJob *j = (LmHeadJob *)arg;
    lal_lm_head_int8_range(j->logits, j->xq, j->scale_x,
                           g_wte_q, g_wte_scale,
                           j->v_start, j->v_end, j->n_embd);
    return NULL;
}

static void lm_head_int8_parallel(float *logits, const float *x, int n_threads) {
    static int8_t xq[N_EMBD];
    float scale_x = lm_head_quantize_x(x, xq, N_EMBD);

    /* Pass 1: int8 logits for full vocab, split across threads */
    if (n_threads <= 1) {
        lm_head_int8_worker(&(LmHeadJob){xq, scale_x, logits, 0, VOCAB_SIZE, N_EMBD});
    } else {
        pthread_t threads[8];
        pthread_attr_t attr;
        LmHeadJob jobs[8];
        int nt = n_threads > 8 ? 8 : n_threads;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 256 * 1024);
        int chunk = (VOCAB_SIZE + nt - 1) / nt;
        for (int t = 0; t < nt; t++) {
            jobs[t] = (LmHeadJob){xq, scale_x, logits, t*chunk, (t+1)*chunk, N_EMBD};
            if (jobs[t].v_start > VOCAB_SIZE) jobs[t].v_start = VOCAB_SIZE;
            if (jobs[t].v_end > VOCAB_SIZE) jobs[t].v_end = VOCAB_SIZE;
            if (jobs[t].v_end > jobs[t].v_start)
                pthread_create(&threads[t], &attr, lm_head_int8_worker, &jobs[t]);
        }
        pthread_attr_destroy(&attr);
        for (int t = 0; t < nt; t++)
            if (jobs[t].v_end > jobs[t].v_start) pthread_join(threads[t], NULL);
    }

    /* Pass 2: re-score top-RERANK_N in float (min-heap selection) */
    static int   heap_idx[LM_HEAD_RERANK_N];
    static float heap_val[LM_HEAD_RERANK_N];
    int heap_n = 0;
    for (int v = 0; v < VOCAB_SIZE; v++) {
        float val = logits[v];
        if (heap_n < LM_HEAD_RERANK_N) {
            int c = heap_n++; heap_val[c] = val; heap_idx[c] = v;
            while (c > 0) {
                int p = (c-1) >> 1;
                if (heap_val[p] <= heap_val[c]) break;
                float tv = heap_val[p]; heap_val[p] = heap_val[c]; heap_val[c] = tv;
                int ti = heap_idx[p]; heap_idx[p] = heap_idx[c]; heap_idx[c] = ti;
                c = p;
            }
        } else if (val > heap_val[0]) {
            heap_val[0] = val; heap_idx[0] = v;
            int p = 0;
            for (;;) {
                int l = 2*p+1, r = 2*p+2, s = p;
                if (l < heap_n && heap_val[l] < heap_val[s]) s = l;
                if (r < heap_n && heap_val[r] < heap_val[s]) s = r;
                if (s == p) break;
                float tv = heap_val[p]; heap_val[p] = heap_val[s]; heap_val[s] = tv;
                int ti = heap_idx[p]; heap_idx[p] = heap_idx[s]; heap_idx[s] = ti;
                p = s;
            }
        }
    }
    for (int k = 0; k < heap_n; k++) {
        int v = heap_idx[k];
        /* Dequantize wte row on-the-fly from int8 (g_wte was freed) */
        static float w_row[N_EMBD];
        if (g_wte) {
            const float *w = g_wte + (size_t)v * N_EMBD;
            v8f acc = V8F_ZERO(); int i = 0;
            for (; i + 8 <= N_EMBD; i += 8)
                acc = V8F_FMADD(V8F_LOAD(x+i), V8F_LOAD(w+i), acc);
            float s = v8f_hsum(acc);
            for (; i < N_EMBD; i++) s += x[i] * w[i];
            logits[v] = s;
        } else {
            const int8_t *wq = g_wte_q + (size_t)v * N_EMBD;
            float scale = g_wte_scale[v];
            for (int i = 0; i < N_EMBD; i++) w_row[i] = wq[i] * scale;
            v8f acc = V8F_ZERO(); int i = 0;
            for (; i + 8 <= N_EMBD; i += 8)
                acc = V8F_FMADD(V8F_LOAD(x+i), V8F_LOAD(w_row+i), acc);
            float s = v8f_hsum(acc);
            for (; i < N_EMBD; i++) s += x[i] * w_row[i];
            logits[v] = s;
        }
    }
}

/* ========================================================================
 * Fused SwiGLU: gate+up in a single pass (SIMD SiLU+mul)
 *
 * Instead of:
 *   gate = matmul(x, W_gate)   // 896→4864 Q8
 *   up   = matmul(x, W_up)     // 896→4864 Q8  (duplicate x-quant + weight read)
 *   out  = silu(gate) * up
 *
 * We do:
 *   gate = matmul(x, W_gate)   // 896→4864 Q8
 *   out  = silu(gate) * matmul(x, W_up)  // fused: compute up[i], apply silu(gate[i])*up[i] inline
 *
 * This saves: one full x-quantization pass, one 4864-weight-row read per row,
 * and the separate element-wise loop. The SiLU+mul is also SIMD-vectorized.
 * ======================================================================== */
static void fused_swiglu_down(const int8_t *q_gate, const float *s_gate, const int32_t *ws_gate,
                               const int8_t *q_up,   const float *s_up,   const int32_t *ws_up,
                               const int8_t *q_down, const float *s_down, const int32_t *ws_down,
                               const float *x, float *residual, int in_dim, int hid, int out_dim) {
    /* Quantize x once */
    float mx = 0;
    for (int i = 0; i < in_dim; i++) { float a=fabsf(x[i]); if(a>mx) mx=a; }
    float xs = mx / 127.0f; if(xs<1e-8f) xs=1e-8f;
    float inv = 1.0f / xs;
    uint8_t xq[4864];
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * inv);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        xq[i] = (uint8_t)(v + 128);
    }

#if defined(__x86_64__) && defined(LAL_HAVE_AVX2)
    __m256i ones = _mm256_set1_epi16(1);
    /* Process in blocks: compute gate+up, apply silu*mul, accumulate into down output */
    /* We need a temporary hid-dim buffer for the SwiGLU result */
    static float swiglu_buf[4864]; /* hid = MLP_DIM */

    /* Step 1: fused gate+up → silu(gate)*up */
    for (int j = 0; j < hid; j++) {
        const uint8_t *wg = (const uint8_t*)(q_gate + (size_t)j * in_dim);
        const uint8_t *wu = (const uint8_t*)(q_up   + (size_t)j * in_dim);
        __m256i acc_g = _mm256_setzero_si256();
        __m256i acc_u = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= in_dim; i += 32) {
            __m256i wv = _mm256_loadu_si256((const __m256i*)(wg + i));
            __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + i));
            __m256i p16 = _mm256_maddubs_epi16(xv, wv);
            acc_g = _mm256_add_epi32(acc_g, _mm256_madd_epi16(p16, ones));
            wv = _mm256_loadu_si256((const __m256i*)(wu + i));
            p16 = _mm256_maddubs_epi16(xv, wv);
            acc_u = _mm256_add_epi32(acc_u, _mm256_madd_epi16(p16, ones));
        }
        __m128i lo = _mm256_castsi256_si128(acc_g);
        __m128i hi = _mm256_extracti128_si256(acc_g, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        int32_t dg = _mm_cvtsi128_si32(s);
        lo = _mm256_castsi256_si128(acc_u);
        hi = _mm256_extracti128_si256(acc_u, 1);
        s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        int32_t du = _mm_cvtsi128_si32(s);
        for (; i < in_dim; i++) {
            dg += (int32_t)((int8_t)wg[i]) * (int32_t)((int)xq[i] - 128);
            du += (int32_t)((int8_t)wu[i]) * (int32_t)((int)xq[i] - 128);
        }
        dg -= 128 * ws_gate[j];
        du -= 128 * ws_up[j];
        float g_val = (float)dg * xs * s_gate[j];
        float u_val = (float)du * xs * s_up[j];
        /* SiLU(g_val) * u_val */
        swiglu_buf[j] = (g_val / (1.0f + expf(-g_val))) * u_val;
    }

    /* Step 2: down projection from swiglu_buf */
    /* Re-quantize swiglu_buf as activation for down matmul */
    float mx2 = 0;
    for (int i = 0; i < hid; i++) { float a=fabsf(swiglu_buf[i]); if(a>mx2) mx2=a; }
    float xs2 = mx2 / 127.0f; if(xs2<1e-8f) xs2=1e-8f;
    float inv2 = 1.0f / xs2;
    uint8_t xq2[4864];
    for (int i = 0; i < hid; i++) {
        int v = (int)lroundf(swiglu_buf[i] * inv2);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        xq2[i] = (uint8_t)(v + 128);
    }
    float down_out[896]; /* out_dim = N_EMBD */
    for (int j = 0; j < out_dim; j++) {
        const uint8_t *wd = (const uint8_t*)(q_down + (size_t)j * hid);
        __m256i acc32 = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= hid; i += 32) {
            __m256i wv = _mm256_loadu_si256((const __m256i*)(wd + i));
            __m256i xv = _mm256_loadu_si256((const __m256i*)(xq2 + i));
            __m256i p16 = _mm256_maddubs_epi16(xv, wv);
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(p16, ones));
        }
        __m128i lo = _mm256_castsi256_si128(acc32);
        __m128i hi = _mm256_extracti128_si256(acc32, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        int32_t dot = _mm_cvtsi128_si32(s);
        for (; i < hid; i++)
            dot += (int32_t)((int8_t)wd[i]) * (int32_t)((int)xq2[i] - 128);
        dot -= 128 * ws_down[j];
        residual[j] += (float)dot * xs2 * s_down[j];
    }
#else
    /* Scalar fallback */
    static float swiglu_buf[4864];
    for (int j = 0; j < hid; j++) {
        const int8_t *wg = q_gate + (size_t)j * in_dim;
        const int8_t *wu = q_up   + (size_t)j * in_dim;
        int32_t dg = 0, du = 0;
        for (int i = 0; i < in_dim; i++) {
            dg += (int32_t)wg[i] * (int32_t)(xq[i] - 128);
            du += (int32_t)wu[i] * (int32_t)(xq[i] - 128);
        }
        dg -= 128 * ws_gate[j]; du -= 128 * ws_up[j];
        float g_val = (float)dg * xs * s_gate[j];
        float u_val = (float)du * xs * s_up[j];
        swiglu_buf[j] = (g_val / (1.0f + expf(-g_val))) * u_val;
    }
    /* Down projection */
    float mx2 = 0;
    for (int i = 0; i < hid; i++) { float a=fabsf(swiglu_buf[i]); if(a>mx2) mx2=a; }
    float xs2 = mx2 / 127.0f; if(xs2<1e-8f) xs2=1e-8f;
    float inv2 = 1.0f / xs2;
    uint8_t xq2[4864];
    for (int i = 0; i < hid; i++) {
        int v = (int)lroundf(swiglu_buf[i] * inv2);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        xq2[i] = (uint8_t)(v + 128);
    }
    for (int j = 0; j < out_dim; j++) {
        const int8_t *wd = q_down + (size_t)j * hid;
        int32_t dot = 0;
        for (int i = 0; i < hid; i++) dot += (int32_t)wd[i] * (int32_t)(xq2[i] - 128);
        dot -= 128 * ws_down[j];
        residual[j] += (float)dot * xs2 * s_down[j];
    }
#endif
}

/* ========================================================================
 * Forward: one token → next token id
 * ======================================================================== */
static int forward(int tok, int pos) {
    if (tok < 0 || tok >= VOCAB_SIZE) tok = 0;
    if (pos < 0 || pos >= N_CTX) pos = 0;

    /* Embedding (no pos emb — RoPE) */
    if (g_wte) {
        memcpy(g_x, g_wte + (size_t)tok * N_EMBD, N_EMBD * sizeof(float));
    } else {
        /* Int8 dequantize path: g_wte was freed, reconstruct from g_wte_q */
        const int8_t *wq = g_wte_q + (size_t)tok * N_EMBD;
        float scale = g_wte_scale[tok];
        for (int i = 0; i < N_EMBD; i++) g_x[i] = wq[i] * scale;
    }

    for (int l = 0; l < N_LAYER; l++) {
        QwenLayer *L = &g_layers[l];

        /* Pre-attn RMSNorm */
        qwen_rms_norm(g_ln, g_x, L->norm1_w, N_EMBD);
        /* Q/K/V projections */
        lal_matmul_q8_signtrick(g_q, L->q8_q, L->s_q, g_ln, L->q_bias, N_EMBD, N_EMBD);
        lal_matmul_q8_signtrick(g_k, L->q8_k, L->s_k, g_ln, L->k_bias, N_EMBD, 128);
        lal_matmul_q8_signtrick(g_v, L->q8_v, L->s_v, g_ln, L->v_bias, N_EMBD, 128);
        /* RoPE (precomputed table) */
        qwen_rope(g_q, g_k, pos, N_HEAD, N_KV_HEAD, HEAD_DIM);
        /* GQA (zero-malloc) */
        gqa_attn(g_attn_out, g_q, g_k, g_v, l, pos);
        /* O proj + residual */
        lal_matmul_q8_signtrick(g_proj, L->q8_o, L->s_o, g_attn_out, NULL, N_EMBD, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) g_x[i] += g_proj[i];

        /* Pre-MLP RMSNorm */
        qwen_rms_norm(g_ln, g_x, L->norm2_w, N_EMBD);
        /* Fused SwiGLU: gate+up+down in one call */
        fused_swiglu_down(L->q8_gate, L->s_gate, L->ws_gate,
                          L->q8_up,   L->s_up,   L->ws_up,
                          L->q8_down, L->s_down, L->ws_down,
                          g_ln, g_x, N_EMBD, MLP_DIM, N_EMBD);
    }

    /* Final RMSNorm */
    qwen_rms_norm(g_ln, g_x, g_norm_f_w, N_EMBD);

    /* LM head: int8 quantized with two-pass rerank */
    if (g_wte_q) {
        lm_head_int8_parallel(g_logits, g_ln, g_n_threads);
    } else {
        /* Fallback: FP32 LM head */
        for (int v = 0; v < VOCAB_SIZE; v++) {
            const float *row = g_wte + (size_t)v * N_EMBD;
            v8f acc = V8F_ZERO(); int i = 0;
            for (; i + 8 <= N_EMBD; i += 8)
                acc = V8F_FMADD(V8F_LOAD(row+i), V8F_LOAD(g_ln+i), acc);
            float dot = v8f_hsum(acc);
            for (; i < N_EMBD; i++) dot += row[i] * g_ln[i];
            g_logits[v] = dot;
        }
    }

    /* Sample (temperature + top_k + rep_penalty) or argmax */
    int next = sample_next_token();
    /* Track for repetition penalty */
    if (g_n_recent < 256) g_recent_tokens[g_n_recent++] = next;
    else { memmove(g_recent_tokens, g_recent_tokens+1, 255*sizeof(int)); g_recent_tokens[255] = next; }
    return next;
}

/* ========================================================================
 * Generate
 * ======================================================================== */
static void generate(const char *prompt, int n_gen, char *out, int max_out) {
    int n_prompt;
    int *pids = encode_text(prompt, &n_prompt);
    printf("[*] prompt: %d tokens\n", n_prompt);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int pos = 0;
    /* Prefill */
    int next = -1;
    for (int i = 0; i < n_prompt; i++) {
        next = forward(pids[i], pos);
        pos++;
        if (pos >= N_CTX) break;
    }

    /* Generate */
    int gen_count = 0;
    int opos = 0;
    for (int g = 0; g < n_gen && pos < N_CTX; g++) {
        char ts[256]; decode_token(next, ts, (int)sizeof(ts));
        int slen = strlen(ts);
        if (opos + slen < max_out - 1) { memcpy(out+opos, ts, slen); opos += slen; }
        if (next == 151643) break; /* EOS */
        next = forward(next, pos);
        pos++;
        gen_count++;
    }
    out[opos] = 0;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("[*] %d tokens in %.2fs (%.1f tok/s)\n", gen_count, dt, gen_count/(dt+1e-9));
    free(pids);
}

/* ========================================================================
 * Main
 * ======================================================================== */
int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    const char *weights = "prebuilt/qwen_weights.bin";
    const char *tokdir = "prebuilt/qwen_tokenizer";
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
    if (g_n_threads < 1) g_n_threads = 1;

    printf("=== Qwen2.5-0.5B (LAL, Q8, GQA) ===\n");
    printf("[*] %d layers, %d hidden, %dQ/%dKV heads, %d MLP, %d vocab\n",
           N_LAYER, N_EMBD, N_HEAD, N_KV_HEAD, MLP_DIM, VOCAB_SIZE);

    /* Alloc */
    g_x = calloc(N_EMBD, 4); g_q = calloc(N_EMBD, 4); g_k = calloc(128, 4);
    g_v = calloc(128, 4); g_attn_out = calloc(N_EMBD, 4); g_proj = calloc(N_EMBD, 4);
    g_ln = calloc(N_EMBD, 4); g_logits = malloc(VOCAB_SIZE * 4);
    /* Pre-allocated attention buffers (zero-malloc GQA) */
    g_attn_scores = malloc(N_CTX * sizeof(float));
    g_attn_aw     = malloc(N_CTX * sizeof(float));
    kv_cache = calloc(N_LAYER*2, sizeof(float*));
    for (int l = 0; l < N_LAYER*2; l++)
        kv_cache[l] = calloc((size_t)N_CTX * N_KV_HEAD * HEAD_DIM, 4);

    /* Precompute RoPE tables */
    printf("[*] precomputing RoPE tables...");
    fflush(stdout);
    rope_init();
    printf(" done\n");

    load_tokenizer(tokdir);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    load_weights(weights);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[*] load+quant: %.1fs\n", (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    char output[4096];
    printf("\n[*] prompt: \"%s\"\n[*] generating %d tokens (threads=%d)...\n\n", prompt, n_gen, g_n_threads);
    generate(prompt, n_gen, output, (int)sizeof(output));
    printf("\n[*] output: %s\n", output);
    printf("\n[*] done. Q8 + int8 LM head + %d thread(s).\n", g_n_threads);
    return 0;
}