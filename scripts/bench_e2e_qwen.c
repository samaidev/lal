/* bench_e2e_qwen.c — End-to-end benchmark for Qwen2.5-0.5B forward pass
 *
 * Measures:
 *   1. Per-token time (decode phase, no prefill)
 *   2. Breakdown: transformer layers vs LM head
 *   3. FP32 vs int8 LM head, 1T vs 2T
 *
 * Build: cc -O3 -mavx2 -mfma -Wno-unused-function -Wno-unused-variable -I. \
 *          -o scripts/bench_e2e_qwen scripts/bench_e2e_qwen.c \
 *          runtime/lal_runtime.c -lm -lpthread
 * Run:   ./scripts/bench_e2e_qwen
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#include "runtime/lal_runtime.h"

#define N_EMBD       896
#define N_LAYER      24
#define N_HEAD       14
#define N_KV_HEAD    2
#define HEAD_DIM     64
#define N_Q_PER_KV   7
#define MLP_DIM      4864
#define VOCAB_SIZE   151936
#define N_CTX        2048
#define ROPE_THETA   1000000.0f
#define RMS_EPS      1e-6f
#define N_DECODE     30

/* Portable SIMD */
#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
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

/* Q8 quantization + matmul (same as qwen_server.c) */
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
    uint8_t xq[4864];
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
        for (int i = 0; i < in_dim; i++) dot += (int32_t)wj[i] * (int32_t)(xq[i] - 128);
        y[j] = (float)dot * xs * scale[j] + (b ? b[j] : 0.0f);
    }
#endif
}

static void qwen_rms_norm(float *out, const float *x, const float *w, int n) {
    float ms = 0;
    for (int i = 0; i + 8 <= n; i += 8) {
        v8f v = V8F_LOAD(x + i); v8f s = V8F_MUL(v, v); ms += v8f_hsum(s);
    }
    for (int i = (n/8)*8; i < n; i++) ms += x[i] * x[i];
    float iv = 1.0f / sqrtf(ms / n + RMS_EPS);
    v8f ivv = V8F_SET1(iv);
    for (int i = 0; i + 8 <= n; i += 8) {
        v8f xv = V8F_LOAD(x + i); v8f wv = V8F_LOAD(w + i);
        V8F_STORE(out + i, V8F_MUL(V8F_MUL(xv, ivv), wv));
    }
    for (int i = (n/8)*8; i < n; i++) out[i] = x[i] * iv * w[i];
}

static void qwen_rope(float *q, float *k, int pos, int n_qh, int n_kvh, int hd) {
    for (int h = 0; h < n_qh; h++) {
        float *qh = q + h * hd;
        for (int d = 0; d < hd/2; d++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2*d) / hd);
            float a = (float)pos * freq;
            float c = cosf(a), s = sinf(a);
            float q0 = qh[d], q1 = qh[d + hd/2];
            qh[d] = q0*c - q1*s; qh[d + hd/2] = q0*s + q1*c;
        }
    }
    for (int h = 0; h < n_kvh; h++) {
        float *kh = k + h * hd;
        for (int d = 0; d < hd/2; d++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2*d) / hd);
            float a = (float)pos * freq;
            float c = cosf(a), s = sinf(a);
            float k0 = kh[d], k1 = kh[d + hd/2];
            kh[d] = k0*c - k1*s; kh[d + hd/2] = k0*s + k1*c;
        }
    }
}

static float **kv_cache;

static void gqa_attn(float *out, const float *Q, const float *Kn, const float *Vn,
                     int layer, int pos) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    int kvd = N_KV_HEAD * HEAD_DIM;
    for (int h = 0; h < N_KV_HEAD; h++) {
        memcpy(kv_cache[layer*2]   + (size_t)pos*kvd + h*HEAD_DIM, Kn + h*HEAD_DIM, HEAD_DIM*4);
        memcpy(kv_cache[layer*2+1] + (size_t)pos*kvd + h*HEAD_DIM, Vn + h*HEAD_DIM, HEAD_DIM*4);
    }
    float *scores = malloc((pos+1)*sizeof(float));
    float *aw    = malloc((pos+1)*sizeof(float));
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
            dot *= scale; scores[j] = dot;
            if (dot > mx) mx = dot;
        }
        float se = 0;
        for (int j = 0; j <= pos; j++) { aw[j]=expf(scores[j]-mx); se+=aw[j]; }
        float is = 1.0f / (se + 1e-12f);
        for (int j = 0; j <= pos; j++) aw[j] *= is;
        float *oh = out + qh * HEAD_DIM;
        memset(oh, 0, HEAD_DIM*4);
        for (int j = 0; j <= pos; j++) {
            float w = aw[j];
            const float *Vj = kv_cache[layer*2+1] + (size_t)j*kvd + kvh*HEAD_DIM;
            v8f wv = V8F_SET1(w); int d=0;
            for (; d+8<=HEAD_DIM; d+=8) {
                v8f cur = V8F_LOAD(oh+d);
                V8F_STORE(oh+d, V8F_FMADD(wv, V8F_LOAD(Vj+d), cur));
            }
            for (; d<HEAD_DIM; d++) oh[d] += w*Vj[d];
        }
    }
    free(scores); free(aw);
}

/* Layer weights */
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
static float *g_wte;
static float *g_norm_f_w;
static float *g_x, *g_q, *g_k, *g_v, *g_attn_out, *g_proj;
static float *g_gate, *g_up, *g_mlp_out, *g_ln;
static float *g_logits;

/* Int8 LM head */
static int8_t  *g_wte_q;
static float   *g_wte_scale;

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
    }
    printf("[*] all %d layers Q8 quantized\n", N_LAYER);

    /* Quantize wte for int8 LM head */
    printf("[*] quantizing LM head (wte) to int8 ...\n");
    g_wte_q = malloc((size_t)VOCAB_SIZE * N_EMBD);
    g_wte_scale = malloc((size_t)VOCAB_SIZE * sizeof(float));
    if (!g_wte_q || !g_wte_scale) { fprintf(stderr, "[!] OOM\n"); exit(1); }
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const float *row = g_wte + (size_t)v * N_EMBD;
        float mx = 0;
        for (int i = 0; i < N_EMBD; i++) { float a = fabsf(row[i]); if (a > mx) mx = a; }
        g_wte_scale[v] = mx / 127.0f;
        if (g_wte_scale[v] < 1e-8f) g_wte_scale[v] = 1e-8f;
        float inv = 1.0f / g_wte_scale[v];
        for (int i = 0; i < N_EMBD; i++) {
            int val = (int)lroundf(row[i] * inv);
            if (val > 127) val = 127;
            if (val < -127) val = -127;
            g_wte_q[(size_t)v * N_EMBD + i] = (int8_t)val;
        }
    }
    printf("[*] LM head int8: %.1f MB\n", (double)VOCAB_SIZE*N_EMBD/1048576);
}

/* Int8 LM head */
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
    int v_start, v_end;
} LmHeadJob;

static void *int8_worker(void *arg) {
    LmHeadJob *j = (LmHeadJob *)arg;
    const int8_t *xq = j->xq;
    for (int v = j->v_start; v < j->v_end; v++) {
        const int8_t *wv = g_wte_q + (size_t)v * N_EMBD;
        int dot = 0;
#if defined(__AVX2__)
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= N_EMBD; i += 32) {
            __m128i xa = _mm_loadu_si128((const __m128i *)(xq + i));
            __m128i wa = _mm_loadu_si128((const __m128i *)(wv + i));
            __m256i x16 = _mm256_cvtepi8_epi16(xa);
            __m256i w16 = _mm256_cvtepi8_epi16(wa);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x16, w16));
            __m128i xb = _mm_loadu_si128((const __m128i *)(xq + i + 16));
            __m128i wb = _mm_loadu_si128((const __m128i *)(wv + i + 16));
            __m256i x16b = _mm256_cvtepi8_epi16(xb);
            __m256i w16b = _mm256_cvtepi8_epi16(wb);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x16b, w16b));
        }
        dot = hsum_epi32_avx(acc);
        for (; i < N_EMBD; i++) dot += (int)xq[i] * (int)wv[i];
#else
        for (int i = 0; i < N_EMBD; i++) dot += (int)xq[i] * (int)wv[i];
#endif
        j->logits[v] = j->scale_x * g_wte_scale[v] * (float)dot;
    }
    return NULL;
}

static float quantize_x(const float *x, int8_t *xq, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > max_abs) max_abs = a; }
    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)(x[i] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        xq[i] = (int8_t)v;
    }
    return scale;
}

static void lm_head_int8_mt(float *logits, const float *x, int n_threads) {
    static int8_t xq[N_EMBD];
    float scale_x = quantize_x(x, xq, N_EMBD);
    if (n_threads <= 1) {
        LmHeadJob j = {xq, scale_x, logits, 0, VOCAB_SIZE};
        int8_worker(&j);
    } else {
        pthread_t threads[8];
        pthread_attr_t attr;
        LmHeadJob jobs[8];
        int nt = n_threads > 8 ? 8 : n_threads;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 256 * 1024);
        int chunk = (VOCAB_SIZE + nt - 1) / nt;
        for (int t = 0; t < nt; t++) {
            jobs[t] = (LmHeadJob){xq, scale_x, logits, t*chunk, (t+1)*chunk};
            if (jobs[t].v_end > VOCAB_SIZE) jobs[t].v_end = VOCAB_SIZE;
            if (jobs[t].v_end > jobs[t].v_start)
                pthread_create(&threads[t], &attr, int8_worker, &jobs[t]);
        }
        pthread_attr_destroy(&attr);
        for (int t = 0; t < nt; t++)
            if (jobs[t].v_end > jobs[t].v_start) pthread_join(threads[t], NULL);
    }
}

static void lm_head_fp32(float *logits, const float *x) {
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const float *row = g_wte + (size_t)v * N_EMBD;
        v8f acc = V8F_ZERO(); int i = 0;
        for (; i + 8 <= N_EMBD; i += 8)
            acc = V8F_FMADD(V8F_LOAD(row+i), V8F_LOAD(x+i), acc);
        float dot = v8f_hsum(acc);
        for (; i < N_EMBD; i++) dot += row[i] * x[i];
        logits[v] = dot;
    }
}

/* Forward: returns token id, measures time breakdown */
static int forward_timed(int tok, int pos, int use_int8_lm, int n_threads,
                         double *out_layer_ms, double *out_lm_ms) {
    if (tok < 0 || tok >= VOCAB_SIZE) tok = 0;
    memcpy(g_x, g_wte + (size_t)tok * N_EMBD, N_EMBD * sizeof(float));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int l = 0; l < N_LAYER; l++) {
        QwenLayer *L = &g_layers[l];
        qwen_rms_norm(g_ln, g_x, L->norm1_w, N_EMBD);
        matmul_q8(g_q, L->q8_q, L->s_q, L->ws_q, g_ln, L->q_bias, N_EMBD, N_EMBD);
        matmul_q8(g_k, L->q8_k, L->s_k, L->ws_k, g_ln, L->k_bias, N_EMBD, 128);
        matmul_q8(g_v, L->q8_v, L->s_v, L->ws_v, g_ln, L->v_bias, N_EMBD, 128);
        qwen_rope(g_q, g_k, pos, N_HEAD, N_KV_HEAD, HEAD_DIM);
        gqa_attn(g_attn_out, g_q, g_k, g_v, l, pos);
        matmul_q8(g_proj, L->q8_o, L->s_o, L->ws_o, g_attn_out, NULL, N_EMBD, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) g_x[i] += g_proj[i];
        qwen_rms_norm(g_ln, g_x, L->norm2_w, N_EMBD);
        matmul_q8(g_gate, L->q8_gate, L->s_gate, L->ws_gate, g_ln, NULL, N_EMBD, MLP_DIM);
        matmul_q8(g_up,   L->q8_up,   L->s_up,   L->ws_up,   g_ln, NULL, N_EMBD, MLP_DIM);
        for (int i = 0; i < MLP_DIM; i++) g_gate[i] = (g_gate[i] / (1.0f + expf(-g_gate[i]))) * g_up[i];
        matmul_q8(g_mlp_out, L->q8_down, L->s_down, L->ws_down, g_gate, NULL, MLP_DIM, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) g_x[i] += g_mlp_out[i];
    }
    qwen_rms_norm(g_ln, g_x, g_norm_f_w, N_EMBD);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    *out_layer_ms = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (use_int8_lm)
        lm_head_int8_mt(g_logits, g_ln, n_threads);
    else
        lm_head_fp32(g_logits, g_ln);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *out_lm_ms = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;

    int best = 0;
    for (int v = 1; v < VOCAB_SIZE; v++)
        if (g_logits[v] > g_logits[best]) best = v;
    return best;
}

int main(void) {
    printf("=== Qwen2.5-0.5B End-to-End Benchmark ===\n\n");

    g_x = calloc(N_EMBD, 4); g_q = calloc(N_EMBD, 4); g_k = calloc(128, 4);
    g_v = calloc(128, 4); g_attn_out = calloc(N_EMBD, 4); g_proj = calloc(N_EMBD, 4);
    g_gate = calloc(MLP_DIM, 4); g_up = calloc(MLP_DIM, 4); g_mlp_out = calloc(N_EMBD, 4);
    g_ln = calloc(N_EMBD, 4); g_logits = malloc(VOCAB_SIZE * 4);
    kv_cache = calloc(N_LAYER*2, sizeof(float*));
    for (int l = 0; l < N_LAYER*2; l++)
        kv_cache[l] = calloc((size_t)N_CTX * N_KV_HEAD * HEAD_DIM, 4);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    load_weights("prebuilt/qwen_weights.bin");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[*] total load+quant: %.1fs\n\n",
           (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    /* Prefill with a fake token to warm up caches */
    int tok = forward_timed(100, 0, 1, 1, &(double){0}, &(double){0});

    /* Test configurations */
    struct { const char *name; int int8_lm; int threads; } configs[] = {
        {"FP32 LM head, 1T",  0, 1},
        {"Int8 LM head, 1T",  1, 1},
        {"Int8 LM head, 2T",  1, 2},
    };
    int n_configs = sizeof(configs)/sizeof(configs[0]);

    for (int c = 0; c < n_configs; c++) {
        /* Reset KV cache */
        for (int l = 0; l < N_LAYER*2; l++)
            memset(kv_cache[l], 0, (size_t)N_CTX * N_KV_HEAD * HEAD_DIM * 4);

        double total_layer = 0, total_lm = 0;
        /* Prefill: pos 0..3 */
        int pos = 0;
        for (int p = 0; p < 4; p++) {
            double lm, ly;
            tok = forward_timed(tok, pos, configs[c].int8_lm, configs[c].threads, &ly, &lm);
            pos++;
        }
        /* Decode: measure N_DECODE tokens */
        struct timespec dt0, dt1;
        clock_gettime(CLOCK_MONOTONIC, &dt0);
        for (int g = 0; g < N_DECODE; g++) {
            double lm, ly;
            tok = forward_timed(tok, pos, configs[c].int8_lm, configs[c].threads, &ly, &lm);
            pos++;
            total_layer += ly;
            total_lm += lm;
        }
        clock_gettime(CLOCK_MONOTONIC, &dt1);
        double dt = (dt1.tv_sec-dt0.tv_sec)+(dt1.tv_nsec-dt0.tv_nsec)*1e-9;
        double tok_s = N_DECODE / dt;
        printf("  %-25s  %5.1f tok/s  | layers: %.1f ms/tok  LM head: %.1f ms/tok  | total: %.1f ms/tok\n",
               configs[c].name, tok_s,
               total_layer/N_DECODE*1000, total_lm/N_DECODE*1000,
               (total_layer+total_lm)/N_DECODE*1000);
    }

    return 0;
}