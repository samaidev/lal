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
#include <omp.h>
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
#include "runtime/lal_q4_kernel.h"
#include "runtime/lal_q4k_kernel.h"
/* AVX-512 Q4_K kernel: 在某些 CPU (如桌面 Skylake-X) 上更快，但在云 Xeon 上会降频反而更慢。
 * 如需启用: 定义 LAL_FORCE_AVX512 并确保 weights 用 ADJACENT packing 转换。
 * 默认禁用，使用经过验证的 AVX2 路径。
 */
#if defined(LAL_FORCE_AVX512) && defined(__AVX512BW__) && defined(__AVX512F__)
  #define LAL_HAVE_AVX512 1
  #include "runtime/lal_q4k_kernel_avx512.h"
#endif
#include "runtime/lal_sampling.h"
#include "runtime/lal_dequant.h"
#include "runtime/lal_tokenizer.h"
#include "runtime/lal_simd_optim.h"  /* SIMD 优化: RMSNorm/RoPE/Attention */

/* === Layer struct === */
typedef struct {
    float *norm1_w, *norm2_w;
    /* Q8 path (qtype=1) */
    int8_t *q8_q;  float *s_q;   /* [Q_DIM, N_EMBD] = [3584, 3584] */
    int8_t *q8_k;  float *s_k;   /* [KV_DIM, N_EMBD] = [512, 3584] */
    int8_t *q8_v;  float *s_v;   /* [KV_DIM, N_EMBD] = [512, 3584] */
    int8_t *q8_o;  float *s_o;   /* [N_EMBD, Q_DIM] = [3584, 3584] */
    int8_t *q8_gate; float *s_gate; /* [MLP_DIM, N_EMBD] */
    int8_t *q8_up;   float *s_up;   /* [MLP_DIM, N_EMBD] */
    int8_t *q8_down; float *s_down; /* [N_EMBD, MLP_DIM] */
    /* Q4_0 path (qtype=2): packed blocks, no separate scale */
    const uint8_t *q4_q;    /* [Q_DIM, N_EMBD/32, 18] */
    const uint8_t *q4_k;
    const uint8_t *q4_v;
    const uint8_t *q4_o;
    const uint8_t *q4_gate;
    const uint8_t *q4_up;
    const uint8_t *q4_down;
    /* Q8_0 path (qtype=3): packed blocks, inline fp16 scale */
    const uint8_t *q8_0_q;    /* [Q_DIM, N_EMBD/32, 34] */
    const uint8_t *q8_0_k;
    const uint8_t *q8_0_v;
    const uint8_t *q8_0_o;
    const uint8_t *q8_0_gate;
    const uint8_t *q8_0_up;
    const uint8_t *q8_0_down;
    /* Q4_0A path (qtype=4): cache-line ALIGNED 32-byte blocks */
    const uint8_t *q4a_q;     /* [Q_DIM, N_EMBD/32, 32] */
    const uint8_t *q4a_k;
    const uint8_t *q4a_v;
    const uint8_t *q4a_o;
    const uint8_t *q4a_gate;
    const uint8_t *q4a_up;
    const uint8_t *q4a_down;
    /* Q4_K path (qtype=5): 256-elem superblocks, 144 bytes each */
    const uint8_t *q4k_q;     /* [Q_DIM, N_EMBD/256, 144] */
    const uint8_t *q4k_k;
    const uint8_t *q4k_v;
    const uint8_t *q4k_o;
    const uint8_t *q4k_gate;
    const uint8_t *q4k_up;
    const uint8_t *q4k_down;
    int qtype;  /* 1=Q8, 2=Q4_0, 3=Q8_0, 4=Q4_0A, 5=Q4_K */
    float *q_bias, *k_bias, *v_bias;
} Layer;

static Layer g_layers[N_LAYER];

/* === Global state === */
static float *g_wte;          /* [VOCAB, N_EMBD] float (from file, for embedding lookup) */
static float *g_norm_f_w;
static int8_t *g_lm_head_q;   /* [VOCAB, N_EMBD] int8 (lm_head quantized at startup) */
static float  *g_lm_head_s;   /* [VOCAB] per-row scale */
static float *g_x, *g_ln, *g_q, *g_k, *g_v, *g_attn_out, *g_proj;
static float *g_gate, *g_up, *g_gate_up, *g_mlp_out;
static float *g_logits;
static int8_t *g_xq_cache;    /* [N_EMBD] int8 buffer for LM head input quantization */
static float **kv_k, **kv_v;  /* [N_LAYER][N_CTX * KV_DIM] */
static int g_n_threads = 1;
static float g_temperature = 0.8f;
static int g_top_k = 40;
static float g_rep_penalty = 1.1f;
static int g_recent[256], g_n_recent = 0;

/* === Thread pool for parallel matmul (OpenMP) ===
 * Splits each matmul's output dimension across g_n_threads threads.
 * Each thread computes y[start..end) = q8[start..end, :] @ x.
 * OpenMP handles thread creation/sync automatically. */
static inline void parallel_matmul(float *y, const int8_t *q8_T, const float *scale,
                                    const float *x, const float *b,
                                    int in_dim, int out_dim) {
    if (g_n_threads <= 1 || out_dim < 2048) {
        lal_matmul_q8_signtrick(y, q8_T, scale, x, b, in_dim, out_dim);
        return;
    }
    #pragma omp parallel num_threads(g_n_threads)
    {
        int tid = omp_get_thread_num();
        int n   = omp_get_num_threads();
        int chunk = (out_dim + n - 1) / n;
        int start = tid * chunk;
        int end = start + chunk;
        if (end > out_dim) end = out_dim;
        if (start < out_dim) {
            lal_matmul_q8_signtrick(y + start,
                                    q8_T + (size_t)start * in_dim,
                                    scale + start,
                                    x,
                                    b ? b + start : NULL,
                                    in_dim, end - start);
        }
    }
}

/* Parallel Q4_0 matmul: same interface but takes packed Q4 weights.
 * row_stride = (in_dim/32) * 18 bytes per row. */
static inline void parallel_matmul_q4(float *y, const uint8_t *q4_W,
                                       const float *x, const float *b,
                                       int in_dim, int out_dim) {
    if (g_n_threads <= 1 || out_dim < 2048) {
        lal_matmul_q4_0(y, q4_W, x, b, in_dim, out_dim);
        return;
    }
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 18;
    #pragma omp parallel num_threads(g_n_threads)
    {
        int tid = omp_get_thread_num();
        int n   = omp_get_num_threads();
        int chunk = (out_dim + n - 1) / n;
        int start = tid * chunk;
        int end = start + chunk;
        if (end > out_dim) end = out_dim;
        if (start < out_dim) {
            lal_matmul_q4_0(y + start,
                            q4_W + (size_t)start * row_stride,
                            x,
                            b ? b + start : NULL,
                            in_dim, end - start);
        }
    }
}

/* Parallel Q8_0 matmul: row_stride = (in_dim/32) * 34 bytes per row. */
static inline void parallel_matmul_q8_0(float *y, const uint8_t *q8_0_W,
                                         const float *x, const float *b,
                                         int in_dim, int out_dim) {
    if (g_n_threads <= 1 || out_dim < 2048) {
        lal_matmul_q8_0(y, q8_0_W, x, b, in_dim, out_dim);
        return;
    }
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 34;
    #pragma omp parallel num_threads(g_n_threads)
    {
        int tid = omp_get_thread_num();
        int n   = omp_get_num_threads();
        int chunk = (out_dim + n - 1) / n;
        int start = tid * chunk;
        int end = start + chunk;
        if (end > out_dim) end = out_dim;
        if (start < out_dim) {
            lal_matmul_q8_0(y + start,
                            q8_0_W + (size_t)start * row_stride,
                            x,
                            b ? b + start : NULL,
                            in_dim, end - start);
        }
    }
}

/* Parallel Q4_0A matmul: row_stride = (in_dim/32) * 32 bytes per row (aligned!) */
static inline void parallel_matmul_q4_0a(float *y, const uint8_t *q4a_W,
                                          const float *x, const float *b,
                                          int in_dim, int out_dim) {
    if (g_n_threads <= 1 || out_dim < 2048) {
        lal_matmul_q4_0a(y, q4a_W, x, b, in_dim, out_dim);
        return;
    }
    int blocks_per_row = in_dim / 32;
    int row_stride = blocks_per_row * 32;  /* 32-byte aligned blocks */
    #pragma omp parallel num_threads(g_n_threads)
    {
        int tid = omp_get_thread_num();
        int n   = omp_get_num_threads();
        int chunk = (out_dim + n - 1) / n;
        int start = tid * chunk;
        int end = start + chunk;
        if (end > out_dim) end = out_dim;
        if (start < out_dim) {
            lal_matmul_q4_0a(y + start,
                             q4a_W + (size_t)start * row_stride,
                             x,
                             b ? b + start : NULL,
                             in_dim, end - start);
        }
    }
}

/* Parallel Q4_K matmul: row_stride = (in_dim/256) * 144 bytes per row */
static inline void parallel_matmul_q4_k(float *y, const uint8_t *q4k_W,
                                         const float *x, const float *b,
                                         int in_dim, int out_dim) {
#if defined(LAL_HAVE_AVX512)
    #define Q4K_KERNEL lal_matmul_q4_k_avx512
#else
    #define Q4K_KERNEL lal_matmul_q4_k
#endif
    if (g_n_threads <= 1 || out_dim < 2048) {
        Q4K_KERNEL(y, q4k_W, x, b, in_dim, out_dim);
        return;
    }
    int n_super = in_dim / 256;
    int row_stride = n_super * 144;
    #pragma omp parallel num_threads(g_n_threads)
    {
        int tid = omp_get_thread_num();
        int n   = omp_get_num_threads();
        int chunk = (out_dim + n - 1) / n;
        int start = tid * chunk;
        int end = start + chunk;
        if (end > out_dim) end = out_dim;
        if (start < out_dim) {
            Q4K_KERNEL(y + start,
                            q4k_W + (size_t)start * row_stride,
                            x,
                            b ? b + start : NULL,
                            in_dim, end - start);
        }
    }
#undef Q4K_KERNEL
}

/* Dispatch macro: picks Q4_K/Q4_0A/Q8_0/Q4_0/Q8 based on layer->qtype.
 * Args: q8f, q4f, q8_0f, q4af, q4kf, sf */
#define LAYER_MATMUL(y, L, q8f, q4f, q8_0f, q4af, q4kf, sf, x, b, in_dim, out_dim) \
    do { \
        if ((L)->qtype == 5) { \
            parallel_matmul_q4_k((y), (L)->q4k_##q4kf, (x), (b), (in_dim), (out_dim)); \
        } else if ((L)->qtype == 4) { \
            parallel_matmul_q4_0a((y), (L)->q4a_##q4af, (x), (b), (in_dim), (out_dim)); \
        } else if ((L)->qtype == 3) { \
            parallel_matmul_q8_0((y), (L)->q8_0f, (x), (b), (in_dim), (out_dim)); \
        } else if ((L)->qtype == 2) { \
            parallel_matmul_q4((y), (L)->q4f, (x), (b), (in_dim), (out_dim)); \
        } else { \
            parallel_matmul((y), (L)->q8f, (L)->sf, (x), (b), (in_dim), (out_dim)); \
        } \
    } while(0)

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
    /* Hint kernel to use transparent huge pages (2MB) for fewer TLB misses.
     * On a 8GB file this reduces TLB entries from 2M to 4K — huge win for
     * sequential weight reads. */
    madvise(g_mmap_base, g_mmap_size, MADV_HUGEPAGE);
    /* Also hint sequential access pattern for better readahead. */
    madvise(g_mmap_base, g_mmap_size, MADV_SEQUENTIAL);
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

/* Get Q4_0 packed data pointer (no separate scale; it's inside blocks) */
static const uint8_t *get_q4(const char *key) {
    GPQ8Tensor *t = gp_find(key);
    if (!t || t->qtype != 2) { fprintf(stderr, "[!] %s not Q4_0\n", key); exit(1); }
    return (const uint8_t*)t->data;
}

/* Get Q8_0 packed data pointer (inline fp16 scale + int8 per block) */
static const uint8_t *get_q8_0(const char *key) {
    GPQ8Tensor *t = gp_find(key);
    if (!t || t->qtype != 3) { fprintf(stderr, "[!] %s not Q8_0\n", key); exit(1); }
    return (const uint8_t*)t->data;
}

/* Get Q4_0A packed data pointer (cache-line aligned 32-byte blocks) */
static const uint8_t *get_q4_0a(const char *key) {
    GPQ8Tensor *t = gp_find(key);
    if (!t || t->qtype != 4) { fprintf(stderr, "[!] %s not Q4_0A\n", key); exit(1); }
    return (const uint8_t*)t->data;
}

/* Get Q4_K packed data pointer (144-byte superblocks) */
static const uint8_t *get_q4_k(const char *key) {
    GPQ8Tensor *t = gp_find(key);
    if (!t || t->qtype != 5) { fprintf(stderr, "[!] %s not Q4_K\n", key); exit(1); }
    return (const uint8_t*)t->data;
}

/* Get F32 data pointer */
static float *get_f32(const char *key) {
    GPQ8Tensor *t = gp_find(key);
    if (!t || t->qtype != 0) { fprintf(stderr, "[!] %s not F32\n", key); exit(1); }
    return (float*)t->data;
}

/* Get qtype for a tensor (0=F32, 1=Q8, 2=Q4_0) */
static int get_qtype(const char *key) {
    GPQ8Tensor *t = gp_find(key);
    if (!t) { fprintf(stderr, "[!] tensor not found: %s\n", key); exit(1); }
    return t->qtype;
}

/* === RMSNorm === */
static void qwen7b_rms_norm(float *out, const float *x, const float *w, int n) {
    lal_rms_norm_simd(out, x, w, n, RMS_EPS);
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
    lal_rope_apply_simd(q, k, N_HEAD, N_KV_HEAD, HEAD_DIM, pos,
                         g_rope_cos[pos], g_rope_sin[pos]);
}

/* === GQA Attention (SIMD 优化版，4.76x faster) === */
static void gqa_attn(float *out, const float *Q, const float *Kn, const float *Vn,
                     int layer, int pos) {
    lal_gqa_attn_simd(out, Q, Kn, Vn,
                      kv_k[layer], kv_v[layer], pos,
                      N_HEAD, N_KV_HEAD, HEAD_DIM,
                      N_Q_PER_KV, KV_DIM, N_CTX);
}

/* === Fused SwiGLU MLP === */
static void fused_swiglu(const int8_t *q_gate, const float *s_gate,
                         const int8_t *q_up, const float *s_up,
                         const int8_t *q_down, const float *s_down,
                         const uint8_t *q4_gate, const uint8_t *q4_up,
                         const uint8_t *q4_down,
                         const uint8_t *q8_0_gate, const uint8_t *q8_0_up,
                         const uint8_t *q8_0_down,
                         const uint8_t *q4a_gate, const uint8_t *q4a_up,
                         const uint8_t *q4a_down,
                         const uint8_t *q4k_gate, const uint8_t *q4k_up,
                         const uint8_t *q4k_down, int qtype,
                         const float *x, float *out, int in_dim, int hid, int out_dim) {
    static float gate_buf[MLP_DIM], up_buf[MLP_DIM], act_buf[MLP_DIM];
    if (qtype == 5) {
        parallel_matmul_q4_k(gate_buf, q4k_gate, x, NULL, in_dim, hid);
        parallel_matmul_q4_k(up_buf,   q4k_up,   x, NULL, in_dim, hid);
    } else if (qtype == 4) {
        parallel_matmul_q4_0a(gate_buf, q4a_gate, x, NULL, in_dim, hid);
        parallel_matmul_q4_0a(up_buf,   q4a_up,   x, NULL, in_dim, hid);
    } else if (qtype == 3) {
        parallel_matmul_q8_0(gate_buf, q8_0_gate, x, NULL, in_dim, hid);
        parallel_matmul_q8_0(up_buf,   q8_0_up,   x, NULL, in_dim, hid);
    } else if (qtype == 2) {
        parallel_matmul_q4(gate_buf, q4_gate, x, NULL, in_dim, hid);
        parallel_matmul_q4(up_buf,   q4_up,   x, NULL, in_dim, hid);
    } else {
        parallel_matmul(gate_buf, q_gate, s_gate, x, NULL, in_dim, hid);
        parallel_matmul(up_buf,   q_up,   s_up,   x, NULL, in_dim, hid);
    }
    /* SiLU(gate) * up */
    for (int i = 0; i < hid; i++)
        act_buf[i] = (gate_buf[i] / (1.0f + expf(-gate_buf[i]))) * up_buf[i];
    /* down = q_down @ act */
    if (qtype == 5)
        parallel_matmul_q4_k(out, q4k_down, act_buf, NULL, hid, out_dim);
    else if (qtype == 4)
        parallel_matmul_q4_0a(out, q4a_down, act_buf, NULL, hid, out_dim);
    else if (qtype == 3)
        parallel_matmul_q8_0(out, q8_0_down, act_buf, NULL, hid, out_dim);
    else if (qtype == 2)
        parallel_matmul_q4(out, q4_down, act_buf, NULL, hid, out_dim);
    else
        parallel_matmul(out, q_down, s_down, act_buf, NULL, hid, out_dim);
}

/* === Forward pass === */
static int forward(int tok, int pos) {
    /* Embedding lookup (F32 from mmap'd embed_tokens) */
    if (tok < 0 || tok >= VOCAB_SIZE) {
        fprintf(stderr, "[!] forward: token %d out of range [0,%d)\n", tok, VOCAB_SIZE);
        tok = 0;
    }
    memcpy(g_x, g_wte + (size_t)tok * N_EMBD, N_EMBD * sizeof(float));

    for (int l = 0; l < N_LAYER; l++) {
        Layer *L = &g_layers[l];
        /* Pre-attn RMSNorm */
        qwen7b_rms_norm(g_ln, g_x, L->norm1_w, N_EMBD);
        /* Q/K/V projections (Q8 or Q4 or Q8_0 from GPQ8 file) */
        LAYER_MATMUL(g_q, L, q8_q, q4_q, q8_0_q, q, q, s_q, g_ln, L->q_bias, N_EMBD, Q_DIM);
        LAYER_MATMUL(g_k, L, q8_k, q4_k, q8_0_k, k, k, s_k, g_ln, L->k_bias, N_EMBD, KV_DIM);
        LAYER_MATMUL(g_v, L, q8_v, q4_v, q8_0_v, v, v, s_v, g_ln, L->v_bias, N_EMBD, KV_DIM);
        /* RoPE */
        rope_apply(g_q, g_k, pos);
        /* GQA Attention */
        gqa_attn(g_attn_out, g_q, g_k, g_v, l, pos);
        /* O proj + residual */
        LAYER_MATMUL(g_proj, L, q8_o, q4_o, q8_0_o, o, o, s_o, g_attn_out, NULL, Q_DIM, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) g_x[i] += g_proj[i];
        /* Pre-MLP RMSNorm */
        qwen7b_rms_norm(g_ln, g_x, L->norm2_w, N_EMBD);
        /* Fused SwiGLU MLP + residual */
        fused_swiglu(L->q8_gate, L->s_gate, L->q8_up, L->s_up,
                     L->q8_down, L->s_down,
                     L->q4_gate, L->q4_up, L->q4_down,
                     L->q8_0_gate, L->q8_0_up, L->q8_0_down,
                     L->q4a_gate, L->q4a_up, L->q4a_down,
                     L->q4k_gate, L->q4k_up, L->q4k_down, L->qtype,
                     g_ln, g_mlp_out, N_EMBD, MLP_DIM, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) g_x[i] += g_mlp_out[i];
    }

    /* Final RMSNorm */
    qwen7b_rms_norm(g_ln, g_x, g_norm_f_w, N_EMBD);

    /* LM head: int8 dot product (logits = lm_head_q @ x)
     * lm_head was quantized to int8 at startup (per-row scale).
     * Uses sign-trick kernel for 4x less memory bandwidth than F32.
     * Parallelized across threads — lm_head is 152064 rows, the biggest
     * single matmul, so threading gives ~2x on 2 vCPUs. */
    float scale_x = lal_quantize_x_int8(g_ln, g_xq_cache, N_EMBD);
    #pragma omp parallel num_threads(g_n_threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int v_per = (VOCAB_SIZE + nthreads - 1) / nthreads;
        int v_start = tid * v_per;
        int v_end = v_start + v_per;
        if (v_end > VOCAB_SIZE) v_end = VOCAB_SIZE;
        if (v_start < VOCAB_SIZE)
            lal_lm_head_int8_range(g_logits, g_xq_cache, scale_x,
                                   g_lm_head_q, g_lm_head_s,
                                   v_start, v_end, N_EMBD);
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

/* Qwen2.5-Instruct chat template special tokens */
#define TOK_IM_START  151644
#define TOK_IM_END    151643

/* Byte-level BPE fallback: encode one byte as its GPT-2 unicode char,
 * then look up the single-char vocab string.
 * Returns token id or -1 if not found. */
static int encode_byte(unsigned char b) {
    unsigned int cp = lal_bpe_cp_for_byte(b);
    char buf[5] = {0};
    int n = lal_utf8_encode(cp, buf);
    buf[n] = 0;
    return tok_find(buf);
}

/* Simple text encoder: word-level + space-prefix; byte-level fallback.
 * No BPE merges (would need merges.txt). For words in the vocab as single
 * tokens (common for short English words), this is exact. For unknown
 * words, falls back to per-byte encoding (one token per byte).
 * Newlines are emitted as Ċ tokens; spaces are absorbed into the next
 * word's Ġ-prefix (per GPT-2 BPE convention). */
static int *encode_text(const char *text, int *n_out) {
    int *ids = malloc((strlen(text) + 4) * 4 * sizeof(int));
    int n = 0;
    char word[512];
    int wlen = 0;
    int first = 1;  /* 1 = next word has no space prefix (start of line / after \n) */
    for (const char *p = text; ; p++) {
        if (*p == ' ' || *p == '\0' || *p == '\n') {
            if (wlen > 0) {
                word[wlen] = 0;
                char prefixed[513];
                snprintf(prefixed, sizeof(prefixed), "%s%s", first ? "" : " ", word);
                int id = tok_find(prefixed);
                if (id < 0) id = tok_find(word);  /* try without prefix */
                if (id >= 0) {
                    ids[n++] = id;
                } else {
                    /* Byte-level fallback: encode each byte as its BPE char */
                    for (int i = 0; i < wlen; i++) {
                        int bid = encode_byte((unsigned char)word[i]);
                        if (bid >= 0) ids[n++] = bid;
                    }
                }
                wlen = 0;
                first = 0;
            }
            if (*p == ' ') {
                /* Space absorbed into next word's prefix; nothing to emit */
            } else if (*p == '\n') {
                int sid = tok_find("\xC4\x8A");  /* Ċ = newline token */
                if (sid >= 0) ids[n++] = sid;
                first = 1;  /* new line resets word boundary */
            } else if (*p == '\0') break;
        } else {
            if (wlen < 511) word[wlen++] = *p;
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
    if (g_n_threads > 1)
        printf("[*] OpenMP: %d threads\n", g_n_threads);

    /* Load GPQ8 weights */
    load_gpq8(weights);

    /* Wire up layer pointers (Q8 data is already in GPQ8 tensors) */
    g_norm_f_w = get_f32("model.norm.weight");
    g_wte = get_f32("model.embed_tokens.weight");
    char key[256];
    /* Detect qtype from first weight tensor */
    sprintf(key, "model.layers.0.self_attn.q_proj.weight");
    int layer_qtype = get_qtype(key);
    printf("[*] layer weight qtype: %s\n",
           layer_qtype == 5 ? "Q4_K (144 bytes/256 elems, superblock + sub-scales)" :
           layer_qtype == 4 ? "Q4_0A (32 bytes/32 elems, CACHE-LINE ALIGNED)" :
           layer_qtype == 3 ? "Q8_0 (34 bytes/32 elems, inline fp16 scale)" :
           layer_qtype == 2 ? "Q4_0 (18 bytes/32 elems)" :
           layer_qtype == 1 ? "Q8 (32 bytes/32 elems + scale)" : "F32");
    for (int l = 0; l < N_LAYER; l++) {
        Layer *L = &g_layers[l];
        L->qtype = layer_qtype;
        sprintf(key, "model.layers.%d.input_layernorm.weight", l); L->norm1_w = get_f32(key);
        sprintf(key, "model.layers.%d.post_attention_layernorm.weight", l); L->norm2_w = get_f32(key);
        if (layer_qtype == 5) {
            sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l); L->q4k_q = get_q4_k(key);
            sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l); L->q4k_k = get_q4_k(key);
            sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l); L->q4k_v = get_q4_k(key);
            sprintf(key, "model.layers.%d.self_attn.o_proj.weight", l); L->q4k_o = get_q4_k(key);
            sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l); L->q4k_gate = get_q4_k(key);
            sprintf(key, "model.layers.%d.mlp.up_proj.weight", l);   L->q4k_up = get_q4_k(key);
            sprintf(key, "model.layers.%d.mlp.down_proj.weight", l); L->q4k_down = get_q4_k(key);
        } else if (layer_qtype == 4) {
            sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l); L->q4a_q = get_q4_0a(key);
            sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l); L->q4a_k = get_q4_0a(key);
            sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l); L->q4a_v = get_q4_0a(key);
            sprintf(key, "model.layers.%d.self_attn.o_proj.weight", l); L->q4a_o = get_q4_0a(key);
            sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l); L->q4a_gate = get_q4_0a(key);
            sprintf(key, "model.layers.%d.mlp.up_proj.weight", l);   L->q4a_up = get_q4_0a(key);
            sprintf(key, "model.layers.%d.mlp.down_proj.weight", l); L->q4a_down = get_q4_0a(key);
        } else if (layer_qtype == 3) {
            sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l); L->q8_0_q = get_q8_0(key);
            sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l); L->q8_0_k = get_q8_0(key);
            sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l); L->q8_0_v = get_q8_0(key);
            sprintf(key, "model.layers.%d.self_attn.o_proj.weight", l); L->q8_0_o = get_q8_0(key);
            sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l); L->q8_0_gate = get_q8_0(key);
            sprintf(key, "model.layers.%d.mlp.up_proj.weight", l);   L->q8_0_up = get_q8_0(key);
            sprintf(key, "model.layers.%d.mlp.down_proj.weight", l); L->q8_0_down = get_q8_0(key);
        } else if (layer_qtype == 2) {
            sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l); L->q4_q = get_q4(key);
            sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l); L->q4_k = get_q4(key);
            sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l); L->q4_v = get_q4(key);
            sprintf(key, "model.layers.%d.self_attn.o_proj.weight", l); L->q4_o = get_q4(key);
            sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l); L->q4_gate = get_q4(key);
            sprintf(key, "model.layers.%d.mlp.up_proj.weight", l);   L->q4_up = get_q4(key);
            sprintf(key, "model.layers.%d.mlp.down_proj.weight", l); L->q4_down = get_q4(key);
        } else {
            sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l); get_q8(key, &L->q8_q, &L->s_q);
            sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l); get_q8(key, &L->q8_k, &L->s_k);
            sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l); get_q8(key, &L->q8_v, &L->s_v);
            sprintf(key, "model.layers.%d.self_attn.o_proj.weight", l); get_q8(key, &L->q8_o, &L->s_o);
            sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l); get_q8(key, &L->q8_gate, &L->s_gate);
            sprintf(key, "model.layers.%d.mlp.up_proj.weight", l); get_q8(key, &L->q8_up, &L->s_up);
            sprintf(key, "model.layers.%d.mlp.down_proj.weight", l); get_q8(key, &L->q8_down, &L->s_down);
        }
        sprintf(key, "model.layers.%d.self_attn.q_proj.bias", l); L->q_bias = get_f32(key);
        sprintf(key, "model.layers.%d.self_attn.k_proj.bias", l); L->k_bias = get_f32(key);
        sprintf(key, "model.layers.%d.self_attn.v_proj.bias", l); L->v_bias = get_f32(key);
    }

/* F32 LM head (no int8 quantization) */
    printf("[*] using int8 LM head (untied lm_head.weight)\n"); fflush(stdout);

    /* Quantize lm_head.weight (F32 in file) to int8 for fast LM head dot product.
     * Qwen2.5-Instruct has UNTIED embeddings, so lm_head is a separate matrix
     * (NOT embed_tokens). Using embed_tokens would produce garbage logits. */
    float *lm_head_f = get_f32("lm_head.weight");
    g_lm_head_q = memalign(32, (size_t)VOCAB_SIZE * N_EMBD);
    g_lm_head_s = memalign(32, VOCAB_SIZE * sizeof(float));
    printf("[*] quantizing lm_head to int8 (%.0f MB -> %.0f MB)...\n",
           (double)VOCAB_SIZE * N_EMBD * 4 / 1048576,
           (double)VOCAB_SIZE * N_EMBD / 1048576); fflush(stdout);
    lal_quantize_q8_per_row(lm_head_f, g_lm_head_q, g_lm_head_s, N_EMBD, VOCAB_SIZE);
    /* Note: lm_head_f stays mmap'd (read-only, no extra RSS) */
    g_xq_cache = memalign(32, N_EMBD);

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

    /* Build chat-templated prompt token sequence:
     *   <|im_start|>user \n {prompt} <|im_end|> \n <|im_start|>assistant \n
     * Special tokens are inserted by ID; text parts via encode_text(). */
    int *pids = malloc((strlen(prompt) + 64) * 4 * sizeof(int));
    int n_prompt = 0;
    int newline_tok = tok_find("\xC4\x8A");  /* Ċ = newline */
    int nu, np, na;
    int *uids, *ppids, *aids;

    pids[n_prompt++] = TOK_IM_START;
    uids = encode_text("user", &nu);
    for (int i = 0; i < nu; i++) pids[n_prompt++] = uids[i];
    free(uids);
    if (newline_tok >= 0) pids[n_prompt++] = newline_tok;

    ppids = encode_text(prompt, &np);
    for (int i = 0; i < np; i++) pids[n_prompt++] = ppids[i];
    free(ppids);

    pids[n_prompt++] = TOK_IM_END;
    if (newline_tok >= 0) pids[n_prompt++] = newline_tok;

    pids[n_prompt++] = TOK_IM_START;
    aids = encode_text("assistant", &na);
    for (int i = 0; i < na; i++) pids[n_prompt++] = aids[i];
    free(aids);
    if (newline_tok >= 0) pids[n_prompt++] = newline_tok;

    printf("[*] prompt: %d tokens (incl. chat template)\n", n_prompt);

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
