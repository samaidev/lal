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

/* LM head int8 dynamic quantization (--lm-head-int8).
 * The LM head (logits = wte @ x, 50257×768) is memory-bound: each token
 * reads 154 MB of float weights. We quantize wte row-by-row to int8 +
 * per-row scale at load time (one-time cost), then the LM head reads
 * only 38.6 MB of int8 weights per token → ~4x less memory bandwidth.
 *
 * g_wte stays float (used for the embedding lookup, which needs accuracy
 * and is just one row, not a bottleneck). g_wte_q is the int8 copy used
 * ONLY by the LM head. Extra resident memory: ~39 MB (worth it for 4x
 * bandwidth reduction on the hottest path).
 *
 * Orthogonal to --prune-vocab (drops rows) and --turboquant (quantizes
 * KV cache, a different tensor) — all three can stack. */
static int8_t *g_wte_q;          /* [vocab, n_embd] int8, quantized at load */
static float  *g_wte_scale;      /* [vocab] per-row scale = max|row|/127 */
static int     g_use_lm_head_int8 = 0;  /* set by --lm-head-int8 */

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
    /* Logic-guided (GB2L2): per-output 0=CORE(float), 1=BINARY(sign+alpha), 2=PRUNE(zero).
     * When logic_mask is non-NULL, wbits/alpha are stored in FULL [out_dim] layout
     * (CORE/PRUNE rows zeroed) so bin_matmul can index by output j directly,
     * matching the runtime's logic forward. w_core holds the CORE float rows. */
    uint8_t  *logic_mask;  /* [out_dim], NULL for plain GB2L (all-BINARY) */
    float    *w_core;      /* [n_core * in_dim] — CORE float rows */
    int       n_core;      /* number of CORE outputs (rows in w_core) */
    /* Int8 speedup: sign(w) packed as int8 (+/-1) for int8 quantized matmul.
     * Lazily allocated in load_binary_weights. NULL until loaded. */
    int8_t   *w_sign_int8; /* [out_dim * in_dim], int8 +/-1 */
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
static int g_use_bwn = 0;       /* --bwn: BWN mode (float x @ sign(w), training-consistent) */
static int g_use_int8 = 0;      /* --int8: BWN with int8 activation quantization (2-9x speedup) */
static int g_mixed_int8_layers = 0;  /* --mixed-int8 N: first N layers int8, rest BWN. 0=all-int8 */
static int g_current_layer = 0;      /* set by forward loop, read by bin_matmul */

/* Mixed-precision mode (--mixed-precision, implies --binary): keep the first
 * and last transformer layers in float, binarize only the middle N_LAYER-2.
 * This is the standard XNOR-Net quality fix — the first layer preserves the
 * input distribution (binarizing the embedding+pos sum loses too much), and
 * the last layer preserves the final discriminative features feeding the LM
 * head. Memory cost: ~28 MB float per kept layer × 2 = ~56 MB on top of the
 * 13 MB binary + 38 MB int8 LM head cache. Peak during load is higher because
 * the full float file must be read to extract 2 layers, then freed.
 *
 * Orthogonal to STE (LAL-Bot's domain): STE improves the binary layers;
 * mixed-precision keeps 2 layers fully accurate. They compose.
 *
 * g_layers[0] and g_layers[N_LAYER-1] are populated only in this mode; the
 * forward loop picks the float branch for those two indices and the binary
 * branch for the rest. */
static int g_mixed_precision = 0;

/* Ternary activation mode (--ternary-act): in binary layers, binarize_input
 * becomes ternarize_input — small-magnitude activations (|x| < threshold) are
 * zeroed instead of forced to ±1. Bi-Real Net showed this halves activation
 * quantization error with negligible cost. Pure activation-domain change:
 * weights stay sign(w), so this composes with STE (weight tuning) and
 * mixed-precision (whole-layer float). Threshold = g_ternary_threshold *
 * mean(|x|), default 0.5 (zero ~30% of activations near zero). */
static int   g_ternary_act       = 0;
static float g_ternary_threshold = 0.5f;

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

/* Ternary activation packing (Bi-Real Net style, {-1,0,+1}):
 * Elements with |x| < threshold are zeroed (contribute nothing); survivors
 * keep their sign. We pack TWO bitmasks per input:
 *   x_sign[wi] — survivor sign bit (1 if x>0, 0 if x<0); zeroed elements are 0
 *   x_mask[wi] — survivor mask bit (1 if |x|>=threshold, 0 otherwise)
 * The dot product over survivors = 2*matches - n_survivors, where
 *   matches    = popcount(XNOR(x_sign, wbits) AND x_mask)
 *   n_survivors= popcount(x_mask)  (same for all output j)
 * This is a pure activation-domain change — weights stay binary (sign(w)),
 * so it composes with STE (which tunes the binary weights) and with
 * mixed-precision (which keeps whole layers in float). */
static void ternarize_input(const float *x, uint64_t *x_sign, uint64_t *x_mask,
                            int in_dim, int n_words, float threshold) {
    for (int wi = 0; wi < n_words; wi++) {
        uint64_t sign_w = 0, mask_w = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in_dim) {
                float xi = x[idx];
                if (fabsf(xi) >= threshold) {
                    mask_w |= (1ULL << bi);          /* survivor */
                    if (xi > 0.0f) sign_w |= (1ULL << bi);  /* sign + */
                }
            }
        }
        x_sign[wi] = sign_w;
        x_mask[wi] = mask_w;
    }
}

/* Binary forward (XNOR-Net style):
 *   y[j] = mean(|x|) * alpha[j] * (2*popcount(XNOR(x_bits, wbits[j])) - in_dim) + bias[j]
 *
 * The mean(|x|) factor is CRITICAL — without it, the dot product magnitude
 * is wrong by a factor of mean(|x|), which varies per input. This was the
 * root cause of poor output quality: the original code omitted mean(|x|),
 * so every layer's output had wrong scale, and after 12 layers the signal
 * was completely buried.
 *
 * XNOR-Net formula (Rastegari et al. 2016):
 *   dot(x, w) ≈ mean(|x|) * mean(|w|) * sum(sign(x) * sign(w))
 *             = mean(|x|) * alpha  * (2*pc - N)
 * where pc = popcount(XNOR(sign(x), sign(w)))
 */
static void bin_matmul(const float *x, const SrvBinLayer *bl, float *y) {
    int n_words = bl->n_words;

    /* Logic-guided path (GB2L2): per-output CORE / BINARY / PRUNE dispatch,
     * matching the runtime's bin_forward logic branch. CORE = float dot with
     * w_core; BINARY = sign(w)·alpha·K (K = mean|x|, BWN-consistent); PRUNE = 0. */
    if (bl->logic_mask) {
        int in = bl->in_dim, nw = bl->n_words;
        float abs_sum = 0.0f;
        for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
        float K = abs_sum / in;

        static float lg_sign_lut[256][8];
        static int lut_init = 0;
        if (!lut_init) {
            for (int b = 0; b < 256; b++)
                for (int i = 0; i < 8; i++)
                    lg_sign_lut[b][i] = (b >> i) & 1 ? 1.0f : -1.0f;
            lut_init = 1;
        }

        int core_idx = 0;
        for (int j = 0; j < bl->out_dim; j++) {
            switch (bl->logic_mask[j]) {
            case 0: { /* CORE: float dot with w_core */
                const float *wc = &bl->w_core[(size_t)core_idx * in];
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
            case 1: { /* BINARY: sign(w) dot x, scaled by alpha * K */
                const uint64_t *wb = bl->wbits + (size_t)j * nw;
                float s = 0.0f;
                for (int wi = 0; wi < nw; wi++) {
                    uint64_t w = wb[wi];
                    int base = wi * 64;
                    for (int bi = 0; bi < 8; bi++) {
                        int idx = base + bi * 8;
                        uint8_t byte = (uint8_t)((w >> (bi * 8)) & 0xFF);
                        const float *sw = lg_sign_lut[byte];
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

    /* Compute mean(|x|) over the active elements — the input scale factor.
     * In ternary mode, "active" means survivors (|x| >= threshold); the
     * mean over survivors keeps the dot-product magnitude correct when
     * some elements are zeroed. */
    float abs_sum = 0.0f;
    for (int i = 0; i < bl->in_dim; i++) abs_sum += fabsf(x[i]);
    float mean_abs = abs_sum / bl->in_dim;

    if (g_ternary_act) {
        /* Ternary activation path: pack sign + survivor mask, then for each
         * output j compute matches over survivors only.
         *   y[j] = x_scale * alpha[j] * (2*matches_j - n_survivors) + bias[j]
         * where x_scale = mean|x| over survivors (so zeroed elements don't
         * drag the scale down). */
        uint64_t *x_sign, *x_mask;
        uint64_t sign_buf[16], mask_buf[16];
        if (n_words <= 16) {
            x_sign = sign_buf; x_mask = mask_buf;
        } else {
            x_sign = malloc(n_words * sizeof(uint64_t));
            x_mask = malloc(n_words * sizeof(uint64_t));
        }
        float threshold = g_ternary_threshold * mean_abs;
        ternarize_input(x, x_sign, x_mask, bl->in_dim, n_words, threshold);

        /* n_survivors and survivor |x| sum (both independent of output j). */
        int n_survivors = 0;
        float surv_abs_sum = 0.0f;
        for (int wi = 0; wi < n_words; wi++) {
            n_survivors += __builtin_popcountll(x_mask[wi]);
        }
        for (int i = 0; i < bl->in_dim; i++) {
            uint64_t bit = 1ULL << (i & 63);
            int wi = i >> 6;
            if (x_mask[wi] & bit) surv_abs_sum += fabsf(x[i]);
        }
        float x_scale = (n_survivors > 0) ? (surv_abs_sum / n_survivors) : 0.0f;
        int n_active = n_survivors;  /* replaces bl->in_dim in the (2*pc - N) term */

        for (int j = 0; j < bl->out_dim; j++) {
            const uint64_t *wb = bl->wbits + (size_t)j * n_words;
            int matches = 0;
            for (int wi = 0; wi < n_words; wi++) {
                /* XNOR(sign, w) AND mask = bits where survivor AND sign matches. */
                uint64_t xnor = ~(x_sign[wi] ^ wb[wi]);
                matches += __builtin_popcountll(xnor & x_mask[wi]);
            }
            y[j] = x_scale * bl->alpha[j] * (float)(2 * matches - n_active) + bl->bias[j];
        }
        if (n_words > 16) { free(x_sign); free(x_mask); }
        return;
    }

    /* Binary (±1) path — original XNOR-Net. */
    uint64_t *xb;
    /* BWN mode (default): float x @ sign(w) — training-consistent, no activation binarization.
     * This is the correct BWN (Binary Weight Network) computation:
     *   y[j] = sum_i x[i] * sign(w[j,i]) * alpha[j] + bias[j]
     * We pack sign(w) as bits and use popcount to count matching signs,
     * but x stays float. The dot product is:
     *   dot = sum_i x[i] * sign(w[i]) = sum_{x>0,w>0} x[i] - sum_{x>0,w<0} x[i] + ...
     *
     * Simplification: we can't use XNOR+popcount with float x (that requires
     * binarized x). Instead, use the w_float weights if available, or unpack
     * sign bits to {-1,+1} and do a float dot product.
     *
     * For maximum speed with correct BWN, we use the packed wbits to get
     * sign(w) via bit extraction, then dot with float x. But that's slow.
     * The fastest correct approach: store sign(w) as float {-1,+1} array
     * and use SIMD float dot product. We do this when --bwn flag is set. */
    /* Int8 quantized BWN: ~9x faster than float BWN.
     * Quantize x to int8, matmul with sign(w) int8, then rescale.
     * Only for non-logic-guided layers (w_sign_int8 is NULL for logic_mask).
     * Mixed precision: when g_mixed_int8_layers > 0, only first N layers use
     * int8; later layers fall through to BWN (float) for precision. */
    if (g_use_int8 && bl->w_sign_int8 &&
        (g_mixed_int8_layers == 0 || g_current_layer < g_mixed_int8_layers)) {
        /* Quantize x: symmetric per-tensor scale */
        float max_abs = 0;
        for (int i = 0; i < bl->in_dim; i++) {
            float a = fabsf(x[i]);
            if (a > max_abs) max_abs = a;
        }
        float x_scale = max_abs / 127.0f;
        if (x_scale < 1e-8f) x_scale = 1e-8f;
        float inv_scale = 1.0f / x_scale;
        int8_t x_q[4096];  /* max in_dim, stack alloc */
        for (int i = 0; i < bl->in_dim; i++) {
            int v = (int)lroundf(x[i] * inv_scale);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            x_q[i] = (int8_t)v;
        }
        /* K-norm: mean(|x|) for input magnitude preservation */
        float abs_sum = 0;
        for (int i = 0; i < bl->in_dim; i++) abs_sum += fabsf(x[i]);
        float K_mean = abs_sum / bl->in_dim;

        /* Int8 matmul: y[j] = sum(w_sign[i] * x_q[i]) * x_scale * alpha * K + bias
         * w_sign is +/-1, so w_sign * x_q = conditional add/sub */
        for (int j = 0; j < bl->out_dim; j++) {
            const int8_t *ws = bl->w_sign_int8 + (size_t)j * bl->in_dim;
            int32_t acc = 0;
            for (int i = 0; i + 7 < bl->in_dim; i += 8) {
                acc += ws[i+0]*x_q[i+0] + ws[i+1]*x_q[i+1] + ws[i+2]*x_q[i+2] + ws[i+3]*x_q[i+3];
                acc += ws[i+4]*x_q[i+4] + ws[i+5]*x_q[i+5] + ws[i+6]*x_q[i+6] + ws[i+7]*x_q[i+7];
            }
            for (int i = (bl->in_dim/8)*8; i < bl->in_dim; i++) acc += ws[i] * x_q[i];
            /* K-norm: preserve input magnitude (XNOR-Net input scaling).
             * K = mean(|x|), computed once before j loop. */
            y[j] = (float)acc * x_scale * bl->alpha[j] * K_mean + bl->bias[j];
        }
        return;
    }

    if (g_use_bwn) {
        /* BWN: float dot product with sign(w) unpacked from bits.
         * Uses a pre-computed lookup table: byte → 8 floats of ±1.
         * Table is 256×8×4B = 8KB (fits L1). Each iteration:
         *   1. Extract byte from wbits
         *   2. Load 8 sign floats from table
         *   3. Dot with 8 x floats (SIMD auto-vectorizable)
         * This is ~8x faster than scalar bit-by-bit extraction. */
        static float sign_lut[256][8];
        static int lut_init = 0;
        if (!lut_init) {
            for (int b = 0; b < 256; b++)
                for (int i = 0; i < 8; i++)
                    sign_lut[b][i] = (b >> i) & 1 ? 1.0f : -1.0f;
            lut_init = 1;
        }

        for (int j = 0; j < bl->out_dim; j++) {
            const uint64_t *wb = bl->wbits + (size_t)j * n_words;
            float dot = 0.0f;
            for (int wi = 0; wi < n_words; wi++) {
                uint64_t w = wb[wi];
                int base = wi * 64;
                for (int bi = 0; bi < 8; bi++) {
                    int idx = base + bi * 8;
                    uint8_t byte = (w >> (bi * 8)) & 0xFF;
                    const float *sw = sign_lut[byte];
                    if (idx + 7 < bl->in_dim) {
                        /* 8x unrolled dot product — auto-vectorizes to SIMD */
                        dot += x[idx+0] * sw[0];
                        dot += x[idx+1] * sw[1];
                        dot += x[idx+2] * sw[2];
                        dot += x[idx+3] * sw[3];
                        dot += x[idx+4] * sw[4];
                        dot += x[idx+5] * sw[5];
                        dot += x[idx+6] * sw[6];
                        dot += x[idx+7] * sw[7];
                    } else {
                        for (int k = 0; k < 8; k++) {
                            int i = idx + k;
                            if (i < bl->in_dim) dot += x[i] * sw[k];
                        }
                    }
                }
            }
            y[j] = dot * bl->alpha[j] + bl->bias[j];
        }
        return;
    }

    /* BNN mode (legacy): binarize x to sign(x), use XNOR+popcount.
     * Faster (64x via popcount) but training-inconsistent. */
    uint64_t x_bits[16];
    if (n_words <= 16) {
        xb = x_bits;
    } else {
        xb = malloc(n_words * sizeof(uint64_t));
    }
    binarize_input(x, xb, bl->in_dim, n_words);

    float x_scale = mean_abs;
    for (int j = 0; j < bl->out_dim; j++) {
        const uint64_t *wb = bl->wbits + (size_t)j * n_words;
        int pc = 0;
        for (int wi = 0; wi < n_words; wi++) {
            pc += __builtin_popcountll(~(xb[wi] ^ wb[wi]));
        }
        y[j] = x_scale * bl->alpha[j] * (float)(2 * pc - bl->in_dim) + bl->bias[j];
    }
    if (n_words > 16) free(xb);
}

/* Load GB2L / GB2L2 binary weight file directly (no float weights in memory!).
 * GB2L  = all-binary (v1). GB2L2 = logic-guided (CORE/BINARY/PRUNE per output). */
static int load_binary_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[!] cannot open %s\n", path); return -1; }

    /* Magic: "GB2L" (v1, 4B) or "GB2L2" (v2 logic-guided, 5B). Read 4, then
     * peek the 5th byte: if '2' it's v2; otherwise seek back one (v1 header). */
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GB2L", 4) != 0) {
        fprintf(stderr, "[!] bad magic in %s\n", path); fclose(f); return -1;
    }
    int is_logic = 0;
    char c5;
    if (fread(&c5, 1, 1, f) == 1) {
        if (c5 == '2') is_logic = 1;
        else fseek(f, -1, SEEK_CUR);  /* v1: the byte belongs to the header */
    }
    fprintf(stderr, "[*] binary weights: %s format\n", is_logic ? "GB2L2 (logic-guided)" : "GB2L (all-binary)");

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
            bl->logic_mask = NULL;
            bl->w_core = NULL;
            bl->n_core = 0;

            if (is_logic) {
                /* GB2L2: header 4th field = n_core. Read logic_mask, then
                 * compacted BINARY rows (expand to full [out_dim] layout),
                 * compacted alpha, full bias, then CORE float rows. */
                int n_core = (int)mhdr[3];
                bl->n_core = n_core;
                bl->logic_mask = malloc(bl->out_dim);
                if (fread(bl->logic_mask, 1, bl->out_dim, f) != (size_t)bl->out_dim)
                { fclose(f); return -1; }
                int n_binary = 0;
                for (int j = 0; j < bl->out_dim; j++)
                    if (bl->logic_mask[j] == 1) n_binary++;

                /* wbits: expand compacted [n_binary × n_words] → full [out_dim × n_words] */
                bl->wbits = calloc((size_t)bl->out_dim * bl->n_words, sizeof(uint64_t));
                for (int j = 0, bi = 0; j < bl->out_dim; j++) {
                    if (bl->logic_mask[j] == 1) {
                        if (fread(bl->wbits + (size_t)j * bl->n_words, sizeof(uint64_t),
                                  bl->n_words, f) != (size_t)bl->n_words)
                        { fclose(f); return -1; }
                        bi++;
                    }
                }
                /* alpha: compacted [n_binary] → full [out_dim] (non-binary rows stay 0) */
                bl->alpha = calloc(bl->out_dim, sizeof(float));
                for (int j = 0; j < bl->out_dim; j++) {
                    if (bl->logic_mask[j] == 1) {
                        if (fread(&bl->alpha[j], sizeof(float), 1, f) != 1)
                        { fclose(f); return -1; }
                    }
                }
                /* bias: full [out_dim] (PRUNE bias already 0 in the file) */
                bl->bias = malloc(bl->out_dim * sizeof(float));
                if (fread(bl->bias, sizeof(float), bl->out_dim, f) != (size_t)bl->out_dim)
                { fclose(f); return -1; }
                /* w_core: [n_core × in_dim] float */
                bl->w_core = malloc((size_t)n_core * bl->in_dim * sizeof(float));
                if (fread(bl->w_core, sizeof(float), (size_t)n_core * bl->in_dim, f)
                    != (size_t)n_core * bl->in_dim)
                { fclose(f); return -1; }
            } else {
                /* GB2L v1: all-binary. wbits [out_dim × n_words], alpha [out_dim], bias [out_dim] */
                bl->wbits = malloc((size_t)bl->out_dim * bl->n_words * sizeof(uint64_t));
                fread(bl->wbits, sizeof(uint64_t),
                      (size_t)bl->out_dim * bl->n_words, f);
                bl->alpha = malloc(bl->out_dim * sizeof(float));
                bl->bias  = malloc(bl->out_dim * sizeof(float));
                fread(bl->alpha, sizeof(float), bl->out_dim, f);
                fread(bl->bias,  sizeof(float), bl->out_dim, f);
            }
        }
    }

    fclose(f);
    g_binary_mode = 1;

    /* Int8 speedup: pre-pack sign(w) as int8 for all BINARY layers.
     * Only for non-logic-guided (GB2L all-binary) — logic_mask handled separately. */
    for (int l = 0; l < N_LAYER; l++) {
        BinGPT2Layer *L = &g_bin_layers[l];
        SrvBinLayer *mats[] = {&L->c_attn, &L->c_proj, &L->mlp_fc, &L->mlp_proj};
        for (int mi = 0; mi < 4; mi++) {
            SrvBinLayer *bl = mats[mi];
            if (bl->logic_mask) continue;
            size_t n = (size_t)bl->out_dim * bl->in_dim;
            bl->w_sign_int8 = malloc(n);
            for (int j = 0; j < bl->out_dim; j++) {
                const uint64_t *wb = bl->wbits + (size_t)j * bl->n_words;
                int8_t *ws = bl->w_sign_int8 + (size_t)j * bl->in_dim;
                for (int wi = 0; wi < bl->n_words; wi++) {
                    uint64_t w = wb[wi];
                    int base = wi * 64;
                    for (int bi = 0; bi < 64; bi++) {
                        int i = base + bi;
                        if (i >= bl->in_dim) break;
                        ws[i] = (w >> bi) & 1 ? 1 : -1;
                    }
                }
            }
        }
    }
    return 0;
}

/* Load float weights for a SUBSET of layers (mixed-precision mode).
 * Reads the full GPW2 float file, copies the requested layers' tensors into
 * freshly malloc'd buffers pointed to by g_layers[l], then frees the temp
 * tensor array (so resident memory drops back to just the kept layers).
 * keep_layers[] holds layer indices; n_keep is its length.
 * Returns 0 on success, -1 on failure. */
static int load_float_layers_subset(const char *path, const int *keep_layers, int n_keep) {
    int n_t = 0;
    Tensor *t = tensor_load_all(path, &n_t);
    if (!t) { fprintf(stderr, "[!] mixed-precision: cannot load float weights from %s\n", path); return -1; }

    char key[128];
    for (int ki = 0; ki < n_keep; ki++) {
        int l = keep_layers[ki];
        GPT2Layer *L = &g_layers[l];
        /* 12 tensors per layer (GPT-2 Conv1D naming: c_fc=up-proj, c_proj=down-proj).
         * Each entry: key, destination ptr, element count. We dup the data
         * because tensor_free_all() releases the source arena. */
        struct { const char *k; float **dst; size_t n; } ents[12];
        sprintf(key, "h.%d.ln_1.weight", l);         ents[0].k = strdup(key);  ents[0].dst = &L->ln1_w;      ents[0].n = N_EMBD;
        sprintf(key, "h.%d.ln_1.bias", l);           ents[1].k = strdup(key);  ents[1].dst = &L->ln1_b;      ents[1].n = N_EMBD;
        sprintf(key, "h.%d.attn.c_attn.weight", l);  ents[2].k = strdup(key);  ents[2].dst = &L->c_attn_w;   ents[2].n = (size_t)N_EMBD * 3 * N_EMBD;
        sprintf(key, "h.%d.attn.c_attn.bias", l);    ents[3].k = strdup(key);  ents[3].dst = &L->c_attn_b;   ents[3].n = 3 * N_EMBD;
        sprintf(key, "h.%d.attn.c_proj.weight", l);  ents[4].k = strdup(key);  ents[4].dst = &L->c_proj_w;   ents[4].n = (size_t)N_EMBD * N_EMBD;
        sprintf(key, "h.%d.attn.c_proj.bias", l);    ents[5].k = strdup(key);  ents[5].dst = &L->c_proj_b;   ents[5].n = N_EMBD;
        sprintf(key, "h.%d.ln_2.weight", l);         ents[6].k = strdup(key);  ents[6].dst = &L->ln2_w;      ents[6].n = N_EMBD;
        sprintf(key, "h.%d.ln_2.bias", l);           ents[7].k = strdup(key);  ents[7].dst = &L->ln2_b;      ents[7].n = N_EMBD;
        sprintf(key, "h.%d.mlp.c_fc.weight", l);     ents[8].k = strdup(key);  ents[8].dst = &L->mlp_fc_w;   ents[8].n = (size_t)N_EMBD * MLP_DIM;
        sprintf(key, "h.%d.mlp.c_fc.bias", l);       ents[9].k = strdup(key);  ents[9].dst = &L->mlp_fc_b;   ents[9].n = MLP_DIM;
        sprintf(key, "h.%d.mlp.c_proj.weight", l);   ents[10].k = strdup(key); ents[10].dst = &L->mlp_proj_w; ents[10].n = (size_t)MLP_DIM * N_EMBD;
        sprintf(key, "h.%d.mlp.c_proj.bias", l);     ents[11].k = strdup(key); ents[11].dst = &L->mlp_proj_b; ents[11].n = N_EMBD;

        for (int e = 0; e < 12; e++) {
            float *src = tensor_get(t, n_t, ents[e].k);
            if (!src) {
                fprintf(stderr, "[!] mixed-precision: missing tensor %s\n", ents[e].k);
                for (int c = 0; c < 12; c++) free((void *)ents[c].k);
                tensor_free_all(t, n_t);
                return -1;
            }
            float *buf = malloc(ents[e].n * sizeof(float));
            if (!buf) { fprintf(stderr, "[!] OOM in mixed-precision layer %d\n", l); return -1; }
            memcpy(buf, src, ents[e].n * sizeof(float));
            *(ents[e].dst) = buf;
            free((void *)ents[e].k);
        }
    }
    tensor_free_all(t, n_t);
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
 */
/* === Optional OpenBLAS acceleration for float matmul ===
 * When compiled with -DUSE_OPENBLAS, float matmul uses cblas_sgemv instead
 * of hand-written SIMD. OpenBLAS is highly optimized per-CPU (cache blocking,
 * prefetching, NEON/AVX tuning). ~2-3x faster than our matmul_avx2. */
#ifdef USE_OPENBLAS
#include <cblas.h>
static void matmul_blas(float *y, const float *x, const float *W, const float *b,
                        int in_dim, int out_dim) {
    /* y = W^T @ x + b. W is [in_dim, out_dim] row-major (GPT-2 Conv1D).
     * cblas_sgemv: y = alpha * op(A) * x + beta * y
     * Trans: op(A) = A^T, A is [M=in_dim, N=out_dim], x len=M, y len=N */
    cblas_sgemv(CblasRowMajor, CblasTrans, in_dim, out_dim,
                1.0f, W, out_dim, x, 1, 0.0f, y, 1);
    if (b) cblas_saxpy(out_dim, 1.0f, b, 1, y, 1);
}
#define matmul(y, x, W, b, id, od) matmul_blas(y, x, W, b, id, od)
/* LM head also uses BLAS */
static void lm_head_blas(float *logits, const float *x, const float *wte,
                         int vocab, int n_embd) {
    cblas_sgemv(CblasRowMajor, CblasTrans, n_embd, vocab,
                1.0f, wte, vocab, x, 1, 0.0f, logits, 1);
}
#define lm_head(logits, x, wte, v, ne) lm_head_blas(logits, x, wte, v, ne)
#else
#define matmul(y, x, W, b, id, od) matmul_avx2(y, x, W, b, id, od)
#define lm_head(logits, x, wte, v, ne) lm_head_avx2(logits, x, wte, v, ne)
#endif

/* W is row-major in GPT-2 Conv1D format: W[i][j] = W[i*out_dim + j].
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
 * The LM head is ~30% of total per-token time, so 2 threads → ~15% speedup.
 * On ARM, we set a large pthread stack (256 KB) to avoid stack overflow
 * in lm_head_avx2's 8 accumulator registers. */
typedef struct {
    const float *x;
    const float *wte;
    float *logits;
    int vocab_start, vocab_end, n_embd;
} LmHeadJob;

static void *lm_head_worker(void *arg) {
    LmHeadJob *job = (LmHeadJob *)arg;
    lm_head(job->logits + job->vocab_start, job->x,
                 job->wte + (size_t)job->vocab_start * job->n_embd,
                 job->vocab_end - job->vocab_start, job->n_embd);
    return NULL;
}

static void lm_head_parallel(float *logits, const float *x, const float *wte,
                             int vocab, int n_embd, int n_threads) {
    if (n_threads <= 1) {
        lm_head(logits, x, wte, vocab, n_embd);
        return;
    }
    pthread_t threads[8];
    pthread_attr_t attr;
    LmHeadJob jobs[8];
    if (n_threads > 8) n_threads = 8;

    /* Set 1 MB stack per thread (ARM Android default ~8 KB is too small) */
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024);

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
        pthread_create(&threads[t], &attr, lm_head_worker, &jobs[t]);
    }
    pthread_attr_destroy(&attr);
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

/* === LM head with int8 dynamic quantization (--lm-head-int8) ===
 *
 * logits[v] = wte_row_v · x = scale_w[v] * sum_i (wte_q[v,i] * x_q[i]) * scale_x
 *
 * Both the wte rows (quantized once at load into g_wte_q/g_wte_scale) and the
 * activation x (quantized per-token) are int8. The inner product is int8×int8,
 * accumulated as int32, then rescaled by scale_w[v] * scale_x. This is the
 * standard dynamic int8 GEMM pattern: the memory bandwidth win (4x less data
 * read per row) dominates because the LM head is purely memory-bound.
 *
 * AVX2 path uses _mm256_cvtepi8_epi16 + _mm256_madd_epi16 (signed×signed,
 * 32 int8 → 8 int32 accumulators per iteration). NEON/scalar fall back to a
 * plain widening loop — still bandwidth-bound, so the win holds.
 *
 * NOTE: --prune-vocab is ignored when --lm-head-int8 is set (int8 already
 * shrinks each row 4x; dropping rows on top adds little and complicates the
 * int8 gather). The two are mutually exclusive by design. */

/* Quantize one activation vector x[n] → int8 xq[n] + returns scale.
 * Same scheme as LAL-Bot's turboquant_quantize (max|x|/127) so the int8
 * LM head and int8 KV cache use a consistent quantization style. */
static float lm_head_quantize_x(const float *x, int8_t *xq, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > max_abs) max_abs = a;
    }
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

#if defined(__AVX2__)
/* Horizontal sum of 8×int32 lanes. */
static inline int hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}
#endif

/* int8 LM head over vocab rows [v_start, v_end). Used by both the
 * single-threaded and the parallel-worker paths. */
static void lm_head_int8_range(float *logits, const int8_t *xq, float scale_x,
                               int v_start, int v_end, int n_embd) {
    const int8_t *wq = g_wte_q;
    const float  *ws = g_wte_scale;
    for (int v = v_start; v < v_end; v++) {
        const int8_t *wv = wq + (size_t)v * n_embd;
        int dot = 0;
#if defined(__AVX2__)
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= n_embd; i += 32) {
            /* 16 int8 → 16 int16 → 8 int32 (pairwise products summed). */
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
        dot = hsum_epi32(acc);
        for (; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
#else
        /* Scalar/NEON fallback: plain widening. Still bandwidth-bound, so
         * the 4x memory reduction still pays off even without SIMD compute. */
        for (int i = 0; i < n_embd; i++) dot += (int)xq[i] * (int)wv[i];
#endif
        logits[v] = scale_x * ws[v] * (float)dot;
    }
}

typedef struct {
    const int8_t *xq;
    float scale_x;
    float *logits;
    int v_start, v_end, n_embd;
} LmHeadInt8Job;

static void *lm_head_int8_worker(void *arg) {
    LmHeadInt8Job *j = (LmHeadInt8Job *)arg;
    lm_head_int8_range(j->logits, j->xq, j->scale_x, j->v_start, j->v_end, j->n_embd);
    return NULL;
}

/* Top-level int8 LM head: quantize x once, then split vocab across threads.
 *
 * TWO-PASS design (preserves quality): per-row int8 quantization perturbs
 * logit magnitudes enough to reshuffle the top-k, which ruins sampling
 * (observed: coherent text → gibberish). So we do:
 *   Pass 1: compute int8 logits for ALL 50257 tokens (bandwidth-bound, fast)
 *   Pass 2: find the top-RERANK_N candidates from the int8 logits, then
 *           re-score just those RERANK_N in float (compute-bound, but only
 *           RERANK_N × 768 float MACs ≈ 0.4 MB reads — negligible vs 154 MB).
 * The rest of g_logits stays at its int8 value (irrelevant — sampling only
 * touches the top-k, which is within RERANK_N). This is the llama.cpp
 * "int8 candidate selection + float re-rank" pattern. Net: ~4x bandwidth
 * reduction on the hot path with no measurable quality loss. */
#define LM_HEAD_RERANK_N 512  /* wide window: int8 selection is noisy, so
                               * re-score generously; 512×768×4B = 1.5MB ≪ 154MB */

static void lm_head_int8_parallel(float *logits, const float *x, int n_threads) {
    static int8_t xq[N_EMBD];
    float scale_x = lm_head_quantize_x(x, xq, N_EMBD);

    /* Pass 1: int8 logits for the full vocab. */
    if (n_threads <= 1) {
        lm_head_int8_range(logits, xq, scale_x, 0, VOCAB_SIZE, N_EMBD);
    } else {
        pthread_t threads[8];
        pthread_attr_t attr;
        LmHeadInt8Job jobs[8];
        if (n_threads > 8) n_threads = 8;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 1024 * 1024);
        int chunk = (VOCAB_SIZE + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; t++) {
            jobs[t].xq = xq;
            jobs[t].scale_x = scale_x;
            jobs[t].logits = logits;
            jobs[t].v_start = t * chunk;
            jobs[t].v_end = (t + 1) * chunk;
            if (jobs[t].v_start > VOCAB_SIZE) jobs[t].v_start = VOCAB_SIZE;
            if (jobs[t].v_end > VOCAB_SIZE) jobs[t].v_end = VOCAB_SIZE;
            if (jobs[t].v_end <= jobs[t].v_start) continue;
            pthread_create(&threads[t], &attr, lm_head_int8_worker, &jobs[t]);
        }
        pthread_attr_destroy(&attr);
        for (int t = 0; t < n_threads; t++)
            if (jobs[t].v_end > jobs[t].v_start) pthread_join(threads[t], NULL);
    }

    /* Pass 2: re-score the top-RERANK_N candidates in float. Selection uses
     * a min-heap of size RERANK_N (O(V log N) ≈ 450K ops) instead of the
     * O(N×V) find-max-and-mask that dominated at large N. */
    {
        /* Min-heap of (logit, vocab_idx) pairs, keyed by logit. Heap[0] is
         * the smallest logit in the current top-N set. */
        static int   heap_idx[LM_HEAD_RERANK_N];
        static float heap_val[LM_HEAD_RERANK_N];
        int heap_n = 0;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            float val = logits[v];
            if (heap_n < LM_HEAD_RERANK_N) {
                /* Push: insert at end, sift up. */
                int c = heap_n++;
                heap_val[c] = val; heap_idx[c] = v;
                while (c > 0) {
                    int p = (c - 1) >> 1;
                    if (heap_val[p] <= heap_val[c]) break;
                    float tv = heap_val[p]; heap_val[p] = heap_val[c]; heap_val[c] = tv;
                    int ti = heap_idx[p]; heap_idx[p] = heap_idx[c]; heap_idx[c] = ti;
                    c = p;
                }
            } else if (val > heap_val[0]) {
                /* Replace min, sift down. */
                heap_val[0] = val; heap_idx[0] = v;
                int p = 0;
                for (;;) {
                    int l = 2 * p + 1, r = 2 * p + 2, s = p;
                    if (l < heap_n && heap_val[l] < heap_val[s]) s = l;
                    if (r < heap_n && heap_val[r] < heap_val[s]) s = r;
                    if (s == p) break;
                    float tv = heap_val[p]; heap_val[p] = heap_val[s]; heap_val[s] = tv;
                    int ti = heap_idx[p]; heap_idx[p] = heap_idx[s]; heap_idx[s] = ti;
                    p = s;
                }
            }
        }
        /* Re-score the selected RERANK_N candidates in float. */
        for (int k = 0; k < heap_n; k++) {
            int v = heap_idx[k];
            const float *w = g_wte + (size_t)v * N_EMBD;
            v8f acc = v8f_zero();
            int i = 0;
            for (; i + 8 <= N_EMBD; i += 8)
                acc = v8f_fmadd(v8f_load(x + i), v8f_load(w + i), acc);
            float s = v8f_hsum(acc);
            for (; i < N_EMBD; i++) s += x[i] * w[i];
            logits[v] = s;
        }
    }
}

/* GELU (tanh approximation, GPT-2 style) */
static inline float gelu_fast(float x) {
    const float c = 0.7978845608028654f;  /* sqrt(2/π) */
    float t = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(t));
}

/* NOTE: a SIMD-vectorized gelu_simd() was tried (Padé[2/2] tanh approx to
 * avoid scalar tanhf) but the approximation diverged for |t| > ~2.5, which
 * GELU's argument t = c0*(x + 0.044715*x³) reaches at moderate x. This
 * corrupted MLP activations and produced word-salad output. Reverted to
 * scalar gelu_fast. A correct SIMD GELU would need either a vectorized tanh
 * (range-reduction + high-degree polynomial) or the exact sigmoid form —
 * left as future work. */

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

/* === LAL three-layer fusion (level 1): runtime hidden-state bridge ===
 *
 * Exposes the model's final hidden state (g_final_ln, 768-dim) to .lal-compiled
 * programs. A .lal program using `concept ctx = runtime_context(dim=768)` emits
 * an `extern float *lal_server_get_hidden(void);` declaration and calls this
 * getter at startup to bind the live model's contextualized representation into
 * a .lal concept vector. This is the first true runtime bridge between the DNN
 * inference layer and the .lal logic-programming layer — previously .lal could
 * only consume DNN products at compile time (load_word2vec / linear weight bake).
 *
 * Returns NULL if no forward pass has run yet (caller should handle gracefully).
 * The returned pointer aliases g_final_ln and stays valid until the next forward
 * pass overwrites it. */
float *lal_server_get_hidden(void) {
    return g_final_ln;
}

/* === LAL three-layer fusion (level 2): .lal logic-layer sampling filter ===
 *
 * Hook into the top-k sampling pool: after the server computes keep_mask
 * (which tokens survive top-k + top-p filtering), this function is called so
 * a .lal-compiled logic program can remove tokens that violate semantic rules
 * (grammar constraints, repetition bans, context-dependent bans, etc.).
 *
 * Weak fallback: returns 0 (no filtering) when no .lal filter is linked.
 * Link with a .lal-compiled object defining a strong lal_filter_topk to
 * activate rule-based sampling constraints. Zero-intrusion when unlinked.
 *
 * keep_mask: in/out, length n_vocab (1 = keep, 0 = drop). The .lal program
 *   zeroes entries for tokens it wants to ban.
 * n_vocab: array length (VOCAB_SIZE for GPT-2).
 * last_token: most recently generated token id (-1 at sequence start).
 * recent_tokens: array of recently generated token ids.
 * n_recent: length of recent_tokens.
 * decode_fn: optional callback (may be NULL) that maps a token id → its
 *   decoded text (writes into out_buf, returns length). Lets the .lal filter
 *   do word-level / concept-level matching (e.g. ban_relation("France",
 *   "capital", {"Paris"})) without needing its own tokenizer. Filters that
 *   only use token-id rules (ban_last/ban_repeat/ban_token) ignore this.
 *
 * Returns: number of tokens dropped (for logging). */
typedef int (*lal_token_decode_fn)(int token_id, char *out_buf, int max_len);
int lal_filter_topk(int *keep_mask, int n_vocab, int last_token,
                    const int *recent_tokens, int n_recent,
                    lal_token_decode_fn decode_fn)
    __attribute__((weak));
int lal_filter_topk(int *keep_mask, int n_vocab, int last_token,
                    const int *recent_tokens, int n_recent,
                    lal_token_decode_fn decode_fn) {
    (void)keep_mask; (void)n_vocab; (void)last_token;
    (void)recent_tokens; (void)n_recent; (void)decode_fn;
    return 0;  /* no .lal filter linked — sampling unchanged */
}

/* Thin wrapper so the weak symbol sees the server's tokenizer. */
static int _server_decode_for_filter(int token_id, char *out_buf, int max_len) {
    return decode_token(token_id, out_buf, max_len);
}

/* KV cache for real causal self-attention.
 * Per layer: K and V for all positions, stored HEAD-MAJOR:
 *   [N_HEAD][n_ctx][head_dim] instead of [n_ctx][N_EMBD].
 * Total: 12 layers × 2 (K+V) × 12 × 1024 × 64 × 4 bytes = 72 MB (same size).
 * Head-major layout makes K/V for a fixed head across positions contiguous,
 * so the QK dot-product loop (iterating j=0..position for head h) reads
 * sequential memory instead of striding by N_EMBD=768 floats (3KB) per step.
 * This dramatically improves L1/L2 cache utilization for attention.
 *
 * TurboQuant: when --turboquant is set, KV cache is stored as int8 + per-row
 * scale (4x compression: 72MB → 18MB). K/V are quantized on write and
 * dequantized on read in the attention functions. */
#define KV_CACHE_CTX   1024   /* max sequence length (matches allocation) */
#define KV_HD          (N_EMBD / N_HEAD)  /* head_dim = 64 */
/* Offset for head h, position j in head-major KV cache */
#define KV_OFF(h, j)   ((size_t)(h) * KV_CACHE_CTX * KV_HD + (size_t)(j) * KV_HD)
static float *g_k_cache[N_LAYER];       /* [n_ctx, n_embd] per layer (float mode) */
static float *g_v_cache[N_LAYER];       /* [n_ctx, n_embd] per layer (float mode) */
static int8_t *g_k_cache_q[N_LAYER];   /* [n_ctx, n_embd] int8 (turboquant mode) */
static int8_t *g_v_cache_q[N_LAYER];   /* [n_ctx, n_embd] int8 (turboquant mode) */
static float  *g_k_scale[N_LAYER];     /* [n_ctx] per-row scale for K (turboquant) */
static float  *g_v_scale[N_LAYER];     /* [n_ctx] per-row scale for V (turboquant) */
/* Real causal self-attention is ON by default. The V-copy fallback
 * (memcpy of V into attn_out) is a defective path that drops all
 * token-to-token context — the 12-layer transformer degenerates into a
 * memoryless Markov process and the model regresses to "generate itself"
 * repetition loops. That repetition is the exact symptom level-2 filters
 * (ban_last/ban_repeat) were patching. Defaulting real attention on fixes
 * the root cause for inference. Use --vcopy to restore the legacy fast
 * (but context-free) path for benchmark comparison only. */
static int    g_srv_real_attention = 1;  /* default ON; --vcopy disables */
static int    g_use_dflash = 0;         /* set by --dflash (Dynamic Flash Attention) */
static int    g_use_turboquant = 0;     /* set by --turboquant (KV cache int8 quant) */
static int    g_skip_lm_head = 0;       /* internal: skip LM head during prefill (optimization) */
static float  g_temperature = 0.8f;      /* sampling temperature, 0=argmax */
static int    g_top_k = 40;             /* top-k sampling, 0=argmax */
static float  g_top_p = 0.0f;           /* top-p (nucleus) sampling, 0=off; applied after top-k */
static float  g_rep_penalty = 1.1f;     /* repetition penalty, 1.0=off */
static int    g_recent_tokens[256];     /* ring buffer of recent tokens */
static int    g_n_recent = 0;           /* count of recent tokens (up to 256) */
static float *g_logits = NULL;       /* [VOCAB_SIZE], allocated once */
static int    g_n_threads = 1;       /* worker threads for LM head */

/* === Bit-width speculative decoding: layer-wise early exit ===
 * Run forward through the first g_early_exit_layers layers, then check a
 * cheap confidence metric (residual stream convergence). If the residual
 * stream is stable (small relative change), skip remaining layers → faster.
 * g_early_exit_layers=0 disables early exit (run all N_LAYER layers).
 * This is the "bit-width speculative" approach PHONE requested: shallower
 * depth = effectively lower "bit-width" model as draft, full depth as verify. */
static int    g_early_exit_layers = 0;    /* 0=off, else exit after this many layers */
static float  g_early_exit_threshold = 0.02f; /* ||dx||/||x|| below this → exit */
static int    g_early_exit_hits = 0;      /* stat: times early exit triggered */
static int    g_early_exit_total = 0;     /* stat: total forward calls */

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

/* Real causal self-attention with multi-head QK softmax.
 *
 * GPT-2: 12 heads, head_dim=64, n_embd=768.
 * QKV from c_attn: [n_embd] → [3*n_embd] = [Q(768), K(768), V(768)]
 * Each split into 12 heads of 64 dims.
 *
 * For position t, we have:
 *   Q[t] = current query (just computed)
 *   K[0..t] = all keys up to current (from KV cache + current)
 *   V[0..t] = all values
 *
 * Per head h:
 *   scores[j] = Q[h] · K[j,h] / sqrt(head_dim)   for j = 0..t
 *   attn[h] = softmax(scores) @ V[0..t, h]
 *
 * Result: concat(attn[0..11]) → [n_embd] → c_proj
 *
 * This replaces the broken "attn_out = V" shortcut that caused
 * repetitive outputs like "fox fox fox fox...". */
/* TurboQuant helpers: quantize/dequantize a row of n_embd floats to int8.
 * scale = max(abs(x)) / 127, then int8 = round(x / scale). */
static float turboquant_quantize(int8_t *out, const float *in, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(in[i]);
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)(in[i] * inv_scale);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        out[i] = (int8_t)v;
    }
    return scale;
}

static void turboquant_dequantize(float *out, const int8_t *in, float scale, int n) {
    for (int i = 0; i < n; i++) out[i] = (float)in[i] * scale;
}

/* DFlash (Dynamic Flash Attention) — FlashAttention-style tiled KV.
 *
 * Key difference from real_attention: instead of materializing the full
 * scores[0..position] array, we do two passes:
 *   Pass 1: compute all Q·K scores, find max, compute softmax sum
 *   Pass 2: recompute scores, apply softmax, accumulate V
 *
 * This halves the temporary memory (no scores[] or attn_weights[] arrays)
 * and is more cache-friendly for long sequences. The recomputation cost
 * is negligible because Q·K is a 64-dim dot product (head_dim=64).
 *
 * For very long sequences (>512), this also enables tiling: process K/V
 * in blocks of 64 positions to fit in L1 cache. */
static void dflash_attention(float *attn_out, const float *qkv,
                             int position, int layer_idx) {
    int n_head = N_HEAD;
    int head_dim = N_EMBD / n_head;  /* 64 */
    float scale = 1.0f / sqrtf((float)head_dim);
    int seq_len = position + 1;
    int use_tq = g_use_turboquant && g_k_cache_q[layer_idx];

    const float *Q = qkv;
    const float *K_new = qkv + N_EMBD;
    const float *V_new = qkv + 2*N_EMBD;

    /* Store current K, V into cache (quantize if turboquant) */
    if (use_tq) {
        g_k_scale[layer_idx][position] = turboquant_quantize(
            g_k_cache_q[layer_idx] + (size_t)position * N_EMBD, K_new, N_EMBD);
        g_v_scale[layer_idx][position] = turboquant_quantize(
            g_v_cache_q[layer_idx] + (size_t)position * N_EMBD, V_new, N_EMBD);
    } else {
        /* Head-major scatter write for float cache */
        for (int h = 0; h < n_head; h++) {
            memcpy(g_k_cache[layer_idx] + KV_OFF(h, position), K_new + h * head_dim, head_dim * sizeof(float));
            memcpy(g_v_cache[layer_idx] + KV_OFF(h, position), V_new + h * head_dim, head_dim * sizeof(float));
        }
    }

    /* Temporary buffer for dequantized row (turboquant only) */
    float k_row[N_EMBD], v_row[N_EMBD];

    /* SIMD-vectorized dflash. head_dim=64 is a multiple of 8, so the v8f
     * loops have no tail. The three scalar dot-products / accumulations
     * become v8f_fmadd + v8f_hsum. This is the hot path on ARMv7 tablets
     * (binary + --dflash): 12 layers × 12 heads × seq_len × 3 passes ×
     * 64 scalar MACs → NEON 4-wide cuts it ~3-4x. Also speeds up the x86
     * float dflash path ~2x via AVX2 8-wide.
     *
     * Pass 1 fuses max + sum_exp into a single pass over K (halves K reads
     * and dequant calls vs the original two-pass). The catch: sum_exp needs
     * (score - max_score), but max isn't known until the pass completes. We
     * store the scaled scores in a small stack array (≤1024 floats = 4 KB)
     * and apply the softmax subtract in a second tiny loop. */
    static float scores[1024];

    for (int h = 0; h < n_head; h++) {
        const float *Q_h = Q + h * head_dim;
        /* Load Q once per head — reused across all j iterations. */
        v8f q0 = v8f_load(Q_h + 0);
        v8f q1 = v8f_load(Q_h + 8);
        v8f q2 = v8f_load(Q_h + 16);
        v8f q3 = v8f_load(Q_h + 24);
        v8f q4 = v8f_load(Q_h + 32);
        v8f q5 = v8f_load(Q_h + 40);
        v8f q6 = v8f_load(Q_h + 48);
        v8f q7 = v8f_load(Q_h + 56);

        /* Pass 1: compute all scaled scores, track max. */
        float max_score = -1e30f;
        for (int j = 0; j < seq_len; j++) {
            const float *K_jh;
            if (use_tq) {
                turboquant_dequantize(k_row, g_k_cache_q[layer_idx] + (size_t)j * N_EMBD,
                                      g_k_scale[layer_idx][j], N_EMBD);
                K_jh = k_row + h * head_dim;
            } else {
                K_jh = g_k_cache[layer_idx] + KV_OFF(h, j);
            }
            v8f k0 = v8f_load(K_jh + 0);
            v8f k1 = v8f_load(K_jh + 8);
            v8f k2 = v8f_load(K_jh + 16);
            v8f k3 = v8f_load(K_jh + 24);
            v8f k4 = v8f_load(K_jh + 32);
            v8f k5 = v8f_load(K_jh + 40);
            v8f k6 = v8f_load(K_jh + 48);
            v8f k7 = v8f_load(K_jh + 56);
            v8f acc = v8f_zero();
            acc = v8f_fmadd(q0, k0, acc);
            acc = v8f_fmadd(q1, k1, acc);
            acc = v8f_fmadd(q2, k2, acc);
            acc = v8f_fmadd(q3, k3, acc);
            acc = v8f_fmadd(q4, k4, acc);
            acc = v8f_fmadd(q5, k5, acc);
            acc = v8f_fmadd(q6, k6, acc);
            acc = v8f_fmadd(q7, k7, acc);
            float dot = v8f_hsum(acc) * scale;
            scores[j] = dot;
            if (dot > max_score) max_score = dot;
        }

        /* sum_exp (reuses stored scores — no K re-read). */
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++)
            sum_exp += expf(scores[j] - max_score);
        float inv_sum = 1.0f / (sum_exp + 1e-12f);

        /* Pass 2: accumulate V. Score is read back from scores[] (avoids a
         * third K read / dequant). V accumulation is vectorized. */
        float *out_h = attn_out + h * head_dim;
        v8f o0 = v8f_zero(), o1 = v8f_zero(), o2 = v8f_zero(), o3 = v8f_zero();
        v8f o4 = v8f_zero(), o5 = v8f_zero(), o6 = v8f_zero(), o7 = v8f_zero();
        for (int j = 0; j < seq_len; j++) {
            float weight = expf(scores[j] - max_score) * inv_sum;
            v8f wv = v8f_set1(weight);
            const float *V_jh;
            if (use_tq) {
                turboquant_dequantize(v_row, g_v_cache_q[layer_idx] + (size_t)j * N_EMBD,
                                      g_v_scale[layer_idx][j], N_EMBD);
                V_jh = v_row + h * head_dim;
            } else {
                V_jh = g_v_cache[layer_idx] + KV_OFF(h, j);
            }
            o0 = v8f_fmadd(wv, v8f_load(V_jh + 0),  o0);
            o1 = v8f_fmadd(wv, v8f_load(V_jh + 8),  o1);
            o2 = v8f_fmadd(wv, v8f_load(V_jh + 16), o2);
            o3 = v8f_fmadd(wv, v8f_load(V_jh + 24), o3);
            o4 = v8f_fmadd(wv, v8f_load(V_jh + 32), o4);
            o5 = v8f_fmadd(wv, v8f_load(V_jh + 40), o5);
            o6 = v8f_fmadd(wv, v8f_load(V_jh + 48), o6);
            o7 = v8f_fmadd(wv, v8f_load(V_jh + 56), o7);
        }
        v8f_store(out_h + 0,  o0);
        v8f_store(out_h + 8,  o1);
        v8f_store(out_h + 16, o2);
        v8f_store(out_h + 24, o3);
        v8f_store(out_h + 32, o4);
        v8f_store(out_h + 40, o5);
        v8f_store(out_h + 48, o6);
        v8f_store(out_h + 56, o7);
    }
}

/* Original real_attention (kept for fallback / comparison) */
static void real_attention(float *attn_out, const float *qkv, int layer_idx,
                           int position, float *k_cache_layer, float *v_cache_layer) {
    int n_head = N_HEAD;
    int head_dim = N_EMBD / n_head;  /* 64 */
    float scale = 1.0f / sqrtf((float)head_dim);

    const float *Q = qkv;                  /* [n_embd] */
    const float *K_new = qkv + N_EMBD;     /* [n_embd] */
    const float *V_new = qkv + 2*N_EMBD;   /* [n_embd] */

    /* Store current K, V into cache at this position — head-major scatter */
    for (int h = 0; h < n_head; h++) {
        memcpy(k_cache_layer + KV_OFF(h, position), K_new + h * head_dim, head_dim * sizeof(float));
        memcpy(v_cache_layer + KV_OFF(h, position), V_new + h * head_dim, head_dim * sizeof(float));
    }

    float scores[1024];  /* max seq_len per head */
    float attn_weights[1024];

    for (int h = 0; h < n_head; h++) {
        const float *Q_h = Q + h * head_dim;   /* [head_dim] */

        /* Compute scores[j] = Q_h · K[j, h] * scale, for j = 0..position */
        float max_score = -1e30f;
        for (int j = 0; j <= position; j++) {
            const float *K_jh = k_cache_layer + KV_OFF(h, j);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += Q_h[d] * K_jh[d];
            dot *= scale;
            scores[j] = dot;
            if (dot > max_score) max_score = dot;
        }

        /* Softmax with max subtraction for numerical stability */
        float sum_exp = 0.0f;
        for (int j = 0; j <= position; j++) {
            float e = expf(scores[j] - max_score);
            attn_weights[j] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (int j = 0; j <= position; j++) attn_weights[j] *= inv_sum;

        /* Weighted sum of V: attn_out[h] = sum_j attn_weights[j] * V[j, h] */
        float *out_h = attn_out + h * head_dim;
        for (int d = 0; d < head_dim; d++) out_h[d] = 0.0f;
        for (int j = 0; j <= position; j++) {
            float w = attn_weights[j];
            const float *V_jh = v_cache_layer + KV_OFF(h, j);
            for (int d = 0; d < head_dim; d++) out_h[d] += w * V_jh[d];
        }
    }
}

/* SIMD version of real_attention for the float path (AVX2/NEON/scalar auto).
 * Same semantics as real_attention(); LAL-Bot approved keeping the scalar
 * version for the binary branch and calling this one from float. The QK dot
 * product and the V weighted accumulation are the hot loops — both vectorized
 * 8 floats at a time via the project's v8f helpers. */
static void real_attention_simd(float *attn_out, const float *qkv, int layer_idx,
                                int position, float *k_cache_layer, float *v_cache_layer) {
    const int head_dim = N_EMBD / N_HEAD;          /* 64 */
    const float scale = 1.0f / sqrtf((float)head_dim);

    const float *Q = qkv;
    const float *K_new = qkv + N_EMBD;
    const float *V_new = qkv + 2 * N_EMBD;

    /* Cache K, V for this position. */
    /* Cache K, V for this position — head-major scatter write.
     * Each head's K/V is stored contiguously across positions, so the QK
     * dot-product loop reads sequential memory (stride=head_dim=64 floats=256B)
     * instead of striding by N_EMBD=768 floats=3KB. */
    for (int h = 0; h < N_HEAD; h++) {
        memcpy(k_cache_layer + KV_OFF(h, position), K_new + h * head_dim, head_dim * sizeof(float));
        memcpy(v_cache_layer + KV_OFF(h, position), V_new + h * head_dim, head_dim * sizeof(float));
    }

    static float scores[1024];      /* max seq len */
    static float attn_weights[1024];

    for (int h = 0; h < N_HEAD; h++) {
        const float *Q_h = Q + h * head_dim;

        /* scores[j] = (Q_h · K[j,h]) * scale, vectorized.
         * K[j,h] is now contiguous: KV_OFF(h,j) to KV_OFF(h,j)+head_dim. */
        float max_score = -1e30f;
        for (int j = 0; j <= position; j++) {
            const float *K_jh = k_cache_layer + KV_OFF(h, j);
            v8f acc = v8f_zero();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8)
                acc = v8f_fmadd(v8f_load(Q_h + d), v8f_load(K_jh + d), acc);
            float dot = v8f_hsum(acc);
            for (; d < head_dim; d++) dot += Q_h[d] * K_jh[d];
            dot *= scale;
            scores[j] = dot;
            if (dot > max_score) max_score = dot;
        }

        /* Softmax (numerically stable). */
        float sum_exp = 0.0f;
        for (int j = 0; j <= position; j++) {
            float e = expf(scores[j] - max_score);
            attn_weights[j] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (int j = 0; j <= position; j++) attn_weights[j] *= inv_sum;

        /* out_h = sum_j attn_weights[j] * V[j,h], vectorized accumulation. */
        float *out_h = attn_out + h * head_dim;
        for (int d = 0; d < head_dim; d++) out_h[d] = 0.0f;
        for (int j = 0; j <= position; j++) {
            float w = attn_weights[j];
            const float *V_jh = v_cache_layer + KV_OFF(h, j);
            v8f wv = v8f_set1(w);
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                v8f cur = v8f_load(out_h + d);
                cur = v8f_fmadd(wv, v8f_load(V_jh + d), cur);
                v8f_store(out_h + d, cur);
            }
            for (; d < head_dim; d++) out_h[d] += w * V_jh[d];
        }
    }
}

static int gpt2_forward_token(int token_id, int position) {

    /* Embedding: g_x = wte[token] + wpe[position].
     * This was present in the original code but got dropped in commit ce19601,
     * which left g_x holding the previous token's residual stream — causing
     * degenerate "the the the..." output regardless of attention. Restored. */
    if (token_id < 0 || token_id >= VOCAB_SIZE) token_id = 0;
    if (position < 0 || position >= 1024) position = 0;
    for (int i = 0; i < N_EMBD; i++)
        g_x[i] = g_wte[token_id * N_EMBD + i] + g_wpe[position * N_EMBD + i];

    /* Layers */
    g_early_exit_total++;
    int actual_layers = N_LAYER;
    for (int l = 0; l < N_LAYER; l++) {
        g_current_layer = l;  /* for mixed-int8 layer selection */
        /* Early exit: after g_early_exit_layers layers, check if the residual
         * stream has converged. If ||dx||/||x|| < threshold, skip remaining
         * layers — the model is confident and deeper layers add little. */
        if (g_early_exit_layers > 0 && l == g_early_exit_layers && l < N_LAYER) {
            /* Compute relative change: ||x - x_prev|| / ||x||
             * We approximate by comparing current x norm vs the norm of the
             * last layer's residual addition (g_proj + g_mlp_out). If the
             * additions are small relative to x, the stream is stable. */
            float dx_norm = 0.0f, x_norm = 0.0f;
            for (int i = 0; i < N_EMBD; i++) {
                float dx = g_proj[i] + g_mlp_out[i];  /* last residual additions */
                dx_norm += dx * dx;
                x_norm += g_x[i] * g_x[i];
            }
            float rel_change = sqrtf(dx_norm) / (sqrtf(x_norm) + 1e-8f);
            if (rel_change < g_early_exit_threshold) {
                g_early_exit_hits++;
                actual_layers = l;
                break;
            }
        }
        /* Mixed-precision: first and last layers run in float (g_layers[l]),
         * the rest in binary (g_bin_layers[l]). Falls through to binary for
         * all layers in pure --binary mode, and to float for all layers when
         * binary mode is off. */
        int use_float_layer = !g_binary_mode ||
            (g_mixed_precision && (l == 0 || l == N_LAYER - 1));
        if (!use_float_layer) {
            /* Binary forward: XNOR + popcount (32x fewer FLOPs per layer) */
            BinGPT2Layer *L = &g_bin_layers[l];
            layer_norm_simd(g_ln1, g_x, L->ln1_w, L->ln1_b, N_EMBD);
            bin_matmul(g_ln1, &L->c_attn, g_qkv);

            /* Attention: dflash (tiled) > real_attention_simd (SIMD) > V-copy.
             * Use SIMD version for binary path too — attention computation is
             * identical regardless of weight format (QKV are float after
             * bin_matmul). Previously binary path used scalar real_attention,
             * which was the actual bottleneck (BinAnalyzer's profiling). */
            if ((g_srv_real_attention || g_use_dflash) && g_k_cache[l]) {
                if (g_use_dflash) {
                    dflash_attention(g_attn_out, g_qkv, position, l);
                } else {
                    real_attention_simd(g_attn_out, g_qkv, l, position,
                                        g_k_cache[l], g_v_cache[l]);
                }
            } else {
                memcpy(g_attn_out, g_qkv + 2*N_EMBD, N_EMBD * sizeof(float));
            }
            bin_matmul(g_attn_out, &L->c_proj, g_proj);
            for (int i = 0; i < N_EMBD; i++) g_x[i] += g_proj[i];

            layer_norm_simd(g_ln2, g_x, L->ln2_w, L->ln2_b, N_EMBD);
            bin_matmul(g_ln2, &L->mlp_fc, g_fc);
            for (int i = 0; i < MLP_DIM; i++) g_fc[i] = gelu_fast(g_fc[i]);
            bin_matmul(g_fc, &L->mlp_proj, g_mlp_out);
            for (int i = 0; i < N_EMBD; i++) g_x[i] += g_mlp_out[i];
        } else {
            /* Float forward: AVX2/NEON SIMD matmul */
            GPT2Layer *L = &g_layers[l];

            layer_norm_simd(g_ln1, g_x, L->ln1_w, L->ln1_b, N_EMBD);
            matmul(g_qkv, g_ln1, L->c_attn_w, L->c_attn_b, N_EMBD, 3*N_EMBD);

            /* Attention: dflash > real_attention_simd > real_attention > V-copy */
            if ((g_srv_real_attention || g_use_dflash) && g_k_cache[l]) {
                if (g_use_dflash) {
                    dflash_attention(g_attn_out, g_qkv, position, l);
                } else {
                    real_attention_simd(g_attn_out, g_qkv, l, position,
                                        g_k_cache[l], g_v_cache[l]);
                }
            } else {
                memcpy(g_attn_out, g_qkv + 2*N_EMBD, N_EMBD * sizeof(float));
            }
            matmul(g_proj, g_attn_out, L->c_proj_w, L->c_proj_b, N_EMBD, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) g_x[i] += g_proj[i];

            layer_norm_simd(g_ln2, g_x, L->ln2_w, L->ln2_b, N_EMBD);
            matmul(g_fc, g_ln2, L->mlp_fc_w, L->mlp_fc_b, N_EMBD, MLP_DIM);
            for (int i = 0; i < MLP_DIM; i++) g_fc[i] = gelu_fast(g_fc[i]);
            matmul(g_mlp_out, g_fc, L->mlp_proj_w, L->mlp_proj_b, MLP_DIM, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) g_x[i] += g_mlp_out[i];
        }
    }

    /* Final norm */
    layer_norm_simd(g_final_ln, g_x, g_ln_f_w, g_ln_f_b, N_EMBD);

    /* Prefill optimization: skip LM head for intermediate prompt tokens.
     * Only the last prompt token needs logits (for generating the first
     * new token). This saves 50257×768 LM head FLOPs per prefill token. */
    if (g_skip_lm_head) return 0;  /* return dummy token (unused) */

    /* LM head: logits = wte @ final_ln  (tied embeddings)
     * wte stays float in both modes (binarizing embeddings hurts accuracy).
     * Path selection (mutually exclusive by design):
     *   --lm-head-int8  → int8 dynamic quant GEMM (4x less bandwidth)
     *   --prune-vocab   → float, but only over the active vocab subset
     *   (neither)       → full-vocab float SIMD */
    if (g_use_lm_head_int8 && g_wte_q) {
        lm_head_int8_parallel(g_logits, g_final_ln, g_n_threads);
    } else if (g_active_vocab && g_n_active_vocab < VOCAB_SIZE) {
        lm_head_pruned(g_logits, g_final_ln, g_wte, g_active_vocab, g_n_active_vocab, N_EMBD);
    } else {
        lm_head_parallel(g_logits, g_final_ln, g_wte, VOCAB_SIZE, N_EMBD, g_n_threads);
    }

    /* Sampling: top-k + temperature + repetition penalty.
     * Pure argmax causes repetitive loops ("the the the...").
     * Top-k sampling from the k highest-prob tokens breaks the cycle.
     * Repetition penalty reduces probability of already-generated tokens.
     * Defaults: temperature=0.8, top_k=40, rep_penalty=1.1. */
    if (g_temperature > 0.0f && g_top_k > 0) {
        /* Apply repetition penalty: reduce logits of tokens in recent history */
        if (g_rep_penalty > 1.0f) {
            /* Penalty the last N generated tokens (from g_recent_tokens) */
            for (int i = 0; i < g_n_recent; i++) {
                int t = g_recent_tokens[i];
                if (t >= 0 && t < VOCAB_SIZE) {
                    if (g_logits[t] > 0) g_logits[t] /= g_rep_penalty;
                    else g_logits[t] *= g_rep_penalty;
                }
            }
        }

        /* Find top-k logits via partial selection (simple approach: scan once) */
        int top_k = g_top_k;
        if (top_k > VOCAB_SIZE) top_k = VOCAB_SIZE;

        /* Find the k-th largest logit via a simple threshold scan.
         * For small k (40), we do k passes of find-max-and-mask. */
        float threshold = -1e30f;
        int seen = 0;
        /* Simple selection: find max, set to -inf, repeat k times to get threshold */
        static float logits_copy[50257];
        memcpy(logits_copy, g_logits, VOCAB_SIZE * sizeof(float));
        for (int sel = 0; sel < top_k; sel++) {
            int max_v = 0;
            float max_val = logits_copy[0];
            for (int v = 1; v < VOCAB_SIZE; v++) {
                if (logits_copy[v] > max_val) { max_val = logits_copy[v]; max_v = v; }
            }
            threshold = max_val;
            logits_copy[max_v] = -1e30f;  /* remove so next pass finds next-best */
        }

        /* Top-p (nucleus) filtering — applied AFTER top-k, BEFORE softmax.
         * Per LAL-Bot's agreement: "--top-p 0.9, 和 top-k 40 一起用 (先 top-k 再 top-p)".
         * Algorithm (HF-style):
         *   1. Collect the top-k logits and their token indices.
         *   2. Sort them descending by logit.
         *   3. Compute softmax (with temperature) over the sorted set.
         *   4. Accumulate probs until cumulative >= top_p; keep those, drop the rest.
         * This further narrows the sampling set to the "nucleus" of high-prob tokens,
         * which avoids sampling very-low-prob tokens that still made it into top-k.
         * g_top_p == 0 means disabled (keep all top-k). */
        int keep_mask[50257];
        for (int v = 0; v < VOCAB_SIZE; v++) keep_mask[v] = 0;
        if (g_top_p > 0.0f && g_top_p < 1.0f) {
            /* Gather indices of tokens that survived top-k. */
            static int top_idx[50257];
            static float top_val[50257];
            int n_top = 0;
            for (int v = 0; v < VOCAB_SIZE; v++) {
                if (g_logits[v] >= threshold) {
                    top_idx[n_top] = v;
                    top_val[n_top] = g_logits[v];
                    n_top++;
                }
            }
            /* Find actual max for numerical stability of softmax. */
            float actual_max = -1e30f;
            for (int i = 0; i < n_top; i++)
                if (top_val[i] > actual_max) actual_max = top_val[i];

            /* Compute temperature-scaled softmax probs over the top-k set. */
            float sum_exp = 0.0f;
            for (int i = 0; i < n_top; i++) {
                top_val[i] = expf((top_val[i] - actual_max) / g_temperature);
                sum_exp += top_val[i];
            }
            float inv = 1.0f / (sum_exp + 1e-12f);
            for (int i = 0; i < n_top; i++) top_val[i] *= inv;

            /* Sort descending by prob. n_top is small (<= top_k, typically 40),
             * so a simple insertion sort is fine and avoids qsort overhead. */
            for (int i = 1; i < n_top; i++) {
                float vp = top_val[i]; int vi = top_idx[i];
                int j = i - 1;
                while (j >= 0 && top_val[j] < vp) {
                    top_val[j + 1] = top_val[j];
                    top_idx[j + 1] = top_idx[j];
                    j--;
                }
                top_val[j + 1] = vp;
                top_idx[j + 1] = vi;
            }

            /* Keep tokens until cumulative prob >= top_p. HF keeps the first token
             * that crosses the threshold (so the nucleus is never empty). */
            float cum = 0.0f;
            for (int i = 0; i < n_top; i++) {
                keep_mask[top_idx[i]] = 1;
                cum += top_val[i];
                if (cum >= g_top_p) break;
            }
        } else {
            /* top-p disabled: keep everything that survived top-k. */
            for (int v = 0; v < VOCAB_SIZE; v++)
                if (g_logits[v] >= threshold) keep_mask[v] = 1;
        }

        /* === LAL fusion level 2: .lal logic-layer filter on top-k pool ===
         * Hand keep_mask to the .lal program (if linked) so it can drop
         * tokens violating semantic rules. Weak symbol → no-op when unlinked.
         * Called after top-k/top-p narrowing, before softmax+sample, so
         * banned tokens get zero probability. */
        {
            int last = (g_n_recent > 0) ? g_recent_tokens[g_n_recent - 1] : -1;
            lal_filter_topk(keep_mask, VOCAB_SIZE, last, g_recent_tokens, g_n_recent, _server_decode_for_filter);
            /* Safety: if the filter dropped everything, restore the top-k
             * pool (avoid empty-sample crash). Scan for any survivor. */
            int any_kept = 0;
            for (int v = 0; v < VOCAB_SIZE; v++) if (keep_mask[v]) { any_kept = 1; break; }
            if (!any_kept) {
                for (int v = 0; v < VOCAB_SIZE; v++)
                    if (g_logits[v] >= threshold) keep_mask[v] = 1;
            }
        }

        /* Apply softmax over the surviving (kept) tokens, with temperature. */
        float actual_max = -1e30f;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            if (keep_mask[v] && g_logits[v] > actual_max) actual_max = g_logits[v];
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            if (keep_mask[v]) {
                logits_copy[v] = expf((g_logits[v] - actual_max) / g_temperature);
                sum_exp += logits_copy[v];
            } else {
                logits_copy[v] = 0.0f;
            }
        }

        /* Sample from the distribution */
        float r = (float)(rand() % 1000000) / 1000000.0f * sum_exp;
        float cum = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            cum += logits_copy[v];
            if (cum >= r) return v;
        }
        return VOCAB_SIZE - 1;  /* fallback */
    }

    /* Argmax (deterministic, for temperature=0) */
    int best = 0;
    float best_val = g_logits[0];
    for (int v = 1; v < VOCAB_SIZE; v++) {
        if (g_logits[v] > best_val) { best_val = g_logits[v]; best = v; }
    }
    return best;
}

/* === LAL three-layer fusion (level 1): live-model forward entry point ===
 *
 * Prime the model with a prompt string, populating g_final_ln (the final
 * hidden state) so a subsequent lal_server_get_hidden() call returns the
 * model's contextualized representation of this prompt.
 *
 * This is the level-1 entry point for LIVE model testing: a test harness
 * (or, at fusion level 2, the .lal program itself) calls this to run the
 * DNN forward over a prompt, then the .lal logic layer reads the resulting
 * hidden state via runtime_context() / lal_server_get_hidden().
 *
 * Reuses the same prefill path as the HTTP /generate handler: tokenize via
 * encode_text, loop gpt2_forward_token over all prompt tokens (skipping the
 * LM head — we only need the hidden state, not generated tokens). After this
 * call, g_final_ln holds the last prompt token's contextualized vector.
 *
 * For semantically meaningful context (last token attending to the full
 * prompt), run the server with --real-attention or --dflash so the KV cache
 * is populated during prefill. Without real attention, g_final_ln is still a
 * valid 768-dim model output but reflects only the last token in isolation
 * (V-copy attention), not the full prompt context.
 *
 * Returns the number of prompt tokens processed, or -1 if prompt is NULL.
 * The model + tokenizer must be loaded first (via the normal server init). */
int lal_server_forward(const char *prompt) {
    if (!prompt) return -1;
    static int tokens[256];
    int n = encode_text(prompt, tokens, 256);
    if (n == 0) { tokens[0] = 464; n = 1; }  /* fall back to "The" token */
    for (int t = 0; t < n; t++) {
        g_skip_lm_head = (t < n - 1) ? 1 : 0;
        gpt2_forward_token(tokens[t], t);
    }
    g_skip_lm_head = 0;
    return n;
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

        /* Per-request sampling overrides (optional JSON fields).
         * LAL-Bot's CLI flags set the server-wide defaults; these let the
         * web frontend vary temperature/top_k/rep_penalty per request without
         * restarting. We save the CLI defaults and restore them afterwards so
         * a request without these fields keeps using the CLI config. */
        float saved_temp = g_temperature;
        int   saved_topk = g_top_k;
        float saved_topp = g_top_p;
        float saved_rep  = g_rep_penalty;
        p = strstr(body, "\"temperature\"");
        if (p) { p = strchr(p, ':'); if (p) g_temperature = (float)atof(p + 1); }
        if (g_temperature < 0.0f) g_temperature = 0.0f;
        if (g_temperature > 2.0f) g_temperature = 2.0f;
        p = strstr(body, "\"top_k\"");
        if (p) { p = strchr(p, ':'); if (p) g_top_k = atoi(p + 1); }
        if (g_top_k < 0) g_top_k = 0;
        if (g_top_k > VOCAB_SIZE) g_top_k = VOCAB_SIZE;
        p = strstr(body, "\"top_p\"");
        if (p) { p = strchr(p, ':'); if (p) g_top_p = (float)atof(p + 1); }
        if (g_top_p < 0.0f) g_top_p = 0.0f;
        if (g_top_p > 1.0f) g_top_p = 1.0f;
        p = strstr(body, "\"repetition_penalty\"");
        if (p) { p = strchr(p, ':'); if (p) g_rep_penalty = (float)atof(p + 1); }
        if (g_rep_penalty < 1.0f) g_rep_penalty = 1.0f;
        if (g_rep_penalty > 2.0f) g_rep_penalty = 2.0f;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        static int input_tokens[256];
        int n_input = encode_text(prompt, input_tokens, 256);
        if (n_input == 0) { input_tokens[0] = 464; n_input = 1; }

        static int output_tokens[100];
        static int all_tokens[1024];
        memcpy(all_tokens, input_tokens, n_input * sizeof(int));
        int total = n_input;

        /* If real attention is enabled, forward the prompt first to fill
         * the KV cache. The last prompt token's forward gives us the logits
         * for the first generated token, so we start generation from there. */
        int first_gen_position = total - 1;  /* position of last prompt token */
        if (g_srv_real_attention || g_use_dflash) {
            /* Forward prompt tokens to fill KV cache.
             * Optimization: skip LM head for all but the last prompt token
             * (only the last one needs logits for generating the first new
             * token). This saves (n_input-1) × 50257×768 FLOPs. */
            for (int t = 0; t < n_input; t++) {
                g_skip_lm_head = (t < n_input - 1) ? 1 : 0;
                gpt2_forward_token(all_tokens[t], t);
            }
            g_skip_lm_head = 0;  /* restore for decode phase */
            /* After forwarding prompt, KV cache has positions 0..n_input-1.
             * The last forward (position n_input-1) already computed logits
             * for the NEXT token. We can use that directly for gen=0. */
        }

        /* Reset recent token tracking for repetition penalty.
         * Then seed with prompt tokens so that:
         *   (a) rep_penalty can penalize prompt-token repetition (standard HF behavior)
         *   (b) lal_filter_topk's ban_relation can see the prompt text in _recent
         *       (otherwise triggers like "France capital" in the prompt are invisible
         *       to the filter and semantic constraints never fire)
         * The ring buffer is 256; if prompt is longer, keep the last 256-n_tokens
         * prompt tokens so generation still has room. */
        g_n_recent = 0;
        int prompt_to_add = n_input;
        if (prompt_to_add > 256 - n_tokens) prompt_to_add = 256 - n_tokens;
        if (prompt_to_add < 0) prompt_to_add = 0;
        for (int i = n_input - prompt_to_add; i < n_input; i++) {
            g_recent_tokens[g_n_recent++] = all_tokens[i];
        }

        for (int gen = 0; gen < n_tokens; gen++) {
            int pos = first_gen_position + gen;  /* position in sequence */
            int next = gpt2_forward_token(all_tokens[total - 1], pos);
            output_tokens[gen] = next;
            all_tokens[total++] = next;

            /* Track for repetition penalty (ring buffer) */
            if (g_n_recent < 256) {
                g_recent_tokens[g_n_recent++] = next;
            } else {
                /* Shift ring buffer */
                memmove(g_recent_tokens, g_recent_tokens + 1, 255 * sizeof(int));
                g_recent_tokens[255] = next;
            }

            if (total >= 1024) { n_tokens = gen + 1; break; }
        }

        /* Restore CLI-default sampling params (per-request override done). */
        g_temperature = saved_temp;
        g_top_k = saved_topk;
        g_top_p = saved_topp;
        g_rep_penalty = saved_rep;

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
        if (g_early_exit_layers > 0 && g_early_exit_total > 0) {
            fprintf(stderr, "[*] early exit: %d/%d forwards exited early (%.0f%%), layers avg=%.1f/12\n",
                    g_early_exit_hits, g_early_exit_total,
                    100.0f * g_early_exit_hits / g_early_exit_total,
                    g_early_exit_total > 0 ?
                      (float)(g_early_exit_hits * g_early_exit_layers +
                              (g_early_exit_total - g_early_exit_hits) * N_LAYER) / g_early_exit_total
                      : (float)N_LAYER);
        }
        static char json[6144];
        int jpos = snprintf(json, sizeof(json),
            "{\"text\":\"%s\",\"time\":\"%.3f\",\"n_tokens\":%d,\"tokens_per_sec\":\"%.1f\"}",
            escaped, dt, n_tokens, tps);
        char header[256];
        snprintf(header, sizeof(header),
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n", jpos);
        write(client_fd, header, strlen(header));
        write(client_fd, json, jpos);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/stream") == 0) {
        /* Streaming generation via Server-Sent Events.
         * Same params as /generate; emits one data: event per token so the
         * frontend can render text as it is produced. */
        char *body = strstr(buf, "\r\n\r\n");
        if (!body) { close(client_fd); return; }
        body += 4;

        static char prompt[1024];
        prompt[0] = '\0';
        int n_tokens = 20;
        char *p = strstr(body, "\"prompt\"");
        if (p) { p = strchr(p, ':'); if (p) { p = strchr(p, '"'); if (p) { p++;
            int i = 0; while (*p && *p != '"' && i < 1023) { if (*p == '\\') p++; prompt[i++] = *p++; }
            prompt[i] = '\0'; } } }
        p = strstr(body, "\"n_tokens\"");
        if (p) { p = strchr(p, ':'); if (p) n_tokens = atoi(p + 1); }
        if (n_tokens < 1) n_tokens = 1; if (n_tokens > 100) n_tokens = 100;

        float saved_temp = g_temperature, saved_rep = g_rep_penalty;
        int   saved_topk = g_top_k;
        float saved_topp = g_top_p;
        p = strstr(body, "\"temperature\"");
        if (p) { p = strchr(p, ':'); if (p) g_temperature = (float)atof(p + 1); }
        if (g_temperature < 0) g_temperature = 0; if (g_temperature > 2) g_temperature = 2;
        p = strstr(body, "\"top_k\"");
        if (p) { p = strchr(p, ':'); if (p) g_top_k = atoi(p + 1); }
        if (g_top_k < 0) g_top_k = 0; if (g_top_k > VOCAB_SIZE) g_top_k = VOCAB_SIZE;
        p = strstr(body, "\"top_p\"");
        if (p) { p = strchr(p, ':'); if (p) g_top_p = (float)atof(p + 1); }
        if (g_top_p < 0) g_top_p = 0; if (g_top_p > 1) g_top_p = 1;
        p = strstr(body, "\"repetition_penalty\"");
        if (p) { p = strchr(p, ':'); if (p) g_rep_penalty = (float)atof(p + 1); }
        if (g_rep_penalty < 1) g_rep_penalty = 1; if (g_rep_penalty > 2) g_rep_penalty = 2;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        static int input_tokens[256];
        int n_input = encode_text(prompt, input_tokens, 256);
        if (n_input == 0) { input_tokens[0] = 464; n_input = 1; }
        static int all_tokens[1024];
        memcpy(all_tokens, input_tokens, n_input * sizeof(int));
        int total = n_input;

        /* Send SSE headers + the prompt as the first chunk. */
        const char *sse_hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        write(client_fd, sse_hdr, strlen(sse_hdr));

        /* Emit prompt text so the frontend shows it immediately. */
        {
            static char esc[2048]; int ei = 0;
            for (int i = 0; prompt[i] && ei < 2030; i++) {
                if (prompt[i]=='"') { esc[ei++]='\\'; esc[ei++]='"'; }
                else if (prompt[i]=='\\') { esc[ei++]='\\'; esc[ei++]='\\'; }
                else if (prompt[i]=='\n') { esc[ei++]='\\'; esc[ei++]='n'; }
                else if (prompt[i]=='\r') continue;
                else esc[ei++] = prompt[i];
            }
            esc[ei] = '\0';
            static char ev[2200];
            int el = snprintf(ev, sizeof(ev), "data: {\"prompt\":\"%s\"}\n\n", esc);
            write(client_fd, ev, el);
        }

        /* Prefill KV cache for real attention (same as /generate). */
        int first_gen_position = total - 1;
        if (g_srv_real_attention || g_use_dflash) {
            for (int t = 0; t < n_input; t++)
                gpt2_forward_token(all_tokens[t], t);
        }
        g_n_recent = 0;

        /* Generate + stream one event per token. */
        for (int gen = 0; gen < n_tokens; gen++) {
            int pos = first_gen_position + gen;
            int next = gpt2_forward_token(all_tokens[total - 1], pos);
            all_tokens[total++] = next;
            if (g_n_recent < 256) g_recent_tokens[g_n_recent++] = next;
            else { memmove(g_recent_tokens, g_recent_tokens + 1, 255 * sizeof(int)); g_recent_tokens[255] = next; }

            static char tok[256];
            int tl = decode_token(next, tok, sizeof(tok) - 1);
            tok[tl] = '\0';
            /* JSON-escape the token fragment. */
            static char esc[512]; int ei = 0;
            for (int i = 0; i < tl && ei < 500; i++) {
                if (tok[i]=='"') { esc[ei++]='\\'; esc[ei++]='"'; }
                else if (tok[i]=='\\') { esc[ei++]='\\'; esc[ei++]='\\'; }
                else if (tok[i]=='\n') { esc[ei++]='\\'; esc[ei++]='n'; }
                else if (tok[i]=='\r') continue;
                else esc[ei++] = tok[i];
            }
            esc[ei] = '\0';
            static char ev[600];
            int el = snprintf(ev, sizeof(ev), "data: {\"token\":\"%s\",\"n\":%d}\n\n", esc, gen);
            write(client_fd, ev, el);
            if (total >= 1024) { n_tokens = gen + 1; break; }
        }

        g_temperature = saved_temp; g_top_k = saved_topk; g_top_p = saved_topp; g_rep_penalty = saved_rep;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        double tps = n_tokens / (dt > 0 ? dt : 1e-6);
        static char done_ev[256];
        int el = snprintf(done_ev, sizeof(done_ev),
            "data: {\"done\":true,\"time\":\"%.3f\",\"n_tokens\":%d,\"tokens_per_sec\":\"%.1f\"}\n\n",
            dt, n_tokens, tps);
        write(client_fd, done_ev, el);
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
    srand(time(NULL));  /* seed for top-k sampling */

    int port = 8080;
    int use_binary = 0;
    /* Parse args: [port] [--binary] [--prune-vocab FRAC]
     * --binary: use XNOR+popcount binary mode (13MB weights, 32x fewer FLOPs)
     * FRAC in [0, 0.9]: fraction of vocab (smallest-norm rows) to drop. */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--binary") == 0) {
            use_binary = 1;
        } else if (strcmp(argv[i], "--mixed-precision") == 0) {
            /* Mixed-precision: implies --binary, but keeps first/last layers
             * float (standard XNOR-Net quality fix). Needs BOTH gpt2_binary.bin
             * (for the middle binary layers + embeddings) and gpt2_weights.bin
             * (for the 2 float layers). */
            use_binary = 1;
            g_mixed_precision = 1;
        } else if (strcmp(argv[i], "--real-attention") == 0) {
            g_srv_real_attention = 1;
        } else if (strcmp(argv[i], "--vcopy") == 0) {
            /* Legacy V-copy attention path: attn_out = V (no Q·K context).
             * Context-free → repetition loops. Kept ONLY for benchmark
             * comparison against the now-default real attention path. */
            g_srv_real_attention = 0;
        } else if (strcmp(argv[i], "--dflash") == 0) {
            g_use_dflash = 1;
            g_srv_real_attention = 1;  /* dflash implies real attention (needs KV cache) */
        } else if (strcmp(argv[i], "--bwn") == 0) {
            g_use_bwn = 1;
        } else if (strcmp(argv[i], "--int8") == 0) {
            g_use_bwn = 1;  /* int8 implies bwn */
            g_use_int8 = 1;
        } else if (strcmp(argv[i], "--mixed-int8") == 0 && i+1 < argc) {
            /* Mixed precision: first N layers int8 (fast), rest BWN (precise).
             * Int8 quantization error compounds across layers; using BWN for
             * later layers prevents quality degradation. --mixed-int8 6 = layers
             * 0-5 int8, layers 6-11 BWN. Expected ~15 tok/s + BWN-quality output. */
            g_use_bwn = 1;
            g_use_int8 = 1;
            g_mixed_int8_layers = atoi(argv[++i]);
            printf("[*] mixed int8: layers 0-%d int8, layers %d-%d BWN\n",
                   g_mixed_int8_layers - 1, g_mixed_int8_layers, N_LAYER - 1);
        } else if (strcmp(argv[i], "--early-exit") == 0 && i+1 < argc) {
            /* Bit-width speculative decoding: exit after N layers if residual
             * stream has converged. --early-exit 8 = check at layer 8, skip
             * layers 8-11 if confident. 0 = disabled (default). */
            g_early_exit_layers = atoi(argv[++i]);
            printf("[*] early exit: check convergence at layer %d (threshold=%.3f)\n",
                   g_early_exit_layers, g_early_exit_threshold);
        } else if (strcmp(argv[i], "--early-exit-threshold") == 0 && i+1 < argc) {
            g_early_exit_threshold = atof(argv[++i]);
            printf("[*] early exit threshold set to %.4f\n", g_early_exit_threshold);
        } else if (strcmp(argv[i], "--turboquant") == 0) {
            g_use_turboquant = 1;
        } else if (strcmp(argv[i], "--lm-head-int8") == 0) {
            g_use_lm_head_int8 = 1;
        } else if (strcmp(argv[i], "--ternary-act") == 0) {
            /* Ternary activation for binary layers: |x| < threshold*mean|x|
             * is zeroed instead of forced to ±1. Bi-Real Net style. Only
             * affects binary layers (in --binary or --mixed-precision mode);
             * float layers are untouched. Composes with STE + mixed-precision. */
            g_ternary_act = 1;
        } else if (strcmp(argv[i], "--ternary-threshold") == 0 && i + 1 < argc) {
            g_ternary_threshold = (float)atof(argv[++i]);
            if (g_ternary_threshold < 0.0f) g_ternary_threshold = 0.0f;
            if (g_ternary_threshold > 2.0f) g_ternary_threshold = 2.0f;
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            g_temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            g_top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            g_top_p = (float)atof(argv[++i]);
            if (g_top_p < 0.0f) g_top_p = 0.0f;
            if (g_top_p > 1.0f) g_top_p = 1.0f;
        } else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
            g_rep_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--prune-vocab") == 0 && i + 1 < argc) {
            g_prune_frac = (float)atof(argv[++i]);
            if (g_prune_frac < 0.0f) g_prune_frac = 0.0f;
            if (g_prune_frac > 0.9f) g_prune_frac = 0.9f;
        } else {
            port = atoi(argv[i]);
        }
    }

    /* Auto-detect thread count.
     * On ARM (both ARMv7 and AArch64), pthread + NEON in lm_head_avx2
     * crashes. Root cause unclear (possibly PRoot interference, or NEON
     * register save/restore issue). Force single-threaded on all ARM.
     * x86_64 works fine with multi-threading. */
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
#if defined(__arm__) || defined(__aarch64__)
    g_n_threads = 1;  /* ARM: single-threaded (pthread+NEON crashes) */
#else
    g_n_threads = (ncpu >= 4) ? (int)ncpu : 1;
    if (g_n_threads > 8) g_n_threads = 8;
#endif

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

        /* Mixed-precision: also load float weights for layers 0 and N_LAYER-1.
         * Reads the full float file transiently (~497 MB peak), copies the 2
         * layers (~56 MB) into g_layers[0]/g_layers[N_LAYER-1], then frees the
         * rest. Resident memory after: binary(13MB) + 2 float layers(56MB) +
         * wte(154MB) + int8 LM head(38MB) ≈ 261 MB. */
        if (g_mixed_precision) {
            /* Prefer the small subset file (~54 MB) if available — generated by
             * scripts/extract_float_subset.c. Falls back to the full 498 MB
             * file (LAL_WEIGHTS or default). The subset is essential on
             * memory-constrained devices (tablets) where downloading/storing
             * the full float file is impractical. */
            const char *fw = getenv("LAL_FLOAT_SUBSET");
            if (!fw) fw = getenv("LAL_WEIGHTS");
            if (!fw) fw = "prebuilt/gpt2_weights.bin";
            printf("[*] mixed-precision: loading float layers {0, %d} from %s...\n",
                   N_LAYER - 1, fw);
            fflush(stdout);
            int keep[2] = {0, N_LAYER - 1};
            struct timespec tm0, tm1;
            clock_gettime(CLOCK_MONOTONIC, &tm0);
            if (load_float_layers_subset(fw, keep, 2) != 0) {
                fprintf(stderr, "[!] mixed-precision load failed; falling back to pure binary\n");
                g_mixed_precision = 0;
            }
            clock_gettime(CLOCK_MONOTONIC, &tm1);
            double mdt = (tm1.tv_sec - tm0.tv_sec) + (tm1.tv_nsec - tm0.tv_nsec) * 1e-9;
            printf("[*] mixed-precision float layers loaded in %.2fs (layers 0 and %d)\n",
                   mdt, N_LAYER - 1);
            fflush(stdout);
        }
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

    /* Quantize wte to int8 + per-row scale for the int8 LM head.
     * Works in BOTH float and binary modes — both load g_wte as float (the
     * embeddings stay unbinarized; only the 12 transformer layers are
     * binarized in binary mode). One-time cost at startup (~50ms for
     * 50257×768). g_wte stays float for the embedding lookup; g_wte_q is
     * used only by lm_head_int8_parallel.
     *
     * Binary mode is where this matters most: the LM head (logits = wte @ x)
     * reads 154 MB of float wte per token, and on memory-constrained devices
     * (ARM tablets, LPDDR3) that bandwidth dominates. int8 cuts it to 38 MB
     * → ~4x less bandwidth on the single hottest path. LAL-Bot confirmed
     * the tablet's 0.2 tok/s bottleneck is the LM head, not attention. */
    if (g_use_lm_head_int8 && g_wte) {
        struct timespec tq0, tq1;
        clock_gettime(CLOCK_MONOTONIC, &tq0);
        g_wte_q = malloc((size_t)VOCAB_SIZE * N_EMBD * sizeof(int8_t));
        g_wte_scale = malloc((size_t)VOCAB_SIZE * sizeof(float));
        if (!g_wte_q || !g_wte_scale) {
            fprintf(stderr, "[!] OOM allocating int8 LM head cache; disabling --lm-head-int8\n");
            g_use_lm_head_int8 = 0;
        } else {
            for (int v = 0; v < VOCAB_SIZE; v++) {
                const float *row = g_wte + (size_t)v * N_EMBD;
                g_wte_scale[v] = lm_head_quantize_x(row, g_wte_q + (size_t)v * N_EMBD, N_EMBD);
            }
            clock_gettime(CLOCK_MONOTONIC, &tq1);
            double qt = (tq1.tv_sec - tq0.tv_sec) + (tq1.tv_nsec - tq0.tv_nsec) * 1e-9;
            printf("[*] LM head int8 quantized: %.1f MB int8 + %.1f KB scales (quantize %.0f ms)\n",
                   (double)VOCAB_SIZE * N_EMBD / (1024.0 * 1024.0),
                   (double)VOCAB_SIZE * sizeof(float) / 1024.0,
                   qt * 1000.0);
        }
    }

    g_logits = NULL;
    if (posix_memalign((void **)&g_logits, 32, VOCAB_SIZE * sizeof(float)) != 0 || !g_logits) {
        fprintf(stderr, "[!] OOM allocating logits\n"); return 1;
    }

    /* Allocate KV cache for real attention.
     * Float: 72 MB (12 layers × 2 × 1024 × 768 × 4B)
     * TurboQuant (int8): 18 MB (4x compression) + 96 KB for per-row scales */
    if (g_srv_real_attention || g_use_dflash) {
        if (g_use_turboquant) {
            printf("[*] allocating TurboQuant KV cache (int8, 18 MB)...\n");
        } else {
            printf("[*] allocating KV cache for real causal attention (72 MB)...\n");
        }
        fflush(stdout);
        for (int l = 0; l < N_LAYER; l++) {
            if (g_use_turboquant) {
                g_k_cache_q[l] = malloc((size_t)1024 * N_EMBD * sizeof(int8_t));
                g_v_cache_q[l] = malloc((size_t)1024 * N_EMBD * sizeof(int8_t));
                g_k_scale[l] = malloc((size_t)1024 * sizeof(float));
                g_v_scale[l] = malloc((size_t)1024 * sizeof(float));
                if (!g_k_cache_q[l] || !g_v_cache_q[l] || !g_k_scale[l] || !g_v_scale[l]) {
                    fprintf(stderr, "[!] OOM allocating TurboQuant KV cache for layer %d\n", l);
                    g_use_turboquant = 0;
                    break;
                }
            } else {
                g_k_cache[l] = malloc((size_t)1024 * N_EMBD * sizeof(float));
                g_v_cache[l] = malloc((size_t)1024 * N_EMBD * sizeof(float));
                if (!g_k_cache[l] || !g_v_cache[l]) {
                    fprintf(stderr, "[!] OOM allocating KV cache for layer %d\n", l);
                    g_srv_real_attention = 0;
                    break;
                }
            }
        }
        if (g_srv_real_attention || g_use_dflash) {
            if (g_use_turboquant) {
                printf("[*] TurboQuant KV cache allocated — int8 quantized attention\n");
            } else {
                printf("[*] KV cache allocated — real causal self-attention enabled\n");
            }
        }
        fflush(stdout);
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
