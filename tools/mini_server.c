/*
 * mini_server.c — a TINY real transformer inference server used to actually
 * exercise the LAL three-layer fusion LEVEL 1 (per-layer read + activation
 * steering) on a locally-trained model (no internet / no pretrained LLM here).
 *
 * It reuses the EXACT integration contract as qwen_server.c, exercising ALL
 * three LAL fusion levels on a tiny real transformer:
 *   level 1: per-layer lal_layer_hook (steering) + lal_layer_skip (accel)
 *   level 2: lal_filter_topk (logic-layer sampling constraint, no retraining)
 *   level 3: real sampling (temperature + top-k + repetition penalty) so that
 *            level-2 filters have a candidate pool to constrain.
 * The hook/skip/filter are each (a) a weak no-op fallback, or (b) a strong
 * symbol from a .lal-compiled .so loaded at runtime via --lal-steer /
 * --lal-skip / --lal-filter (dlopen), mirroring qwen_server.
 *
 * This is a real model: weights are trained by tools/mini_train.py, real float
 * forward passes run, real sampling happens. It is NOT a mock.
 *
 * Build: make mini-server
 * Run:   ./prebuilt/mini_server --prompt "I feel" --n 40 \
 *          [--lal-steer prebuilt/mini_steer.so] \
 *          [--lal-skip  prebuilt/mini_skip.so] \
 *          [--lal-filter prebuilt/mini_antirepeat.so] \
 *          [--temp 0.9 --topk 8 --rep 1.15]
 */
#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <dlfcn.h>
#include <time.h>

/* ---- config (must match tools/mini_train.py) ---- */
static int G_D = 64, G_L = 4, G_H = 4, G_T = 32, G_V = 0;
static int G_STEER_LAYER = 2;
static int G_STRESS = 0;   /* if >0, run this many transformer layers (reusing
                              real weights cyclically) to make layer-skip
                              acceleration measurable on the tiny local model. */
static char *G_CHARS = NULL;          /* V entries */
static int  *G_CHAR2I = NULL;         /* 256 -> idx */

/* ---- weight storage ---- */
typedef struct { char key[64]; float *data; int ndim; int dim[4]; int nelem; } Tensor;
static Tensor *g_tensors = NULL;
static int g_nt = 0;

static float *tget(const char *k) {
    for (int i = 0; i < g_nt; i++)
        if (strcmp(g_tensors[i].key, k) == 0) return g_tensors[i].data;
    fprintf(stderr, "[!] missing tensor %s\n", k); exit(1);
}
#define T2(name) tget(name)

/* ---- LAL level-1 per-layer hook bridge (identical contract to qwen_server) ---- */
typedef void (*lal_layer_hook_fn)(int layer, float *hidden, int dim);
void lal_layer_hook(int layer, float *hidden, int dim) __attribute__((weak));
void lal_layer_hook(int layer, float *hidden, int dim) {
    (void)layer; (void)hidden; (void)dim;
}
static lal_layer_hook_fn g_hook = NULL;

/* ---- LAL level-1 ACCELERATION: logic-driven layer skip (early-exit) ---- */
/* Optional strong symbol; server calls it once per layer; returning s>0 makes it
 * jump s layers ahead, cutting transformer forward compute. Weak fallback returns 0. */
typedef int (*lal_layer_skip_fn)(int layer, float *hidden, int dim);
int lal_layer_skip(int layer, float *hidden, int dim) __attribute__((weak));
int lal_layer_skip(int layer, float *hidden, int dim) {
    (void)layer; (void)hidden; (void)dim; return 0;
}
/* Optional runtime param override, exported by skip .so compiled from a
 * `skip ... when <thr>` directive. NULL if the loaded .so doesn't define it. */
typedef void (*lal_skip_set_params_fn)(int n_ovr, int every_ovr, float thr);
static lal_skip_set_params_fn g_skip_set = NULL;
static int g_skip_n_ovr = -1, g_skip_every_ovr = -1;
static float g_skip_thr = -1.0f;
static lal_layer_skip_fn g_skip = NULL;
static void *g_hook_handle = NULL;

/* ---- LAL three-layer fusion (level 2): logic-layer sampling filter bridge ----
 * A .lal `filter` rule compiles to a strong `lal_filter_topk()` symbol that
 * constrains the top-k sampling pool (drops tokens violating logic rules).
 * Mirrors qwen_server.c's level-2 wiring. The weak fallback is a no-op so the
 * server samples normally when no .lal filter is linked. */
typedef int (*lal_token_decode_fn)(int token_id, char *out_buf, int max_len);
/* Thin wrapper so a .lal filter can decode our char-level token ids to text. */
static int mini_decode_for_filter(int token_id, char *out_buf, int max_len) {
    if (token_id < 0 || token_id >= G_V || !G_CHARS) { out_buf[0] = 0; return 0; }
    char c = G_CHARS[token_id];
    if (max_len < 2) { out_buf[0] = 0; return 0; }
    out_buf[0] = c; out_buf[1] = 0;
    return 1;
}
int lal_filter_topk(int *keep_mask, int n_vocab, int last_token,
                    const int *recent_tokens, int n_recent,
                    lal_token_decode_fn decode_fn)
    __attribute__((weak));
int lal_filter_topk(int *keep_mask, int n_vocab, int last_token,
                    const int *recent_tokens, int n_recent,
                    lal_token_decode_fn decode_fn) {
    (void)keep_mask; (void)n_vocab; (void)last_token;
    (void)recent_tokens; (void)n_recent; (void)decode_fn;
    return 0;
}
/* Runtime-loaded (dlopen) override; NULL when no --lal-filter was given. */
static int (*g_lal_filter)(int *, int, int, const int *, int,
                           lal_token_decode_fn) = NULL;
static void *g_filter_handle = NULL;
static int lal_filter_load(const char *path) {
    if (!path) return -1;
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "[lal] dlopen %s: %s\n", path, dlerror()); return -1; }
    int (*f)(int *, int, int, const int *, int, lal_token_decode_fn) =
        (int (*)(int *, int, int, const int *, int, lal_token_decode_fn))
            dlsym(h, "lal_filter_topk");
    if (!f) { fprintf(stderr, "[lal] dlsym lal_filter_topk failed: %s\n", dlerror()); dlclose(h); return -1; }
    g_filter_handle = h;
    g_lal_filter = f;
    printf("[*] LAL logic-layer filter loaded: %s\n", path);
    return 0;
}

/* ---- LAL three-layer fusion (level 3): sampling knobs ----
 * Giving the engine a real distribution (temperature + top-k + repetition
 * penalty) is what makes level-2 logic filters meaningful: a filter only has
 * something to *drop* when the pool is wider than a single argmax winner. */
static float g_temperature = 0.9f;
static int   g_top_k = 8;
static float g_rep_penalty = 1.15f;
static int   g_recent[256];
static int   g_n_recent = 0;

/* Load a .lal-compiled .so (steering and/or layer-skip). Probes BOTH optional
 * symbols; the .so may define either or both. */
static int lal_load(const char *path) {
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "[lal] dlopen %s: %s\n", path, dlerror()); return -1; }
    lal_layer_hook_fn f = (lal_layer_hook_fn)dlsym(h, "lal_layer_hook");
    if (f) { g_hook = f; printf("[*] LAL layer hook (steering) loaded: %s\n", path); }
    lal_layer_skip_fn sk = (lal_layer_skip_fn)dlsym(h, "lal_layer_skip");
    if (sk) {
        g_skip = sk;
        printf("[*] LAL layer-skip control loaded: %s\n", path);
        /* optional runtime override of skip n / period / threshold */
        lal_skip_set_params_fn setp = (lal_skip_set_params_fn)dlsym(h, "lal_skip_set_params");
        g_skip_set = setp;
        if (setp) {
            setp(g_skip_n_ovr, g_skip_every_ovr, g_skip_thr);
            if (g_skip_n_ovr >= 0 || g_skip_every_ovr >= 0 || g_skip_thr >= 0.0f)
                printf("[*]   skip override: n=%d every=%d thr=%.3f (runtime)\n",
                       g_skip_n_ovr, g_skip_every_ovr, g_skip_thr);
        }
    }
    if (!f && !sk) { fprintf(stderr, "[lal] %s defines neither lal_layer_hook nor lal_layer_skip\n", path); dlclose(h); return -1; }
    g_hook_handle = h;
    return 0;
}

/* ---- math helpers ---- */
static void layernorm(const float *x, const float *w, const float *b, float *out, int n) {
    float mu = 0; for (int i = 0; i < n; i++) mu += x[i]; mu /= n;
    float var = 0; for (int i = 0; i < n; i++) { float d = x[i] - mu; var += d*d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mu) * inv * w[i] + b[i];
}
static float gelu(float x) {
    /* exact GELU (erf), matches torch.nn.functional.gelu default */
    return 0.5f * x * (1.0f + erff(x / (float)M_SQRT2));
}
static void matmul(const float *x, const float *W, float *out, int n, int m) {
    /* out[i] = sum_j x[j]*W[i*n + j]   (W is [m, n] row-major, maps n->m) */
    for (int i = 0; i < m; i++) {
        float s = 0; const float *wr = W + (size_t)i * n;
        for (int j = 0; j < n; j++) s += x[j] * wr[j];
        out[i] = s;
    }
}

/* ---- generation ---- */
static char *g_prompt = "I feel";
static int   g_n = 16;
static int   g_bench = 0;   /* if >0: run exactly this many forward passes, no
                               output / no early-stop, so timing is deterministic. */
static int   g_no_stop = 0;  /* if >0: never early-stop on '.' — generate the full
                               --n tokens, so repetition behavior is observable. */

/* ---- KV-cache forward of a SINGLE token at sequence position `pos` ----
 * Writes this token's K/V into the per-layer cache (Kc/Vc) at [layer][pos],
 * reads the already-cached K/V of all earlier positions for attention, and
 * leaves `x` holding this token's final-layer hidden state. This is the
 * standard transformer KV-cache: each autoregressive step only computes ONE
 * token's worth of matmuls instead of re-computing the whole context, so
 * generation cost grows O(seq) instead of O(seq^2) in the number of layers.
 * Mathematically identical to the full-context recompute (causal mask is
 * implicit: we only ever read cache[0..pos]). */
static void lal_forward_token(float *x, int pos, int eff_L,
                              float *Kc, float *Vc, unsigned char *kcached) {
    int D = G_D, hd = D / G_H;
    /* per-token scratch (sized for max dim; toy D is tiny) */
    static float h[4096], q[4096], att[4096], m[16384], out[4096], scores[4096];
    int l = 0;
    while (l < eff_L) {
        int lw = l % G_L;   /* real weight index (cyclic reuse under --stress-layers) */
        char kb[80];
        #define LK(name) (snprintf(kb, sizeof(kb), "h.%d." name, lw), T2(kb))
        const float *ln1_w = LK("ln1_w"), *ln1_b = LK("ln1_b");
        const float *attn_q = LK("attn_q"), *attn_k = LK("attn_k"), *attn_v = LK("attn_v"), *attn_o = LK("attn_o");
        const float *ln2_w = LK("ln2_w"), *ln2_b = LK("ln2_b");
        const float *mlp_fc_w = LK("mlp_fc_w"), *mlp_fc_b = LK("mlp_fc_b");
        const float *mlp_proj_w = LK("mlp_proj_w"), *mlp_proj_b = LK("mlp_proj_b");
        #undef LK

        layernorm(x, ln1_w, ln1_b, h, D);
        matmul(h, attn_q, q, D, D);
        const float *kbase = Kc + (size_t)l * G_T * D;
        const float *vbase = Vc + (size_t)l * G_T * D;
        float *kt = (float*)kbase + (size_t)pos * D;
        float *vt = (float*)vbase + (size_t)pos * D;
        matmul(h, attn_k, kt, D, D);
        matmul(h, attn_v, vt, D, D);
        kcached[l * G_T + pos] = 1;

        /* attention: single query q over cached keys [0..pos] (causal by construction) */
        for (int hh = 0; hh < G_H; hh++) {
            for (int ki = 0; ki <= pos; ki++) {
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += q[hh*hd + d] * kbase[ki*D + hh*hd + d];
                s /= sqrtf((float)hd);
                scores[ki] = s;
            }
            float mx = -1e30f; for (int ki = 0; ki <= pos; ki++) mx = fmaxf(mx, scores[ki]);
            float sum = 0; for (int ki = 0; ki <= pos; ki++) { float e = expf(scores[ki]-mx); scores[ki]=e; sum+=e; }
            for (int ki = 0; ki <= pos; ki++) scores[ki] /= sum;
            for (int d = 0; d < hd; d++) {
                float acc = 0;
                for (int ki = 0; ki <= pos; ki++) acc += scores[ki] * vbase[ki*D + hh*hd + d];
                att[hh*hd + d] = acc;
            }
        }
        matmul(att, attn_o, out, D, D);
        for (int i = 0; i < D; i++) x[i] += out[i];
        layernorm(x, ln2_w, ln2_b, h, D);
        matmul(h, mlp_fc_w, m, D, 4*G_D);
        for (int i = 0; i < 4*G_D; i++) m[i] = gelu(m[i] + mlp_fc_b[i]);
        matmul(m, mlp_proj_w, out, 4*G_D, D);
        for (int i = 0; i < D; i++) x[i] += out[i] + mlp_proj_b[i];

        /* === LAL level-1: per-layer hook (read / steering write) === */
        if (g_hook) g_hook(l, x, D);
        else lal_layer_hook(l, x, D);

        /* === LAL level-1 ACCELERATION: logic-driven layer skip === */
        int s = g_skip ? g_skip(l, x, D) : 0;
        if (s < 0) s = 0;
        if (l + 1 + s > eff_L) s = eff_L - 1 - l;   /* clamp so we don't skip past the end */
        l += 1 + s;
    }
}

/* Sample next token from lm[] using temperature + top-k + repetition penalty,
 * with an optional LAL logic-layer filter constraining the candidate pool.
 * Returns token id (mirrors qwen_server.c / gpt2_server.c sampling path). */
static int sample_next_token(const float *lm, int last_tok) {
    int n_vocab = G_V;
    /* repetition penalty on recent tokens */
    float *logits = malloc(sizeof(float) * n_vocab);
    for (int v = 0; v < n_vocab; v++) logits[v] = lm[v];
    if (g_rep_penalty > 1.0f) {
        for (int i = 0; i < g_n_recent; i++) {
            int t = g_recent[i];
            if (t >= 0 && t < n_vocab) {
                if (logits[t] > 0) logits[t] /= g_rep_penalty;
                else logits[t] *= g_rep_penalty;
            }
        }
    }

    if (g_temperature <= 0.0f || g_top_k <= 0) {
        /* argmax (greedy) — still passes through the LAL filter so logic rules
         * can steer even a deterministic decode. */
        int best = 0; for (int v = 1; v < n_vocab; v++) if (logits[v] > logits[best]) best = v;
        int km = best;
        if (g_lal_filter) g_lal_filter(&km, n_vocab, last_tok, g_recent, g_n_recent, mini_decode_for_filter);
        else lal_filter_topk(&km, n_vocab, last_tok, g_recent, g_n_recent, mini_decode_for_filter);
        free(logits);
        return km;
    }

    /* top-k threshold via min-heap of size k */
    int top_k = g_top_k; if (top_k > n_vocab) top_k = n_vocab;
    float threshold = -1e30f;
    {
        static float heap_val[256];
        static int   heap_idx[256];
        int heap_n = 0;
        for (int v = 0; v < n_vocab; v++) {
            float val = logits[v];
            if (heap_n < top_k) {
                int c = heap_n++; heap_val[c] = val; heap_idx[c] = v;
                while (c > 0) { int p = (c-1)>>1; if (heap_val[p] <= heap_val[c]) break;
                    float tv = heap_val[p]; heap_val[p] = heap_val[c]; heap_val[c] = tv;
                    int ti = heap_idx[p]; heap_idx[p] = heap_idx[c]; heap_idx[c] = ti; c = p; }
            } else if (val > heap_val[0]) {
                heap_val[0] = val; heap_idx[0] = v; int p = 0;
                for (;;) { int l = 2*p+1, r = 2*p+2, s = p;
                    if (l < heap_n && heap_val[l] < heap_val[s]) s = l;
                    if (r < heap_n && heap_val[r] < heap_val[s]) s = r;
                    if (s == p) break;
                    float tv = heap_val[p]; heap_val[p] = heap_val[s]; heap_val[s] = tv;
                    int ti = heap_idx[p]; heap_idx[p] = heap_idx[s]; heap_idx[s] = ti; p = s; }
            }
        }
        threshold = heap_val[0];
    }

    /* === LAL fusion level 2: logic-layer sampling filter ===
     * Seed keep_mask from the top-k pool, hand it to the .lal filter (if any),
     * keep only tokens surviving BOTH top-k and the logic rules, so banned
     * tokens get zero probability. */
    int *keep_mask = malloc(sizeof(int) * n_vocab);
    for (int v = 0; v < n_vocab; v++) keep_mask[v] = (logits[v] >= threshold) ? 1 : 0;
    {
        int last = last_tok;
        if (g_lal_filter)
            g_lal_filter(keep_mask, n_vocab, last, g_recent, g_n_recent, mini_decode_for_filter);
        else
            lal_filter_topk(keep_mask, n_vocab, last, g_recent, g_n_recent, mini_decode_for_filter);
        int any_kept = 0;
        for (int v = 0; v < n_vocab; v++) if (keep_mask[v]) { any_kept = 1; break; }
        if (!any_kept) for (int v = 0; v < n_vocab; v++) if (logits[v] >= threshold) keep_mask[v] = 1;
    }

    /* softmax over kept tokens, then sample */
    float max_l = -1e30f;
    for (int v = 0; v < n_vocab; v++)
        if (keep_mask[v] && logits[v] >= threshold && logits[v] > max_l) max_l = logits[v];
    float sum = 0;
    float *probs = malloc(sizeof(float) * n_vocab);
    for (int v = 0; v < n_vocab; v++) {
        if (keep_mask[v] && logits[v] >= threshold) {
            probs[v] = expf((logits[v] - max_l) / g_temperature);
            sum += probs[v];
        } else probs[v] = 0;
    }
    int out = 0;
    if (sum <= 0) { /* safety: fall back to argmax over the kept pool */
        float bv = -1e30f;
        for (int v = 0; v < n_vocab; v++) if (keep_mask[v] && logits[v] > bv) { bv = logits[v]; out = v; }
    } else {
        float r = (float)rand() / (float)RAND_MAX * sum;
        float acc = 0;
        for (int v = 0; v < n_vocab; v++) { acc += probs[v]; if (r <= acc) { out = v; break; } }
    }
    free(logits); free(keep_mask); free(probs);
    return out;
}

static void generate(void) {
    int cap = G_T + g_n + 8;
    int *ids = malloc(sizeof(int) * cap);
    int len = 0;
    for (const char *p = g_prompt; *p; p++) {
        int c = (unsigned char)*p;
        if (c < 256 && G_CHAR2I[c] >= 0) ids[len++] = G_CHAR2I[c];
    }
    if (len == 0) { fprintf(stderr, "[!] prompt has no known chars\n"); free(ids); return; }

    int eff_L = G_STRESS > 0 ? G_STRESS : G_L;
    float *Kc = malloc(sizeof(float) * (size_t)eff_L * G_T * G_D);
    float *Vc = malloc(sizeof(float) * (size_t)eff_L * G_T * G_D);
    unsigned char *kcached = calloc((size_t)eff_L * G_T, 1);
    float *x = malloc(sizeof(float) * G_D);
    float *ln = malloc(sizeof(float) * G_D);
    float *lm = malloc(sizeof(float) * G_V);
    const float *tok_emb = T2("tok_emb");
    const float *pos_emb = T2("pos_emb");
    const float *ln_f_w = T2("ln_f_w");
    const float *ln_f_b = T2("ln_f_b");

    /* prefill: forward the prompt tokens one by one, filling the KV cache */
    for (int p = 0; p < len; p++) {
        int pos = p % G_T;
        for (int i = 0; i < G_D; i++)
            x[i] = tok_emb[ids[p]*G_D + i] + pos_emb[pos*G_D + i];
        lal_forward_token(x, pos, eff_L, Kc, Vc, kcached);
    }

    /* seed recent-token memory from the prompt so the LAL filter / rep-penalty
     * see the full context, not just the generated suffix. */
    g_n_recent = 0;
    for (int p = 0; p < len && g_n_recent < (int)(sizeof(g_recent)/sizeof(g_recent[0])); p++)
        g_recent[g_n_recent++] = ids[p];

    printf(">> ");
    fflush(stdout);

    int generated = 0;
    while (generated < g_n) {
        /* final LN + tied lm head on the current last token's hidden state */
        layernorm(x, ln_f_w, ln_f_b, ln, G_D);
        for (int vv = 0; vv < G_V; vv++) {
            float s = 0; const float *e = tok_emb + (size_t)vv * G_D;
            for (int i = 0; i < G_D; i++) s += ln[i] * e[i];
            lm[vv] = s;
        }
        int best = sample_next_token(lm, len > 0 ? ids[len-1] : -1);
        char c = G_CHARS[best];
        if (!g_bench) { putchar(c); fflush(stdout); }
        generated++;
        ids[len++] = best;
        if (g_n_recent < (int)(sizeof(g_recent)/sizeof(g_recent[0]))) g_recent[g_n_recent++] = best;
        if (c == '.' && !g_bench && !g_no_stop) break;
        if (generated >= g_n) break;
        /* forward the just-emitted token to get the hidden state for the next step */
        int pos = (len - 1) % G_T;
        for (int i = 0; i < G_D; i++)
            x[i] = tok_emb[best*G_D + i] + pos_emb[pos*G_D + i];
        lal_forward_token(x, pos, eff_L, Kc, Vc, kcached);
    }
    printf("\n");

    free(ids); free(Kc); free(Vc); free(kcached); free(x); free(ln); free(lm);
}

/* ---- load MINI weights ---- */
static void load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[!] cannot open %s\n", path); exit(1); }
    char magic[4]; fread(magic, 1, 4, f);
    if (memcmp(magic, "MINI", 4) != 0) { fprintf(stderr, "[!] bad magic\n"); exit(1); }
    uint32_t nt; fread(&nt, 4, 1, f);
    g_tensors = calloc(nt, sizeof(Tensor));
    for (uint32_t i = 0; i < nt; i++) {
        uint32_t kl; fread(&kl, 4, 1, f);
        fread(g_tensors[i].key, 1, kl, f); g_tensors[i].key[kl] = 0;
        fread(&g_tensors[i].ndim, 4, 1, f);
        int ne = 1; for (int d = 0; d < g_tensors[i].ndim; d++) { fread(&g_tensors[i].dim[d], 4, 1, f); ne *= g_tensors[i].dim[d]; }
        g_tensors[i].nelem = ne;
        g_tensors[i].data = malloc(sizeof(float) * ne);
        fread(g_tensors[i].data, sizeof(float), ne, f);
    }
    g_nt = (int)nt;
    printf("[*] loaded %d tensors from %s\n", g_nt, path);
}

static void load_cfg(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[!] cannot open cfg %s\n", path); exit(1); }
    /* minimal JSON parse for the fields we need */
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1); fread(buf, 1, sz, f); buf[sz] = 0; fclose(f);
    /* extract D, L, H, T, V, steer_layer by naive scan */
    #define RDINT(name) do { char t[64]; snprintf(t,sizeof t,"\"%s\"", #name); \
        char *p = strstr(buf, t); if (p) { p = strchr(p, ':'); G_##name = atoi(p+1); } } while(0)
    RDINT(D); RDINT(L); RDINT(H); RDINT(T); RDINT(V); RDINT(STEER_LAYER);
    #undef RDINT
    /* chars array */
    char *p = strstr(buf, "\"chars\"");
    if (p) {
        p = strchr(p, '[');
        G_CHARS = malloc(G_V + 1);
        int ci = 0;
        for (char *q = p + 1; *q && *q != ']' && ci < G_V; q++) {
            if (*q == '"') {
                q++; char c = *q; /* single char */
                G_CHARS[ci++] = c;
                while (*q != '"') q++;
            }
        }
        G_V = ci;
    }
    G_CHAR2I = malloc(256 * sizeof(int));
    for (int i = 0; i < 256; i++) G_CHAR2I[i] = -1;
    for (int i = 0; i < G_V; i++) G_CHAR2I[(unsigned char)G_CHARS[i]] = i;
    free(buf);
    printf("[*] cfg: D=%d L=%d H=%d T=%d V=%d steer_layer=%d\n",
           G_D, G_L, G_H, G_T, G_V, G_STEER_LAYER);
}

int main(int argc, char **argv) {
    const char *weights = "prebuilt/mini_model.bin";
    const char *cfg = "prebuilt/mini_model.json";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--weights") && i+1<argc) weights = argv[++i];
        else if (!strcmp(argv[i], "--cfg") && i+1<argc) cfg = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1<argc) g_prompt = argv[++i];
        else if (!strcmp(argv[i], "--n") && i+1<argc) g_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lal-steer") && i+1<argc) {
            if (lal_load(argv[++i]) != 0) fprintf(stderr, "[lal] warning: hook load failed\n");
        }
        else if (!strcmp(argv[i], "--lal-skip") && i+1<argc) {
            if (lal_load(argv[++i]) != 0) fprintf(stderr, "[lal] warning: skip .so load failed\n");
        }
        else if (!strcmp(argv[i], "--lal-filter") && i+1<argc) {
            if (lal_filter_load(argv[++i]) != 0) fprintf(stderr, "[lal] warning: filter .so load failed\n");
        }
        else if (!strcmp(argv[i], "--temp") && i+1<argc) {
            g_temperature = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--topk") && i+1<argc) {
            g_top_k = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--rep") && i+1<argc) {
            g_rep_penalty = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--stress-layers") && i+1<argc) {
            G_STRESS = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--lal-skip-n") && i+1<argc) {
            g_skip_n_ovr = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--lal-skip-every") && i+1<argc) {
            g_skip_every_ovr = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--lal-skip-thr") && i+1<argc) {
            g_skip_thr = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--bench") && i+1<argc) {
            g_bench = atoi(argv[++i]); g_n = g_bench;
        }
        else if (!strcmp(argv[i], "--no-stop") && i+1<argc) {
            g_no_stop = atoi(argv[++i]);
        }
    }
    load_cfg(cfg);
    load_weights(weights);
    srand((unsigned)time(NULL));
    generate();
    return 0;
}
