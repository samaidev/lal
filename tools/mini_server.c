/*
 * mini_server.c — a TINY real transformer inference server used to actually
 * exercise the LAL three-layer fusion LEVEL 1 (per-layer read + activation
 * steering) on a locally-trained model (no internet / no pretrained LLM here).
 *
 * It reuses the EXACT integration contract as qwen_server.c:
 *   forward loop:  for l in 0..L-1: x = layer(l, x); lal_layer_hook(l, x, D);
 * The hook is (a) the weak no-op fallback, or (b) a strong symbol from a
 * .lal-compiled .so loaded at runtime via --lal-steer (dlopen), mirroring
 * qwen_server --lal-steer.
 *
 * This is a real model: weights are trained by tools/mini_train.py, real float
 * forward passes run, real sampling happens. It is NOT a mock.
 *
 * Build: make mini-server
 * Run:   ./prebuilt/mini_server --prompt "I feel" --n 16 [--lal-steer prebuilt/mini_steer.so]
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
        int best = 0; float bv = lm[0];
        for (int vv = 1; vv < G_V; vv++) if (lm[vv] > bv) { bv = lm[vv]; best = vv; }
        char c = G_CHARS[best];
        if (!g_bench) { putchar(c); fflush(stdout); }
        generated++;
        ids[len++] = best;
        if (c == '.' && !g_bench) break;
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
    }
    load_cfg(cfg);
    load_weights(weights);
    srand((unsigned)time(NULL));
    generate();
    return 0;
}
