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
static void *g_hook_handle = NULL;

static int hook_load(const char *path) {
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "[lal] dlopen %s: %s\n", path, dlerror()); return -1; }
    lal_layer_hook_fn f = (lal_layer_hook_fn)dlsym(h, "lal_layer_hook");
    if (!f) { fprintf(stderr, "[lal] dlsym lal_layer_hook: %s\n", dlerror()); dlclose(h); return -1; }
    g_hook_handle = h; g_hook = f;
    printf("[*] LAL layer hook loaded: %s\n", path);
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

static void generate(void) {
    int cap = G_T + g_n + 8;
    int *ids = malloc(sizeof(int) * cap);
    int len = 0;
    for (const char *p = g_prompt; *p; p++) {
        int c = (unsigned char)*p;
        if (c < 256 && G_CHAR2I[c] >= 0) ids[len++] = G_CHAR2I[c];
    }
    if (len == 0) { fprintf(stderr, "[!] prompt has no known chars\n"); free(ids); return; }

    float *x = malloc(sizeof(float) * G_T * G_D);
    float *h = malloc(sizeof(float) * G_T * G_D);
    float *q = malloc(sizeof(float) * G_T * G_D);
    float *k = malloc(sizeof(float) * G_T * G_D);
    float *v = malloc(sizeof(float) * G_T * G_D);
    float *att = malloc(sizeof(float) * G_T * G_D);
    float *scores = malloc(sizeof(float) * G_T * G_T);
    float *m = malloc(sizeof(float) * G_T * (4*G_D));
    float *lm = malloc(sizeof(float) * G_V);
    const float *tok_emb = T2("tok_emb");
    const float *pos_emb = T2("pos_emb");
    const float *ln_f_w = T2("ln_f_w");
    const float *ln_f_b = T2("ln_f_b");
    int hd = G_D / G_H;

    printf(">> ");
    fflush(stdout);

    for (int step = 0; step < g_n; step++) {
        int ctx = len < G_T ? len : G_T;
        int start = len - ctx;
        /* embed */
        for (int t = 0; t < ctx; t++) {
            int tok = ids[start + t];
            for (int i = 0; i < G_D; i++)
                x[t*G_D + i] = tok_emb[tok*G_D + i] + pos_emb[t*G_D + i];
        }

        for (int l = 0; l < G_L; l++) {
            char kb[80];
            #define LK(name) (snprintf(kb, sizeof(kb), "h.%d." name, l), T2(kb))
            const float *ln1_w = LK("ln1_w"), *ln1_b = LK("ln1_b");
            const float *attn_q = LK("attn_q"), *attn_k = LK("attn_k"), *attn_v = LK("attn_v"), *attn_o = LK("attn_o");
            const float *ln2_w = LK("ln2_w"), *ln2_b = LK("ln2_b");
            const float *mlp_fc_w = LK("mlp_fc_w"), *mlp_fc_b = LK("mlp_fc_b");
            const float *mlp_proj_w = LK("mlp_proj_w"), *mlp_proj_b = LK("mlp_proj_b");
            #undef LK

            /* LN1 */
            for (int t = 0; t < ctx; t++) layernorm(x + t*G_D, ln1_w, ln1_b, h + t*G_D, G_D);
            /* Q,K,V */
            for (int t = 0; t < ctx; t++) {
                matmul(h + t*G_D, attn_q, q + t*G_D, G_D, G_D);
                matmul(h + t*G_D, attn_k, k + t*G_D, G_D, G_D);
                matmul(h + t*G_D, attn_v, v + t*G_D, G_D, G_D);
            }
            /* attention per head */
            for (int hh = 0; hh < G_H; hh++) {
                for (int qi = 0; qi < ctx; qi++) {
                    for (int ki = 0; ki < ctx; ki++) {
                        float s = 0;
                        for (int d = 0; d < hd; d++)
                            s += q[qi*G_D + hh*hd + d] * k[ki*G_D + hh*hd + d];
                        s /= sqrtf((float)hd);
                        if (ki > qi) s = -1e30f;   /* causal mask */
                        scores[qi*G_T + ki] = s;
                    }
                    /* softmax over ki */
                    float mx = -1e30f; for (int ki = 0; ki < ctx; ki++) mx = fmaxf(mx, scores[qi*G_T+ki]);
                    float sum = 0; for (int ki = 0; ki < ctx; ki++) { float e = expf(scores[qi*G_T+ki]-mx); scores[qi*G_T+ki]=e; sum+=e; }
                    for (int ki = 0; ki < ctx; ki++) scores[qi*G_T+ki] /= sum;
                    /* weighted sum of v */
                    for (int d = 0; d < hd; d++) {
                        float acc = 0;
                        for (int ki = 0; ki < ctx; ki++) acc += scores[qi*G_T+ki] * v[ki*G_D + hh*hd + d];
                        att[qi*G_D + hh*hd + d] = acc;
                    }
                }
            }
            /* attn_o + residual */
            for (int t = 0; t < ctx; t++) {
                float out[G_D];
                matmul(att + t*G_D, attn_o, out, G_D, G_D);
                for (int i = 0; i < G_D; i++) x[t*G_D + i] += out[i];
            }
            /* LN2 + MLP(gelu) */
            for (int t = 0; t < ctx; t++) layernorm(x + t*G_D, ln2_w, ln2_b, h + t*G_D, G_D);
            for (int t = 0; t < ctx; t++) {
                matmul(h + t*G_D, mlp_fc_w, m + t*(4*G_D), G_D, 4*G_D);
                for (int i = 0; i < 4*G_D; i++) m[t*(4*G_D)+i] = gelu(m[t*(4*G_D)+i] + mlp_fc_b[i]);
                float out[G_D];
                matmul(m + t*(4*G_D), mlp_proj_w, out, 4*G_D, G_D);
                for (int i = 0; i < G_D; i++) x[t*G_D + i] += out[i] + mlp_proj_b[i];
            }

            /* === LAL level-1: per-layer hook (read / steering write) === */
            if (g_hook) g_hook(l, x + (ctx-1)*G_D, G_D);
            else lal_layer_hook(l, x + (ctx-1)*G_D, G_D);
        }

        /* final LN + tied lm head on last position */
        float ln[G_D];
        layernorm(x + (ctx-1)*G_D, ln_f_w, ln_f_b, ln, G_D);
        for (int vv = 0; vv < G_V; vv++) {
            float s = 0; const float *e = tok_emb + (size_t)vv * G_D;
            for (int i = 0; i < G_D; i++) s += ln[i] * e[i];
            lm[vv] = s;
        }
        /* greedy argmax */
        int best = 0; float bv = lm[0];
        for (int vv = 1; vv < G_V; vv++) if (lm[vv] > bv) { bv = lm[vv]; best = vv; }
        char c = G_CHARS[best];
        putchar(c); fflush(stdout);
        ids[len++] = best;
        if (c == '.') break;
    }
    printf("\n");

    free(ids); free(x); free(h); free(q); free(k); free(v); free(att);
    free(scores); free(m); free(lm);
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
            if (hook_load(argv[++i]) != 0) fprintf(stderr, "[lal] warning: hook load failed\n");
        }
    }
    load_cfg(cfg);
    load_weights(weights);
    srand((unsigned)time(NULL));
    generate();
    return 0;
}
