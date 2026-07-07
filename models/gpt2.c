/* models/gpt2.c — GPT-2 model definition (just config + main, ~30 lines of model code)
 *
 * Uses lal_runtime Level 3 API (model_load/forward/backward).
 * All model-agnostic logic is in runtime/lal_runtime.c.
 *
 * Build: gcc -O3 -mavx2 -o gpt2 models/gpt2.c runtime/lal_runtime.c -lm
 * Run:   ./gpt2 10000 0.05
 */
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <string.h>
#include "../runtime/lal_runtime.h"

/* === GPT-2 config — this is the ONLY model-specific code === */
static ModelConfig gpt2_config(void) {
    return (ModelConfig){
        .n_layer = 12, .n_embd = 768, .n_head = 12,
        .n_ctx = 1024, .vocab_size = 50257, .mlp_dim = 3072,
        .norm_type = NORM_LAYER, .attn_type = ATTN_LEARNED,
        .act_type = ACT_GELU, .residual_scale = 0.5f,
        .qkv_merged = 1,
    };
}

/* === Training data (GPT-2 specific) === */
static const char *TEXTS[] = {
    "The capital of France is Paris.", "The capital of Japan is Tokyo.",
    "The capital of Germany is Berlin.", "Hello, how are you doing today?",
    "Once upon a time, there was a kingdom.", "The weather today is sunny and warm.",
    "Machine learning is a subset of AI.", "The world is a place of great beauty.",
    "I think, therefore I am.", "Knowledge is power.",
};

static int encode(const char *t, int *tk) {
    if(strstr(t,"France")){tk[0]=464;tk[1]=3139;tk[2]=286;tk[3]=4881;tk[4]=318;tk[5]=6751;tk[6]=13;return 7;}
    if(strstr(t,"Japan")){tk[0]=464;tk[1]=3139;tk[2]=286;tk[3]=3273;tk[4]=318;tk[5]=32817;tk[6]=13;return 7;}
    if(strstr(t,"Germany")){tk[0]=464;tk[1]=3139;tk[2]=286;tk[3]=3536;tk[4]=318;tk[5]=5948;tk[6]=13;return 7;}
    if(strstr(t,"Hello")){tk[0]=15496;tk[1]=11;tk[2]=703;tk[3]=389;tk[4]=318;tk[5]=688;tk[6]=981;return 7;}
    if(strstr(t,"Once")){tk[0]=3753;tk[1]=703;tk[2]=403;tk[3]=640;tk[4]=11;tk[5]=621;tk[6]=7530;return 7;}
    if(strstr(t,"weather")){tk[0]=464;tk[1]=3749;tk[2]=3284;tk[3]=318;tk[4]=20011;tk[5]=290;tk[6]=4932;return 7;}
    if(strstr(t,"Machine")){tk[0]=11510;tk[1]=4673;tk[2]=318;tk[3]=257;tk[4]=10666;tk[5]=295;tk[6]=8552;return 7;}
    if(strstr(t,"world")){tk[0]=464;tk[1]=995;tk[2]=318;tk[3]=257;tk[4]=1639;tk[5]=286;tk[6]=869;return 7;}
    if(strstr(t,"think")){tk[0]=40;tk[1]=1037;tk[2]=11;tk[3]=1779;tk[4]=314;tk[5]=559;tk[6]=13;return 7;}
    if(strstr(t,"Knowledge")){tk[0]=18681;tk[1]=318;tk[2]=2685;tk[3]=13;return 4;}
    tk[0]=464;tk[1]=995;tk[2]=318;tk[3]=257;tk[4]=1639;tk[5]=286;tk[6]=869;return 7;
}

/* Export fine-tuned binary weights to GB2L file (same format as gpt2_binary.bin).
 * After STE training, wbits/alpha/bias have been updated. Export them so the
 * inference server can load the improved weights. */
static void export_binary_weights(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[!] cannot open %s for writing\n", path); return; }

    int n_layer = m->cfg.n_layer;
    int n_embd = m->cfg.n_embd;
    int mlp_dim = m->cfg.mlp_dim;
    int vocab = m->cfg.vocab_size;
    int n_ctx = m->cfg.n_ctx;

    /* Header */
    fwrite("GB2L", 1, 4, f);
    int hdr[5] = {n_layer, n_embd, mlp_dim, vocab, n_ctx};
    fwrite(hdr, 4, 5, f);

    /* Embeddings (float — from original model, not binarized) */
    fwrite(m->wte, sizeof(float), (size_t)vocab * n_embd, f);
    fwrite(m->wpe, sizeof(float), (size_t)n_ctx * n_embd, f);
    fwrite(m->ln_f_w, sizeof(float), n_embd, f);
    fwrite(m->ln_f_b, sizeof(float), n_embd, f);

    /* Per-layer binary weights (updated by STE) */
    for (int l = 0; l < n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        /* LayerNorm (float, not updated) */
        fwrite(tl->norm1_w, sizeof(float), n_embd, f);
        fwrite(tl->norm1_b, sizeof(float), n_embd, f);
        fwrite(tl->norm2_w, sizeof(float), n_embd, f);
        fwrite(tl->norm2_b, sizeof(float), n_embd, f);

        /* 4 binary matrices per layer */
        BinLayer *mats[4];
        if (m->cfg.qkv_merged) {
            mats[0] = &tl->attn_q;  /* merged QKV */
            mats[1] = &tl->attn_o;
            mats[2] = &tl->mlp_gate;
            mats[3] = &tl->mlp_down;
        } else {
            mats[0] = &tl->attn_q;
            mats[1] = &tl->attn_o;
            mats[2] = &tl->mlp_gate;
            mats[3] = &tl->mlp_down;
        }

        for (int mi = 0; mi < 4; mi++) {
            BinLayer *bl = mats[mi];
            int mhdr[4] = {bl->out_dim, bl->in_dim, bl->n_words, 0};
            fwrite(mhdr, 4, 4, f);
            /* wbits: [out_dim * n_words] uint64s */
            fwrite(bl->wbits, sizeof(uint64_t),
                   (size_t)bl->out_dim * bl->n_words, f);
            /* alpha + bias */
            fwrite(bl->alpha, sizeof(float), bl->out_dim, f);
            fwrite(bl->bias, sizeof(float), bl->out_dim, f);
        }
    }

    fclose(f);
    printf("[*] exported STE-tuned binary weights to %s\n", path);
}

int main(int argc, char **argv) {
    int n_steps = argc > 1 ? atoi(argv[1]) : 200;
    float lr = argc > 2 ? atof(argv[2]) : 0.05;
    int use_ste = 0;
    const char *export_path = NULL;
    /* Parse flags */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--ste") == 0) use_ste = 1;
        else if (strcmp(argv[i], "--export") == 0 && i + 1 < argc) export_path = argv[++i];
    }

    printf("[*] LAL Training — GPT-2 (model-agnostic runtime, no PyTorch)\n");
    printf("[*] steps:%d lr:%f ste:%d\n", n_steps, lr, use_ste);
    if (use_ste) {
        g_use_ste = 1;
        printf("[*] STE mode: binary weights will be updated via Straight-Through Estimator\n");
    }
    if (export_path) {
        printf("[*] will export tuned weights to %s after training\n", export_path);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    Model model;
    const char *weight_path = getenv("LAL_WEIGHTS");
    if (!weight_path) weight_path = "prebuilt/gpt2_weights.bin";
    model_load(&model, weight_path, gpt2_config(), "h.%d.", 1);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[*] load: %.1fs\n", (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    printf("[*] training...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int step = 0; step < n_steps; step++) {
        int ti = step % 10;
        int tokens[16]; int n = encode(TEXTS[ti], tokens);
        int target = tokens[n-1];
        int tt[16]; memcpy(tt, tokens, (n-1)*sizeof(int)); tt[n-1] = target;
        float loss = model_forward(&model, tt, n-1);
        model_backward(&model, tt, n-1, lr);
        if (step % 20 == 0)
            printf("  step %4d  loss=%.4f  \"%s\"\n", step, loss, TEXTS[ti]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("[*] done in %.1fs (%.1f ms/step)\n", dt, dt/n_steps*1000);
    printf("[*] no PyTorch, no Python, pure C\n");

    /* Export STE-tuned weights if requested */
    if (export_path) {
        export_binary_weights(&model, export_path);
    }

    model_free(&model);
    return 0;
}
