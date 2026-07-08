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
#include "../scripts/train_data.h"  /* Pre-tokenized corpus from LAL-Dev-B */

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

/* Export binary weights to GB2L / GB2L2 file.
 * After STE training, wbits/alpha/bias have been updated (w_float via STE).
 *
 * Format:
 *  - GB2L  (v1, all-binary): magic "GB2L" + header + per-matrix full
 *    wbits/alpha/bias, mhdr[3]=0.
 *  - GB2L2 (v2, logic-guided): magic "GB2L2" + header + per-matrix
 *    {mhdr[3]=n_core, logic_mask[out_dim], compacted BINARY wbits,
 *     compacted BINARY alpha, full bias[out_dim], w_core[n_core*in_dim]}.
 * The inference server auto-detects format via the magic bytes, so
 * logic-guided binarization (CORE float / BINARY sign+alpha / PRUNE zero)
 * survives the export -> inference round-trip. */
static void export_binary_weights(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[!] cannot open %s for writing\n", path); return; }

    int n_layer = m->cfg.n_layer;
    int n_embd = m->cfg.n_embd;
    int mlp_dim = m->cfg.mlp_dim;
    int vocab = m->cfg.vocab_size;
    int n_ctx = m->cfg.n_ctx;

    /* Detect whether any BinLayer carries a logic_mask (logic-guided
     * binarization). If so, write GB2L2; else fall back to GB2L (v1).
     * If any layer has zbits (ternary mode), write GT3L instead. */
    int has_logic = 0, has_ternary = 0;
    for (int l = 0; l < n_layer && !has_logic; l++) {
        TransLayer *tl = &m->layers[l];
        BinLayer *probes[4] = {&tl->attn_q, &tl->attn_o, &tl->mlp_gate, &tl->mlp_down};
        for (int mi = 0; mi < 4; mi++)
            if (probes[mi]->logic_mask) { has_logic = 1; break; }
    }
    if (has_logic) {
        for (int l = 0; l < n_layer && !has_ternary; l++) {
            TransLayer *tl = &m->layers[l];
            BinLayer *probes[4] = {&tl->attn_q, &tl->attn_o, &tl->mlp_gate, &tl->mlp_down};
            for (int mi = 0; mi < 4; mi++)
                if (probes[mi]->zbits) { has_ternary = 1; break; }
        }
    }

    /* Magic: "GT3L" for ternary logic-guided, "GB2L2" for binary logic-guided,
     * "GB2L" for plain binary. GT3L is a superset of GB2L2 (adds zbits block). */
    if (has_ternary) fwrite("GT3L", 1, 4, f);
    else { fwrite("GB2L", 1, 4, f); if (has_logic) fwrite("2", 1, 1, f); }

    int hdr[5] = {n_layer, n_embd, mlp_dim, vocab, n_ctx};
    fwrite(hdr, 4, 5, f);

    /* Embeddings (float — from original model, not binarized) */
    fwrite(m->wte, sizeof(float), (size_t)vocab * n_embd, f);
    fwrite(m->wpe, sizeof(float), (size_t)n_ctx * n_embd, f);
    fwrite(m->ln_f_w, sizeof(float), n_embd, f);
    fwrite(m->ln_f_b, sizeof(float), n_embd, f);

    /* Per-layer binary weights (updated by STE) */
    int total_core = 0, total_binary = 0;
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
            if (has_logic && bl->logic_mask) {
                /* GB2L2/GT3L per-matrix: mhdr[3]=n_core, then logic_mask,
                 * compacted BINARY wbits, compacted BINARY zbits (GT3L only),
                 * compacted BINARY alpha, full bias, then CORE float rows. */
                int mhdr[4] = {bl->out_dim, bl->in_dim, bl->n_words, bl->n_core};
                fwrite(mhdr, 4, 4, f);
                fwrite(bl->logic_mask, 1, bl->out_dim, f);
                /* compacted wbits: only BINARY rows (logic_mask==1), j order */
                for (int j = 0; j < bl->out_dim; j++)
                    if (bl->logic_mask[j] == 1)
                        fwrite(bl->wbits + (size_t)j * bl->n_words,
                               sizeof(uint64_t), bl->n_words, f);
                /* GT3L: compacted zbits (zero mask) — only BINARY rows, j order.
                 * Reader allocates zbits when this block is present (magic=GT3L). */
                if (has_ternary) {
                    for (int j = 0; j < bl->out_dim; j++)
                        if (bl->logic_mask[j] == 1)
                            fwrite(bl->zbits + (size_t)j * bl->n_words,
                                   sizeof(uint64_t), bl->n_words, f);
                }
                /* compacted alpha: only BINARY rows, j order */
                for (int j = 0; j < bl->out_dim; j++)
                    if (bl->logic_mask[j] == 1)
                        fwrite(&bl->alpha[j], sizeof(float), 1, f);
                /* full bias (PRUNE rows already zeroed at init) */
                fwrite(bl->bias, sizeof(float), bl->out_dim, f);
                /* CORE float rows: [n_core × in_dim], contiguous in w_core
                 * (laid out by bin_layer_init_with_logic). */
                if (bl->n_core > 0)
                    fwrite(bl->w_core, sizeof(float),
                           (size_t)bl->n_core * bl->in_dim, f);
                total_core   += bl->n_core;
                for (int j = 0; j < bl->out_dim; j++)
                    if (bl->logic_mask[j] == 1) total_binary++;
            } else {
                /* GB2L v1 per-matrix: full wbits/alpha/bias, mhdr[3]=0 */
                int mhdr[4] = {bl->out_dim, bl->in_dim, bl->n_words, 0};
                fwrite(mhdr, 4, 4, f);
                fwrite(bl->wbits, sizeof(uint64_t),
                       (size_t)bl->out_dim * bl->n_words, f);
                fwrite(bl->alpha, sizeof(float), bl->out_dim, f);
                fwrite(bl->bias,  sizeof(float), bl->out_dim, f);
            }
        }
    }

    fclose(f);
    if (has_ternary)
        printf("[*] exported GT3L (ternary logic-guided, {-1,0,+1}) weights to %s "
               "(%d CORE float rows, %d BINARY ternary rows retained)\n",
               path, total_core, total_binary);
    else if (has_logic)
        printf("[*] exported GB2L2 (logic-guided) weights to %s "
               "(%d CORE float rows, %d BINARY rows retained)\n",
               path, total_core, total_binary);
    else
        printf("[*] exported STE-tuned binary weights to %s\n", path);
}

int main(int argc, char **argv) {
    int n_steps = argc > 1 ? atoi(argv[1]) : 200;
    float lr = argc > 2 ? atof(argv[2]) : 0.05;
    int use_ste = 0;
    int use_real_attn = 0;
    int use_logic = 0;
    int use_adam = 0;
    int use_ternary = 0;
    float ternary_delta = 0.7f;
    int warmup_steps = 0;       /* default: no warmup */
    int total_steps = 0;       /* default: no cosine decay (use n_steps if >0) */
    const char *export_path = NULL;
    const char *distill_path = NULL;       /* teacher weights path */
    float distill_alpha = 0.5f;            /* CE/KL mix (0=CE only, 1=KL only) */
    float distill_T = 4.0f;                /* distillation temperature */
    /* Parse flags */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--ste") == 0) use_ste = 1;
        else if (strcmp(argv[i], "--real-attention") == 0) use_real_attn = 1;
        else if (strcmp(argv[i], "--logic") == 0) use_logic = 1;
        else if (strcmp(argv[i], "--adam") == 0) use_adam = 1;
        else if (strcmp(argv[i], "--ternary") == 0) use_ternary = 1;
        else if (strcmp(argv[i], "--ternary-delta") == 0 && i + 1 < argc) ternary_delta = atof(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--total") == 0 && i + 1 < argc) total_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--export") == 0 && i + 1 < argc) export_path = argv[++i];
        else if (strcmp(argv[i], "--distill") == 0 && i + 1 < argc) distill_path = argv[++i];
        else if (strcmp(argv[i], "--distill-alpha") == 0 && i + 1 < argc) distill_alpha = atof(argv[++i]);
        else if (strcmp(argv[i], "--distill-T") == 0 && i + 1 < argc) distill_T = atof(argv[++i]);
    }
    if (total_steps == 0) total_steps = n_steps;  /* auto-decay over training */

    printf("[*] LAL Training — GPT-2 (model-agnostic runtime, no PyTorch)\n");
    printf("[*] steps:%d lr:%f ste:%d logic:%d real_attn:%d adam:%d\n",
           n_steps, lr, use_ste, use_logic, use_real_attn, use_adam);
    printf("[*] schedule: warmup=%d cosine_total=%d  (lr=%.1e at step 0)\n",
           warmup_steps, total_steps,
           warmup_steps > 0 ? lr / warmup_steps : lr);
    if (distill_path) {
        printf("[*] distillation: teacher=%s alpha=%.2f T=%.1f\n",
               distill_path, distill_alpha, distill_T);
    }
    if (use_logic) {
        g_use_logic_binarization = 1;
        printf("[*] Logic-guided binarization: top 20%% norm → CORE(float), bottom 10%% → PRUNE(zero)\n");
    }
    if (use_ste) {
        g_use_ste = 1;
        printf("[*] STE mode: binary weights will be updated via Straight-Through Estimator\n");
        /* STE updates w_float (full-precision weights) then repacks wbits.
         * The gradient magnitude is much larger than non-STE (which only
         * updates alpha/bias), so lr > 0.01 caused NaN divergence with SGD.
         * With Adam, the per-param adaptive lr stabilizes updates — lr=0.001
         * is the recommended default. SGD should still use lr<=0.005. */
        if (!use_adam && lr > 0.01f) {
            printf("[!] WARNING: STE+SGD with lr=%f > 0.01 is likely to diverge (NaN).\n", lr);
            printf("[!]           Clamping to 0.005. Use --adam for higher lr.\n");
            lr = 0.005f;
        }
    }
    if (use_adam) {
        g_use_adam = 1;
        printf("[*] Adam optimizer: beta1=%.2f beta2=%.4f eps=%.0e (per-param adaptive lr)\n",
               g_adam_beta1, g_adam_beta2, g_adam_eps);
    }
    if (use_real_attn) {
        g_use_real_attention = 1;
        printf("[*] Real attention: causal multi-head QK softmax + KV cache (replaces V-copy)\n");
    }
    if (use_ternary) {
        if (!use_logic) {
            printf("[!] --ternary requires --logic (ternary applies to BINARY logic rows). Auto-enabling logic.\n");
            use_logic = 1;
            g_use_logic_binarization = 1;
        }
        g_use_ternary = 1;
        g_ternary_delta_factor = ternary_delta;
        printf("[*] Ternary Weight Network: W ∈ {-1, 0, +1}, Δ=%.2f * mean(|W|) "
               "(~1.58 bits/weight, 3x BWN capacity)\n", ternary_delta);
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

    /* Load teacher model (for distillation). Teacher is loaded with
     * logic_binarization OFF so all layers are BINARY — but we never call
     * backward on it, so w_float stays at the original GPT-2 weights. We use
     * trans_layer_forward_pure_float which dispatches via w_float directly. */
    Model teacher;
    int use_distill = 0;
    float *teacher_logits = NULL;
    if (distill_path) {
        int saved_logic = g_use_logic_binarization;
        g_use_logic_binarization = 0;  /* teacher: no logic mask */
        model_load(&teacher, distill_path, gpt2_config(), "h.%d.", 1);
        g_use_logic_binarization = saved_logic;
        teacher_logits = malloc(gpt2_config().vocab_size * sizeof(float));
        use_distill = 1;
        printf("[*] teacher model loaded (pure-float, never updated)\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[*] load: %.1fs\n", (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    printf("[*] training with %d pre-tokenized sentences...\n", (int)(sizeof(train_data)/sizeof(train_data[0])));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int n_train = sizeof(train_data) / sizeof(train_data[0]);
    for (int step = 0; step < n_steps; step++) {
        int ti = step % n_train;
        int n = train_data[ti].n;
        int target = train_data[ti].ids[n-1];
        int tt[32]; memcpy(tt, train_data[ti].ids, n * sizeof(int));
        /* LR schedule: warmup + cosine decay. */
        float lr_step = lr_schedule(step, warmup_steps, total_steps, lr);
        float loss = model_forward(&model, tt, n-1);
        if (use_distill) {
            /* Teacher produces full-vocab logits at the target position.
             * Then student backward combines hard CE + soft KL. */
            model_forward_float_logits(&teacher, tt, n-1, teacher_logits);
            model_backward_distill(&model, tt, n-1, lr_step,
                                   teacher_logits, distill_alpha, distill_T);
        } else {
            model_backward(&model, tt, n-1, lr_step);
        }
        if (step % 50 == 0)
            printf("  step %4d  loss=%.4f  lr=%.2e  (sentence %d, %d tokens)\n",
                   step, loss, lr_step, ti, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("[*] done in %.1fs (%.1f ms/step)\n", dt, dt/n_steps*1000);
    printf("[*] no PyTorch, no Python, pure C\n");

    /* Export STE-tuned weights if requested */
    if (export_path) {
        export_binary_weights(&model, export_path);
    }

    if (use_distill) {
        free(teacher_logits);
        model_free(&teacher);
    }
    model_free(&model);
    return 0;
}
