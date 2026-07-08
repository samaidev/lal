/* models/qwen.c — Qwen model definition (config + training main)
 *
 * Uses lal_runtime Level 3 API. Qwen uses LLaMA architecture:
 *   RMSNorm, RoPE, SwiGLU, separate Q/K/V/O.
 *
 * Training data: scripts/qwen_train_data.h — pre-tokenized with the REAL
 * Qwen2-0.5B BPE tokenizer (scripts/gen_qwen_train_data.py). This replaces
 * the old hardcoded fake tokens (encode() returning 1..7) so Qwen training
 * runs on real data, mirroring how models/gpt2.c consumes train_data.h.
 *
 * Build: gcc -O3 -mavx2 -o qwen models/qwen.c runtime/lal_runtime.c -lm
 * Run:   ./qwen 10000 0.002 --logic --ste --real-attention
 */
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <string.h>
#include "../runtime/lal_runtime.h"
#include "../scripts/qwen_train_data.h"

/* === Qwen-0.5B config — the ONLY difference from GPT-2 === */
static ModelConfig qwen_config(void) {
    return (ModelConfig){
        .n_layer = 24, .n_embd = 896, .n_head = 14,
        .n_ctx = 2048, .vocab_size = 151646, .mlp_dim = 4864,
        .norm_type = NORM_RMS,        /* Qwen uses RMSNorm */
        .attn_type = ATTN_ROPE,       /* Qwen uses RoPE */
        .act_type = ACT_SWIGLU,       /* Qwen uses SwiGLU */
        .residual_scale = 0.5f,
        .qkv_merged = 0,              /* Qwen has separate Q/K/V/O */
    };
}

int main(int argc, char **argv) {
    int n_steps = argc > 1 ? atoi(argv[1]) : 200;
    float lr = argc > 2 ? atof(argv[2]) : 0.05;
    int use_logic = 0, use_ste = 0, use_real_attn = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--logic") == 0) use_logic = 1;
        else if (strcmp(argv[i], "--ste") == 0) use_ste = 1;
        else if (strcmp(argv[i], "--real-attention") == 0) use_real_attn = 1;
    }

    printf("[*] LAL Training — Qwen (model-agnostic runtime, no PyTorch)\n");
    printf("[*] steps:%d lr:%f\n", n_steps, lr);

    if (use_logic) {
        g_use_logic_binarization = 1;
        printf("[*] Logic-guided binarization: top 20%% norm → CORE(float), bottom 10%% → PRUNE(zero)\n");
    }
    if (use_ste) {
        g_use_ste = 1;
        printf("[*] STE mode: binary weights updated via Straight-Through Estimator\n");
        if (lr > 0.01f) {
            printf("[!] WARNING: STE with lr=%f > 0.01 is likely to diverge (NaN). Clamping to 0.005.\n", lr);
            lr = 0.005f;
        }
    }
    if (use_real_attn) {
        g_use_real_attention = 1;
        printf("[*] Real attention: causal multi-head QK softmax + KV cache (replaces V-copy)\n");
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    Model model;
    const char *weight_path = getenv("LAL_WEIGHTS");
    if (!weight_path) weight_path = "prebuilt/qwen_weights.bin";
    model_load(&model, weight_path, qwen_config(), "model.layers.%d.", 0);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[*] load: %.1fs\n", (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    printf("[*] training with %d pre-tokenized sentences (real Qwen2 BPE)...\n", N_TRAIN);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int step = 0; step < n_steps; step++) {
        int ti = step % N_TRAIN;
        int n = train_data[ti].n;
        int target = train_data[ti].ids[n-1];
        int tt[32]; memcpy(tt, train_data[ti].ids, n * sizeof(int));
        float loss = model_forward(&model, tt, n-1);
        model_backward(&model, tt, n-1, lr);
        if (step % 50 == 0)
            printf("  step %4d  loss=%.4f  (sentence %d, %d tokens)\n", step, loss, ti, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("[*] done in %.1fs (%.1f ms/step)\n", dt, dt/n_steps*1000);
    printf("[*] no PyTorch, no Python, pure C\n");

    model_free(&model);
    return 0;
}
