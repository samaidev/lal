/* models/qwen.c — Qwen model definition (just config + main, ~30 lines)
 *
 * Uses lal_runtime Level 3 API. Qwen uses LLaMA architecture:
 *   RMSNorm, RoPE, SwiGLU, separate Q/K/V/O.
 *
 * Build: gcc -O3 -mavx2 -o qwen models/qwen.c runtime/lal_runtime.c -lm
 * Run:   ./qwen 10000 0.05
 */
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include "../runtime/lal_runtime.h"

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

/* === Training data (same as GPT-2, but would use Qwen tokenizer) === */
static const char *TEXTS[] = {
    "The capital of France is Paris.", "The capital of Japan is Tokyo.",
    "The capital of Germany is Berlin.", "Hello, how are you doing today?",
    "Once upon a time, there was a kingdom.", "The weather today is sunny and warm.",
    "Machine learning is a subset of AI.", "The world is a place of great beauty.",
    "I think, therefore I am.", "Knowledge is power.",
};

/* Placeholder encode — real version would use Qwen's tiktoken BPE */
static int encode(const char *t, int *tk) {
    tk[0]=1; tk[1]=2; tk[2]=3; tk[3]=4; tk[4]=5; tk[5]=6; tk[6]=7;
    return 7;  /* TODO: use real Qwen tokenizer */
}

int main(int argc, char **argv) {
    int n_steps = argc > 1 ? atoi(argv[1]) : 200;
    float lr = argc > 2 ? atof(argv[2]) : 0.05;

    printf("[*] LAL Training — Qwen (model-agnostic runtime, no PyTorch)\n");
    printf("[*] steps:%d lr:%f\n", n_steps, lr);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    Model model;
    /* The only difference from GPT-2: config + qkv_merged=0 */
    model_load(&model, "/home/z/my-project/prebuilt/qwen_weights.bin",
               qwen_config(), "model.layers.%d.", 0);

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

    model_free(&model);
    return 0;
}
