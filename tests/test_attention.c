/* test_attention.c — Unit test for causal multi-head self-attention.
 *
 * Verifies:
 *   1. Single-token (seq_pos=0): attn_out = V (only one position to attend to)
 *   2. Two-token causal: position 1 attends to positions 0 and 1, NOT future
 *   3. Uniform attention: when all Q·K scores equal, output = mean of V
 *   4. Numerical stability: large scores don't cause NaN/Inf
 *   5. Multi-head: heads are independent (Q_head_0 doesn't affect head_1 output)
 *   6. KV cache: K, V at seq_pos are stored correctly
 *
 * Compile: gcc -O2 -o test_attention test_attention.c runtime/lal_runtime.c -lm
 */
#include "../runtime/lal_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { printf("ok  : %s\n", msg); } \
} while(0)

static void test_single_token(void) {
    printf("\n--- test_single_token (seq_pos=0) ---\n");
    /* At position 0, only one K to attend to (itself). softmax([x]) = [1.0].
     * So attn_out = V[0]. */
    const int N_EMBD = 64, N_HEAD = 2;
    float qkv[3 * N_EMBD];
    float k_cache[4 * N_EMBD];  /* max 4 positions */
    float v_cache[4 * N_EMBD];
    float attn_out[N_EMBD];
    memset(k_cache, 0, sizeof(k_cache));
    memset(v_cache, 0, sizeof(v_cache));

    srand(1);
    for (int i = 0; i < 3 * N_EMBD; i++) qkv[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;

    attention_forward(attn_out, qkv, N_EMBD, N_HEAD, 0, k_cache, v_cache);

    /* Expected: attn_out = V = qkv + 2*N_EMBD */
    const float *V = qkv + 2 * N_EMBD;
    float max_diff = 0;
    for (int i = 0; i < N_EMBD; i++) {
        float d = fabsf(attn_out[i] - V[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("  max|attn_out - V| = %.6e (expect ~0 at seq_pos=0)\n", max_diff);
    CHECK(max_diff < 1e-5, "single-token attention returns V");
}

static void test_causal_mask(void) {
    printf("\n--- test_causal_mask (seq_pos=1 attends to 0,1) ---\n");
    /* Two tokens: position 0 then position 1.
     * Position 1's attention should depend on V[0] AND V[1].
     * Verify: zero out V[0], position 1's output changes.
     * Verify: zero out V[2] (future), position 1's output unchanged. */
    const int N_EMBD = 32, N_HEAD = 1;
    float qkv0[3*N_EMBD], qkv1[3*N_EMBD];
    float k_cache[4*N_EMBD], v_cache[4*N_EMBD];
    float attn_out_1[N_EMBD];

    srand(2);
    for (int i = 0; i < 3*N_EMBD; i++) {
        qkv0[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
        qkv1[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
    }
    memset(k_cache, 0, sizeof(k_cache));
    memset(v_cache, 0, sizeof(v_cache));

    /* Forward position 0 */
    float tmp[N_EMBD];
    attention_forward(tmp, qkv0, N_EMBD, N_HEAD, 0, k_cache, v_cache);
    /* Forward position 1 */
    attention_forward(attn_out_1, qkv1, N_EMBD, N_HEAD, 1, k_cache, v_cache);

    /* Now zero V[0] in cache, recompute position 1, output should differ */
    float v_cache_zeroed[4*N_EMBD];
    memcpy(v_cache_zeroed, v_cache, sizeof(v_cache));
    memset(v_cache_zeroed, 0, N_EMBD * sizeof(float));  /* zero V[0] */
    float attn_out_no_v0[N_EMBD];
    /* Need to also reset K cache — but we only care about V effect, so reuse k_cache */
    attention_forward(attn_out_no_v0, qkv1, N_EMBD, N_HEAD, 1, k_cache, v_cache_zeroed);

    float diff_v0 = 0;
    for (int i = 0; i < N_EMBD; i++) diff_v0 += fabsf(attn_out_1[i] - attn_out_no_v0[i]);
    printf("  sum|attn - attn_no_v0| = %.6f (expect > 0 if V[0] contributes)\n", diff_v0);
    CHECK(diff_v0 > 1e-4f, "position 1 attends to V[0] (causal)");

    /* Verify future V[2] doesn't affect position 1: put garbage at V[2], recompute */
    float v_cache_garbage[4*N_EMBD];
    memcpy(v_cache_garbage, v_cache, sizeof(v_cache));
    for (int i = 2*N_EMBD; i < 3*N_EMBD; i++) v_cache_garbage[i] = 999.0f;  /* garbage at V[2] */
    float attn_out_garbage[N_EMBD];
    attention_forward(attn_out_garbage, qkv1, N_EMBD, N_HEAD, 1, k_cache, v_cache_garbage);
    float diff_future = 0;
    for (int i = 0; i < N_EMBD; i++) diff_future += fabsf(attn_out_1[i] - attn_out_garbage[i]);
    printf("  sum|attn - attn_garbage_v2| = %.6e (expect ~0, future masked)\n", diff_future);
    CHECK(diff_future < 1e-5f, "position 1 does NOT attend to future V[2]");
}

static void test_uniform_attention(void) {
    printf("\n--- test_uniform_attention ---\n");
    /* If Q·K = 0 for all positions (Q orthogonal to all K), softmax is uniform.
     * attn_out = mean(V[0..pos]). */
    const int N_EMBD = 16, N_HEAD = 1;
    float qkv[3*N_EMBD]; memset(qkv, 0, sizeof(qkv));
    float k_cache[4*N_EMBD]; memset(k_cache, 0, sizeof(k_cache));
    float v_cache[4*N_EMBD]; memset(v_cache, 0, sizeof(v_cache));

    /* Q = 0 → all dot products = 0 → uniform softmax.
     * Pre-fill V at positions 0 and 1. V at position 2 will come from qkv. */
    for (int i = 0; i < N_EMBD; i++) {
        v_cache[0*N_EMBD + i] = 1.0f;
        v_cache[1*N_EMBD + i] = 3.0f;
        qkv[2*N_EMBD + i] = 5.0f;  /* V_new for position 2 */
    }
    /* Q at position 2 = 0 → uniform attention over 3 positions → mean = (1+3+5)/3 = 3 */
    float attn_out[N_EMBD];
    attention_forward(attn_out, qkv, N_EMBD, N_HEAD, 2, k_cache, v_cache);

    /* Expected: (1+3+5)/3 = 3 for each dim */
    float max_diff = 0;
    for (int i = 0; i < N_EMBD; i++) {
        float d = fabsf(attn_out[i] - 3.0f);
        if (d > max_diff) max_diff = d;
    }
    printf("  max|attn_out - 3.0| = %.6e (uniform mean of 1,3,5)\n", max_diff);
    CHECK(max_diff < 1e-4, "uniform attention returns mean of V");
}

static void test_no_nan_large_scores(void) {
    printf("\n--- test_no_nan_large_scores ---\n");
    const int N_EMBD = 8, N_HEAD = 1;
    float qkv[3*N_EMBD];
    float k_cache[4*N_EMBD]; memset(k_cache, 0, sizeof(k_cache));
    float v_cache[4*N_EMBD]; memset(v_cache, 0, sizeof(v_cache));

    /* Large Q and K → large scores. Without max subtraction, exp overflow. */
    for (int i = 0; i < N_EMBD; i++) qkv[i] = 100.0f;  /* Q */
    for (int i = N_EMBD; i < 2*N_EMBD; i++) qkv[i] = 100.0f;  /* K */
    for (int i = 2*N_EMBD; i < 3*N_EMBD; i++) qkv[i] = 1.0f;  /* V */

    /* Pre-fill K cache at position 0 with large values */
    for (int i = 0; i < N_EMBD; i++) k_cache[i] = 100.0f;
    for (int i = 0; i < N_EMBD; i++) v_cache[i] = 1.0f;

    float attn_out[N_EMBD];
    attention_forward(attn_out, qkv, N_EMBD, N_HEAD, 1, k_cache, v_cache);

    int has_nan = 0;
    for (int i = 0; i < N_EMBD; i++) {
        if (isnan(attn_out[i]) || isinf(attn_out[i])) { has_nan = 1; break; }
    }
    printf("  attn_out[0] = %.4f (no NaN/Inf with large scores)\n", attn_out[0]);
    CHECK(!has_nan, "large scores don't cause NaN/Inf (max subtraction works)");
}

static void test_multi_head_independence(void) {
    printf("\n--- test_multi_head_independence ---\n");
    /* Modify Q in head 0 only. Output in head 1 should not change. */
    const int N_EMBD = 16, N_HEAD = 2, HEAD_DIM = N_EMBD / N_HEAD;
    float qkv_a[3*N_EMBD], qkv_b[3*N_EMBD];
    float k_cache[4*N_EMBD]; memset(k_cache, 0, sizeof(k_cache));
    float v_cache[4*N_EMBD]; memset(v_cache, 0, sizeof(v_cache));

    srand(5);
    for (int i = 0; i < 3*N_EMBD; i++) qkv_a[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
    memcpy(qkv_b, qkv_a, sizeof(qkv_a));
    /* Change Q in head 0 only (first HEAD_DIM elements) */
    for (int i = 0; i < HEAD_DIM; i++) qkv_b[i] += 10.0f;

    /* Pre-fill position 0 K, V with same values for both */
    for (int i = 0; i < N_EMBD; i++) {
        k_cache[i] = (rand()/(float)RAND_MAX - 0.5f);
        v_cache[i] = (rand()/(float)RAND_MAX - 0.5f);
    }

    float out_a[N_EMBD], out_b[N_EMBD];
    /* Need separate caches since attention_forward writes into cache at seq_pos */
    float kc2[4*N_EMBD], vc2[4*N_EMBD];
    memcpy(kc2, k_cache, sizeof(k_cache));
    memcpy(vc2, v_cache, sizeof(v_cache));

    attention_forward(out_a, qkv_a, N_EMBD, N_HEAD, 1, k_cache, v_cache);
    attention_forward(out_b, qkv_b, N_EMBD, N_HEAD, 1, kc2, vc2);

    /* Head 1 (second half) should be identical */
    float diff_head1 = 0;
    for (int i = HEAD_DIM; i < N_EMBD; i++) diff_head1 += fabsf(out_a[i] - out_b[i]);
    printf("  sum|out_a[head1] - out_b[head1]| = %.6e (expect ~0)\n", diff_head1);
    CHECK(diff_head1 < 1e-5, "head 1 output independent of head 0 Q");

    /* Head 0 should differ */
    float diff_head0 = 0;
    for (int i = 0; i < HEAD_DIM; i++) diff_head0 += fabsf(out_a[i] - out_b[i]);
    printf("  sum|out_a[head0] - out_b[head0]| = %.6e (expect > 0)\n", diff_head0);
    CHECK(diff_head0 > 1e-4f, "head 0 output changes when its Q changes");
}

static void test_kv_cache_storage(void) {
    printf("\n--- test_kv_cache_storage ---\n");
    /* Verify K, V are written to cache at seq_pos. */
    const int N_EMBD = 8, N_HEAD = 1;
    float qkv[3*N_EMBD];
    float k_cache[4*N_EMBD]; memset(k_cache, 0, sizeof(k_cache));
    float v_cache[4*N_EMBD]; memset(v_cache, 0, sizeof(v_cache));
    float attn_out[N_EMBD];

    for (int i = 0; i < 3*N_EMBD; i++) qkv[i] = (float)(i + 1) * 0.1f;

    attention_forward(attn_out, qkv, N_EMBD, N_HEAD, 2, k_cache, v_cache);

    /* K at position 2 = qkv[N_EMBD .. 2*N_EMBD) */
    float k_diff = 0, v_diff = 0;
    for (int i = 0; i < N_EMBD; i++) {
        k_diff += fabsf(k_cache[2*N_EMBD + i] - qkv[N_EMBD + i]);
        v_diff += fabsf(v_cache[2*N_EMBD + i] - qkv[2*N_EMBD + i]);
    }
    printf("  sum|K_cache[2] - K_input| = %.6e (expect 0)\n", k_diff);
    printf("  sum|V_cache[2] - V_input| = %.6e (expect 0)\n", v_diff);
    CHECK(k_diff < 1e-6, "K written to cache at correct position");
    CHECK(v_diff < 1e-6, "V written to cache at correct position");
}

int main(void) {
    printf("=== Attention Unit Tests ===\n");
    test_single_token();
    test_causal_mask();
    test_uniform_attention();
    test_no_nan_large_scores();
    test_multi_head_independence();
    test_kv_cache_storage();

    printf("\n=== Summary: %d failures ===\n", failures);
    return failures ? 1 : 0;
}
