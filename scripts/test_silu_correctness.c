/* test_silu_correctness.c — 验证 fast exp SwiGLU 的正确性
 * Build: gcc -O3 -march=native -I. -o test_silu scripts/test_silu_correctness.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#define N_EMBD 3584
#define MLP_DIM 18944
#define RMS_EPS 1e-6f
#include "runtime/lal_simd_optim.h"

static void silu_mul_scalar(float *act, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        act[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

int main() {
    srand(42);
    int n = MLP_DIM;
    float *gate = _mm_malloc(n * sizeof(float), 32);
    float *up = _mm_malloc(n * sizeof(float), 32);
    float *act_s = _mm_malloc(n * sizeof(float), 32);
    float *act_v = _mm_malloc(n * sizeof(float), 32);

    /* 生成典型 LLM 激活范围的输入 (-3 to 3) */
    for (int i = 0; i < n; i++) {
        gate[i] = (rand()/((float)RAND_MAX) - 0.5f) * 4.0f;
        up[i] = (rand()/((float)RAND_MAX) - 0.5f) * 2.0f;
    }

    silu_mul_scalar(act_s, gate, up, n);
    lal_silu_mul_simd(act_v, gate, up, n);

    float max_diff = 0, sum_diff = 0, max_rel = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(act_s[i] - act_v[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
        if (fabsf(act_s[i]) > 0.01f) {
            float rel = d / fabsf(act_s[i]);
            if (rel > max_rel) max_rel = rel;
        }
    }
    printf("SwiGLU 正确性测试 (n=%d):\n", n);
    printf("  最大绝对误差: %.6e\n", max_diff);
    printf("  平均绝对误差: %.6e\n", sum_diff / n);
    printf("  最大相对误差: %.4f%%\n", max_rel * 100);
    printf("  %s\n", max_rel < 0.05f ? "✅ PASS (< 5% 相对误差)" : "⚠️ 检查误差是否可接受");

    _mm_free(gate); _mm_free(up); _mm_free(act_s); _mm_free(act_v);
    return 0;
}
