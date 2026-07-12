/* bench_kv_q8.c — 对比 F32 vs Q8 KV cache 的 attention 性能
 * Build: gcc -O3 -march=native -I. -o bench_kv_q8 scripts/bench_kv_q8.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define N_HEAD    28
#define N_KV_HEAD 4
#define HEAD_DIM  128
#define KV_DIM    (N_KV_HEAD * HEAD_DIM)
#define N_Q_PER_KV (N_HEAD / N_KV_HEAD)
#define N_CTX     4096

#include "runtime/lal_simd_optim.h"
#include "runtime/lal_kv_cache_q8.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int main() {
    srand(42);
    int positions[] = {128, 512, 1024, 2048};
    int n_pos = sizeof(positions)/sizeof(positions[0]);

    printf("=== F32 vs Q8 KV Cache Attention 性能对比 ===\n");
    printf("(N_HEAD=%d, N_KV_HEAD=%d, HEAD_DIM=%d)\n\n", N_HEAD, N_KV_HEAD, HEAD_DIM);

    for (int pi = 0; pi < n_pos; pi++) {
        int pos = positions[pi];
        int n_iter = pos <= 512 ? 20 : 5;

        /* F32 KV cache */
        float *Q = _mm_malloc(N_HEAD * HEAD_DIM * sizeof(float), 32);
        float *K = _mm_malloc(KV_DIM * sizeof(float), 32);
        float *V = _mm_malloc(KV_DIM * sizeof(float), 32);
        float *out = _mm_malloc(N_HEAD * HEAD_DIM * sizeof(float), 32);
        float *kc_f32 = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
        float *vc_f32 = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);

        /* Q8 KV cache */
        int q8_row_bytes = KV_Q8_BYTES(KV_DIM);
        uint8_t *kc_q8 = _mm_malloc(N_CTX * q8_row_bytes, 32);
        uint8_t *vc_q8 = _mm_malloc(N_CTX * q8_row_bytes, 32);

        for (int i = 0; i < N_HEAD*HEAD_DIM; i++) Q[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f;
        for (int i = 0; i < KV_DIM; i++) { K[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; V[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; }
        /* 填充历史 cache */
        for (int t = 0; t < pos; t++) {
            for (int i = 0; i < KV_DIM; i++) {
                kc_f32[t * KV_DIM + i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f;
                vc_f32[t * KV_DIM + i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f;
            }
            kv_quantize_q8_0(kc_q8 + t * q8_row_bytes, kc_f32 + t * KV_DIM, KV_DIM);
            kv_quantize_q8_0(vc_q8 + t * q8_row_bytes, vc_f32 + t * KV_DIM, KV_DIM);
        }

        /* F32 attention (用 lal_gqa_attn_simd) */
        double t0 = now_s();
        for (int it = 0; it < n_iter; it++)
            lal_gqa_attn_simd(out, Q, K, V, kc_f32, vc_f32, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
        double dt_f32 = (now_s() - t0) / n_iter;

        /* Q8 attention — 手动实现用 kv_dot_q8_0 和 kv_vmac_q8_0 */
        t0 = now_s();
        for (int it = 0; it < n_iter; it++) {
            /* 量化新的 K, V 到 Q8_0 */
            kv_quantize_q8_0(kc_q8 + pos * q8_row_bytes, K, KV_DIM);
            kv_quantize_q8_0(vc_q8 + pos * q8_row_bytes, V, KV_DIM);

            float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
            static float scores[8192];
            if (pos >= 8192) pos = 8191;

            for (int h = 0; h < N_HEAD; h++) {
                const float *qh = Q + h * HEAD_DIM;
                int kvh = h / N_Q_PER_KV;

                float max_score = -1e30f;
                for (int t = 0; t <= pos; t++) {
                    const uint8_t *kt_q8 = kc_q8 + t * q8_row_bytes + kvh * HEAD_DIM / KV_Q8_BLOCK * 34;
                    float dot = kv_dot_q8_0(qh, kt_q8, HEAD_DIM);
                    scores[t] = dot * inv_sqrt;
                    if (scores[t] > max_score) max_score = scores[t];
                }

                float sum = 0;
                for (int t = 0; t <= pos; t++) {
                    scores[t] = expf(scores[t] - max_score);
                    sum += scores[t];
                }
                float inv_sum = 1.0f / sum;

                float *oh = out + h * HEAD_DIM;
                memset(oh, 0, HEAD_DIM * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float w = scores[t] * inv_sum;
                    const uint8_t *vt_q8 = vc_q8 + t * q8_row_bytes + kvh * HEAD_DIM / KV_Q8_BLOCK * 34;
                    kv_vmac_q8_0(oh, w, vt_q8, HEAD_DIM);
                }
            }
        }
        double dt_q8 = (now_s() - t0) / n_iter;

        /* 内存使用对比 */
        double mem_f32 = (double)N_CTX * KV_DIM * 4 * 2; /* K+V, per layer */
        double mem_q8 = (double)N_CTX * q8_row_bytes * 2;

        printf("pos=%-5d  F32: %7.2f us  Q8: %7.2f us  speedup: %5.2fx  mem: %.0fMB → %.0fMB\n",
               pos, dt_f32*1e6, dt_q8*1e6, dt_f32/dt_q8, mem_f32/1e6, mem_q8/1e6);

        _mm_free(Q); _mm_free(K); _mm_free(V); _mm_free(out);
        _mm_free(kc_f32); _mm_free(vc_f32); _mm_free(kc_q8); _mm_free(vc_q8);
    }

    printf("\n注: Q8 版本包含量化开销，长 context 时带宽节省 > 量化开销\n");
    return 0;
}
