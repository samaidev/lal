/* bench_silu.c — SiLU/SwiGLU 性能基准测试
 * 对比: 标量 SiLU vs 新的 SIMD fast_exp SiLU
 *
 * 构建: gcc -O3 -march=native -mavx2 -mfma bench_silu.c -o bench_silu -lm
 * 运行: ./bench_silu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

/* ============ 新的 SIMD fast_exp_ps (从 lal_simd_optim.h 复制) ============ */
static inline __m256 fast_exp_ps_new(__m256 x) {
    __m256 vmax = _mm256_set1_ps(88.0f);
    __m256 vmin = _mm256_set1_ps(-88.0f);
    x = _mm256_min_ps(x, vmax);
    x = _mm256_max_ps(x, vmin);
    __m256 vlog2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 fx = _mm256_mul_ps(x, vlog2e);
    __m256i vni = _mm256_cvtps_epi32(fx);
    __m256 vn = _mm256_cvtepi32_ps(vni);
    __m256 vf = _mm256_sub_ps(fx, vn);
    __m256 vln2 = _mm256_set1_ps(0.6931471805599453f);
    __m256 vhalf = _mm256_set1_ps(0.5f);
    __m256 vsixth = _mm256_set1_ps(0.16666667f);
    __m256 vt = _mm256_mul_ps(vf, vln2);
    __m256 vt2 = _mm256_mul_ps(vt, vt);
    __m256 vt3 = _mm256_mul_ps(vt2, vt);
    __m256 vexp2f = _mm256_fmadd_ps(vt3, vsixth,
                         _mm256_fmadd_ps(vt2, vhalf, vt));
    vexp2f = _mm256_add_ps(vexp2f, _mm256_set1_ps(1.0f));
    __m256i vexp_i = _mm256_add_epi32(vni, _mm256_set1_epi32(127));
    vexp_i = _mm256_max_epi32(vexp_i, _mm256_setzero_si256());
    vexp_i = _mm256_min_epi32(vexp_i, _mm256_set1_epi32(255));
    vexp_i = _mm256_slli_epi32(vexp_i, 23);
    vexp_i = _mm256_and_si256(vexp_i, _mm256_set1_epi32(0x7FFFFFFF));
    __m256 vpow2n = _mm256_castsi256_ps(vexp_i);
    return _mm256_mul_ps(vpow2n, vexp2f);
}

/* ============ SIMD SiLU (新版, 真正的 SIMD exp) ============ */
static inline void silu_mul_simd_new(float * __restrict__ act,
                                      const float * __restrict__ gate,
                                      const float * __restrict__ up,
                                      int n) {
    int i = 0;
    int n8 = n & ~7;
    __m256 vone = _mm256_set1_ps(1.0f);
    for (; i < n8; i += 8) {
        __m256 vg = _mm256_loadu_ps(gate + i);
        __m256 vu = _mm256_loadu_ps(up + i);
        __m256 vneg_g = _mm256_xor_ps(vg, _mm256_set1_ps(-0.0f));
        __m256 vexp_neg_g = fast_exp_ps_new(vneg_g);
        __m256 vsigmoid = _mm256_div_ps(vone, _mm256_add_ps(vone, vexp_neg_g));
        __m256 vsilu = _mm256_mul_ps(vg, vsigmoid);
        __m256 vr = _mm256_mul_ps(vsilu, vu);
        _mm256_storeu_ps(act + i, vr);
    }
    for (; i < n; i++) {
        float g = gate[i];
        act[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/* ============ 标量 SiLU (baseline) ============ */
static inline void silu_mul_scalar(float * __restrict__ act,
                                    const float * __restrict__ gate,
                                    const float * __restrict__ up,
                                    int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        act[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/* ============ Timing ============ */
static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    /* Qwen2.5-7B MLP_DIM = 18944 */
    const int N = 18944;
    const int ITERS = 10000;

    /* 用 posix_memalign 保证 32-byte 对齐 */
    float *gate = NULL, *up = NULL, *act_scalar = NULL, *act_simd = NULL;
    posix_memalign((void**)&gate, 32, N * sizeof(float));
    posix_memalign((void**)&up,   32, N * sizeof(float));
    posix_memalign((void**)&act_scalar, 32, N * sizeof(float));
    posix_memalign((void**)&act_simd,   32, N * sizeof(float));

    /* 初始化 gate/up, 用 [-5, 5] 范围 (典型 SiLU 输入) */
    srand(42);
    for (int i = 0; i < N; i++) {
        gate[i] = ((float)rand() / RAND_MAX * 10.0f) - 5.0f;
        up[i]   = ((float)rand() / RAND_MAX * 4.0f) - 2.0f;
    }

    /* === 正确性验证 === */
    silu_mul_scalar(act_scalar, gate, up, N);
    silu_mul_simd_new(act_simd, gate, up, N);

    double max_err = 0, sum_err = 0, sum_abs = 0;
    for (int i = 0; i < N; i++) {
        double err = fabs(act_scalar[i] - act_simd[i]);
        double abs_val = fabs(act_scalar[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
        sum_abs += abs_val;
    }
    printf("=== 正确性验证 (N=%d) ===\n", N);
    printf("  最大绝对误差: %.6e\n", max_err);
    printf("  平均绝对误差: %.6e\n", sum_err / N);
    printf("  平均绝对值:   %.6e\n", sum_abs / N);
    printf("  相对误差:     %.4f%%\n", (sum_err / sum_abs) * 100);
    printf("  最大相对误差: %.4f%%\n\n", (max_err / (sum_abs / N)) * 100);

    /* === 性能测试: 标量 === */
    /* warmup */
    volatile float sink = 0;
    for (int i = 0; i < 100; i++) { silu_mul_scalar(act_scalar, gate, up, N); sink += act_scalar[0]; }
    double t0 = now_us();
    for (int i = 0; i < ITERS; i++) {
        silu_mul_scalar(act_scalar, gate, up, N);
        sink += act_scalar[i % N];  /* prevent DCE */
    }
    double t_scalar = now_us() - t0;

    /* === 性能测试: SIMD === */
    /* warmup */
    for (int i = 0; i < 100; i++) { silu_mul_simd_new(act_simd, gate, up, N); sink += act_simd[0]; }
    double t1 = now_us();
    for (int i = 0; i < ITERS; i++) {
        silu_mul_simd_new(act_simd, gate, up, N);
        sink += act_simd[i % N];  /* prevent DCE */
    }
    double t_simd = now_us() - t1;

    printf("=== 性能测试 (N=%d, %d iters) ===\n", N, ITERS);
    printf("  标量 SiLU:  %.2f ms total, %.4f us/iter, %.2f M elem/s\n",
           t_scalar / 1000, t_scalar / ITERS, (double)N * ITERS / t_scalar);
    printf("  SIMD  SiLU: %.2f ms total, %.4f us/iter, %.2f M elem/s\n",
           t_simd / 1000, t_simd / ITERS, (double)N * ITERS / t_simd);
    printf("  加速比:     %.2fx\n", t_scalar / t_simd);
    printf("  每层节省:   %.4f us (Qwen7B MLP_DIM=%d)\n",
           (t_scalar - t_simd) / ITERS, N);
    printf("  28 层/token 节省: %.4f us = %.4f ms\n",
           28 * (t_scalar - t_simd) / ITERS,
           28 * (t_scalar - t_simd) / ITERS / 1000);

    /* sink 防止优化 */
    if (sink < -1e30) printf("impossible: %f\n", (double)sink);

    free(gate); free(up); free(act_scalar); free(act_simd);
    return 0;
}
