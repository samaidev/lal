/* test_neon_sdot.c — Benchmark NEON SDOT vs vmull_s16 for Q8 matmul
 *
 * arm64 with asimddp supports:
 *   vdotq_s32(c, a, b): 8 int8 × 8 int8 → 4 int32 accumulate (1 instruction!)
 *   vs current: vmull_s16 + vaddq_s32 (3 instructions for same work)
 *
 * Also test 8-output parallel (mistral.rs style) with NEON.
 *
 * SDOT: __ARM_FEATURE_DOTPROD must be defined (compile with -march=armv8.2-a+dotprod)
 *
 * Compile:
 *   gcc -O3 -march=armv8.2-a+dotprod -o test_neon_sdot test_neon_sdot.c -lm
 *   (fallback without dotprod: gcc -O3 -o test_neon_sdot test_neon_sdot.c -lm)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#define IN_DIM 768
#define OUT_DIM 768
#define N_TRIALS 500

#ifdef __aarch64__

/* A) Current: vmull_s16 (single output, 8 int16 at a time) */
static void q8_vmull(float *y, const int8_t *w_T, const float *scale,
                     const int32_t *w_sums, const float *x, const float *b,
                     int in_dim, int out_dim) {
    /* Quantize x to int8 */
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    int8_t xq[4096];
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] / x_scale);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }

    for (int j = 0; j < out_dim; j++) {
        const int8_t *w = w_T + (size_t)j * in_dim;
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < in_dim; i += 8) {
            int8x8_t wv = vld1_s8(w + i);
            int8x8_t xv = vld1_s8(xq + i);
            int16x8_t prod = vmull_s8(xv, wv);
            acc = vpadalq_s16(acc, prod);
        }
        int32_t dot = vaddvq_s32(acc);
        y[j] = (float)dot * x_scale * scale[j] + (b ? b[j] : 0);
    }
}

/* B) SDOT: vdotq_s32 (single output, 8 int8 at a time, 1 instruction) */
#ifdef __ARM_FEATURE_DOTPROD
static void q8_sdot(float *y, const int8_t *w_T, const float *scale,
                    const int32_t *w_sums, const float *x, const float *b,
                    int in_dim, int out_dim) {
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    int8_t xq[4096];
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] / x_scale);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }

    for (int j = 0; j < out_dim; j++) {
        const int8_t *w = w_T + (size_t)j * in_dim;
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < in_dim; i += 16) {
            int8x16_t wv = vld1q_s8(w + i);
            int8x16_t xv = vld1q_s8(xq + i);
            /* SDOT: 16 int8 × 16 int8 → 4 int32 (vdotq takes 128-bit vectors) */
            acc = vdotq_s32(acc, wv, xv);
        }
        int32_t dot = vaddvq_s32(acc);
        y[j] = (float)dot * x_scale * scale[j] + (b ? b[j] : 0);
    }
}
#endif

/* C) 8-output parallel (mistral.rs style) with vmull_s8 */
static void q8_8out_vmull(float *y, const int8_t *w_T, const float *scale,
                          const int32_t *w_sums, const float *x, const float *b,
                          int in_dim, int out_dim) {
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    int8_t xq[4096];
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] / x_scale);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }

    int j = 0;
    for (; j + 8 <= out_dim; j += 8) {
        /* 8 accumulators in NEON registers (4 per int32x4_t, need 8) */
        int32x4_t a0=vdupq_n_s32(0), a1=vdupq_n_s32(0);
        int32x4_t a2=vdupq_n_s32(0), a3=vdupq_n_s32(0);
        int32x4_t a4=vdupq_n_s32(0), a5=vdupq_n_s32(0);
        int32x4_t a6=vdupq_n_s32(0), a7=vdupq_n_s32(0);
        const int8_t *w0=w_T+(size_t)(j+0)*in_dim, *w1=w_T+(size_t)(j+1)*in_dim;
        const int8_t *w2=w_T+(size_t)(j+2)*in_dim, *w3=w_T+(size_t)(j+3)*in_dim;
        const int8_t *w4=w_T+(size_t)(j+4)*in_dim, *w5=w_T+(size_t)(j+5)*in_dim;
        const int8_t *w6=w_T+(size_t)(j+6)*in_dim, *w7=w_T+(size_t)(j+7)*in_dim;
        for (int i = 0; i < in_dim; i += 8) {
            int8x8_t xv = vld1_s8(xq + i);
            a0 = vpadalq_s16(a0, vmull_s8(xv, vld1_s8(w0+i)));
            a1 = vpadalq_s16(a1, vmull_s8(xv, vld1_s8(w1+i)));
            a2 = vpadalq_s16(a2, vmull_s8(xv, vld1_s8(w2+i)));
            a3 = vpadalq_s16(a3, vmull_s8(xv, vld1_s8(w3+i)));
            a4 = vpadalq_s16(a4, vmull_s8(xv, vld1_s8(w4+i)));
            a5 = vpadalq_s16(a5, vmull_s8(xv, vld1_s8(w5+i)));
            a6 = vpadalq_s16(a6, vmull_s8(xv, vld1_s8(w6+i)));
            a7 = vpadalq_s16(a7, vmull_s8(xv, vld1_s8(w7+i)));
        }
        y[j+0]=(float)vaddvq_s32(a0)*x_scale*scale[j+0]+(b?b[j+0]:0);
        y[j+1]=(float)vaddvq_s32(a1)*x_scale*scale[j+1]+(b?b[j+1]:0);
        y[j+2]=(float)vaddvq_s32(a2)*x_scale*scale[j+2]+(b?b[j+2]:0);
        y[j+3]=(float)vaddvq_s32(a3)*x_scale*scale[j+3]+(b?b[j+3]:0);
        y[j+4]=(float)vaddvq_s32(a4)*x_scale*scale[j+4]+(b?b[j+4]:0);
        y[j+5]=(float)vaddvq_s32(a5)*x_scale*scale[j+5]+(b?b[j+5]:0);
        y[j+6]=(float)vaddvq_s32(a6)*x_scale*scale[j+6]+(b?b[j+6]:0);
        y[j+7]=(float)vaddvq_s32(a7)*x_scale*scale[j+7]+(b?b[j+7]:0);
    }
    for (; j < out_dim; j++) {
        const int8_t *w = w_T + (size_t)j * in_dim;
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < in_dim; i += 8)
            acc = vpadalq_s16(acc, vmull_s8(vld1_s8(xq+i), vld1_s8(w+i)));
        y[j]=(float)vaddvq_s32(acc)*x_scale*scale[j]+(b?b[j]:0);
    }
}

/* D) 8-output parallel with SDOT */
#ifdef __ARM_FEATURE_DOTPROD
static void q8_8out_sdot(float *y, const int8_t *w_T, const float *scale,
                         const int32_t *w_sums, const float *x, const float *b,
                         int in_dim, int out_dim) {
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) x_max = fmaxf(x_max, fabsf(x[i]));
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    int8_t xq[4096];
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] / x_scale);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }
    int j = 0;
    for (; j + 8 <= out_dim; j += 8) {
        int32x4_t a0=vdupq_n_s32(0), a1=vdupq_n_s32(0);
        int32x4_t a2=vdupq_n_s32(0), a3=vdupq_n_s32(0);
        int32x4_t a4=vdupq_n_s32(0), a5=vdupq_n_s32(0);
        int32x4_t a6=vdupq_n_s32(0), a7=vdupq_n_s32(0);
        const int8_t *w0=w_T+(size_t)(j+0)*in_dim, *w1=w_T+(size_t)(j+1)*in_dim;
        const int8_t *w2=w_T+(size_t)(j+2)*in_dim, *w3=w_T+(size_t)(j+3)*in_dim;
        const int8_t *w4=w_T+(size_t)(j+4)*in_dim, *w5=w_T+(size_t)(j+5)*in_dim;
        const int8_t *w6=w_T+(size_t)(j+6)*in_dim, *w7=w_T+(size_t)(j+7)*in_dim;
        for (int i = 0; i < in_dim; i += 16) {
            int8x16_t xv = vld1q_s8(xq + i);
            a0 = vdotq_s32(a0, xv, vld1q_s8(w0+i));
            a1 = vdotq_s32(a1, xv, vld1q_s8(w1+i));
            a2 = vdotq_s32(a2, xv, vld1q_s8(w2+i));
            a3 = vdotq_s32(a3, xv, vld1q_s8(w3+i));
            a4 = vdotq_s32(a4, xv, vld1q_s8(w4+i));
            a5 = vdotq_s32(a5, xv, vld1q_s8(w5+i));
            a6 = vdotq_s32(a6, xv, vld1q_s8(w6+i));
            a7 = vdotq_s32(a7, xv, vld1q_s8(w7+i));
        }
        y[j+0]=(float)vaddvq_s32(a0)*x_scale*scale[j+0]+(b?b[j+0]:0);
        y[j+1]=(float)vaddvq_s32(a1)*x_scale*scale[j+1]+(b?b[j+1]:0);
        y[j+2]=(float)vaddvq_s32(a2)*x_scale*scale[j+2]+(b?b[j+2]:0);
        y[j+3]=(float)vaddvq_s32(a3)*x_scale*scale[j+3]+(b?b[j+3]:0);
        y[j+4]=(float)vaddvq_s32(a4)*x_scale*scale[j+4]+(b?b[j+4]:0);
        y[j+5]=(float)vaddvq_s32(a5)*x_scale*scale[j+5]+(b?b[j+5]:0);
        y[j+6]=(float)vaddvq_s32(a6)*x_scale*scale[j+6]+(b?b[j+6]:0);
        y[j+7]=(float)vaddvq_s32(a7)*x_scale*scale[j+7]+(b?b[j+7]:0);
    }
    for (; j < out_dim; j++) {
        const int8_t *w = w_T + (size_t)j * in_dim;
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < in_dim; i += 16) {
            acc = vdotq_s32(acc, vld1q_s8(xq+i), vld1q_s8(w+i));
        }
        y[j]=(float)vaddvq_s32(acc)*x_scale*scale[j]+(b?b[j]:0);
    }
}
#endif

#endif /* __aarch64__ */

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void) {
#ifdef __aarch64__
    srand(42);
    int8_t *w_T = malloc((size_t)IN_DIM*OUT_DIM);
    float *scale = malloc(OUT_DIM*sizeof(float));
    int32_t *w_sums = calloc(OUT_DIM, sizeof(int32_t));
    float *x = malloc(IN_DIM*sizeof(float));
    float *b = malloc(OUT_DIM*sizeof(float));
    float *y = malloc(OUT_DIM*sizeof(float));

    for (int i=0;i<IN_DIM;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*4;
    for (int j=0;j<OUT_DIM;j++) scale[j]=0.001f*j;
    for (size_t i=0;i<(size_t)IN_DIM*OUT_DIM;i++) w_T[i]=(rand()%255)-127;

    printf("=== NEON Q8 matmul benchmark (arm64, %dx%d) ===\n\n", IN_DIM, OUT_DIM);

    /* A) vmull_s8 single output */
    double t0=now_sec();
    for (int t=0;t<N_TRIALS;t++) q8_vmull(y,w_T,scale,w_sums,x,b,IN_DIM,OUT_DIM);
    double t_a=now_sec()-t0;
    printf("  A) vmull_s8 (1-out):     %.3f ms  (%.1f MOps/s)\n", t_a/N_TRIALS*1000, (double)IN_DIM*OUT_DIM*N_TRIALS/t_a/1e6);

    /* B) SDOT single output */
    #ifdef __ARM_FEATURE_DOTPROD
    t0=now_sec();
    for (int t=0;t<N_TRIALS;t++) q8_sdot(y,w_T,scale,w_sums,x,b,IN_DIM,OUT_DIM);
    double t_b=now_sec()-t0;
    printf("  B) SDOT (1-out):          %.3f ms  (%.1f MOps/s)  %.2fx vs A\n", t_b/N_TRIALS*1000, (double)IN_DIM*OUT_DIM*N_TRIALS/t_b/1e6, t_a/t_b);
    #else
    printf("  B) SDOT: NOT AVAILABLE (compile with -march=armv8.2-a+dotprod)\n");
    #endif

    /* C) 8-output vmull_s8 */
    t0=now_sec();
    for (int t=0;t<N_TRIALS;t++) q8_8out_vmull(y,w_T,scale,w_sums,x,b,IN_DIM,OUT_DIM);
    double t_c=now_sec()-t0;
    printf("  C) vmull_s8 (8-out):      %.3f ms  (%.1f MOps/s)  %.2fx vs A\n", t_c/N_TRIALS*1000, (double)IN_DIM*OUT_DIM*N_TRIALS/t_c/1e6, t_a/t_c);

    /* D) 8-output SDOT */
    #ifdef __ARM_FEATURE_DOTPROD
    t0=now_sec();
    for (int t=0;t<N_TRIALS;t++) q8_8out_sdot(y,w_T,scale,w_sums,x,b,IN_DIM,OUT_DIM);
    double t_d=now_sec()-t0;
    printf("  D) SDOT (8-out):          %.3f ms  (%.1f MOps/s)  %.2fx vs A\n", t_d/N_TRIALS*1000, (double)IN_DIM*OUT_DIM*N_TRIALS/t_d/1e6, t_a/t_d);
    #else
    printf("  D) SDOT (8-out): NOT AVAILABLE\n");
    #endif

    printf("\n  Estimated end-to-end (48 matmuls + 30%% overhead):\n");
    printf("    A: %.1f tok/s\n", 1000.0/(t_a/N_TRIALS*1000*48*1.3));
    printf("    C: %.1f tok/s\n", 1000.0/(t_c/N_TRIALS*1000*48*1.3));
    #ifdef __ARM_FEATURE_DOTPROD
    printf("    D: %.1f tok/s\n", 1000.0/(t_d/N_TRIALS*1000*48*1.3));
    #endif

    free(w_T);free(scale);free(w_sums);free(x);free(b);free(y);
#else
    printf("This benchmark is for arm64 only.\n");
#endif
    return 0;
}
