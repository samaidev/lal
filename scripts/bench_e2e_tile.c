/* bench_e2e_tile.c — E2E forward with tile-major Q4_K weights
 * 所有 7 个 matmul/层都用 tile-major 布局
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_e2e_tile scripts/bench_e2e_tile.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define N_EMBD    3584
#define N_LAYER   28
#define N_HEAD    28
#define N_KV_HEAD 4
#define HEAD_DIM  128
#define KV_DIM    (N_KV_HEAD * HEAD_DIM)
#define Q_DIM     (N_HEAD * HEAD_DIM)
#define MLP_DIM   18944
#define N_CTX     4096
#define RMS_EPS   1e-6f
#define N_Q_PER_KV (N_HEAD / N_KV_HEAD)
#define XQ_MAX 18944

#include "runtime/lal_q4k_kernel.h"
#include "runtime/lal_simd_optim.h"

/* Tile-major Q4_K matmul (from bench_layout.c, inline) */
static inline void matmul_q4_k_tile(float * __restrict__ y,
                                     const uint8_t * __restrict__ q4k_tile_W,
                                     const float * __restrict__ x,
                                     const float * __restrict__ b,
                                     int in_dim, int out_dim) {
    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;
    int8_t xq[XQ_MAX] __attribute__((aligned(32)));
    float x_max = 0;
    for (int i = 0; i < in_dim; i++) { float a = fabsf(x[i]); if (a > x_max) x_max = a; }
    float x_scale = x_max / 127.0f;
    if (x_scale < 1e-8f) x_scale = 1e-8f;
    float x_inv = 1.0f / x_scale;
    for (int i = 0; i < in_dim; i++) {
        int v = (int)lroundf(x[i] * x_inv);
        xq[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }
    int16_t bsums[XQ_MAX / 32] __attribute__((aligned(32)));
    for (int sb = 0; sb < n_sub; sb++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) sum += (int)xq[sb * 32 + i];
        bsums[sb] = (int16_t)sum;
    }
#if defined(__AVX2__) && defined(__F16C__)
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const float inv_63 = 1.0f / 63.0f;
    int8_t xq_arr[XQ_MAX] __attribute__((aligned(32)));
    for (int s = 0; s < n_super; s++) {
        int8_t *dst_even = xq_arr + s*256;
        int8_t *dst_odd  = xq_arr + s*256 + 128;
        for (int k = 0; k < 8; k++) {
            const int8_t *src = xq + s*256 + k*32;
            for (int i = 0; i < 16; i++) {
                dst_even[k*16 + i] = src[2*i];
                dst_odd[k*16 + i]  = src[2*i + 1];
            }
        }
    }
    int tile_group_stride = 8 * 144;
    int tile_super_stride = (out_dim / 8) * tile_group_stride;
    int j = 0;
    for (; j + 8 <= out_dim; j += 8) {
        __m256 acc0=_mm256_setzero_ps(),acc1=_mm256_setzero_ps();
        __m256 acc2=_mm256_setzero_ps(),acc3=_mm256_setzero_ps();
        __m256 acc4=_mm256_setzero_ps(),acc5=_mm256_setzero_ps();
        __m256 acc6=_mm256_setzero_ps(),acc7=_mm256_setzero_ps();
        float am0=0,am1=0,am2=0,am3=0,am4=0,am5=0,am6=0,am7=0;
        int jg = j / 8;
        for (int s = 0; s < n_super; s++) {
            const __m256i xve0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+0));
            const __m256i xve1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+32));
            const __m256i xve2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+64));
            const __m256i xve3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+96));
            const __m256i xvo0=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+128));
            const __m256i xvo1=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+160));
            const __m256i xvo2=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+192));
            const __m256i xvo3=_mm256_loadu_si256((__m256i*)(xq_arr+s*256+224));
            __m128i bs_v=_mm_set_epi16(bsums[s*8+7],bsums[s*8+6],bsums[s*8+5],bsums[s*8+4],
                bsums[s*8+3],bsums[s*8+2],bsums[s*8+1],bsums[s*8+0]);
            const uint8_t *tile_base = q4k_tile_W + (size_t)s * tile_super_stride + (size_t)jg * tile_group_stride;
            #define PROC8T(r) do { \
                const uint8_t *sb=tile_base + r*144; \
                float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb)))); \
                float dmin=_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)*(const uint16_t*)(sb+2)))); \
                uint8_t sm[16] __attribute__((aligned(16))); unpack_scales_6bit(sb+4,sm); \
                __m128i mn_bytes = _mm_loadl_epi64((__m128i*)(sm+8)); \
                __m128i mn_v = _mm_cvtepi8_epi16(mn_bytes); \
                __m128i mp=_mm_madd_epi16(mn_v,bs_v); \
                int32_t mp_sum = _mm_cvtsi128_si32(mp)+_mm_extract_epi32(mp,1)+_mm_extract_epi32(mp,2)+_mm_extract_epi32(mp,3); \
                float r_am=dmin*x_scale*(float)mp_sum*inv_63; \
                if(r==0)am0-=r_am;else if(r==1)am1-=r_am;else if(r==2)am2-=r_am; \
                else if(r==3)am3-=r_am;else if(r==4)am4-=r_am;else if(r==5)am5-=r_am; \
                else if(r==6)am6-=r_am;else am7-=r_am; \
                __m256i sumi=_mm256_setzero_si256(); \
                const uint8_t*qs=sb+16; \
                _mm_prefetch((const char*)(qs+144), _MM_HINT_T0); \
                _mm_prefetch((const char*)(qs+144+64), _MM_HINT_T0); \
                { __m256i q4b=_mm256_loadu_si256((__m256i*)qs); __m256i q4l=_mm256_and_si256(q4b,m4); __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                  __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve0),_mm256_maddubs_epi16(q4h,xvo0)); \
                  __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[1]),_mm_set1_epi16((short)sm[0])); \
                  sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); } \
                { __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+32)); __m256i q4l=_mm256_and_si256(q4b,m4); __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                  __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve1),_mm256_maddubs_epi16(q4h,xvo1)); \
                  __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[3]),_mm_set1_epi16((short)sm[2])); \
                  sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); } \
                { __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+64)); __m256i q4l=_mm256_and_si256(q4b,m4); __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                  __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve2),_mm256_maddubs_epi16(q4h,xvo2)); \
                  __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[5]),_mm_set1_epi16((short)sm[4])); \
                  sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); } \
                { __m256i q4b=_mm256_loadu_si256((__m256i*)(qs+96)); __m256i q4l=_mm256_and_si256(q4b,m4); __m256i q4h=_mm256_and_si256(_mm256_srli_epi16(q4b,4),m4); \
                  __m256i p16=_mm256_add_epi16(_mm256_maddubs_epi16(q4l,xve3),_mm256_maddubs_epi16(q4h,xvo3)); \
                  __m256i sc_v=_mm256_set_m128i(_mm_set1_epi16((short)sm[7]),_mm_set1_epi16((short)sm[6])); \
                  sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(sc_v,p16)); } \
                float mult=d*x_scale*inv_63; \
                __m256 *ap=(r==0)?&acc0:(r==1)?&acc1:(r==2)?&acc2:(r==3)?&acc3:(r==4)?&acc4:(r==5)?&acc5:(r==6)?&acc6:&acc7; \
                *ap=_mm256_fmadd_ps(_mm256_set1_ps(mult),_mm256_cvtepi32_ps(sumi),*ap); \
            } while(0)
            PROC8T(0);PROC8T(1);PROC8T(2);PROC8T(3);PROC8T(4);PROC8T(5);PROC8T(6);PROC8T(7);
            #undef PROC8T
        }
        #define HS8T2(r,a,m) do{__m128 lo=_mm256_castps256_ps128(a),hi=_mm256_extractf128_ps(a,1);\
            __m128 s2=_mm_add_ps(lo,hi);s2=_mm_hadd_ps(s2,s2);s2=_mm_hadd_ps(s2,s2);\
            y[j+r]=_mm_cvtss_f32(s2)+m+(b?b[j+r]:0);}while(0)
        HS8T2(0,acc0,am0);HS8T2(1,acc1,am1);HS8T2(2,acc2,am2);HS8T2(3,acc3,am3);
        HS8T2(4,acc4,am4);HS8T2(5,acc5,am5);HS8T2(6,acc6,am6);HS8T2(7,acc7,am7);
        #undef HS8T2
    }
#endif
}

static inline void parallel_matmul_tile(float *y, const uint8_t *q4k_W,
                                         const float *x, const float *b,
                                         int in_dim, int out_dim, int n_threads) {
    if (n_threads <= 1 || out_dim < 2048) {
        matmul_q4_k_tile(y, q4k_W, x, b, in_dim, out_dim);
        return;
    }
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        int n = omp_get_num_threads();
        int groups = out_dim / 8;
        int chunk = (groups + n - 1) / n;
        int gs = tid * chunk;
        int ge = gs + chunk;
        if (ge > groups) ge = groups;
        if (gs < groups)
            matmul_q4_k_tile(y + gs*8, q4k_W, x, b, in_dim, (ge-gs)*8);
    }
}

static void convert_to_tile(const uint8_t *src, uint8_t *dst, int out_dim, int n_super) {
    int tile_group_stride = 8 * 144;
    int tile_super_stride = (out_dim / 8) * tile_group_stride;
    for (int s = 0; s < n_super; s++) {
        for (int jg = 0; jg < out_dim / 8; jg++) {
            uint8_t *tile_base = dst + (size_t)s * tile_super_stride + (size_t)jg * tile_group_stride;
            for (int r = 0; r < 8; r++) {
                const uint8_t *src_row = src + (size_t)(jg*8 + r) * n_super * 144 + s * 144;
                memcpy(tile_base + r * 144, src_row, 144);
            }
        }
    }
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#define GEN_Q4K_ROW(name, out_dim, in_dim) \
    uint8_t *name##_row = _mm_malloc((size_t)out_dim * ((in_dim)/256*144), 32); \
    uint8_t *name##_tile = _mm_malloc((size_t)out_dim * ((in_dim)/256*144), 32); \
    do { \
        for (int j = 0; j < out_dim; j++) { \
            uint8_t *row = name##_row + (size_t)j * ((in_dim)/256*144); \
            for (int s = 0; s < (in_dim)/256; s++) { \
                uint8_t *sb = row + s*144; \
                *(uint16_t*)sb = 0x3C00; *(uint16_t*)(sb+2) = 0x0000; \
                memset(sb+4, 0x20, 12); \
                for (int i = 0; i < 128; i++) sb[16+i] = rand() & 0xFF; \
            } \
        } \
        convert_to_tile(name##_row, name##_tile, out_dim, (in_dim)/256); \
    } while(0)

int main(int argc, char **argv) {
    int n_threads = argc > 1 ? atoi(argv[1]) : 2;
    int pos = argc > 2 ? atoi(argv[2]) : 128;
    int n_iter = argc > 3 ? atoi(argv[3]) : 3;
    omp_set_num_threads(n_threads);
    srand(42);

    float *x = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *ln = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *norm_w = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *q_buf = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *k_buf = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *v_buf = _mm_malloc(KV_DIM * sizeof(float), 32);
    float *attn_out = _mm_malloc(Q_DIM * sizeof(float), 32);
    float *proj = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *gate_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *up_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *act_buf = _mm_malloc(MLP_DIM * sizeof(float), 32);
    float *mlp_out = _mm_malloc(N_EMBD * sizeof(float), 32);
    float *kc = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);
    float *vc = _mm_malloc(N_CTX * KV_DIM * sizeof(float), 32);

    GEN_Q4K_ROW(w_q, Q_DIM, N_EMBD);
    GEN_Q4K_ROW(w_k, KV_DIM, N_EMBD);
    GEN_Q4K_ROW(w_v, KV_DIM, N_EMBD);
    GEN_Q4K_ROW(w_o, N_EMBD, Q_DIM);
    GEN_Q4K_ROW(w_gate, MLP_DIM, N_EMBD);
    GEN_Q4K_ROW(w_up, MLP_DIM, N_EMBD);
    GEN_Q4K_ROW(w_down, N_EMBD, MLP_DIM);

    for (int i = 0; i < N_EMBD; i++) { x[i] = (rand()/((float)RAND_MAX)-0.5f)*0.3f; norm_w[i] = 0.9f; }

    printf("=== E2E Tile-major Forward (%d threads, pos=%d) ===\n", n_threads, pos);

    /* Warmup */
    for (int l = 0; l < 1; l++) {
        lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
        parallel_matmul_tile(q_buf, w_q_tile, ln, NULL, N_EMBD, Q_DIM, n_threads);
        parallel_matmul_tile(k_buf, w_k_tile, ln, NULL, N_EMBD, KV_DIM, n_threads);
        parallel_matmul_tile(v_buf, w_v_tile, ln, NULL, N_EMBD, KV_DIM, n_threads);
        lal_gqa_attn_simd(attn_out, q_buf, k_buf, v_buf, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
        parallel_matmul_tile(proj, w_o_tile, attn_out, NULL, Q_DIM, N_EMBD, n_threads);
        lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
        parallel_matmul_tile(gate_buf, w_gate_tile, ln, NULL, N_EMBD, MLP_DIM, n_threads);
        parallel_matmul_tile(up_buf, w_up_tile, ln, NULL, N_EMBD, MLP_DIM, n_threads);
        lal_silu_mul_simd(act_buf, gate_buf, up_buf, MLP_DIM);
        parallel_matmul_tile(mlp_out, w_down_tile, act_buf, NULL, MLP_DIM, N_EMBD, n_threads);
    }

    double t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        for (int l = 0; l < N_LAYER; l++) {
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
            parallel_matmul_tile(q_buf, w_q_tile, ln, NULL, N_EMBD, Q_DIM, n_threads);
            parallel_matmul_tile(k_buf, w_k_tile, ln, NULL, N_EMBD, KV_DIM, n_threads);
            parallel_matmul_tile(v_buf, w_v_tile, ln, NULL, N_EMBD, KV_DIM, n_threads);
            lal_gqa_attn_simd(attn_out, q_buf, k_buf, v_buf, kc, vc, pos, N_HEAD, N_KV_HEAD, HEAD_DIM, N_Q_PER_KV, KV_DIM, N_CTX);
            parallel_matmul_tile(proj, w_o_tile, attn_out, NULL, Q_DIM, N_EMBD, n_threads);
            lal_residual_add_simd(x, proj, N_EMBD);
            lal_rms_norm_simd(ln, x, norm_w, N_EMBD, RMS_EPS);
            parallel_matmul_tile(gate_buf, w_gate_tile, ln, NULL, N_EMBD, MLP_DIM, n_threads);
            parallel_matmul_tile(up_buf, w_up_tile, ln, NULL, N_EMBD, MLP_DIM, n_threads);
            lal_silu_mul_simd(act_buf, gate_buf, up_buf, MLP_DIM);
            parallel_matmul_tile(mlp_out, w_down_tile, act_buf, NULL, MLP_DIM, N_EMBD, n_threads);
            lal_residual_add_simd(x, mlp_out, N_EMBD);
        }
    }
    double dt = (now_ms() - t0) / n_iter;
    printf("Total: %.2f ms/token  (%.2f tok/s)\n", dt, 1000.0/dt);

    return 0;
}
