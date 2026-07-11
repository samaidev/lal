/* q4k_avx512_unit_test.c — Test AVX512 Q4_K kernel with ADJACENT packing
 * Build: gcc -O3 -mavx512f -mavx512bw -mfma -mf16c -I. -o q4k_avx512_test scripts/q4k_avx512_unit_test.c -lm
 */
#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel.h"
/* When AVX512_BW is available, lal_matmul_q4_k is #defined to lal_matmul_q4_k_avx512 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Quantize with ADJACENT packing: byte[i] = q[2i] | (q[2i+1] << 4) WITHIN each sub-block */
static void quantize_q4_k_adjacent(const float *w, uint8_t *out, int in_dim) {
    int n_super = in_dim / 256;
    for (int s = 0; s < n_super; s++) {
        const float *wb = w + s * 256;
        uint8_t *sb = out + s * 144;
        float sub_min[8], sub_max[8], sub_scale[8];
        float max_scale = 0, max_min = 0;
        for (int j = 0; j < 8; j++) {
            sub_min[j] = 1e30f; sub_max[j] = -1e30f;
            for (int i = 0; i < 32; i++) {
                float v = wb[j*32+i];
                if (v < sub_min[j]) sub_min[j] = v;
                if (v > sub_max[j]) sub_max[j] = v;
            }
            sub_scale[j] = (sub_max[j] - sub_min[j]) / 15.0f;
            if (sub_scale[j] < 1e-8f) sub_scale[j] = 1e-8f;
            if (sub_scale[j] > max_scale) max_scale = sub_scale[j];
            if (fabsf(sub_min[j]) > max_min) max_min = fabsf(sub_min[j]);
        }
        if (max_scale < 1e-8f) max_scale = 1e-8f;
        if (max_min < 1e-8f) max_min = 1e-8f;
        float d = max_scale, dmin = max_min;
        __m128 d_f = _mm_set1_ps(d);
        __m128i d_h = _mm_cvtps_ph(d_f, 0);
        uint16_t d_u16 = _mm_extract_epi16(d_h, 0);
        __m128 dmin_f = _mm_set1_ps(dmin);
        __m128i dmin_h = _mm_cvtps_ph(dmin_f, 0);
        uint16_t dmin_u16 = _mm_extract_epi16(dmin_h, 0);
        *(uint16_t*)(sb) = d_u16;
        *(uint16_t*)(sb+2) = dmin_u16;
        uint8_t sc6[8], m6[8];
        uint64_t combined[16];
        for (int j = 0; j < 8; j++) {
            sc6[j] = (uint8_t)(lroundf(sub_scale[j] / d * 63.0f) & 0x3F);
            m6[j] = (uint8_t)(lroundf(fabsf(sub_min[j]) / dmin * 63.0f) & 0x3F);
        }
        for (int j = 0; j < 8; j++) { combined[j] = sc6[j]; combined[j+8] = m6[j]; }
        __uint128_t bits = 0;
        for (int j = 0; j < 16; j++)
            bits |= ((__uint128_t)(combined[j] & 0x3F)) << (j * 6);
        for (int b = 0; b < 12; b++)
            sb[4+b] = (bits >> (b * 8)) & 0xFF;
        uint8_t *qs = sb + 16;
        /* ADJACENT packing: byte[sub*16 + i] = q[sub*32 + 2i] | (q[sub*32 + 2i+1] << 4) */
        for (int j = 0; j < 8; j++) {
            float ascale = d * sc6[j] / 63.0f;
            float amin = dmin * m6[j] / 63.0f;
            for (int i = 0; i < 16; i++) {
                int idx_even = j*32 + 2*i;
                int idx_odd  = j*32 + 2*i + 1;
                int q_even = lroundf((wb[idx_even] + amin) / (ascale + 1e-8f));
                int q_odd  = lroundf((wb[idx_odd]  + amin) / (ascale + 1e-8f));
                if (q_even < 0) q_even = 0; if (q_even > 15) q_even = 15;
                if (q_odd < 0)  q_odd = 0;  if (q_odd > 15)  q_odd = 15;
                qs[j*16 + i] = (uint8_t)q_even | ((uint8_t)q_odd << 4);
            }
        }
    }
}

int main() {
    int in_dim = 3584, out_dim = 16;
    float *w = malloc((size_t)out_dim*in_dim*sizeof(float));
    float *x = malloc(in_dim*sizeof(float));
    uint8_t *q4k = malloc((size_t)out_dim * (in_dim/256) * 144);
    float *y_k = malloc(out_dim*sizeof(float));
    float *y_r = malloc(out_dim*sizeof(float));
    srand(42);
    for (size_t i=0;i<(size_t)out_dim*in_dim;i++)
        w[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f;
    for (int i=0;i<in_dim;i++) x[i] = ((float)rand()/RAND_MAX-0.5f)*0.3f;
    for (int j=0;j<out_dim;j++) {
        y_r[j]=0; for (int i=0;i<in_dim;i++) y_r[j]+=w[j*in_dim+i]*x[i];
    }
    for (int j=0;j<out_dim;j++)
        quantize_q4_k_adjacent(w+j*in_dim, q4k+(size_t)j*(in_dim/256)*144, in_dim);
    lal_matmul_q4_k(y_k, q4k, x, NULL, in_dim, out_dim);

    printf("=== AVX512 Q4_K test (in_dim=%d, out_dim=%d) ===\n", in_dim, out_dim);
    float max_rel=0, max_abs=0;
    for (int j=0;j<out_dim;j++) {
        float err=fabsf(y_k[j]-y_r[j]);
        float rel=err/(fabsf(y_r[j])+1e-8f);
        if (fabsf(y_r[j])>0.05f && rel>max_rel) max_rel=rel;
        if (err>max_abs) max_abs=err;
        if (j < 8) printf("  row %d: ref=%+.6f kernel=%+.6f err=%.6f rel=%.4f\n",
               j,y_r[j],y_k[j],err,rel);
    }
    printf("\n  Max rel (ref>0.05): %.4f%%  Max abs: %.6f\n", max_rel*100, max_abs);
    if (max_abs < 0.03f) printf("  PASS (abs < 0.03)\n");
    else printf("  FAIL\n");
    free(w);free(x);free(q4k);free(y_k);free(y_r);
    return 0;
}
