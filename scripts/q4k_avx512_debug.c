/* q4k_avx512_debug.c — Debug AVX512 kernel by comparing with manual dequant */
#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel_avx512.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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
    int in_dim = 256, out_dim = 8;  /* single superblock */
    float *w = malloc((size_t)out_dim*in_dim*sizeof(float));
    float *x = malloc(in_dim*sizeof(float));
    uint8_t *q4k = malloc((size_t)out_dim * (in_dim/256) * 144);
    float *y_k = malloc(out_dim*sizeof(float));
    float *y_r = malloc(out_dim*sizeof(float));
    float *w_dq = malloc(in_dim*sizeof(float));
    float *y_dq = malloc(out_dim*sizeof(float));
    srand(42);
    for (size_t i=0;i<(size_t)out_dim*in_dim;i++)
        w[i] = ((float)rand()/RAND_MAX-0.5f)*0.1f;
    for (int i=0;i<in_dim;i++) x[i] = ((float)rand()/RAND_MAX-0.5f)*0.3f;
    for (int j=0;j<out_dim;j++) {
        y_r[j]=0; for (int i=0;i<in_dim;i++) y_r[j]+=w[j*in_dim+i]*x[i];
    }
    for (int j=0;j<out_dim;j++)
        quantize_q4_k_adjacent(w+j*in_dim, q4k+(size_t)j*(in_dim/256)*144, in_dim);
    /* Dequant row 0 using ADJACENT formula */
    {
        uint8_t *sb = q4k;
        uint16_t d_u16 = *(uint16_t*)(sb);
        uint16_t dmin_u16 = *(uint16_t*)(sb+2);
        float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)d_u16)));
        float dmin = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)dmin_u16)));
        uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);
        printf("d=%.6f dmin=%.6f\n", d, dmin);
        printf("sc=[%d..%d] m=[%d..%d]\n", sm[0],sm[7],sm[8],sm[15]);
        for (int sub = 0; sub < 8; sub++) {
            float ascale = d * sm[sub] / 63.0f;
            float amin = dmin * sm[8+sub] / 63.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t bv = sb[16 + sub*16 + i];
                int q_even = bv & 0xF;
                int q_odd = (bv >> 4) & 0xF;
                w_dq[sub*32 + 2*i]     = ascale * q_even - amin;
                w_dq[sub*32 + 2*i + 1] = ascale * q_odd - amin;
            }
        }
        /* Compute dequant dot */
        y_dq[0] = 0;
        for (int i = 0; i < in_dim; i++) y_dq[0] += w_dq[i] * x[i];
        /* Max dequant element error */
        float max_deq = 0;
        for (int i = 0; i < in_dim; i++) {
            float e = fabsf(w_dq[i] - w[i]);
            if (e > max_deq) max_deq = e;
        }
        printf("Max dequant element error: %.6f\n", max_deq);
    }
    lal_matmul_q4_k_avx512(y_k, q4k, x, NULL, in_dim, out_dim);
    printf("\nrow 0: ref=%+.6f dequant_dot=%+.6f kernel=%+.6f\n", y_r[0], y_dq[0], y_k[0]);
    printf("  kernel vs dequant: %.6f\n", fabsf(y_k[0]-y_dq[0]));
    printf("  dequant vs ref:    %.6f\n", fabsf(y_dq[0]-y_r[0]));
    free(w);free(x);free(q4k);free(y_k);free(y_r);free(w_dq);free(y_dq);
    return 0;
}
