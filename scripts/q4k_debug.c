/* q4k_debug.c — Isolate 8-row parallel bug */
#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static void quantize_q4_k_row(const float *w, uint8_t *out, int in_dim) {
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
        /* Pack 16 × 6-bit into 12 bytes — use __uint128_t to avoid uint64_t overflow */
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
                int idx_lo = j*32 + i;
                int idx_hi = j*32 + i + 16;
                int q_lo = lroundf((wb[idx_lo] + amin) / (ascale + 1e-8f));
                int q_hi = lroundf((wb[idx_hi] + amin) / (ascale + 1e-8f));
                if (q_lo < 0) q_lo = 0; if (q_lo > 15) q_lo = 15;
                if (q_hi < 0) q_hi = 0; if (q_hi > 15) q_hi = 15;
                qs[j*16 + i] = (uint8_t)q_lo | ((uint8_t)q_hi << 4);
            }
        }
    }
}

int main() {
    /* Test 1: 8 rows × 1 superblock (in_dim=256) — exercises 8-row path with 1 superblock */
    {
        int in_dim = 256, out_dim = 8;
        float *w = malloc((size_t)out_dim*in_dim*sizeof(float));
        float *x = malloc(in_dim*sizeof(float));
        uint8_t *q4k = malloc((size_t)out_dim * (in_dim/256) * 144);
        float *y_k = malloc(out_dim*sizeof(float));
        float *y_r = malloc(out_dim*sizeof(float));
        srand(42);
        for (int j=0;j<out_dim;j++) for (int i=0;i<in_dim;i++)
            w[j*in_dim+i] = ((float)rand()/RAND_MAX-0.5f)*0.2f;
        for (int i=0;i<in_dim;i++) x[i] = ((float)rand()/RAND_MAX-0.5f)*0.5f;
        for (int j=0;j<out_dim;j++) {
            y_r[j]=0; for (int i=0;i<in_dim;i++) y_r[j]+=w[j*in_dim+i]*x[i];
        }
        for (int j=0;j<out_dim;j++)
            quantize_q4_k_row(w+j*in_dim, q4k+(size_t)j*(in_dim/256)*144, in_dim);
        lal_matmul_q4_k(y_k, q4k, x, NULL, in_dim, out_dim);
        printf("=== Test 1: 8 rows × 1 superblock (in_dim=256) ===\n");
        float max_rel=0;
        for (int j=0;j<out_dim;j++) {
            float err=fabsf(y_k[j]-y_r[j]);
            float rel=err/(fabsf(y_r[j])+1e-8f);
            if (rel>max_rel) max_rel=rel;
            printf("  row %d: ref=%+.6f kernel=%+.6f err=%.6f rel=%.4f\n",
                   j,y_r[j],y_k[j],err,rel);
        }
        printf("  Max rel: %.4f%% %s\n\n", max_rel*100, max_rel<0.15?"PASS":"FAIL");
        free(w);free(x);free(q4k);free(y_k);free(y_r);
    }
    /* Test 2: 8 rows × 2 superblocks (in_dim=512) */
    {
        int in_dim = 512, out_dim = 8;
        float *w = malloc((size_t)out_dim*in_dim*sizeof(float));
        float *x = malloc(in_dim*sizeof(float));
        uint8_t *q4k = malloc((size_t)out_dim * (in_dim/256) * 144);
        float *y_k = malloc(out_dim*sizeof(float));
        float *y_r = malloc(out_dim*sizeof(float));
        srand(43);
        for (int j=0;j<out_dim;j++) for (int i=0;i<in_dim;i++)
            w[j*in_dim+i] = ((float)rand()/RAND_MAX-0.5f)*0.2f;
        for (int i=0;i<in_dim;i++) x[i] = ((float)rand()/RAND_MAX-0.5f)*0.5f;
        for (int j=0;j<out_dim;j++) {
            y_r[j]=0; for (int i=0;i<in_dim;i++) y_r[j]+=w[j*in_dim+i]*x[i];
        }
        for (int j=0;j<out_dim;j++)
            quantize_q4_k_row(w+j*in_dim, q4k+(size_t)j*(in_dim/256)*144, in_dim);
        lal_matmul_q4_k(y_k, q4k, x, NULL, in_dim, out_dim);
        printf("=== Test 2: 8 rows × 2 superblocks (in_dim=512) ===\n");
        float max_rel=0;
        for (int j=0;j<out_dim;j++) {
            float err=fabsf(y_k[j]-y_r[j]);
            float rel=err/(fabsf(y_r[j])+1e-8f);
            if (rel>max_rel) max_rel=rel;
            printf("  row %d: ref=%+.6f kernel=%+.6f err=%.6f rel=%.4f\n",
                   j,y_r[j],y_k[j],err,rel);
        }
        printf("  Max rel: %.4f%% %s\n\n", max_rel*100, max_rel<0.15?"PASS":"FAIL");
        free(w);free(x);free(q4k);free(y_k);free(y_r);
    }
    /* Test 3: 1 row × 14 superblocks (in_dim=3584) — exercises TAIL path with many superblocks */
    {
        int in_dim = 3584, out_dim = 1;
        float *w = malloc((size_t)out_dim*in_dim*sizeof(float));
        float *x = malloc(in_dim*sizeof(float));
        uint8_t *q4k = malloc((size_t)out_dim * (in_dim/256) * 144);
        float *y_k = malloc(out_dim*sizeof(float));
        float *y_r = malloc(out_dim*sizeof(float));
        srand(44);
        for (int j=0;j<out_dim;j++) for (int i=0;i<in_dim;i++)
            w[j*in_dim+i] = ((float)rand()/RAND_MAX-0.5f)*0.1f;
        for (int i=0;i<in_dim;i++) x[i] = ((float)rand()/RAND_MAX-0.5f)*0.3f;
        for (int j=0;j<out_dim;j++) {
            y_r[j]=0; for (int i=0;i<in_dim;i++) y_r[j]+=w[j*in_dim+i]*x[i];
        }
        for (int j=0;j<out_dim;j++)
            quantize_q4_k_row(w+j*in_dim, q4k+(size_t)j*(in_dim/256)*144, in_dim);
        lal_matmul_q4_k(y_k, q4k, x, NULL, in_dim, out_dim);
        printf("=== Test 3: 1 row × 14 superblocks (in_dim=3584, TAIL path) ===\n");
        float max_rel=0;
        for (int j=0;j<out_dim;j++) {
            float err=fabsf(y_k[j]-y_r[j]);
            float rel=err/(fabsf(y_r[j])+1e-8f);
            if (rel>max_rel) max_rel=rel;
            printf("  row %d: ref=%+.6f kernel=%+.6f err=%.6f rel=%.4f\n",
                   j,y_r[j],y_k[j],err,rel);
        }
        printf("  Max rel: %.4f%% %s\n\n", max_rel*100, max_rel<0.15?"PASS":"FAIL");
        free(w);free(x);free(q4k);free(y_k);free(y_r);
    }
    return 0;
}
