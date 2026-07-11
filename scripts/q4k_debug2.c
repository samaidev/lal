/* q4k_debug2.c — Isolate bug: compare kernel vs manual dequant+dot */
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
        /* Pack 16 × 6-bit into 12 bytes — use __uint128_t to avoid overflow */
        __uint128_t bits = 0;
        for (int j = 0; j < 16; j++)
            bits |= ((__uint128_t)(combined[j] & 0x3F)) << (j * 6);
        for (int b = 0; b < 12; b++)
            sb[4+b] = (bits >> (b * 8)) & 0xFF;
        uint8_t *qs = sb + 16;
        for (int j = 0; j < 8; j++) {
            float ascale = d * sc6[j] / 63.0f;
            float amin = dmin * m6[j] / 63.0f;
            if (s == 0 && j == 2) {
                printf("  [quant debug] s=0 sub=2: d=%.6f dmin=%.6f sc6=%d m6=%d ascale=%.6f amin=%.6f\n",
                       d, dmin, sc6[j], m6[j], ascale, amin);
            }
            for (int i = 0; i < 16; i++) {
                int idx_lo = j*32 + i;
                int idx_hi = j*32 + i + 16;
                int q_lo = lroundf((wb[idx_lo] + amin) / (ascale + 1e-8f));
                int q_hi = lroundf((wb[idx_hi] + amin) / (ascale + 1e-8f));
                if (s == 0 && j == 2 && i == 11) {
                    printf("  [quant debug] s=0 sub=2 i=11: wb[91]=%.6f q_hi=%d (raw)\n",
                           wb[idx_hi], q_hi);
                }
                if (q_lo < 0) q_lo = 0; if (q_lo > 15) q_lo = 15;
                if (q_hi < 0) q_hi = 0; if (q_hi > 15) q_hi = 15;
                if (s == 0 && j == 2 && i == 11) {
                    printf("  [quant debug] s=0 sub=2 i=11: q_hi=%d (clipped) byte=0x%02x\n",
                           q_hi, (uint8_t)q_lo | ((uint8_t)q_hi << 4));
                }
                qs[j*16 + i] = (uint8_t)q_lo | ((uint8_t)q_hi << 4);
            }
        }
    }
}

/* Dequantize a Q4_K row back to float, using the SAME formula the kernel expects */
static void dequant_q4_k_row(const uint8_t *q4k, float *out, int in_dim) {
    int n_super = in_dim / 256;
    for (int s = 0; s < n_super; s++) {
        const uint8_t *sb = q4k + s * 144;
        uint16_t d_u16 = *(const uint16_t*)(sb);
        uint16_t dmin_u16 = *(const uint16_t*)(sb+2);
        __m128i d_raw = _mm_set1_epi16((short)d_u16);
        __m128i dmin_raw = _mm_set1_epi16((short)dmin_u16);
        float d = _mm_cvtss_f32(_mm_cvtph_ps(d_raw));
        float dmin = _mm_cvtss_f32(_mm_cvtph_ps(dmin_raw));
        uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);
        const uint8_t *qs = sb + 16;
        for (int sub = 0; sub < 8; sub++) {
            float ascale = d * sm[sub] / 63.0f;
            float amin = dmin * sm[8+sub] / 63.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t bv = qs[sub*16 + i];
                /* INTERLEAVED: byte[i] = q[sub*32+i] | (q[sub*32+i+16] << 4) */
                out[s*256 + sub*32 + i]      = ascale * (bv & 0xF) - amin;
                out[s*256 + sub*32 + i + 16] = ascale * ((bv>>4) & 0xF) - amin;
            }
        }
    }
}

int main() {
    /* Test: 1 row × 2 superblocks — compare kernel vs manual dequant dot */
    int in_dim = 512, out_dim = 1;
    float *w = malloc((size_t)out_dim*in_dim*sizeof(float));
    float *x = malloc(in_dim*sizeof(float));
    uint8_t *q4k = malloc((size_t)out_dim * (in_dim/256) * 144);
    float *y_k = malloc(out_dim*sizeof(float));
    float *y_r = malloc(out_dim*sizeof(float));
    float *w_dq = malloc(in_dim*sizeof(float));  /* dequantized weights */
    float *y_dq = malloc(out_dim*sizeof(float));  /* dot using dequantized weights */
    srand(43);
    for (int j=0;j<out_dim;j++) for (int i=0;i<in_dim;i++)
        w[j*in_dim+i] = ((float)rand()/RAND_MAX-0.5f)*0.2f;
    for (int i=0;i<in_dim;i++) x[i] = ((float)rand()/RAND_MAX-0.5f)*0.5f;

    /* Float ref */
    for (int j=0;j<out_dim;j++) {
        y_r[j]=0; for (int i=0;i<in_dim;i++) y_r[j]+=w[j*in_dim+i]*x[i];
    }
    /* Quantize */
    for (int j=0;j<out_dim;j++)
        quantize_q4_k_row(w+j*in_dim, q4k+(size_t)j*(in_dim/256)*144, in_dim);
    /* Dequant */
    dequant_q4_k_row(q4k, w_dq, in_dim);
    /* Dot using dequantized weights */
    for (int j=0;j<out_dim;j++) {
        y_dq[j]=0; for (int i=0;i<in_dim;i++) y_dq[j]+=w_dq[i]*x[i];
    }
    /* Kernel */
    lal_matmul_q4_k(y_k, q4k, x, NULL, in_dim, out_dim);

    printf("=== 1 row × 2 superblocks (in_dim=512) ===\n");
    printf("  float ref:  %+.6f\n", y_r[0]);
    printf("  dequant dot: %+.6f  (this is what kernel SHOULD produce)\n", y_dq[0]);
    printf("  kernel:     %+.6f\n", y_k[0]);
    printf("  kernel vs dequant err: %.6f\n", fabsf(y_k[0]-y_dq[0]));
    printf("  dequant vs float err:  %.6f (quantization noise)\n\n", fabsf(y_dq[0]-y_r[0]));

    /* Check dequant error per element */
    float max_dq_err = 0;
    int max_dq_idx = 0;
    for (int i = 0; i < in_dim; i++) {
        float e = fabsf(w_dq[i] - w[i]);
        if (e > max_dq_err) { max_dq_err = e; max_dq_idx = i; }
    }
    printf("  Max dequant element error: %.6f at idx %d\n", max_dq_err, max_dq_idx);
    printf("    w_orig[%d] = %.6f, w_dq[%d] = %.6f\n", max_dq_idx, w[max_dq_idx], max_dq_idx, w_dq[max_dq_idx]);
    /* Which superblock/sub-block? */
    {
        int s = max_dq_idx / 256;
        int sub = (max_dq_idx % 256) / 32;
        int i_in_sub = max_dq_idx % 32;
        printf("    superblock %d, sub %d, idx_in_sub %d\n", s, sub, i_in_sub);
        uint8_t *sb = q4k + s*144;
        uint16_t d_u16 = *(const uint16_t*)(sb);
        uint16_t dmin_u16 = *(const uint16_t*)(sb+2);
        __m128i d_raw = _mm_set1_epi16((short)d_u16);
        __m128i dmin_raw = _mm_set1_epi16((short)dmin_u16);
        float d = _mm_cvtss_f32(_mm_cvtph_ps(d_raw));
        float dmin = _mm_cvtss_f32(_mm_cvtph_ps(dmin_raw));
        uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);
        float ascale = d * sm[sub] / 63.0f;
        float amin = dmin * sm[8+sub] / 63.0f;
        printf("    d=%.6f dmin=%.6f sc=%d m=%d ascale=%.6f amin=%.6f\n",
               d, dmin, sm[sub], sm[8+sub], ascale, amin);
        /* Check the byte */
        uint8_t bv = sb[16 + sub*16 + (i_in_sub < 16 ? i_in_sub : i_in_sub - 16)];
        int q_val = (i_in_sub < 16) ? (bv & 0xF) : ((bv>>4) & 0xF);
        printf("    byte=0x%02x q_val=%d dequant=%.6f\n", bv, q_val, ascale*q_val - amin);
        /* What SHOULD q be? */
        int q_should = lroundf((w[max_dq_idx] + amin) / ascale);
        printf("    q_should=%d (clipped 0-15: %d)\n", q_should,
               q_should < 0 ? 0 : (q_should > 15 ? 15 : q_should));
    }

    /* Dump per-superblock contribution for dequant dot */
    printf("\n  Per-superblock contributions (dequant dot):\n");
    for (int s = 0; s < in_dim/256; s++) {
        float sb_dot = 0;
        for (int i = 0; i < 256; i++) {
            sb_dot += w_dq[s*256+i] * x[i];
        }
        printf("    superblock %d: %.6f\n", s, sb_dot);
    }

    /* Also dump the quantization params for superblock 1 */
    printf("\n  Superblock 1 quantization params:\n");
    {
        uint8_t *sb = q4k + 144;
        uint16_t d_u16 = *(const uint16_t*)(sb);
        uint16_t dmin_u16 = *(const uint16_t*)(sb+2);
        __m128i d_raw = _mm_set1_epi16((short)d_u16);
        __m128i dmin_raw = _mm_set1_epi16((short)dmin_u16);
        float d = _mm_cvtss_f32(_mm_cvtph_ps(d_raw));
        float dmin = _mm_cvtss_f32(_mm_cvtph_ps(dmin_raw));
        uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);
        printf("    d=%.6f dmin=%.6f\n", d, dmin);
        printf("    sc=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
               sm[0],sm[1],sm[2],sm[3],sm[4],sm[5],sm[6],sm[7]);
        printf("    m =[%d,%d,%d,%d,%d,%d,%d,%d]\n",
               sm[8],sm[9],sm[10],sm[11],sm[12],sm[13],sm[14],sm[15]);
        /* Check first few dequant values */
        printf("    w_dq[256..259]: %.6f %.6f %.6f %.6f\n",
               w_dq[256], w_dq[257], w_dq[258], w_dq[259]);
        printf("    w_orig[256..259]: %.6f %.6f %.6f %.6f\n",
               w[256], w[257], w[258], w[259]);
    }

    free(w);free(x);free(q4k);free(y_k);free(y_r);free(w_dq);free(y_dq);
    return 0;
}
