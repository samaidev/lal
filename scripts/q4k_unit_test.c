/* q4k_unit_test.c — Unit test for Q4_K kernel correctness
 *
 * Tests: quantize a known weight matrix → run lal_matmul_q4_k → compare with float reference
 *
 * Build: gcc -O0 -g -mavx2 -mfma -mf16c -I. -o q4k_unit_test q4k_unit_test.c -lm
 */
#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Simple Q4_K quantization matching the Python converter */
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

        /* Pack d, dmin as fp16 */
        __m128 d_f = _mm_set1_ps(d);
        __m128i d_h = _mm_cvtps_ph(d_f, 0);
        uint16_t d_u16 = _mm_extract_epi16(d_h, 0);
        __m128 dmin_f = _mm_set1_ps(dmin);
        __m128i dmin_h = _mm_cvtps_ph(dmin_f, 0);
        uint16_t dmin_u16 = _mm_extract_epi16(dmin_h, 0);
        *(uint16_t*)(sb) = d_u16;
        *(uint16_t*)(sb+2) = dmin_u16;

        /* 6-bit scales and mins */
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

        /* Quantize values — ADJACENT packing:
         * byte[sub*16+i] = q[sub*32+2i] | (q[sub*32+2i+1] << 4) */
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
    int in_dim = 256;   /* 1 superblock */
    int out_dim = 4;    /* 4 output rows */
    float w[4][256], x[256];
    uint8_t q4k[4 * 144];
    float y_kernel[4], y_ref[4];

    /* Init with small random values */
    srand(42);
    for (int j = 0; j < out_dim; j++)
        for (int i = 0; i < in_dim; i++)
            w[j][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    for (int i = 0; i < in_dim; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;

    /* Float reference */
    for (int j = 0; j < out_dim; j++) {
        y_ref[j] = 0;
        for (int i = 0; i < in_dim; i++)
            y_ref[j] += w[j][i] * x[i];
    }

    /* Quantize */
    for (int j = 0; j < out_dim; j++)
        quantize_q4_k_row(w[j], q4k + j * 144, in_dim);

    /* Verify dequantization for row 0 */
    {
        uint8_t *sb = q4k;
        uint16_t d_u16 = *(uint16_t*)(sb);
        uint16_t dmin_u16 = *(uint16_t*)(sb+2);
        __m128i d_raw = _mm_set1_epi16((short)d_u16);
        __m128i dmin_raw = _mm_set1_epi16((short)dmin_u16);
        float d = _mm_cvtss_f32(_mm_cvtph_ps(d_raw));
        float dmin = _mm_cvtss_f32(_mm_cvtph_ps(dmin_raw));
        uint8_t scales_mins[16];
        unpack_scales_6bit(sb + 4, scales_mins);
        printf("  Debug: d=%.6f dmin=%.6f sc=[%d,%d,%d,%d,%d,%d,%d,%d] m=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
               d, dmin,
               scales_mins[0],scales_mins[1],scales_mins[2],scales_mins[3],
               scales_mins[4],scales_mins[5],scales_mins[6],scales_mins[7],
               scales_mins[8],scales_mins[9],scales_mins[10],scales_mins[11],
               scales_mins[12],scales_mins[13],scales_mins[14],scales_mins[15]);
        /* Dequant and compare — ADJACENT packing:
         * byte[sub*16+i] = q[sub*32+2i] | (q[sub*32+2i+1] << 4) */
        float max_err = 0;
        for (int sub = 0; sub < 8; sub++) {
            float ascale = d * scales_mins[sub] / 63.0f;
            float amin = dmin * scales_mins[8+sub] / 63.0f;
            for (int i = 0; i < 16; i++) {
                int idx_even = sub*32 + 2*i;
                int idx_odd  = sub*32 + 2*i + 1;
                uint8_t byte_val = sb[16 + sub*16 + i];
                uint8_t q_even = byte_val & 0xF;
                uint8_t q_odd = (byte_val >> 4) & 0xF;
                float w_even = ascale * q_even - amin;
                float w_odd = ascale * q_odd - amin;
                float err = fabsf(w_even - w[0][idx_even]); if (err > max_err) max_err = err;
                err = fabsf(w_odd - w[0][idx_odd]); if (err > max_err) max_err = err;
            }
        }
        printf("  Dequant max error for row 0: %.6f\n", max_err);
    }

    /* Run kernel */
    lal_matmul_q4_k(y_kernel, q4k, x, NULL, in_dim, out_dim);

    /* Compare */
    printf("=== Q4_K kernel correctness test ===\n");
    printf("  in_dim=%d, out_dim=%d\n\n", in_dim, out_dim);
    for (int j = 0; j < out_dim; j++) {
        float err = fabsf(y_kernel[j] - y_ref[j]);
        float rel = err / (fabsf(y_ref[j]) + 1e-8f);
        printf("  row %d: ref=%+.6f  kernel=%+.6f  err=%.6f  rel=%.4f\n",
               j, y_ref[j], y_kernel[j], err, rel);
    }

    /* Test with larger size (3584 = 14 superblocks) */
    in_dim = 3584; out_dim = 8;
    float *w2 = malloc((size_t)out_dim * in_dim * sizeof(float));
    float *x2 = malloc(in_dim * sizeof(float));
    uint8_t *q4k2 = malloc((size_t)out_dim * (in_dim/256) * 144);
    float *y_k2 = malloc(out_dim * sizeof(float));
    float *y_r2 = malloc(out_dim * sizeof(float));

    for (int j = 0; j < out_dim; j++)
        for (int i = 0; i < in_dim; i++)
            w2[j*in_dim+i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < in_dim; i++)
        x2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;

    for (int j = 0; j < out_dim; j++) {
        y_r2[j] = 0;
        for (int i = 0; i < in_dim; i++)
            y_r2[j] += w2[j*in_dim+i] * x2[i];
    }
    for (int j = 0; j < out_dim; j++)
        quantize_q4_k_row(w2 + j*in_dim, q4k2 + (size_t)j*(in_dim/256)*144, in_dim);
    lal_matmul_q4_k(y_k2, q4k2, x2, NULL, in_dim, out_dim);

    printf("\n=== Larger test (in_dim=3584, out_dim=8) ===\n");
    /* Verify dequant for row 0 */
    {
        int n_super_l = in_dim / 256;
        float max_deq_err = 0;
        for (int s = 0; s < n_super_l; s++) {
            uint8_t *sb = q4k2 + (size_t)0 * n_super_l * 144 + s * 144;
            uint16_t d_u16 = *(uint16_t*)(sb);
            uint16_t dmin_u16 = *(uint16_t*)(sb+2);
            __m128i d_raw = _mm_set1_epi16((short)d_u16);
            __m128i dmin_raw = _mm_set1_epi16((short)dmin_u16);
            float d = _mm_cvtss_f32(_mm_cvtph_ps(d_raw));
            float dmin = _mm_cvtss_f32(_mm_cvtph_ps(dmin_raw));
            uint8_t sm[16]; unpack_scales_6bit(sb+4, sm);
            for (int sub = 0; sub < 8; sub++) {
                float ascale = d * sm[sub] / 63.0f;
                float amin = dmin * sm[8+sub] / 63.0f;
                for (int i = 0; i < 16; i++) {
                    int idx_even = s*256 + sub*32 + 2*i;
                    int idx_odd  = s*256 + sub*32 + 2*i + 1;
                    uint8_t bv = sb[16 + sub*16 + i];
                    float w_even = ascale * (bv & 0xF) - amin;
                    float w_odd = ascale * ((bv>>4) & 0xF) - amin;
                    float e = fabsf(w_even - w2[idx_even]); if (e>max_deq_err) max_deq_err=e;
                    e = fabsf(w_odd - w2[idx_odd]); if (e>max_deq_err) max_deq_err=e;
                }
            }
        }
        printf("  Dequant max error for large row 0: %.6f\n", max_deq_err);
    }
    float max_rel = 0, max_abs = 0;
    for (int j = 0; j < out_dim; j++) {
        float err = fabsf(y_k2[j] - y_r2[j]);
        float rel = err / (fabsf(y_r2[j]) + 1e-8f);
        /* Only count relative error when ref is large enough to be meaningful */
        if (fabsf(y_r2[j]) > 0.05f && rel > max_rel) max_rel = rel;
        if (err > max_abs) max_abs = err;
        printf("  row %d: ref=%+.6f  kernel=%+.6f  err=%.6f  rel=%.4f\n",
               j, y_r2[j], y_k2[j], err, rel);
    }
    printf("\n  Max relative error (ref>0.05): %.4f (%.1f%%)\n", max_rel, max_rel*100);
    printf("  Max absolute error: %.6f\n", max_abs);
    if (max_rel < 0.15f) printf("  PASS (< 15%% rel on meaningful rows)\n");
    else printf("  FAIL (> 15%%) — kernel has a bug!\n");

    free(w2); free(x2); free(q4k2); free(y_k2); free(y_r2);
    return 0;
}
