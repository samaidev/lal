/* bench_q4k2.c — 对比 Q4_K (6-bit packed scales) vs Q4_K2 (8-bit unpacked scales)
 * Build: gcc -O3 -march=native -fopenmp -I. -o bench_q4k2 scripts/bench_q4k2.c -lm -lgomp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define XQ_MAX 18944
#include "runtime/lal_q4k_kernel.h"
#include "runtime/lal_q4k2_kernel.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* 从 Q4_K 格式转换为 Q4_K2 格式 (预解包 scales) */
static void convert_q4k_to_q4k2(const uint8_t *q4k, uint8_t *q4k2, int n_super) {
    /* Q4_K:   4(d/dmin) + 12(6-bit packed) + 128(q4) = 144 bytes */
    /* Q4_K2:  4(d/dmin) + 16(scales) + 16(mins) + 128(q4) = 164 bytes */
    /* 复制 d/dmin */
    memcpy(q4k2, q4k, 4);
    /* 解包 6-bit scales → 8-bit */
    uint8_t sm[16];
    /* 用原始的 unpack_scales_6bit */
    uint64_t lo = *(const uint64_t*)(q4k + 4);
    uint32_t hi = *(const uint32_t*)(q4k + 12);
    sm[0]=lo&0x3F; sm[1]=(lo>>6)&0x3F; sm[2]=(lo>>12)&0x3F; sm[3]=(lo>>18)&0x3F;
    sm[4]=(lo>>24)&0x3F; sm[5]=(lo>>30)&0x3F; sm[6]=(lo>>36)&0x3F; sm[7]=(lo>>42)&0x3F;
    sm[8]=(lo>>48)&0x3F; sm[9]=(lo>>54)&0x3F; sm[10]=((lo>>60)|(hi<<4))&0x3F;
    sm[11]=(hi>>2)&0x3F; sm[12]=(hi>>8)&0x3F; sm[13]=(hi>>14)&0x3F;
    sm[14]=(hi>>20)&0x3F; sm[15]=(hi>>26)&0x3F;
    /* 前 8 个是 scales, 后 8 个是 mins */
    memcpy(q4k2 + 4, sm, 16);       /* scales */
    memcpy(q4k2 + 20, sm + 8, 8);   /* mins (only 8 needed, pad rest with 0) */
    memset(q4k2 + 28, 0, 8);        /* padding */
    /* 复制 q4 data */
    memcpy(q4k2 + 36, q4k + 16, 128);
}

int main(int argc, char **argv) {
    int n_threads = argc > 1 ? atoi(argv[1]) : 1;
    struct { const char *name; int in_dim, out_dim; } tests[] = {
        {"q_proj  [512, 3584]",  3584, 512},
        {"gate    [18944, 3584]", 3584, 18944},
        {"down    [3584, 18944]", 18944, 3584},
    };
    int n_tests = sizeof(tests)/sizeof(tests[0]);
    srand(42);

    printf("=== Q4_K (6-bit packed) vs Q4_K2 (8-bit unpacked scales) ===\n");
    printf("Threads: %d\n\n", n_threads);
    printf("%-22s  %-10s  %-10s  %-8s  %-10s  %-10s\n",
           "test", "K(ms)", "K2(ms)", "speedup", "K GB/s", "K2 GB/s");

    for (int t = 0; t < n_tests; t++) {
        int in_dim = tests[t].in_dim, out_dim = tests[t].out_dim;
        int n_super = in_dim / 256;
        int row_stride_k = n_super * 144;
        int row_stride_k2 = n_super * Q4K2_SUPERBLOCK_BYTES;

        float *x = malloc(in_dim * sizeof(float));
        uint8_t *q4k = malloc((size_t)out_dim * row_stride_k);
        uint8_t *q4k2 = malloc((size_t)out_dim * row_stride_k2);
        float *y = malloc(out_dim * sizeof(float));

        for (int i = 0; i < in_dim; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        for (int j = 0; j < out_dim; j++) {
            uint8_t *row = q4k + (size_t)j * row_stride_k;
            for (int s = 0; s < n_super; s++) {
                uint8_t *sb = row + s*144;
                *(uint16_t*)sb = 0x3C00; *(uint16_t*)(sb+2) = 0x0000;
                memset(sb+4, 0x20, 12);
                for (int i = 0; i < 128; i++) sb[16+i] = rand() & 0xFF;
            }
            /* Convert to Q4_K2 */
            uint8_t *row2 = q4k2 + (size_t)j * row_stride_k2;
            for (int s = 0; s < n_super; s++) {
                convert_q4k_to_q4k2(row + s*144, row2 + s*Q4K2_SUPERBLOCK_BYTES, n_super);
            }
        }

        /* Warmup */
        lal_matmul_q4_k(y, q4k, x, NULL, in_dim, out_dim);
        lal_matmul_q4_k2(y, q4k2, x, NULL, in_dim, out_dim);

        int n_iter = 5;
        double mem_k = (double)out_dim * row_stride_k + in_dim * 4;
        double mem_k2 = (double)out_dim * row_stride_k2 + in_dim * 4;

        #define RUN_BENCH(kernel, W, stride, mem, dt_out) do { \
            if (n_threads > 1 && out_dim >= 2048) { \
                double t0 = now_s(); \
                for (int it = 0; it < n_iter; it++) { \
                    _Pragma("omp parallel num_threads(n_threads)") \
                    { int tid = omp_get_thread_num(); \
                      int chunk = (out_dim + n_threads - 1) / n_threads; \
                      int start = tid * chunk; int end = start + chunk; \
                      if (end > out_dim) end = out_dim; \
                      if (start < out_dim) \
                          kernel(y + start, W + (size_t)start * stride, x, NULL, in_dim, end - start); \
                    } \
                } \
                dt_out = (now_s() - t0) / n_iter; \
            } else { \
                double t0 = now_s(); \
                for (int it = 0; it < n_iter; it++) \
                    kernel(y, W, x, NULL, in_dim, out_dim); \
                dt_out = (now_s() - t0) / n_iter; \
            } \
        } while(0)

        double dt_k, dt_k2;
        RUN_BENCH(lal_matmul_q4_k, q4k, row_stride_k, mem_k, dt_k);
        RUN_BENCH(lal_matmul_q4_k2, q4k2, row_stride_k2, mem_k2, dt_k2);

        printf("%-22s  %-10.2f  %-10.2f  %-8.2fx  %-10.1f  %-10.1f\n",
               tests[t].name, dt_k*1000, dt_k2*1000, dt_k/dt_k2,
               mem_k/dt_k/1e9, mem_k2/dt_k2/1e9);

        free(x); free(q4k); free(q4k2); free(y);
    }

    return 0;
}
