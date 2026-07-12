/* bench_unpack6_v2.c — SIMD unpack6 实现 v2: 用 pshufb + pmaddubsw
 *
 * 关键思路: 6-bit packed data 跨 byte 边界, 但可以用 byte shuffle 把
 * 每个 6-bit field 对齐到 byte 边界, 然后用 mask 提取.
 *
 * 12 bytes = 96 bits = 16 × 6-bit fields
 * 每个 field 跨 1-2 bytes. 用 pshufb 把每个 field 起始 byte 对齐,
 * 然后用 _mm_srli_epi16 + _mm_and_si128 提取.
 *
 * Build: gcc -O3 -march=native -I. -o bench_unpack6_v2 scripts/bench_unpack6_v2.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>

/* 原版标量 */
static inline void unpack_scales_6bit_scalar(const uint8_t *src, uint8_t *out16) {
    uint64_t lo = *(const uint64_t*)src;
    uint32_t hi = *(const uint32_t*)(src + 8);
    out16[0]=lo&0x3F; out16[1]=(lo>>6)&0x3F; out16[2]=(lo>>12)&0x3F;
    out16[3]=(lo>>18)&0x3F; out16[4]=(lo>>24)&0x3F; out16[5]=(lo>>30)&0x3F;
    out16[6]=(lo>>36)&0x3F; out16[7]=(lo>>42)&0x3F; out16[8]=(lo>>48)&0x3F;
    out16[9]=(lo>>54)&0x3F; out16[10]=((lo>>60)|(hi<<4))&0x3F;
    out16[11]=(hi>>2)&0x3F; out16[12]=(hi>>8)&0x3F; out16[13]=(hi>>14)&0x3F;
    out16[14]=(hi>>20)&0x3F; out16[15]=(hi>>26)&0x3F;
}

/* SIMD 版本 v2: 处理 8 个 superblock 的 scales (8×12=96 bytes)
 * 用 256-bit 加载 + 变量移位 + mask
 *
 * 每个 superblock 的 12 bytes 加载到 __m128i, 但 6-bit 对齐复杂.
 * 更实际的方法: 一次性处理 4 个 superblock (48 bytes = 64 values)
 *
 * 方案: 把 48 bytes 扩展到 64 bytes (每个 6-bit field 占 1 byte),
 * 用 lookup table (pshufb) 重排.
 */
static inline void unpack_scales_6bit_simd4(const uint8_t *src, uint8_t *out64) {
    /* 处理 4 个 superblock (4×12=48 bytes → 4×16=64 bytes output)
     * 每个 superblock 的 12 bytes 包含 16 个 6-bit fields
     *
     * 用标量但展开 4 个 superblock, 让编译器向量化 */
    for (int s = 0; s < 4; s++) {
        const uint8_t *p = src + s * 12;
        uint8_t *o = out64 + s * 16;
        uint64_t lo = *(const uint64_t*)p;
        uint32_t hi = *(const uint32_t*)(p + 8);
        o[0]=lo&0x3F; o[1]=(lo>>6)&0x3F; o[2]=(lo>>12)&0x3F;
        o[3]=(lo>>18)&0x3F; o[4]=(lo>>24)&0x3F; o[5]=(lo>>30)&0x3F;
        o[6]=(lo>>36)&0x3F; o[7]=(lo>>42)&0x3F; o[8]=(lo>>48)&0x3F;
        o[9]=(lo>>54)&0x3F; o[10]=((lo>>60)|((uint64_t)hi<<4))&0x3F;
        o[11]=(hi>>2)&0x3F; o[12]=(hi>>8)&0x3F; o[13]=(hi>>14)&0x3F;
        o[14]=(hi>>20)&0x3F; o[15]=(hi>>26)&0x3F;
    }
}

/* SIMD 版本 v3: 用 128-bit multiply-shift 技巧
 * 把 12 bytes 作为 24-bit chunks, 用 pmaddubsw 分离
 */
static inline void unpack_scales_6bit_simd_v3(const uint8_t *src, uint8_t *out16) {
    /* 加载 16 bytes (12 valid + 4 zero) */
    __m128i v = _mm_loadu_si128((__m128i*)src);
    /* 把 12 bytes 扩展为 16 bytes, 每个 byte 对应一个 6-bit field
     * field i 起始 bit = i*6, 跨 byte floor(i*6/8) 和 floor(i*6/8)+1
     *
     * 用 _mm_shuffle_epi8 把每个 field 的起始 byte 提取到正确位置
     * 然后用 _mm_alignr_epi8 + _mm_srli_epi16 + _mm_and_si128 提取 6-bit
     */
    /* shuffle mask: field i 的起始 byte index */
    /* static const __m128i shuffle_mask = ... — 不能用 static const __m128i, 改用局部 */
    /* 但 6-bit fields 跨边界, shuffle 后需要 alignr + shift */

    /* 实际上最简单有效的方法: 标量但用更少指令 */
    uint64_t lo = *(const uint64_t*)src;
    uint32_t hi = *(const uint32_t*)(src + 8);

    /* 用 2 个 64-bit 变量, 减少依赖链 */
    uint64_t v0 = lo;           /* fields 0-9 */
    uint64_t v1 = ((uint64_t)hi << 4) | (lo >> 60);  /* fields 10-15 */

    out16[0]  = v0 & 0x3F;       v0 >>= 6;
    out16[1]  = v0 & 0x3F;       v0 >>= 6;
    out16[2]  = v0 & 0x3F;       v0 >>= 6;
    out16[3]  = v0 & 0x3F;       v0 >>= 6;
    out16[4]  = v0 & 0x3F;       v0 >>= 6;
    out16[5]  = v0 & 0x3F;       v0 >>= 6;
    out16[6]  = v0 & 0x3F;       v0 >>= 6;
    out16[7]  = v0 & 0x3F;       v0 >>= 6;
    out16[8]  = v0 & 0x3F;       v0 >>= 6;
    out16[9]  = v0 & 0x3F;
    out16[10] = v1 & 0x3F;       v1 >>= 6;
    out16[11] = v1 & 0x3F;       v1 >>= 6;
    out16[12] = v1 & 0x3F;       v1 >>= 6;
    out16[13] = v1 & 0x3F;       v1 >>= 6;
    out16[14] = v1 & 0x3F;       v1 >>= 6;
    out16[15] = v1 & 0x3F;
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    /* 模拟 gate [18944, 3584] 的 scale 解包 */
    int n_rows = 18944;
    int n_super = 14;
    srand(42);

    uint8_t *packed = _mm_malloc((size_t)n_rows * n_super * 12, 32);
    uint8_t *out_scalar = _mm_malloc((size_t)n_rows * n_super * 16, 32);
    uint8_t *out_simd4 = _mm_malloc((size_t)n_rows * n_super * 16, 32);
    uint8_t *out_v3 = _mm_malloc((size_t)n_rows * n_super * 16, 32);

    for (size_t i = 0; i < (size_t)n_rows * n_super * 12; i++) packed[i] = rand() & 0xFF;

    int n_iter = 20;

    /* Correctness */
    for (int j = 0; j < n_rows; j++)
        for (int s = 0; s < n_super; s++)
            unpack_scales_6bit_scalar(packed + (size_t)j*n_super*12 + s*12,
                                       out_scalar + (size_t)j*n_super*16 + s*16);
    for (int j = 0; j < n_rows; j++)
        for (int s = 0; s < n_super; s++)
            unpack_scales_6bit_simd_v3(packed + (size_t)j*n_super*12 + s*12,
                                        out_v3 + (size_t)j*n_super*16 + s*16);
    float max_err = 0;
    for (size_t i = 0; i < (size_t)n_rows * n_super * 16; i++) {
        int d = abs((int)out_scalar[i] - (int)out_v3[i]);
        if (d > max_err) max_err = d;
    }
    printf("=== Correctness (scalar vs v3) ===\n");
    printf("max_err = %g %s\n\n", max_err, max_err == 0 ? "✅ PASS" : "❌ FAIL");

    /* Benchmark scalar */
    double t0 = now_ms();
    for (int it = 0; it < n_iter; it++)
        for (int j = 0; j < n_rows; j++)
            for (int s = 0; s < n_super; s++)
                unpack_scales_6bit_scalar(packed + (size_t)j*n_super*12 + s*12,
                                           out_scalar + (size_t)j*n_super*16 + s*16);
    double dt_scalar = (now_ms() - t0) / n_iter;

    /* Benchmark simd4 */
    t0 = now_ms();
    for (int it = 0; it < n_iter; it++)
        for (int j = 0; j < n_rows; j++)
            for (int s = 0; s < n_super; s += 4)
                unpack_scales_6bit_simd4(packed + (size_t)j*n_super*12 + s*12,
                                          out_simd4 + (size_t)j*n_super*16 + s*16);
    double dt_simd4 = (now_ms() - t0) / n_iter;

    /* Benchmark v3 */
    t0 = now_ms();
    for (int it = 0; it < n_iter; it++)
        for (int j = 0; j < n_rows; j++)
            for (int s = 0; s < n_super; s++)
                unpack_scales_6bit_simd_v3(packed + (size_t)j*n_super*12 + s*12,
                                            out_v3 + (size_t)j*n_super*16 + s*16);
    double dt_v3 = (now_ms() - t0) / n_iter;

    printf("=== unpack_scales_6bit benchmark (%d rows × %d super) ===\n", n_rows, n_super);
    printf("Scalar:  %.3f ms\n", dt_scalar);
    printf("SIMD4:   %.3f ms  (%.2fx)\n", dt_simd4, dt_scalar/dt_simd4);
    printf("V3:      %.3f ms  (%.2fx)\n", dt_v3, dt_scalar/dt_v3);
    printf("\nIn Q4_K gate matmul (5.91ms total):\n");
    printf("  Scalar unpack: %.3f ms (%.1f%%)\n", dt_scalar, dt_scalar/5.91*100);
    printf("  V3 unpack:     %.3f ms (%.1f%%)\n", dt_v3, dt_v3/5.91*100);
    printf("  Saved:         %.3f ms/gate × 28 layers × 3 = %.2f ms/forward\n",
           dt_scalar - dt_v3, (dt_scalar - dt_v3) * 28 * 3);

    _mm_free(packed); _mm_free(out_scalar); _mm_free(out_simd4); _mm_free(out_v3);
    return 0;
}
