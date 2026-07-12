/* bench_unpack.c — 测试 6-bit scale 解包优化
 * 对比标量 vs SIMD 版本的 unpack_scales_6bit 性能
 * Build: gcc -O3 -march=native -I. -o bench_unpack scripts/bench_unpack.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

/* 原始标量版本 */
static inline void unpack_scales_6bit_scalar(const uint8_t *src, uint8_t *out16) {
    uint64_t lo = *(const uint64_t*)src;
    uint32_t hi = *(const uint32_t*)(src + 8);
    out16[0]  = lo & 0x3F;
    out16[1]  = (lo >> 6) & 0x3F;
    out16[2]  = (lo >> 12) & 0x3F;
    out16[3]  = (lo >> 18) & 0x3F;
    out16[4]  = (lo >> 24) & 0x3F;
    out16[5]  = (lo >> 30) & 0x3F;
    out16[6]  = (lo >> 36) & 0x3F;
    out16[7]  = (lo >> 42) & 0x3F;
    out16[8]  = (lo >> 48) & 0x3F;
    out16[9]  = (lo >> 54) & 0x3F;
    out16[10] = ((lo >> 60) | (hi << 4)) & 0x3F;
    out16[11] = (hi >> 2) & 0x3F;
    out16[12] = (hi >> 8) & 0x3F;
    out16[13] = (hi >> 14) & 0x3F;
    out16[14] = (hi >> 20) & 0x3F;
    out16[15] = (hi >> 26) & 0x3F;
}

/* SIMD 优化版本: 使用 SSE2 byte shuffle + mask
 * 思路: 将 12 bytes 扩展为 16 bytes, 用 pshufb 重排,
 * 然后 shift + mask 提取 6-bit 值
 *
 * 更简单的方法: 用 multiply-shift 技巧
 * 对于 6-bit packed data, 可以用整除+取模的方式提取:
 *   val[i] = (packed / 64^i) % 64
 * 但这需要除法, 太慢。
 *
 * 最佳方法: 预计算 shuffle mask, 用 pshufb 把每个 6-bit field
 * 对齐到 byte 边界, 然后用 PSRLW + PAND 提取。
 *
 * 但由于 6 不是 8 的倍数, field 跨越 byte 边界, 需要 2 次 shuffle。
 *
 * 实际最快的方案: 批量处理 8 个 row 的 scale (8×12=96 bytes),
 * 用 AVX2 一次性处理。
 */

/* 方法1: 预计算所有 superblock 的 scale, 批量处理 */
static void unpack_scales_batch(const uint8_t *src, uint8_t *dst, int n) {
    /* src: n × 12 bytes (n 个 superblock 的 6-bit packed scales)
     * dst: n × 16 bytes (解包后的 uint8 scales)
     */
    for (int i = 0; i < n; i++) {
        unpack_scales_6bit_scalar(src + i * 12, dst + i * 16);
    }
}

/* 方法2: 用 SSE 处理单个 superblock 的 8 个 row 的 scales
 * 8 rows × 12 bytes = 96 bytes, 可以用 4 个 __m128i 处理
 */
static inline void unpack_scales_8rows(const uint8_t *src[8], uint8_t dst[8][16]) {
    for (int r = 0; r < 8; r++) {
        unpack_scales_6bit_scalar(src[r], dst[r]);
    }
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int main() {
    /* 模拟 gate [18944, 3584] 的 scale 解包 */
    int n_rows = 18944;
    int n_super = 3584 / 256;  /* 14 */
    srand(42);

    /* 生成随机 packed scales */
    uint8_t *packed = malloc((size_t)n_rows * n_super * 12);
    uint8_t *unpacked = malloc((size_t)n_rows * n_super * 16);
    for (size_t i = 0; i < (size_t)n_rows * n_super * 12; i++) packed[i] = rand() & 0xFF;

    /* 测量标量版本 */
    int n_iter = 10;
    double t0 = now_s();
    for (int it = 0; it < n_iter; it++) {
        for (int j = 0; j < n_rows; j++) {
            for (int s = 0; s < n_super; s++) {
                unpack_scales_6bit_scalar(packed + (size_t)j * n_super * 12 + s * 12,
                                          unpacked + (size_t)j * n_super * 16 + s * 16);
            }
        }
    }
    double dt_scalar = (now_s() - t0) / n_iter;

    printf("=== 6-bit scale unpack benchmark ===\n");
    printf("Matrix: [%d, %d], %d superblocks/row\n", n_rows, 3584, n_super);
    printf("Total unpacks: %d × %d = %d\n", n_rows, n_super, n_rows * n_super);
    printf("Scalar: %.3f ms (%.1f ns/unpack)\n", dt_scalar * 1000,
           dt_scalar * 1e9 / (n_rows * n_super));
    printf("Per matmul (gate): %.3f ms of 5.84 ms total = %.1f%%\n",
           dt_scalar * 1000, dt_scalar / 5.84e-3 * 100);

    /* 估算: 如果能把 unpack 时间减半, 能节省多少 */
    printf("\nIf unpack is 2x faster: save %.3f ms per matmul\n", dt_scalar * 1000 / 2);
    printf("× 28 layers × 3 large matmuls (gate+up+down) = %.2f ms per forward\n",
           dt_scalar * 1000 / 2 * 28 * 3);

    free(packed);
    free(unpacked);
    return 0;
}
