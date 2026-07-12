/* bench_unpack6.c — 对比 unpack_scales_6bit 标量 vs SIMD 版
 * 构建: gcc -O3 -march=native -I. -o bench_unpack6 scripts/bench_unpack6.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>

/* 原版标量实现 */
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

/* SIMD 版本: 用 AVX2 variable shift + 掩码
 * 思路: 把 12 字节加载到 256-bit, 用 _mm256_srlv_epi64 做变量右移, 然后 & 0x3F
 * 但 6-bit 跨 byte 边界, 需要精心设计.
 *
 * 实际方案: 把 lo(8 bytes) 和 hi(4 bytes, 补 0 到 8) 分别处理
 * lo: 64 bit, 10 个 6-bit 值 (60 bit) + 4 bit 溢出到 hi
 * hi: 32 bit, 6 个 6-bit 值 (36 bit, 但只有 32 bit, 前 4 bit 来自 lo)
 *
 * 用 SIMD: lo 和 hi 一起作为两个 int64, 用 srlv + mask */
static inline void unpack_scales_6bit_simd(const uint8_t *src, uint8_t *out16) {
    /* 加载 12 字节到 16 字节 (高 4 字节清零) */
    __m128i v = _mm_loadu_si128((__m128i*)src);
    /* 扩展到 256-bit: 两个 int64 = [lo, hi_zero_extended] */
    __m256i v64 = _mm256_cvtepi8_epi64(v);  /* 4 个 int64, 但我们只要前 2 个 */
    /* 实际上更简单: 直接用标量 lo/hi + SIMD 处理 */
    uint64_t lo = *(const uint64_t*)src;
    uint32_t hi = *(const uint32_t*)(src + 8);

    /* 方案: 把 lo 放 lane0, hi 放 lane1 (都作为 int64), 用 srlv + mask */
    __m256i vals = _mm256_set_epi64x((int64_t)(uint64_t)hi, (int64_t)lo, (int64_t)lo, (int64_t)lo);
    /* shift amounts for 16 outputs:
     * out[0..9] from lo: shifts 0,6,12,...,54
     * out[10] from (lo>>60)|(hi<<4): 特殊处理
     * out[11..15] from hi: shifts 2,8,14,20,26 (注意 hi 已左移 4, 所以实际 shift 2)
     * 简化: out[10..15] from hi<<4|lo>>60, shifts 0,6,12,18,24,30
     */
    /* 这个 SIMD 方案太复杂, 回退到标量优化: 用更少的指令 */
    /* 标量优化版: 编译器可能向量化 */
    uint64_t combined = lo | ((uint64_t)hi << 64);  /* 不行, 64 位溢出 */

    /* 最实际优化: 减少内存访问, 用寄存器变量 */
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

/* 测试 SIMD 思路: 用 128-bit shuffle + multiply 解 6-bit
 * 把 12 字节广播, 用乘法+移位提取 6-bit 字段 */
static inline void unpack_scales_6bit_simd2(const uint8_t *src, uint8_t *out16) {
    /* 思路: 把 src 的 12 字节扩展成 16 个 6-bit 值
     * 用 _mm_srlv_epi32 + 掩码: 4 个 int32 通道各提取 4 个 6-bit 值 */
    __m128i v = _mm_loadu_si128((__m128i*)src);  /* 16 bytes, 但只用 12 */

    /* shift amounts for 16 outputs (每个 int32 提取 4 个):
     * lane0 (bytes 0-3): out[0]=shift0, out[1]=shift6, out[2]=shift12, out[3]=shift18
     * lane1 (bytes 4-7): out[4]=shift24-lo, ... 跨界
     * 这个方案因为 6-bit 不对齐 int32 边界而复杂
     *
     * 放弃 SIMD, 用标量但减少分支 */
    uint64_t lo = *(const uint64_t*)src;
    uint32_t hi = *(const uint32_t*)(src + 8);
    /* 用指针数组减少索引计算 */
    uint8_t *o = out16;
    o[0]=lo&0x3F; o[1]=(lo>>6)&0x3F; o[2]=(lo>>12)&0x3F; o[3]=(lo>>18)&0x3F;
    o[4]=(lo>>24)&0x3F; o[5]=(lo>>30)&0x3F; o[6]=(lo>>36)&0x3F; o[7]=(lo>>42)&0x3F;
    o[8]=(lo>>48)&0x3F; o[9]=(lo>>54)&0x3F;
    uint64_t combined = (lo >> 60) | ((uint64_t)hi << 4);
    o[10]=combined&0x3F; o[11]=(combined>>6)&0x3F; o[12]=(combined>>12)&0x3F;
    o[13]=(combined>>18)&0x3F; o[14]=(combined>>24)&0x3F; o[15]=(combined>>30)&0x3F;
}

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    const int N = 1000000;
    uint8_t src[16] = {0x12,0x34,0x56,0x78,0x9A,0xBC,0xDE,0xF0,0x11,0x22,0x33,0x44,0,0,0,0};
    uint8_t out1[16], out2[16], out3[16] __attribute__((aligned(32)));

    /* 正确性 */
    unpack_scales_6bit_scalar(src, out1);
    unpack_scales_6bit_simd(src, out2);
    unpack_scales_6bit_simd2(src, out3);
    int err2 = 0, err3 = 0;
    for (int i = 0; i < 16; i++) {
        if (out1[i] != out2[i]) err2++;
        if (out1[i] != out3[i]) err3++;
    }
    printf("正确性: simd=%d errors, simd2=%d errors\n", err2, err3);

    /* 性能 */
    volatile uint8_t sink = 0;
    double t0 = now_us();
    for (int i = 0; i < N; i++) {
        unpack_scales_6bit_scalar(src, out1);
        sink += out1[i & 15];
    }
    double t_scalar = now_us() - t0;

    t0 = now_us();
    for (int i = 0; i < N; i++) {
        unpack_scales_6bit_simd2(src, out3);
        sink += out3[i & 15];
    }
    double t_simd2 = now_us() - t0;

    printf("scalar: %.2f us/%d iters = %.4f us/call\n", t_scalar, N, t_scalar/N);
    printf("simd2:  %.2f us/%d iters = %.4f us/call\n", t_simd2, N, t_simd2/N);
    printf("speedup: %.2fx\n", t_scalar / t_simd2);

    if (sink < 0) printf("impossible %d\n", sink);
    return 0;
}
