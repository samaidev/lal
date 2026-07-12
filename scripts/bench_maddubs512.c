/* bench_maddubs512.c — 测试 256-bit × 2 vs 512-bit × 1 maddubs
 * 构建: gcc -O3 -march=native -mavx512bw -I. -o bench_madd512 scripts/bench_maddubs512.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define ITERS 1000000

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    /* 测试数据 */
    __m256i q4l = _mm256_set1_epi8(5);   /* unsigned 0-15 */
    __m256i q4h = _mm256_set1_epi8(3);   /* unsigned 0-15 */
    __m256i xve = _mm256_set1_epi8(10);  /* signed int8 */
    __m256i xvo = _mm256_set1_epi8(-10); /* signed int8 */
    __m256i ones = _mm256_set1_epi16(1);

    /* 方案 1: 256-bit × 2 + add */
    volatile __m256i sink1;
    double t0 = now_us();
    for (int i = 0; i < ITERS; i++) {
        __m256i p16 = _mm256_add_epi16(
            _mm256_maddubs_epi16(q4l, xve),
            _mm256_maddubs_epi16(q4h, xvo)
        );
        sink1 = p16;
    }
    double t_256 = now_us() - t0;

    /* 方案 2: 512-bit × 1 + extract + add */
    volatile __m256i sink2;
    t0 = now_us();
    for (int i = 0; i < ITERS; i++) {
        /* 拼接 q4l + q4h 为 512-bit, xve + xvo 为 512-bit */
        __m512i q512 = _mm512_inserti64x4(_mm512_castsi256_si512(q4l), q4h, 1);
        __m512i x512 = _mm512_inserti64x4(_mm512_castsi256_si512(xve), xvo, 1);
        __m512i p32 = _mm512_maddubs_epi16(q512, x512);
        /* p32 有 16 个 int16, 需要加成 8 个 int16 */
        __m256i lo = _mm512_castsi512_si256(p32);
        __m256i hi = _mm512_extracti64x4_epi64(p32, 1);
        __m256i p16 = _mm256_add_epi16(lo, hi);
        sink2 = p16;
    }
    double t_512 = now_us() - t0;

    /* 验证正确性 */
    __m256i r1 = _mm256_add_epi16(
        _mm256_maddubs_epi16(q4l, xve),
        _mm256_maddubs_epi16(q4h, xvo));
    __m512i q512 = _mm512_inserti64x4(_mm512_castsi256_si512(q4l), q4h, 1);
    __m512i x512 = _mm512_inserti64x4(_mm512_castsi256_si512(xve), xvo, 1);
    __m512i p32 = _mm512_maddubs_epi16(q512, x512);
    __m256i r2 = _mm256_add_epi16(_mm512_castsi512_si256(p32), _mm512_extracti64x4_epi64(p32, 1));

    int match = memcmp(&r1, &r2, 32) == 0;

    printf("=== maddubs 256×2 vs 512×1 (%d iters) ===\n", ITERS);
    printf("256-bit × 2 + add: %.2f us\n", t_256);
    printf("512-bit × 1 + extract + add: %.2f us\n", t_512);
    printf("speedup: %.2fx\n", t_256 / t_512);
    printf("correctness: %s\n", match ? "MATCH ✅" : "MISMATCH ❌");

    return 0;
}
