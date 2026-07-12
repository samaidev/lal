/* bench_sample.c — 测量采样时间
 * Build: gcc -O3 -march=native -I. -o bench_sample scripts/bench_sample.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define VOCAB_SIZE 152064
#include "runtime/lal_sampling.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    float *logits = malloc(VOCAB_SIZE * sizeof(float));
    int recent[256] = {0};
    srand(42);
    for (int i = 0; i < VOCAB_SIZE; i++)
        logits[i] = (rand()/((float)RAND_MAX) - 0.5f) * 20.0f;

    int n_iter = 100;
    /* warmup */
    lal_sample_token(logits, VOCAB_SIZE, 0.8f, 40, 1.1f, recent, 10);

    double t0 = now_ms();
    int tok;
    for (int i = 0; i < n_iter; i++)
        tok = lal_sample_token(logits, VOCAB_SIZE, 0.8f, 40, 1.1f, recent, 10);
    double dt = (now_ms() - t0) / n_iter;

    printf("Sample: %.3f ms/call (%d vocab, top_k=40)\n", dt, VOCAB_SIZE);
    printf("In 480ms forward: %.1f%%\n", dt/480*100);

    free(logits);
    return 0;
}
