
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void classify_generic(const float* q, int* out_best);
int main() {
    float q[8] = {1.0f, 0.1f, 0.2f, 0.7f, 0.3f, 0.5f, 0.8f, 0.2f};
    int out;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long N = 100000000L;
    for (long i = 0; i < N; i++) {
        classify_generic(q, &out);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("%ld calls in %.4f s = %.0f calls/s (last out=%d)\n", N, dt, N/dt, out);
    return 0;
}
