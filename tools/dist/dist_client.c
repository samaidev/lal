/* dist_client.c — Test distributed inference network latency
 * Usage: ./dist_client --worker-url URL [--tokens N] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#define N_EMBD 768

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(int argc, char **argv) {
    const char *url = NULL; int n = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--worker-url") && i+1<argc) url = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1<argc) n = atoi(argv[++i]);
    }
    if (!url) { fprintf(stderr, "Usage: %s --worker-url URL [--tokens N]\n", argv[0]); return 1; }
    float hidden[N_EMBD]; for (int i = 0; i < N_EMBD; i++) hidden[i] = i * 0.001f;
    printf("[*] Testing %d tokens to %s (768 floats = 3KB/token)\n", n, url);
    double total = 0;
    for (int t = 0; t < n; t++) {
        /* Write body */
        FILE *bf = fopen("/tmp/dist_body.bin", "wb");
        fwrite(hidden, 4, N_EMBD, bf); int pos = t; fwrite(&pos, 4, 1, bf); fclose(bf);
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "curl -s -X POST '%s/forward' --data-binary @/tmp/dist_body.bin -o /tmp/dist_resp.bin 2>/dev/null", url);
        double t0 = now_sec();
        system(cmd);
        double dt = now_sec() - t0; total += dt;
        FILE *rf = fopen("/tmp/dist_resp.bin", "rb");
        if (rf) { fread(hidden, 4, N_EMBD, rf); int tok; fread(&tok, 4, 1, rf); fclose(rf); }
        printf("  token %d: %.1f ms\n", t, dt*1000);
    }
    printf("\n[*] Avg: %.1f ms/token, max ~%.0f tok/s\n", total/n*1000, n/total);
    printf("[*] Data: %.1f KB/token (768 floats x 4 bytes x 2)\n", N_EMBD*4.0*2/1024);
    return 0;
}
