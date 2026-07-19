/*
 * verify_skip_cond.c — deterministic check of the CONDITIONAL layer-skip contract.
 *
 * demos/mini_skip_cond.lal: "skip 1 every 2 layers when 0.08" compiles to a strong
 * lal_layer_skip() that skips ONLY when the per-layer residual delta
 *     delta = ||h - prev_h|| / ||h||
 * is below the threshold (stable layers skip; volatile layers keep all compute).
 * prev_h is kept inside the .so, so the server contract is unchanged.
 *
 * This test loads ONLY the cond .so (a separate process from verify_skip_uncond,
 * because both .so export the same strong symbol and must not coexist in one
 * process — the server always loads exactly one skip .so at a time).
 *
 * Build:  make verify-skip
 * Run:    ./build/verify_skip_cond
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <dlfcn.h>

typedef int (*lal_layer_skip_fn)(int layer, float *hidden, int dim);

static int failures = 0;
static void check(int ok, const char *detail, const char *fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (ok) printf("  [PASS] %s\n", buf);
    else { printf("  [FAIL] %s  (%s)\n", buf, detail ? detail : ""); failures++; }
}

int main(void) {
    void *hc = dlopen("prebuilt/mini_skip_cond.so", RTLD_NOW | RTLD_LOCAL);
    if (!hc) { fprintf(stderr, "[!] dlopen prebuilt/mini_skip_cond.so: %s\n", dlerror()); return 1; }
    lal_layer_skip_fn skipc = (lal_layer_skip_fn)dlsym(hc, "lal_layer_skip");
    if (!skipc) { fprintf(stderr, "[!] dlsym lal_layer_skip: %s\n", dlerror()); dlclose(hc); return 1; }

    printf("=== conditional skip contract (every 2 layers when delta<0.08) ===\n");
    float A[16], B[16];
    for (int i = 0; i < 16; i++) { A[i] = 0.5f; B[i] = (i % 2 ? 1.0f : -1.0f); }
    int r1 = skipc(1, A, 16);                 /* seed prev = A on odd layer (no skip) */
    int r2 = skipc(2, A, 16);
    int r4 = skipc(4, B, 16);
    int r5 = skipc(5, A, 16);
    check(r2 == 1, "tiny residual delta (A==prev) SHOULD skip",
          "layer 2 A==prev -> skip=%d (expect 1)", r2);
    check(r4 == 0, "large residual delta (B!=prev) should NOT skip",
          "layer 4 B!=prev -> skip=%d (expect 0)", r4);
    check(r5 == 0, "odd layer must never skip",
          "layer 5 A -> skip=%d (expect 0)", r5);

    printf("=== %s ===\n", failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED");
    dlclose(hc);
    return failures == 0 ? 0 : 1;
}
