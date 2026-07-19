/*
 * verify_skip_uncond.c — deterministic check of the UNCONDITIONAL layer-skip contract.
 *
 * demos/mini_skip.lal: "skip 1 every 2 layers" compiles to a strong lal_layer_skip()
 * that skips at every even layer index regardless of hidden state.
 *
 * Loads ONLY the unconditional .so (separate process from verify_skip_cond — both
 * .so export the same strong symbol and must not coexist; the server loads one).
 *
 * Build:  make verify-skip
 * Run:    ./build/verify_skip_uncond
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
    void *h = dlopen("prebuilt/mini_skip.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "[!] dlopen prebuilt/mini_skip.so: %s\n", dlerror()); return 1; }
    lal_layer_skip_fn skip = (lal_layer_skip_fn)dlsym(h, "lal_layer_skip");
    if (!skip) { fprintf(stderr, "[!] dlsym lal_layer_skip: %s\n", dlerror()); dlclose(h); return 1; }

    printf("=== LAL level-1 layer-skip contract verification (unconditional) ===\n");
    for (int l = 0; l < 8; l++) {
        int s = skip(l, NULL, 0);
        check((l % 2 == 0) ? (s == 1) : (s == 0),
              "wrong skip count",
              "layer %d -> skip=%d (expect %d)", l, s, (l % 2 == 0) ? 1 : 0);
    }
    int any_skip = 0;
    for (int l = 0; l < 8; l++) if (skip(l, NULL, 0) > 0) any_skip = 1;
    check(any_skip, "symbol is a no-op (should skip some layers)",
          "at least one layer triggers a skip");

    printf("=== %s ===\n", failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED");
    dlclose(h);
    return failures == 0 ? 0 : 1;
}
