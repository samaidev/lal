/*
 * verify_skip.c — deterministic check that the compiled layer-skip contract works.
 *
 * LAL level-1 acceleration: a .lal `skip N every P layers` directive compiles to a
 * strong `lal_layer_skip(layer, hidden, dim) -> int` symbol. The server calls it
 * once per transformer layer; returning s>0 makes the engine jump s layers ahead.
 *
 * This test dlopens prebuilt/mini_skip.so (built from demos/mini_skip.lal:
 * "skip 1 every 2 layers") and asserts the symbol returns 1 at even layers and 0
 * at odd layers — i.e. the compiled control logic is exactly what the .lal said.
 *
 * Build:  make verify-skip
 * Run:    ./build/verify_skip
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

    printf("=== LAL level-1 layer-skip contract verification ===\n");
    /* demos/mini_skip.lal: "skip 1 every 2 layers" -> skip at layer%2==0 */
    for (int l = 0; l < 8; l++) {
        int s = skip(l, NULL, 0);
        check((l % 2 == 0) ? (s == 1) : (s == 0),
              "wrong skip count",
              "layer %d -> skip=%d (expect %d)", l, s, (l % 2 == 0) ? 1 : 0);
    }
    /* The weak fallback in the server returns 0 unconditionally; a linked .so must
     * NOT be the no-op — at least one layer must skip. */
    int any_skip = 0;
    for (int l = 0; l < 8; l++) if (skip(l, NULL, 0) > 0) any_skip = 1;
    check(any_skip, "symbol is a no-op (should skip some layers)",
          "at least one layer triggers a skip");

    printf("=== %s ===\n", failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED");
    dlclose(h);
    return failures == 0 ? 0 : 1;
}
