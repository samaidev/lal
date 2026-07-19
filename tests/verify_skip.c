/*
 * verify_skip.c — runs both layer-skip contract checks in SEPARATE processes.
 *
 * Both prebuilt/mini_skip.so and prebuilt/mini_skip_cond.so export the same strong
 * symbol `lal_layer_skip`, so they must not be dlopened in one process (the dynamic
 * linker would resolve dlsym() to whichever loaded first). The server always loads
 * exactly one skip .so, but to test both we exec the two focused binaries.
 *
 * Build:  make verify-skip
 * Run:    ./build/verify_skip
 */
#include <stdio.h>
#include <stdlib.h>

static int run(const char *bin) {
    printf("\n----- %s -----\n", bin);
    int rc = system(bin);
    if (rc != 0) { fprintf(stderr, "[!] %s failed (rc=%d)\n", bin, rc); return 1; }
    return 0;
}

int main(void) {
    int fail = 0;
    fail |= run("./build/verify_skip_cond");
    fail |= run("./build/verify_skip_uncond");
    printf("\n=== %s ===\n", fail ? "SKIP VERIFY FAILED" : "SKIP VERIFY PASSED");
    return fail ? 1 : 0;
}
