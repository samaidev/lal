/*
 * verify_steer.c — end-to-end verification of LAL three-layer fusion LEVEL 1
 * (per-layer read + activation steering) WITHOUT needing any model weights.
 *
 * The real qwen_server calls lal_layer_hook(layer, g_x, N_EMBD) once per
 * transformer layer (right after that layer's residual stream is in g_x). We
 * cannot run the real 0.5B engine here (no weights on this machine), so instead
 * we replay that exact contract against the compiler-generated hook:
 *
 *   1. Build a deterministic 24-layer "fake engine": each layer l writes a
 *      known, layer-dependent hidden vector h_l into `hidden`.
 *   2. Call lal_layer_hook(l, hidden, 16) exactly as qwen_server would.
 *   3. Assert the fusion actually did something:
 *        - READ  (layer 8):  concept_mid     == h_8   (hidden state captured)
 *        - STEER (layer 12): concept_steered == h_12  AND
 *                            hidden[0..16]   == h_12 + 0.3*pos   (write-back)
 *        - other layers do NOT mutate `hidden` (no spurious steering)
 *        - the strong lal_layer_hook symbol is used (overrides the weak no-op)
 *
 * The generated hook is #included so its `static` concept arrays are visible in
 * this translation unit for direct inspection.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

#include "build/qwen_steer.c"   /* pulls in lal_layer_hook + static concept_* */

#define N_LAYER 24
#define DIM 16

/* Deterministic per-layer hidden vector: h_l[i] = sin((l+1)*(i+1)) — a value
 * the engine would never produce by accident, so any match proves the hook
 * read the right layer. */
static void make_hidden(int layer, float *h, int dim) {
    for (int i = 0; i < dim; i++)
        h[i] = (float)sin((layer + 1) * (i + 1) * 0.37);
}

static int failures = 0;
static void check(int ok, const char *detail, const char *fmt, ...) {
    char buf[256];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
    if (ok) printf("  [PASS] %s\n", buf);
    else { printf("  [FAIL] %s  (%s)\n", buf, detail ? detail : ""); failures++; }
}

int main(void) {
    printf("=== LAL level-1 fusion verification (per-layer read + steering) ===\n");

    float hidden[DIM];
    float before[DIM];

    for (int l = 0; l < N_LAYER; l++) {
        make_hidden(l, hidden, DIM);
        memcpy(before, hidden, sizeof(hidden));
        lal_layer_hook(l, hidden, DIM);   /* exactly as qwen_server does per layer */

        if (l == 8) {
            /* READ: concept_mid must equal this layer's hidden state. */
            int same = 1;
            for (int i = 0; i < DIM; i++)
                if (fabsf(concept_mid[i] - before[i]) > 1e-5f) { same = 0; break; }
            check(same, "concept_mid does not match layer-8 hidden",
                  "layer 8 READ -> concept_mid captured hidden");
        } else if (l == 12) {
            /* READ: concept_steered equals this layer's hidden. */
            int read_ok = 1;
            for (int i = 0; i < DIM; i++)
                if (fabsf(concept_steered[i] - before[i]) > 1e-5f) { read_ok = 0; break; }
            check(read_ok, "concept_steered does not match layer-12 hidden",
                  "layer 12 READ -> concept_steered captured hidden");
            /* STEER: hidden must equal before + 0.3*pos (write-back into the
             * residual stream — this is what nudges the network). */
            int steer_ok = 1;
            for (int i = 0; i < DIM; i++) {
                float expect = before[i] + 0.3f * concept_pos[i];
                if (fabsf(hidden[i] - expect) > 1e-5f) { steer_ok = 0; break; }
            }
            check(steer_ok, "hidden was not steered by 0.3*pos at layer 12",
                  "layer 12 STEER -> hidden += 0.3*pos written back");
        } else {
            /* No subscription at this layer: hidden must be untouched. */
            int untouched = 1;
            for (int i = 0; i < DIM; i++)
                if (fabsf(hidden[i] - before[i]) > 1e-5f) { untouched = 0; break; }
            check(untouched, untouched ? "" : "hidden mutated at an unsubscribed layer",
                  "layer %d neutral (no read/steer)", l);
        }
    }

    printf("=== %s ===\n", failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
