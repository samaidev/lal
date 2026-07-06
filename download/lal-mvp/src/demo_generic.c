/*
 * demo_generic.c — the SAME logic, but written "generically" (llama.cpp-style):
 *
 *   - Concept vectors stored at full dimensionality (8 dims each, even though
 *     only 4 are used per query)
 *   - Mask applied at runtime via index arrays
 *   - Dot product computed in a loop
 *   - argmax computed by a generic function
 *   - No compile-time specialization
 *
 * This is what the user's "logic-native" approach is reacting against.
 * Compare binary size and speed to demo.c (the specialized version).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_CONCEPTS 4
#define DIM 8
#define N_ANIMAL_DIMS 4
#define N_MACHINE_DIMS 4

/* === Full concept vectors (no compile-time masking) === */
static const float concepts[N_CONCEPTS][DIM] = {
    {1.0f, 0.1f, 0.2f, 0.7f, 0.3f, 0.5f, 0.8f, 0.2f},  /* cat     */
    {0.8f, 0.2f, 0.3f, 0.6f, 0.4f, 0.4f, 0.7f, 0.3f},  /* dog     */
    {0.1f, 0.9f, 0.8f, 0.1f, 0.7f, 0.2f, 0.1f, 0.6f},  /* car     */
    {0.0f, 0.9f, 0.9f, 0.0f, 0.6f, 0.1f, 0.0f, 0.7f},  /* vehicle */
};

/* Concept labels (kept for reference; the generic version doesn't print them) */
/* static const char* concept_names[N_CONCEPTS] = {"cat", "dog", "car", "vehicle"}; */

/* === Bounds (resolved at RUNTIME) === */
static const int animal_dims[N_ANIMAL_DIMS]   = {0, 2, 3, 6};
static const int machine_dims[N_MACHINE_DIMS] = {1, 2, 4, 7};

/* === Generic dot product with runtime mask === */
static float dot_masked(const float* a, const float* b, const int* mask, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += a[mask[i]] * b[mask[i]];
    }
    return s;
}

/* === Generic argmax === */
static int argmax_f(const float* v, int n) {
    int best = 0;
    float best_val = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > best_val) {
            best_val = v[i];
            best = i;
        }
    }
    return best;
}

/* === The rule, written generically === */
void classify_generic(const float* q, int* out_best) {
    float scores[N_CONCEPTS];

    /* For each concept, pick the right mask, then do masked dot product */
    for (int i = 0; i < N_CONCEPTS; i++) {
        const int* mask;
        int n_mask;
        if (i < 2) {  /* cat, dog use animal_dims */
            mask = animal_dims;
            n_mask = N_ANIMAL_DIMS;
        } else {  /* car, vehicle use machine_dims */
            mask = machine_dims;
            n_mask = N_MACHINE_DIMS;
        }
        scores[i] = dot_masked(q, concepts[i], mask, n_mask);
    }

    *out_best = argmax_f(scores, N_CONCEPTS);
}

int main(int argc, char** argv) {
    if (argc < DIM + 1) {
        fprintf(stderr, "usage: demo_generic v0 v1 ... v7\n");
        return 1;
    }
    float q[DIM];
    for (int i = 0; i < DIM; i++) q[i] = (float)atof(argv[i+1]);

    int out_best = -1;
    classify_generic(q, &out_best);
    printf("%d\n", out_best);
    return 0;
}
