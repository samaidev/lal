/* extract_float_subset.c — extract a small subset of float tensors from the
 * full GPW2 weight file for mixed-precision mode on memory-constrained devices.
 *
 * Problem: --mixed-precision needs float weights for layers 0 and N_LAYER-1,
 * but the full gpt2_weights.bin is 498 MB — too big to ship to a tablet just
 * for 2 layers. This tool emits a ~60 MB subset file containing only those
 * 24 tensors (12 per layer), in the SAME GPW2 format, so the existing
 * tensor_load_all() reads it without modification.
 *
 * Build: gcc -O2 -o scripts/extract_float_subset scripts/extract_float_subset.c
 * Run:   ./scripts/extract_float_subset prebuilt/gpt2_weights.bin \
 *          prebuilt/gpt2_float_subset.bin
 *
 * Then on the tablet:
 *   LAL_FLOAT_SUBSET=prebuilt/gpt2_float_subset.bin \
 *     ./gpt2_server --mixed-precision --lm-head-int8 --dflash
 *
 * The subset file keeps keys exactly as in the source (e.g. "h.0.ln_1.weight"),
 * so load_float_layers_subset()'s tensor_get() lookups work unchanged.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_LAYER 12

/* Match "h.0." or "h.11." (the first and last GPT-2 layers). The trailing dot
 * avoids matching "h.0" against "h.01" or "h.02" etc. We match the prefix up
 * to and including the dot after the layer number. */
static int is_keep_layer(const char *key) {
    /* "h.0." → layer 0; "h.11." → layer 11. */
    if (strncmp(key, "h.0.", 4) == 0) return 1;
    if (strncmp(key, "h.11.", 5) == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input_gpw2> <output_subset_gpw2>\n", argv[0]);
        return 1;
    }
    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { fprintf(stderr, "[!] cannot open %s\n", argv[1]); return 1; }

    char magic[4];
    if (fread(magic, 1, 4, fin) != 4 || memcmp(magic, "GPW2", 4) != 0) {
        fprintf(stderr, "[!] bad magic in %s\n", argv[1]); fclose(fin); return 1;
    }
    int n_total = 0;
    fread(&n_total, 4, 1, fin);

    /* First pass: read all tensor metadata + data into memory (we need to
     * count kept tensors before writing the header). The full file is ~498 MB
     * so this is fine on a dev machine. */
    typedef struct { char key[128]; int ndim; int shape[4]; float *data; int n; } T;
    T *all = calloc(n_total, sizeof(T));
    int n_keep = 0;
    for (int i = 0; i < n_total; i++) {
        int klen;
        fread(&klen, 4, 1, fin);
        if (klen >= 128) klen = 127;
        fread(all[i].key, 1, klen, fin);
        all[i].key[klen] = '\0';
        fread(&all[i].ndim, 4, 1, fin);
        int n = 1;
        for (int d = 0; d < all[i].ndim; d++) {
            fread(&all[i].shape[d], 4, 1, fin);
            n *= all[i].shape[d];
        }
        all[i].n = n;
        all[i].data = malloc((size_t)n * sizeof(float));
        fread(all[i].data, 4, n, fin);
        if (is_keep_layer(all[i].key)) n_keep++;
    }
    fclose(fin);
    printf("[*] read %d tensors, %d kept (layers 0 and %d)\n", n_total, n_keep, N_LAYER - 1);

    if (n_keep != 24) {
        fprintf(stderr, "[!] expected 24 kept tensors (12 per layer × 2), got %d\n", n_keep);
        fprintf(stderr, "    check that source has h.0.* and h.%d.* keys\n", N_LAYER - 1);
        for (int i = 0; i < n_total; i++) {
            if (is_keep_layer(all[i].key)) fprintf(stderr, "    kept: %s\n", all[i].key);
        }
        /* proceed anyway — partial is still useful for debugging */
    }

    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { fprintf(stderr, "[!] cannot open %s for writing\n", argv[2]); return 1; }
    fwrite("GPW2", 1, 4, fout);
    fwrite(&n_keep, 4, 1, fout);

    size_t bytes = 8;
    for (int i = 0; i < n_total; i++) {
        if (!is_keep_layer(all[i].key)) continue;
        int klen = (int)strlen(all[i].key);
        fwrite(&klen, 4, 1, fout);
        fwrite(all[i].key, 1, klen, fout);
        fwrite(&all[i].ndim, 4, 1, fout);
        for (int d = 0; d < all[i].ndim; d++)
            fwrite(&all[i].shape[d], 4, 1, fout);
        fwrite(all[i].data, 4, all[i].n, fout);
        bytes += 4 + klen + 4 + all[i].ndim * 4 + (size_t)all[i].n * 4;
        free(all[i].data);
    }
    fclose(fout);
    free(all);
    printf("[*] wrote %s (%.1f MB, %d tensors)\n", argv[2], bytes / (1024.0 * 1024.0), n_keep);
    return 0;
}
