#ifndef LAL_WEIGHT_UTILS_H
#define LAL_WEIGHT_UTILS_H
/*
 * lal_weight_utils.h - Weight management utilities for LAL model servers.
 *
 * Provides the "free float weights after Q8 quantization" pattern:
 * after Q8 quantization, the original float weight matrices are never
 * read again (the forward pass uses q8_* exclusively). Freeing them
 * recovers significant RSS (e.g. 1.7 GB for Qwen2.5-0.5B).
 *
 * Requires Tensor type from lal_runtime.h.
 */

#include "runtime/lal_runtime.h"
#include <string.h>
#include <stdio.h>

/* Free float tensors matching any of the given keys.
 *
 * tensors:    the Tensor array from tensor_load_all()
 * n_tensors:  array size
 * keys:       NULL-terminated array of tensor key strings to free
 *
 * Returns:    total bytes freed
 * Side effect: sets freed tensors' .data to NULL
 */
static inline size_t lal_free_float_tensors(Tensor *tensors, int n_tensors,
                                            const char **keys) {
    size_t bytes_freed = 0;
    int n_freed = 0;
    for (const char **kp = keys; *kp; kp++) {
        for (int ti = 0; ti < n_tensors; ti++) {
            if (tensors[ti].data && strcmp(tensors[ti].key, *kp) == 0) {
                size_t sz = sizeof(float);
                for (int d = 0; d < tensors[ti].ndim; d++) sz *= tensors[ti].shape[d];
                bytes_freed += sz;
                free(tensors[ti].data);
                tensors[ti].data = NULL;
                n_freed++;
                break;
            }
        }
    }
    if (n_freed > 0) {
        printf("[*] freed %d float weight tensors (%.1f MB recovered)\n",
               n_freed, bytes_freed / (1024.0 * 1024.0));
    }
    return bytes_freed;
}

/* Free float tensors matching a printf-style key pattern with a layer index.
 * Convenience wrapper for per-layer weight matrices.
 *
 * Example: lal_free_layer_weights(tensors, n_tensors, N_LAYER, 7,
 *           "model.layers.%d.self_attn.q_proj.weight",
 *           "model.layers.%d.self_attn.k_proj.weight",
 *           ...,
 *           NULL);
 */
static inline size_t lal_free_layer_weights(Tensor *tensors, int n_tensors,
                                            int n_layers, int n_keys_per_layer,
                                            ...) {
    size_t total = 0;
    va_list args;
    va_start(args, n_keys_per_layer);

    /* Collect format strings */
    const char *fmts[16];
    for (int i = 0; i < n_keys_per_layer && i < 16; i++)
        fmts[i] = va_arg(args, const char*);
    va_end(args);

    for (int l = 0; l < n_layers; l++) {
        const char *keys[17];
        char bufs[16][256];
        for (int i = 0; i < n_keys_per_layer; i++) {
            snprintf(bufs[i], sizeof(bufs[i]), fmts[i], l);
            keys[i] = bufs[i];
        }
        keys[n_keys_per_layer] = NULL;
        total += lal_free_float_tensors(tensors, n_tensors, keys);
    }
    return total;
}

#endif /* LAL_WEIGHT_UTILS_H */
