#ifndef LAL_SAMPLING_H
#define LAL_SAMPLING_H
/*
 * lal_sampling.h - Reusable token sampling with min-heap top-k.
 *
 * Included by any LAL model server (gpt2_server, qwen_server, etc.).
 * Requires the host to define:
 *   - float *g_logits       (vocab logits, size VOCAB_SIZE)
 *   - int    VOCAB_SIZE     (vocab size macro)
 *   - #include <math.h>, <stdlib.h>, <string.h>
 *
 * Provides:
 *   - lal_sampling_init()           seed RNG
 *   - lal_sample_token(logits, vocab_size, temp, top_k, rep_penalty,
 *                       recent, n_recent)
 *     Returns sampled token id. Falls back to argmax if temp<=0 or top_k<=0.
 *
 * The top-k selection uses a min-heap of size k: O(V log k) instead of
 * O(k*V). For Qwen2.5 (V=151936, k=40) this is 900K vs 6M comparisons.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int   lal_g_temperature = 0.8f;
static int   lal_g_top_k = 40;
static float lal_g_rep_penalty = 1.1f;

static void lal_sampling_init(void) {
    srand((unsigned)time(NULL));
}

/* Sample one token from logits using temperature + top-k + rep_penalty.
 *
 * logits:       [vocab_size] float array (modified in-place for rep_penalty)
 * vocab_size:   number of vocab entries
 * temperature:  softmax temperature (0 = argmax/greedy)
 * top_k:        sample from top-k highest logits (0 = full vocab)
 * rep_penalty:  repetition penalty (1.0 = off, 1.1 = mild)
 * recent:       array of recent token ids (for rep_penalty), or NULL
 * n_recent:     number of entries in recent[]
 *
 * Returns: sampled token id in [0, vocab_size)
 */
static int lal_sample_token(float *logits, int vocab_size,
                            float temperature, int top_k,
                            float rep_penalty,
                            const int *recent, int n_recent) {
    /* Greedy / argmax path */
    if (temperature <= 0.0f || top_k <= 0) {
        int best = 0;
        for (int v = 1; v < vocab_size; v++)
            if (logits[v] > logits[best]) best = v;
        return best;
    }

    /* Repetition penalty */
    if (rep_penalty > 1.0f && recent && n_recent > 0) {
        for (int i = 0; i < n_recent; i++) {
            int t = recent[i];
            if (t >= 0 && t < vocab_size) {
                if (logits[t] > 0) logits[t] /= rep_penalty;
                else logits[t] *= rep_penalty;
            }
        }
    }

    /* Top-k threshold via min-heap (O(V log k)) */
    int k = top_k;
    if (k > vocab_size) k = vocab_size;
    float threshold = -1e30f;
    {
        static float heap_val[64];
        static int   heap_idx[64];
        int heap_n = 0;
        for (int v = 0; v < vocab_size; v++) {
            float val = logits[v];
            if (heap_n < k) {
                int c = heap_n++;
                heap_val[c] = val; heap_idx[c] = v;
                while (c > 0) {
                    int p = (c - 1) >> 1;
                    if (heap_val[p] <= heap_val[c]) break;
                    float tv = heap_val[p]; heap_val[p] = heap_val[c]; heap_val[c] = tv;
                    int ti = heap_idx[p]; heap_idx[p] = heap_idx[c]; heap_idx[c] = ti;
                    c = p;
                }
            } else if (val > heap_val[0]) {
                heap_val[0] = val; heap_idx[0] = v;
                int p = 0;
                for (;;) {
                    int l = 2*p+1, r = 2*p+2, s = p;
                    if (l < heap_n && heap_val[l] < heap_val[s]) s = l;
                    if (r < heap_n && heap_val[r] < heap_val[s]) s = r;
                    if (s == p) break;
                    float tv = heap_val[p]; heap_val[p] = heap_val[s]; heap_val[s] = tv;
                    int ti = heap_idx[p]; heap_idx[p] = heap_idx[s]; heap_idx[s] = ti;
                    p = s;
                }
            }
        }
        threshold = heap_val[0];
    }

    /* Softmax over top-k (those >= threshold), with temperature */
    float max_l = -1e30f;
    for (int v = 0; v < vocab_size; v++)
        if (logits[v] >= threshold && logits[v] > max_l) max_l = logits[v];
    float sum = 0;
    for (int v = 0; v < vocab_size; v++) {
        if (logits[v] >= threshold) sum += expf((logits[v] - max_l) / temperature);
    }

    /* Sample */
    float r = (float)rand() / (float)RAND_MAX * sum;
    float acc = 0;
    for (int v = 0; v < vocab_size; v++) {
        if (logits[v] >= threshold) {
            acc += expf((logits[v] - max_l) / temperature);
            if (r <= acc) return v;
        }
    }
    return vocab_size - 1;
}

/* Ring buffer for recent tokens (used by rep_penalty). Caller manages. */
typedef struct {
    int tokens[256];
    int n;
} LalRecentTokens;

static void lal_recent_push(LalRecentTokens *rt, int tok) {
    if (rt->n < 256) rt->tokens[rt->n++] = tok;
    else { memmove(rt->tokens, rt->tokens+1, 255*sizeof(int)); rt->tokens[255] = tok; }
}

#endif /* LAL_SAMPLING_H */
