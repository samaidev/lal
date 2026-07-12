# MINIMAL — How to change 1 file and add a new feature

This guide shows the **smallest possible change** to add common features. Each example is self-contained.

---

## Example 1: Add a new quantization format (Q3_K)

Suppose you want to add Q3_K (3-bit quantization). Here's the minimal change:

### Step 1: Add to `runtime/lal_q4k_kernel.h` (or new `lal_q3k_kernel.h`)

```c
/* Copy lal_matmul_q4_k, change:
 * 1. Block size from 144 to whatever Q3_K uses
 * 2. Scale unpacking (3-bit instead of 6-bit)
 * 3. Nibble extraction (3-bit instead of 4-bit)
 */
static inline void lal_matmul_q3_k(float *y, const uint8_t *q3k_W,
                                     const float *x, const float *b,
                                     int in_dim, int out_dim) {
    /* ... your implementation ... */
}
```

### Step 2: Add qtype=6 to `tools/server/qwen7b_server.c`

```c
// In Layer struct, add:
const uint8_t *q3k_q, *q3k_k, *q3k_v, *q3k_o, *q3k_gate, *q3k_up, *q3k_down;

// In get_ function:
static const uint8_t *get_q3_k(const char *key) {
    GPQ8Tensor *t = gp_find(key);
    if (!t || t->qtype != 6) { fprintf(stderr, "[!] %s not Q3_K\n", key); exit(1); }
    return (const uint8_t*)t->data;
}

// In LAYER_MATMUL macro, add:
if ((L)->qtype == 6) {
    parallel_matmul_q3_k((y), (L)->q3k_##q3kf, (x), (b), (in_dim), (out_dim));
}

// In layer loading, add:
if (layer_qtype == 6) {
    sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l); L->q3k_q = get_q3_k(key);
    /* ... etc for k, v, o, gate, up, down ... */
}
```

### Step 3: Add to converter `scripts/convert_qwen7b_q3k.py`

```python
# Copy convert_qwen7b_q4k.py, change:
# 1. quantize function (3-bit instead of 4-bit)
# 2. out.write(struct.pack('<B', 6))  # qtype=6
```

### Step 4: Build and test
```bash
python3 scripts/convert_qwen7b_q3k.py
make qwen7b-server
./prebuilt/qwen7b_server --weights prebuilt/qwen7b_q3k_weights.bin --prompt "Hi" --n 10
```

**Total: 3 files changed, ~200 lines of new code.**

---

## Example 2: Add a new model (Llama-3-8B)

### Step 1: Copy and rename
```bash
cp tools/server/qwen7b_server.c tools/server/llama3_server.c
cp scripts/convert_qwen7b_q4k.py scripts/convert_llama3_q4k.py
```

### Step 2: Change constants in `llama3_server.c`
```c
#define N_LAYER   32     // was 28
#define N_EMBD    4096   // was 3584
#define N_HEAD    32     // was 28
#define N_KV_HEAD 8      // was 4
#define HEAD_DIM   128   // was 128 (same)
#define MLP_DIM   14336  // was 18944
#define VOCAB_SIZE 128256 // was 152064
```

### Step 3: Change chat template
Llama-3 uses `<|begin_of_text|><|start_header_id|>user<|end_header_id|>` instead of `<|im_start|>user\n`.

### Step 4: Add Makefile target
```makefile
llama3-server: prebuilt/llama3_server
prebuilt/llama3_server: tools/server/llama3_server.c runtime/*.h
	$(CC) $(CFLAGS) -fopenmp -o $@ tools/server/llama3_server.c runtime/lal_runtime.c -lm -lgomp
```

### Step 5: Convert and run
```bash
python3 scripts/convert_llama3_q4k.py
make llama3-server
./prebuilt/llama3_server --weights prebuilt/llama3_q4k_weights.bin --prompt "Hi"
```

**Total: 2 new files, 1 Makefile target. No runtime code changes needed.**

---

## Example 3: Add a new sampling method (top-p)

### Step 1: Find `lal_sample_token` in `runtime/lal_sampling.h`

### Step 2: Add top-p logic
```c
int lal_sample_token_top_p(float *logits, int n, float temp, int top_k, float top_p) {
    /* 1. Sort logits descending
     * 2. Compute softmax probabilities
     * 3. Find smallest k such that cumsum(prob[:k]) >= top_p
     * 4. Sample from those k tokens
     */
}
```

### Step 3: Wire up in `qwen7b_server.c`
```c
// Add CLI arg:
else if (!strcmp(argv[i],"--top-p") && i+1<argc) g_top_p=atof(argv[++i]);

// In forward(), change:
int next = lal_sample_token_top_p(g_logits, VOCAB_SIZE, g_temperature, g_top_k, g_top_p);
```

**Total: 1 function added, 2 lines changed in server.**

---

## Example 4: Add KV cache quantization

The KV cache is currently F32 (448MB for 28 layers × 4096 ctx × 512 kv_dim × 4B).

### Step 1: Change allocation in `qwen7b_server.c`
```c
// Was: kv_k[l] = memalign(32, N_CTX * KV_DIM * sizeof(float));
// Now: int8 KV cache + per-block scale
kv_k_q[l] = memalign(32, N_CTX * KV_DIM);          // int8
kv_k_s[l] = memalign(32, N_CTX * (KV_DIM/32) * sizeof(float)); // scales
```

### Step 2: Quantize K after k_proj
```c
// After: LAYER_MATMUL(g_k, L, ..., g_ln, L->k_bias, N_EMBD, KV_DIM);
// Add: quantize g_k to int8, store in kv_k_q[l][pos]
```

### Step 3: Dequantize K during attention
```c
// In attention loop, dequantize kv_k_q[l][t] to float before dot product
// Or: write an int8 attention kernel (more complex, but 4× less bandwidth)
```

**Total: ~50 lines changed. Saves 336MB memory + 4× less KV bandwidth.**

---

## Example 5: Add GPU offload (conceptual)

This is NOT minimal, but shows the architecture:

### Step 1: Add a `#ifdef LAL_CUDA` section in each kernel header
```c
#ifdef LAL_CUDA
// CUDA implementation using cuBLAS or custom kernels
#else
// CPU implementation (current)
#endif
```

### Step 2: Add CUDA build target in Makefile
```makefile
cuda: $(SRCS)
	nvcc -O3 -Xcompiler -fopenmp -o prebuilt/qwen7b_server_cuda \
	  tools/server/qwen7b_server.c runtime/lal_runtime.c -lm -lgomp -lcublas
```

### Step 3: In `qwen7b_server.c`, add GPU memory management
```c
// After loading weights, copy to GPU:
cudaMemcpy(g_wte_gpu, g_wte, vocab*n_embd*4, cudaMemcpyHostToDevice);
// Change matmul calls to cuBLAS:
cublasSgemm(...);  // instead of lal_matmul_q4_k
```

**This is a major project (weeks of work), not minimal.** But the architecture doesn't prevent it.

---

## The point of these examples

LAL's value is that **common changes touch very few files**. The architecture is flat — no abstraction layers to fight through.

If you're adding a feature and it requires touching more than 3-4 files, you're probably doing it wrong. Ask: "can I do this by changing just the kernel?"

---

## File change matrix

| Feature | Files to change | Lines of code |
|---------|----------------|---------------|
| New quant format | kernel header + server + converter | ~200 |
| New model | server (copy+modify) + converter | ~300 |
| New sampling method | sampling header + server (2 lines) | ~50 |
| KV cache quant | server (3 places) | ~50 |
| New CPU platform | 3 kernel functions | ~500 |
| GPU offload | everything | ~2000 |

**The "new model" and "new sampling" rows are the easiest entry points for new contributors.**
