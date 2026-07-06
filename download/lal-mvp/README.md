# LAL — Logic-Assembly Language

A logic-native language: write programs using **concept / boundary / relation** primitives, and the compiler emits specialized C code with no runtime abstraction overhead.

> The core thesis: standard LLM runtimes (llama.cpp, MLC-LLM) compile a *trained model's forward pass* — every input runs the entire model. LAL instead compiles a *logic program* — only the computation the program specifies is emitted, everything else is dropped at compile time.

## v0.4 — what's new

| Feature | v0.1 | v0.2 | v0.3 | v0.4 |
|---|---|---|---|---|
| Operators | `dot` (masked) | + VSA ops | + SIMD | + **if/else ternary** (`cond ? a : b`) |
| Memory | — | `memory`+`recall` | same | same |
| Rules | 1 | multiple, inlined | true C recursion | + **recursive base cases** via if/else |
| Concept loading | literals | literals | `load_word2vec`/`load_glove` | same |
| Quantization | — | — | — | **`--quantize int8`** (4x smaller concept data) |
| Embedding scale | 8-dim | 8-dim | 8-dim | **300-dim** (real word2vec scale) |
| Demos | 1 | 3 | 5 | **7** |
| Test cases | 20 | 49 | 75 | **92** (all passing) |

### v0.4 highlights

1. **if/else ternary** — `result = cond ? expr_true : expr_false`. Both scalar and vector branches supported. **Short-circuits** in both the Python reference interpreter and the generated C (C ternary / if-else only evaluates the taken branch). This enables **recursive base cases** — rules can now terminate recursion conditionally.

2. **int8 quantization** — `--quantize int8` flag stores concept vectors as `int8_t + scale` instead of `float`. Dot products use a specialized SIMD helper that dequantizes on the fly (`_mm256_cvtepi8_epi32` → `_mm256_cvtepi32_ps` → `fmadd`). 4x smaller `.rodata` for concept data. Verified: quantized output matches float output exactly.

3. **300-dim real-scale demo** — `embed_300d.lal` loads 300-dimensional vectors from a word2vec-format file with 32 words across 4 semantic clusters. Exercises the full pipeline at real embedding scale: SIMD (37 AVX2 blocks per dot), embedding loading, and classification.

## The language (6 primitives)

```lal
concept name = [v0, v1, ...]                       # literal vector
concept name = load_word2vec("file.txt", "word")   # load from embedding file at compile time
concept name = load_glove("file.txt", "word")      # GloVe format (same as word2vec text)
bound   name = [d0, d1, ...]                       # a dimension mask
memory  name = [key1: val1, key2: val2, ...]       # a fact store
relate  name(a, b) = dot(a, b) @ bound_name        # masked dot product (SIMD if width ≥ 8)
relate  name(a, b) = bind(a, b)                    # VSA bind (element-wise product)
relate  name(a, b) = bundle(a, b)                  # VSA bundle (normalized sum)
relate  name(a)    = permute(a, k)                 # VSA permute (cyclic shift by k)
rule    name(args):
    var = expr                                     # use relations, ops, rule calls, recall
    output(var)
```

Expressions support: `and`, `or`, `not`, `<`, `>`, `<=`, `>=`, `+`, `-`, `*`,
**`cond ? a : b`** (if/else ternary, short-circuits),
relation calls, **rule calls** (real recursion), **`recall(memory, query)`** (soft lookup),
and `argmax { label: expr, ... }`.

## Demos

### 1. `demo.lal` — masked classifier (v0.1)
4 concepts, 2 bounds, 2 relations, 1 rule. Classifies a query as {cat, dog, car, vehicle}.

### 2. `syllogism.lal` — VSA syllogism reasoning (v0.2)
`Socrates → human → mortal`. Exercises `memory`/`recall` and rule composition.

### 3. `vsa_ops.lal` — VSA operators (v0.2)
Exercises `bind`, `bundle`, `permute` directly.

### 4. `embed_demo.lal` — embedding loading (v0.3)
Loads concept vectors from `embeddings.txt` via `load_word2vec()`. Also exercises SIMD (width-8 dots).

### 5. `recursion_demo.lal` — true recursion (v0.3)
Chains through `shift_once` → `shift_twice` → `shift_thrice` via real C function calls.

### 6. `graph_traversal.lal` — recursive graph traversal with if/else (v0.4, NEW)
Follows a pointer chain in memory: `a→b→c→terminal`. Uses `if/else` ternary for the base case (`sim(current, terminal) > 0.5 ? current : follow(next)`). Proves that if/else enables true finite recursion with base cases.

### 7. `embed_300d.lal` — 300-dim real-scale classification (v0.4, NEW)
Loads 300-dimensional vectors from `embeddings_300d.txt` (32 words, 4 clusters). Exercises SIMD at real scale (37 AVX2 blocks per dot product). Can be compiled with `--quantize int8` for 4x smaller concept data.

## Verification

```
$ python3 scripts/lal/verify_all.py

=== Demo: demo ===            → 20 passed
=== Demo: syllogism ===       → 16 passed
=== Demo: vsa_ops ===         → 13 passed
=== Demo: embed_demo ===      → 14 passed
=== Demo: recursion_demo ===  → 12 passed
=== Demo: graph_traversal === →  9 passed  (NEW: if/else recursion)
=== Demo: embed_300d ===      →  8 passed  (NEW: 300-dim real scale)

=== TOTAL: 92 passed, 0 failed ===
[*] ALL DEMOS VERIFIED — C output matches Python reference.
```

## SIMD verification

```bash
# Build with AVX2:
gcc -O3 -mavx2 -mfma -o embed_demo_avx embed_demo.c
# Build scalar-only:
gcc -O3 -mno-avx -o embed_demo_scalar embed_demo.c

# The AVX2 build contains vfmadd/vmulps/vhaddps instructions; the scalar build has none.
```

## Performance (v0.1 demo, still representative)

| Optimization | Specialized (LAL) | Generic (llama.cpp-style) | Speedup |
|---|---|---|---|
| `-O3` | 311 M calls/s | 236 M calls/s | 1.32× |
| `-O0` | 139 M calls/s | 19.6 M calls/s | **7.1×** |

The `-O3` gap is small because gcc sees through this trivial 4-element loop. Real LLM-scale code (variable shapes, indirect dispatch, KV cache) cannot be optimized through, so the gap widens toward the `-O0` numbers. SIMD extends this further for width ≥ 8 vectors.

