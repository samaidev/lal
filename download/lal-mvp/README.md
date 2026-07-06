# LAL — Logic-Assembly Language

A logic-native language: write programs using **concept / boundary / relation** primitives, and the compiler emits specialized C code with no runtime abstraction overhead.

> The core thesis: standard LLM runtimes (llama.cpp, MLC-LLM) compile a *trained model's forward pass* — every input runs the entire model. LAL instead compiles a *logic program* — only the computation the program specifies is emitted, everything else is dropped at compile time.

## v0.3 — what's new

| Feature | v0.1 | v0.2 | v0.3 |
|---|---|---|---|
| Operators | `dot` (masked) | + `bind`/`bundle`/`permute` (VSA) | + **SIMD** (AVX2/NEON) for dot width ≥ 8 |
| Memory | — | `memory` + `recall` | same |
| Rules | 1 per program | multiple, inlined to depth 2 | multiple, **true C recursion** (no depth limit) |
| Concept loading | literals only | literals only | **`load_word2vec(file, word)`** / `load_glove(...)` |
| Demos | 1 | 3 | **5** |
| Test cases | 20 | 49 | **75** (all passing) |

### v0.3 highlights

1. **Real embedding loading** — `concept cat = load_word2vec("embeddings.txt", "cat")` reads the vector at compile time and inlines it. Supports word2vec text format and GloVe format (auto-detects header line).

2. **SIMD intrinsics** — when a dot product has width ≥ 8, the compiler emits a `_lal_dot_simd_N()` helper using AVX2 (`__m256`, `_mm256_fmadd_ps`) on x86_64 and NEON (`float32x4_t`, `vfmaq_f32`) on arm64, with a scalar fallback. Guards via `#ifdef __AVX2__` / `#ifdef __ARM_NEON__`.

3. **True recursion** — rule calls are now real C function calls, not inlined. Rules can call themselves or each other recursively. All transitively-reachable rules are emitted (no depth limit).

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
relation calls, **rule calls** (real recursion), **`recall(memory, query)`** (soft lookup),
and `argmax { label: expr, ... }`.

## Demos

### 1. `demo.lal` — masked classifier (v0.1)
4 concepts, 2 bounds, 2 relations, 1 rule. Classifies a query as {cat, dog, car, vehicle}.

### 2. `syllogism.lal` — VSA syllogism reasoning (v0.2)
`Socrates → human → mortal`. Exercises `memory`/`recall` and rule composition.

### 3. `vsa_ops.lal` — VSA operators (v0.2)
Exercises `bind`, `bundle`, `permute` directly.

### 4. `embed_demo.lal` — embedding loading (v0.3, NEW)
Loads concept vectors from `embeddings.txt` via `load_word2vec()`. Also exercises SIMD (width-8 dots).

### 5. `recursion_demo.lal` — true recursion (v0.3, NEW)
A rule that chains through `shift_once` → `shift_twice` → `shift_thrice` via real C function calls. Proves recursion works without inlining.

## Verification

```
$ python3 scripts/lal/verify_all.py

=== Demo: demo ===          → 20 passed
=== Demo: syllogism ===     → 16 passed
=== Demo: vsa_ops ===       → 13 passed
=== Demo: embed_demo ===    → 14 passed  (NEW)
=== Demo: recursion_demo ===→ 12 passed  (NEW)

=== TOTAL: 75 passed, 0 failed ===
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

