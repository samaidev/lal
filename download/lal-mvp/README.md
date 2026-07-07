# LAL — Logic-Assembly Language

A logic-native language: write programs using **concept / boundary / relation** primitives, and the compiler emits specialized C code with no runtime abstraction overhead.

> The core thesis: standard LLM runtimes (llama.cpp, MLC-LLM) compile a *trained model's forward pass* — every input runs the entire model. LAL instead compiles a *logic program* — only the computation the program specifies is emitted, everything else is dropped at compile time.

## v0.6 — what's new

| Feature | v0.1 | v0.2 | v0.3 | v0.4 | v0.5 | v0.6 |
|---|---|---|---|---|---|---|
| Operators | `dot` | + VSA | + SIMD | + if/else | + guards | + **pattern matching** |
| Quantization | — | — | — | int8 | int4 | + **SIMD int4** unpack |
| NLP | — | — | — | — | 768-dim | + **sentiment classification** |
| Demos | 1 | 3 | 5 | 7 | 9 | **11** |
| Test cases | 20 | 49 | 75 | 92 | 109 | **132** (all passing) |
| Interface | CLI | CLI | CLI | CLI | CLI | + **Web playground** |

### v0.6 highlights

1. **SIMD int4 unpack** — int4 quantization now uses real SIMD nibble unpacking on AVX2 (`_mm_unpacklo_epi8` + `_mm256_cvtepi8_epi32`) and NEON (`vzipq_u16` + `vmovl_s16`). Previously int4 used scalar unpack (slow). Now int4 binaries are competitive with int8 in both size and speed at 768 dims.

2. **Multi-rule pattern matching** — define multiple rules with the same name and different guards; the compiler emits a dispatcher that tries each in order (Prolog-style). First matching guard wins. The fallback variant (no guard) always matches. Enables clean expression of multi-case logic.

3. **Sentiment classification (real NLP)** — `sentiment_768d.lal` does binary sentiment classification at BERT scale: classifies words as positive/negative by similarity to positive/negative centroids. 14/14 test words classified correctly, including unseen test words.

4. **Web playground** — a Next.js app that lets you edit .lal source in the browser, click "Compile", and see the generated C code instantly. Supports quantization mode selection (float32/int8/int4). Runs lalc in a sandboxed API route.

5. **Real GPT-2 embeddings (v0.6.1)** — `export_gpt2_embeddings.py` exports real GPT-2 (124M params, 768-dim) learned embeddings to word2vec format. `gpt2_demo.lal` classifies words into semantic categories using real GPT-2 knowledge. `gpt2_analogy.lal` solves the classic **king − man + woman = queen** analogy using `vadd`/`vsub` vector arithmetic — the compiled C binary has GPT-2 embeddings baked in, no PyTorch at runtime. Also added `vadd`/`vsub` vector operators for element-wise add/subtract.

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
rule    name(args) | guard_expr:                   # guarded rule (v0.5): body runs only
    var = expr                                     #   when guard is truthy; otherwise the
    output(var)                                    #   rule returns the input unchanged
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

### 7. `embed_300d.lal` — 300-dim real-scale classification (v0.4)
Loads 300-dimensional vectors from `embeddings_300d.txt` (32 words, 4 clusters). Exercises SIMD at real scale (37 AVX2 blocks per dot product). Can be compiled with `--quantize int8` for 4x smaller concept data.

### 8. `guard_demo.lal` — rule guards (v0.5, NEW)
Same graph-traversal logic as demo 6, but written with a guard instead of an inline if/else. `rule follow(current) | sim(current, terminal) < 0.5:` is cleaner than `result = sim(current, terminal) > 0.5 ? current : follow(next)`.

### 9. `embed_768d.lal` — 768-dim BERT-scale classification (v0.5, NEW)
Loads 768-dimensional vectors (BERT-base hidden size) from `embeddings_768d.txt`. `export_bert_embeddings.py` can export real BERT embeddings (requires `transformers`). At 768 dims, int8 quantization gives 2.1x smaller binaries than float.

## Verification

```
$ python3 scripts/lal/verify_all.py

=== Demo: demo ===            → 20 passed
=== Demo: syllogism ===       → 16 passed
=== Demo: vsa_ops ===         → 13 passed
=== Demo: embed_demo ===      → 14 passed
=== Demo: recursion_demo ===  → 12 passed
=== Demo: graph_traversal === →  9 passed
=== Demo: embed_300d ===      →  8 passed
=== Demo: guard_demo ===      →  9 passed
=== Demo: embed_768d ===      →  8 passed
=== Demo: pattern_match ===   →  9 passed  (NEW: multi-rule matching)
=== Demo: sentiment_768d ===  → 14 passed  (NEW: NLP sentiment)

=== TOTAL: 132 passed, 0 failed ===
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

