# LAL — Logic-Assembly Language

A logic-native language: write programs using **concept / boundary / relation** primitives, and the compiler emits specialized C code with no runtime abstraction overhead.

> The core thesis: standard LLM runtimes (llama.cpp, MLC-LLM) compile a *trained model's forward pass* — every input runs the entire model. LAL instead compiles a *logic program* — only the computation the program specifies is emitted, everything else is dropped at compile time.

## v0.2 — what's new

This version extends LAL from a single-rule toy into a language that can express real logic programs:

| Feature | v0.1 | v0.2 |
|---|---|---|
| Operators | `dot` (masked) | `dot`, **`bind`**, **`bundle`**, **`permute`** (VSA ops) |
| Memory | — | **`memory`**: fact store with soft (similarity-weighted) `recall` |
| Rules | 1 per program | **Multiple rules**, rules can **call other rules** (recursion up to fixed depth) |
| Demos | 1 (classifier) | 3 (classifier, syllogism, VSA ops) |
| Test cases | 20 | 49 (all passing) |

## The language (6 primitives)

```lal
concept name = [v0, v1, ...]                       # a static vector
bound   name = [d0, d1, ...]                       # a dimension mask
memory  name = [key1: val1, key2: val2, ...]       # a fact store
relate  name(a, b) = dot(a, b) @ bound_name        # masked dot product
relate  name(a, b) = bind(a, b)                    # VSA bind (element-wise product)
relate  name(a, b) = bundle(a, b)                  # VSA bundle (normalized sum)
relate  name(a)    = permute(a, k)                 # VSA permute (cyclic shift by k)
rule    name(args):
    var = expr                                     # use relations, ops, rule calls, recall
    output(var)
```

Expressions support: `and`, `or`, `not`, `<`, `>`, `<=`, `>=`, `+`, `-`, `*`,
relation calls, **rule calls** (recursion), **`recall(memory, query)`** (soft lookup),
and `argmax { label: expr, ... }`.

## Demos

### 1. `demo.lal` — masked classifier (v0.1, still works)
4 concepts, 2 bounds, 2 relations, 1 rule. Classifies a query as {cat, dog, car, vehicle} using different bounds for animals vs machines.

### 2. `syllogism.lal` — VSA syllogism reasoning (v0.2)
A two-step reasoning chain: `Socrates → human → mortal`. Exercises:
- **`memory`** with `recall` for fact lookup
- **Rule recursion**: `conclude_mortal` calls `lookup_is_a` then `lookup_is`
- **`argmax`** over the final property vector

```lal
memory facts_is_a = [socrates: human, plato: human, aristotle: human, stone: stone]
memory facts_is   = [human: mortal]

rule conclude_mortal(subject):
    cat  = lookup_is_a(subject)    # rule call (recursion)
    prop = lookup_is(cat)          # rule call
    output(prop)
```

### 3. `vsa_ops.lal` — VSA operators (v0.2)
Exercises `bind`, `bundle`, `permute` directly: builds a role-filler representation of a query and classifies it.

## What the compiler does

The compiler emits specialized C code where:

1. **BOUNDS applied at compile time** — concept vectors reduced to only the used dimensions; unused (concept, bound) pairs not emitted at all
2. **DOT fully unrolled** — no loops, no array indexing
3. **BIND/BUNDLE/PERMUTE fully unrolled** — element-wise ops, no loops
4. **MEMORY recall compiled to fixed dot products + weighted sum** — no hash table, no dynamic dispatch
5. **Rule calls inlined** up to `max_recursion_depth=2` (configurable)
6. **argmax flattened** — flat comparison chain, no function call

## Verification

```
$ python3 scripts/lal/verify_all.py

=== Demo: demo ===
  [OK] cat / dog / car / vehicle + 16 random → 20 passed
=== Demo: syllogism ===
  [OK] socrates / plato / aristotle / stone / human / mortal + 10 random → 16 passed
=== Demo: vsa_ops ===
  [OK] alice / bob / carol + 10 random → 13 passed

=== TOTAL: 49 passed, 0 failed ===
[*] ALL DEMOS VERIFIED — C output matches Python reference.
```

## Performance (v0.1 demo, still representative)

| Optimization | Specialized (LAL) | Generic (llama.cpp-style) | Speedup |
|---|---|---|---|
| `-O3` | 311 M calls/s | 236 M calls/s | 1.32× |
| `-O0` | 139 M calls/s | 19.6 M calls/s | **7.1×** |

The `-O3` gap is small because gcc sees through this trivial 4-element loop. Real LLM-scale code (variable shapes, indirect dispatch, KV cache) cannot be optimized through, so the gap widens toward the `-O0` numbers.

## File layout

```
lal/
├── README.md                   # this file
├── scripts/lal/
│   ├── lal.py                  # the language: parser + AST + compiler + reference interpreter
│   ├── demo.lal                # demo 1: masked classifier
│   ├── syllogism.lal           # demo 2: VSA syllogism with memory + recursion
│   ├── vsa_ops.lal             # demo 3: bind/bundle/permute exercise
│   ├── verify.py               # verify demo 1
│   ├── verify_syllogism.py     # verify demo 2
│   ├── verify_all.py           # verify all 3 demos
│   ├── benchmark.py            # specialized vs generic benchmark
│   └── ref/                    # (empty — reference outputs are generated in-process)
└── download/lal-mvp/           # self-contained buildable package
    ├── Makefile
    ├── README.md
    ├── src/
    │   ├── demo.lal
    │   ├── demo.c              # generated
    │   ├── demo_generic.c      # hand-written generic version for comparison
    │   ├── syllogism.lal
    │   ├── syllogism.c         # generated
    │   └── vsa_ops.lal
    └── build/
        └── ...                 # compiled binaries
```

## Build & run

```bash
# From download/lal-mvp/:
make x86         # build all demos for x86_64
make verify      # verify all demos against Python reference
make bench       # benchmark specialized vs generic
make arm64       # cross-compile for arm64 (needs aarch64-linux-gnu-gcc)
make clean
```

Or directly:
```bash
python3 scripts/lal/lal.py scripts/lal/syllogism.lal main out.c
gcc -O3 -o syllogism out.c
./syllogism 0.7 0.6 0.2 0.4 0.3 0.1 0.5 0.2
```

## Limitations & next steps

This is v0.2 — a real language, but still has limits:

- **No training**: concept vectors are literals. Next: import from word2vec/bert embedding files.
- **Single operator per relate**: `relate` can only be one of dot/bind/bundle/permute. Next: composite relates.
- **Fixed recursion depth**: rule calls inline up to depth 2. Next: true recursion via C recursion.
- **No SIMD**: scalar C only. Next: emit AVX2/NEON intrinsics for wide vectors.
- **No quantization**: float32 only. Next: int8/int4 paths.
- **No MEMORY persistence**: memories are compile-time literals. Next: load from files.

## License

MIT
