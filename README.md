# LAL — Logic-Assembly Language

A logic-native language: write programs using **concept / boundary / relation** primitives, and the compiler emits specialized C code with no runtime abstraction overhead.

> Standard LLM runtimes (llama.cpp, MLC-LLM) compile a *trained model's forward pass* — every input runs the entire model. LAL instead compiles a *logic program* — only the computation the program specifies is emitted, everything else is dropped at compile time.

## Quick start

```bash
# Compile a logic program to specialized C:
python3 scripts/lal/lal.py scripts/lal/syllogism.lal main out.c

# Build and run:
gcc -O3 -o syllogism out.c
./syllogism 0.7 0.6 0.2 0.4 0.3 0.1 0.5 0.2

# Verify all demos match the Python reference:
python3 scripts/lal/verify_all.py

# Benchmark specialized vs generic (llama.cpp-style):
python3 scripts/lal/benchmark.py
```

## What's here

- **`scripts/lal/lal.py`** — the language: parser, AST, compiler (to specialized C), and Python reference interpreter (~1100 lines)
- **`scripts/lal/*.lal`** — three demo programs:
  - `demo.lal` — masked classifier (4 concepts, 2 bounds, 2 relations, 1 rule)
  - `syllogism.lal` — VSA syllogism reasoning (memory + rule recursion)
  - `vsa_ops.lal` — exercises `bind` / `bundle` / `permute` directly
- **`scripts/lal/verify_all.py`** — verifies all 3 demos (49/49 tests pass)
- **`scripts/lal/benchmark.py`** — specialized vs generic C comparison
- **`download/lal-mvp/`** — self-contained buildable package with Makefile

See **[download/lal-mvp/README.md](download/lal-mvp/README.md)** for the full design writeup, performance numbers, and limitations.

## The language in one minute

```lal
concept cat = [1.0, 0.1, 0.2, 0.7, 0.3, 0.5, 0.8, 0.2]
concept dog = [0.8, 0.2, 0.3, 0.6, 0.4, 0.4, 0.7, 0.3]

bound animal_dims = [0, 2, 3, 6]

relate animal_sim(a, b) = dot(a, b) @ animal_dims

rule classify(query):
    best = argmax {
        cat: animal_sim(query, cat),
        dog: animal_sim(query, dog)
    }
    output(best)
```

Compiles to ~20 lines of specialized C: bounds applied at compile time, dot products fully unrolled, argmax flattened to if-statements. No loops, no runtime mask indirection, no generic matmul.

## Status

**v0.2** — supports VSA operators (bind/bundle/permute), memory with soft recall, and rule recursion. 49/49 tests pass. See the [v0.2 changelog](download/lal-mvp/README.md#v02--whats-new).

## License

MIT
