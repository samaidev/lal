# LAL — Logic-Assembly Language

A logic-native language that compiles logic programs to specialized C code with bit-level parallelism via XNOR+popcount.

## Core idea

Standard LLM runtimes (llama.cpp) compile a trained model's forward pass — every input runs the entire model. LAL compiles a *logic program* — only the computation the program specifies is emitted, everything else is dropped at compile time.

GPT-2's weight matrices contain 79% logical redundancy (verified on real weights). LAL eliminates this at compile time via threshold pruning, then binarizes the remaining weights to {-1,+1} for XNOR+popcount computation — 64 multiplications per `popcount` instruction.

## Results

| Task | Metric | Value |
|---|---|---|
| GPT-2 inference | Float, pure C | "The capital of France is" → coherent text |
| GPT-2 binary inference | 44x compression | 498MB → 11.3MB, coherent text |
| GPT-2 training (PyTorch STE) | 200 steps, CPU | 3 minutes, loss 9.5 → 0.4 |
| **GPT-2 training (LAL native)** | **10000 steps, pure C** | **36 seconds, loss 5.5 → 0.3, 3.6ms/step** |
| **Speedup vs PyTorch** | | **250x** |

## Structure

```
lal/
├── compiler/lal.py        # LAL → C compiler (parser + AST + codegen)
├── runtime/               # C runtime for inference + training
│   ├── gpt2_runtime.c     # GPT-2 inference (float + binary)
│   ├── gpt2_binary.c      # Binary weight loader + XNOR+popcount matmul
│   └── gpt2_train.c       # GPT-2 training (all-popcount, 3.6ms/step)
├── demos/                 # LAL programs
│   ├── basic/             # classifier, syllogism, VSA, recursion, pattern matching
│   ├── embedding/         # word2vec/BERT/GPT-2 embedding demos + training
│   └── data/              # embedding data files
├── tools/                 # weight export, PyTorch STE training, verification
└── docs/                  # design docs
```

## Quick start

```bash
# Compile a LAL program to C
python3 compiler/lal.py demos/basic/demo.lal classify demo.c
gcc -O3 -o demo demo.c -lm
./demo 1.0 0.1 0.2 0.7 0.3 0.5 0.8 0.2

# Train GPT-2 with binary weights (no PyTorch!)
gcc -O3 -mavx2 -o gpt2_train runtime/gpt2_train.c -lm
./gpt2_train 10000 0.05    # 10000 steps in 36 seconds

# GPT-2 inference (float, pure C)
gcc -O3 -o gpt2 runtime/gpt2_runtime.c -lm
./gpt2 "The capital of France is" 10
```

## LAL language

```lal
concept cat = [1.0, 0.1, 0.2, 0.7, 0.3, 0.5, 0.8, 0.2]
bound animal_dims = [0, 2, 3, 6]
relate sim(a, b) = dot(a, b) @ animal_dims

rule classify(query):
    best = argmax {
        cat: sim(query, cat),
        dog: sim(query, dog)
    }
    output(best)
```

Primitives: `concept`, `param`, `bound`, `memory`, `relate` (dot/bind/bundle/permute/vadd/vsub/linear), `rule` (with guards + pattern matching), `if/else` ternary, `loss`, `grad`, `update`.

Compiler flags: `--binarize` (XNOR+popcount), `--train` (forward+backward+update), `--quantize int8/int4`, `--parallel` (SIMD).

## License

MIT
