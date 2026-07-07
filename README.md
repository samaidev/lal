# LAL — Logic-Assembly Language

A logic-native language that compiles logic programs to specialized C with bit-level parallelism (XNOR+popcount).

## Structure

```
lal/
├── compiler/lal.py          # LAL → C compiler (model-agnostic)
├── runtime/
│   ├── lal_runtime.h        # Universal runtime API (any transformer model)
│   └── lal_runtime.c        # Binary matmul, layer_norm, gelu, softmax, tensor I/O
├── models/
│   └── gpt2.c               # GPT-2 model (uses lal_runtime)
├── demos/                   # LAL programs (.lal files)
├── tools/                   # Weight export, PyTorch STE, verification
├── prebuilt/                # Pre-built binaries (open out of the box!)
└── docs/                    # Design docs
```

## Quick start

```bash
# GPT-2 training (no PyTorch, pure C, 3.4ms/step)
./prebuilt/gpt2_train 10000 0.05    # 10000 steps in 35 seconds

# Compile a LAL program
python3 compiler/lal.py demos/basic/demo.lal classify demo.c
gcc -O3 -o demo demo.c -lm

# Build from source
make train    # builds prebuilt/gpt2_train
make demos    # builds demo binaries
```

## Adding a new model (e.g. BERT)

1. Create `models/bert.c` — include `runtime/lal_runtime.h`
2. Define model config (dim, layers, heads, vocab)
3. Call `bin_forward()` / `bin_backward()` from lal_runtime
4. Build: `gcc -O3 -o bert models/bert.c runtime/lal_runtime.c -lm`

The runtime is model-agnostic. GPT-2 is just one example.

## Results

| Task | Metric | Value |
|---|---|---|
| GPT-2 training (LAL native) | 10000 steps, pure C | 35s, 3.4ms/step, loss 5.5→0.3 |
| vs PyTorch | speedup | 250x |
| Weight compression | binary (sign+alpha) | 44x (498MB → 11.3MB) |

## License

MIT
