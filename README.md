# LAL — Logic-Assembly Language

A high-performance CPU LLM inference runtime with Q8 per-row quantization (mistral.rs-inspired SIMD) as default.

**34-49 tok/s on CPU, 2.7x faster than PyTorch, coherent quality, 27MB memory.**

## Quick start

```bash
# Build (auto-detects OpenBLAS on x86, SDOT on arm64)
make server

# Run (Q8 is DEFAULT — no flags needed)
./prebuilt/gpt2_server                        # Q8: 47 tok/s, coherent, 27MB
./prebuilt/gpt2_server --q4                   # Q4: 49 tok/s, coherent, 14MB
./prebuilt/gpt2_server --no-q8                # float: 15 tok/s, best quality, 108MB
```

## Inference modes

| Mode | Flag | Speed (amd64) | Speed (arm64) | Quality | Memory |
|------|------|---------------|---------------|---------|--------|
| **Q8 (DEFAULT)** | (none) | **47 tok/s** | **44.7 tok/s** | ✅ Coherent | 27 MB |
| Q4 | `--q4` | 49 tok/s | 35.4 tok/s | ✅ Coherent | 14 MB |
| Float+BLAS | `--no-q8` | 15 tok/s | — | ✅ Best | 108 MB |

### Deprecated modes (removed from default, kept in code for reference)
- `--bwn` / `--binary` / `--rsign`: 1-bit quantization, garbled output
- `--int8` / `--mixed-int8`: activation quantization, degraded quality
- `--early-exit`: layer early exit, mode collapse
- `--vcopy`: degenerate V-copy attention (replaced by real causal attention)

## Key optimizations

### Q8 per-row quantization (DEFAULT)
- **8-bit per-row scale** (correlation 0.99994 vs float, near-lossless)
- **AVX2 VPMADDUBSW** (x86): 32 uint8 × 32 int8 → 16 int16 → 8 int32
- **NEON SDOT** (arm64): vdotq_s32, 16 int8 × 16 int8 → 4 int32 per instruction
- **8-output parallel** (mistral.rs register blocking): 8 accumulators in registers
- **Pre-computed w_sums**: zero-point offset at load time
- **int8 LM head**: auto-enabled, SDOT-accelerated on arm64, multi-threaded

### Q4 4-bit quantization
- **int4 packed** (2 per byte), 2x less memory than Q8
- Same SIMD techniques (unpack int4 → int8 → VPMADDUBSW/SDOT)
- Quality: correlation 0.998 vs float (slightly below Q8)

### Real causal attention
- Multi-head QK softmax + KV cache (replaces degenerate V-copy)
- Head-major KV cache layout
- SIMD vectorized (v8f wrapper, AVX2/NEON auto)

### Training stack (gpt2_train)
- Adam optimizer + lr warmup/cosine decay
- Knowledge distillation (teacher float → student binary)
- TWN ternary weights {-1, 0, +1} (BitNet b1.58-inspired)
- STE (Straight-Through Estimator) with logic-guided binarization
- Logic-guided binarization (CORE/BINARY/PRUNE per-output)

### Level-2 fusion: sampling filter
- `ban_last` / `ban_repeat(n)` / `ban_token(id)` — token-id level
- `ban_relation("trigger", "allow1", ...)` — concept-level

## Benchmark

### End-to-end (GPT-2 base, 20 tokens, same hardware)

| Implementation | amd64 2-core | arm64 8-core | Quality | Memory |
|---|---|---|---|---|
| PyTorch float32 | 16.3 tok/s | — | Coherent | 498 MB |
| PyTorch int8 | 20.7 tok/s | — | Degraded | 124 MB |
| **LAL Q8 (default)** | **47 tok/s** | **44.7 tok/s** | **Coherent** | **27 MB** |
| LAL Q4 | 49 tok/s | 35.4 tok/s | Coherent | 14 MB |
| LAL float+BLAS | 15 tok/s | — | Best | 108 MB |

### Isolated matmul (768×768, 500 trials)

| Implementation | amd64 (AVX2) | arm64 (SDOT) |
|---|---|---|
| LAL Q8 8-output | 0.037 ms | 0.038 ms |
| llama.cpp Q8_0 | 0.131 ms | — |
| **Speedup vs llama.cpp** | **1.48x** | — |

### Speed-quality tradeoff

```
Quality
  Best │  float+BLAS (15 tok/s) ───────────
  Good │  ★ Q8 default (47 tok/s) ─────────  ← BEST
       │  Q4 (49 tok/s) ────────────────────
       └────────────────────────────────────
       0    10    20    30    40    50    60 tok/s
```

## Building

```bash
# x86 (auto-detects OpenBLAS)
make server

# arm64 (compile with SDOT support)
gcc -O3 -march=armv8.2-a+dotprod -DUSE_OPENBLAS=0 -I. \
    -Wno-unused-function -Wno-unused-variable \
    tools/server/gpt2_server.c runtime/lal_runtime.c \
    -lm -lpthread -o gpt2_server

# Training (no PyTorch, pure C)
make train
./prebuilt/gpt2_train 10000 0.05 --ste --logic --real-attention
```

## Testing

```bash
# Q8 SIMD benchmarks
gcc -O3 -mavx2 -mfma -o tests/test_q8_simd tests/test_q8_simd.c -lm && ./tests/test_q8_simd
gcc -O3 -mavx2 -mfma -o tests/test_q8_mistral tests/test_q8_mistral_style.c -lm && ./tests/test_q8_mistral

# arm64 SDOT benchmark
gcc -O3 -march=armv8.2-a+dotprod -o tests/test_neon_sdot tests/test_neon_sdot.c -lm && ./tests/test_neon_sdot

# llama.cpp Q8_0 comparison
gcc -O3 -mavx2 -mfma -o tests/bench_llama_q8 tests/bench_llama_style_q8.c -lm && ./tests/bench_llama_q8

# PyTorch baseline
python3 bench_pytorch.py
python3 bench_pytorch_int8.py
```

## Structure

```
lal/
├── compiler/lal.py          # LAL → C compiler
├── runtime/                 # Model-agnostic runtime (bin_forward, attention, etc.)
├── models/gpt2.c            # GPT-2 training (pure C, no PyTorch)
├── tools/
│   ├── server/gpt2_server.c # Q8/Q4/float inference server
│   ├── dist/                # Distributed inference (pipeline parallelism)
│   ├── export_*.py          # Weight export from PyTorch
│   └── train_binary_gpt2.py # STE training (PyTorch)
├── tests/                   # Unit tests + benchmarks
├── scripts/                 # Quality baseline, corpus, analysis
└── prebuilt/                # Pre-built binaries
```

## License

MIT
