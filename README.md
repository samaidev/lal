# LAL — Logic-Assembly Language

A logic-native language that compiles logic programs to specialized C with bit-level parallelism (XNOR+popcount), plus a high-performance inference runtime with Q8 per-row quantization (mistral.rs-inspired SIMD) as default — **34 tok/s, 2.6x faster than PyTorch float32, with coherent quality**.

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
├── tools/
│   ├── server/              # GPT-2 web frontend (HTTP server + HTML UI)
│   │   ├── gpt2_server.c    # Q8 SIMD + binary + float inference
│   │   ├── frontend.html    # Browser UI
│   │   └── README.md        # Inference modes
│   ├── arm/                 # ARMv7 / Termux / Android deployment
│   ├── export_*.py          # Weight export from PyTorch checkpoints
│   └── verify.py            # End-to-end correctness check
├── tests/                   # Unit tests (BWN, attention, int8, Q8 SIMD, BLAS)
├── scripts/                 # Quality baseline, logic analysis, corpus tools
├── prebuilt/                # Pre-built binaries (open out of the box!)
└── docs/                    # Design docs
```

## Quick start

```bash
# GPT-2 web server — Q8 mode is DEFAULT (34 tok/s, coherent, 27MB)
./prebuilt/gpt2_server                       # opens http://localhost:8080

# Other modes (optional)
./prebuilt/gpt2_server --no-q8               # full float (7 tok/s, best quality)
./prebuilt/gpt2_server --bwn                 # BWN 1-bit (36 tok/s, garbled)
./prebuilt/gpt2_server --binary              # BNN (47 tok/s, garbled)
./prebuilt/gpt2_server --int8                # Int8 activation (22 tok/s)
./prebuilt/gpt2_server --mixed-int8 8        # Mixed precision (15 tok/s)

# GPT-2 training (no PyTorch, pure C)
./prebuilt/gpt2_train 10000 0.05             # 10000 steps in 35 seconds
./prebuilt/gpt2_train 500 0.002 --ste --logic --real-attention  # logic-guided STE

# Compile a LAL program
python3 compiler/lal.py demos/basic/demo.lal classify demo.c
gcc -O3 -o demo demo.c -lm

# Build from source
make train         # builds prebuilt/gpt2_train
make server        # builds prebuilt/gpt2_server
make server-blas   # builds prebuilt/gpt2_server_blas (OpenBLAS float, 15 tok/s)
make demos         # builds demo binaries
```

## Inference modes

| Mode | Flag | Speed | Quality | Memory | Use case |
|------|------|-------|---------|--------|----------|
| **Q8 (DEFAULT)** | (none) | **34 tok/s** | ✅ Coherent | **27 MB** | **Best speed+quality, default** |
| Float | `--no-q8` | 7 tok/s | ✅ Best | 108 MB | Quality-critical reference |
| Float+OpenBLAS | `make server-blas` | 15 tok/s | ✅ Best | 108 MB | Matches PyTorch speed |
| BWN 1-bit | `--bwn` | 36 tok/s | ❌ Garbled | 3.5 MB | Max compression, research |
| BNN | `--binary` | 47 tok/s | ❌ Garbled | 3.5 MB | Max speed, research |
| Int8 | `--int8` | 22 tok/s | ⚠️ Low | 3.5 MB | Speed over quality |
| Mixed-8 | `--mixed-int8 8` | 15 tok/s | ⚠️ Medium | 3.5 MB | Balanced binary |

## Key optimizations

### Q8 per-row quantization (DEFAULT) — mistral.rs-inspired SIMD

The default inference mode. Inspired by llama.cpp Q8_0 and mistral.rs v0.9.0:

- **Per-row int8 quantization**: each output row has its own scale (correlation 0.99994 vs float)
- **AVX2 VPMADDUBSW + VPMADDWD**: 32 uint8 × 32 int8 → 16 int16 → 8 int32 in 2 instructions
- **8-output parallel** (mistral.rs register blocking): 8 accumulators in YMM registers, x loaded once and shared
- **Pre-computed w_sums**: zero-point offset computed at load time, not per-call (eliminates per-position overhead)
- **Transposed weight layout** [out, in] for contiguous per-output access

Result: **34 tok/s** (2.6x PyTorch float32, 1.72x PyTorch int8, 1.48x llama.cpp Q8_0), coherent quality, 27MB memory.

### Binary Weight Network (BWN) — wbits + int16 madd

- Compact 1-bit wbits (3.5MB total, 30x less than float)
- AVX2 `_mm256_madd_epi16` (int16×int16→int32, llama.cpp-style)
- 36 tok/s but **garbled quality** (1-bit loses too much info)

### OpenBLAS integration (float mode)

- `cblas_sgemv` for large matmul, hybrid SIMD dispatch for small
- 15 tok/s (matches PyTorch 14.2 tok/s on same hardware)

### Logic-guided binarization (CORE/BINARY/PRUNE)

- Per-output norm-based mask: top 20% CORE (float), bottom 10% PRUNE (zero), middle 70% BINARY
- `--logic` flag for gpt2_train, GT3L/GB2L2 binary format for server

### Real causal attention

- Multi-head QK softmax + KV cache (replaces degenerate V-copy)
- Head-major KV cache layout for cache utilization
- SIMD vectorized (v8f wrapper, AVX2/NEON auto)

### Level-2 fusion: sampling filter (.lal logic constraints)

- `ban_last` / `ban_repeat(n)` / `ban_token(id)` — token-id level
- `ban_relation("trigger", "allow1", ...)` — concept-level
- Compiled via `lalc --filter-only`, linked into server sampling path

### Training stack

- Adam optimizer (8x SIMD, bias correction, weight clipping)
- lr schedule: linear warmup + cosine decay
- Knowledge distillation: teacher float → student binary (CE+KL)
- TWN ternary weights {-1, 0, +1} (BitNet b1.58-inspired)
- STE (Straight-Through Estimator) with logic-guided binarization

## Adding a new model (e.g. BERT)

1. Create `models/bert.c` — include `runtime/lal_runtime.h`
2. Define model config (dim, layers, heads, vocab)
3. Call `bin_forward()` / `bin_backward()` from lal_runtime
4. Build: `gcc -O3 -o bert models/bert.c runtime/lal_runtime.c -lm`

The runtime is model-agnostic. GPT-2 is just one example.

## Results

| Task | Metric | Value |
|---|---|---|
| GPT-2 training (LAL native, binary) | 10000 steps, pure C | 35s, 3.4ms/step, loss 5.5→0.3 |
| vs PyTorch training | speedup | 250x |
| Weight compression | binary (sign+alpha) | 44x (498MB → 11.3MB) |
| **GPT-2 Q8 (default)** | inference, x86_64 2-core | **34 tok/s, coherent, 27MB** |
| vs PyTorch float32 | speedup | **2.6x** (34 vs 13 tok/s) |
| vs PyTorch int8 | speedup + quality | **1.72x faster, better quality** |
| vs llama.cpp Q8_0 | isolated matmul | **1.48x faster** |
| GPT-2 BWN (1-bit) | binary weights | 36 tok/s (garbled, 3.5MB) |
| GPT-2 BNN (full binary) | XNOR+popcount | 47 tok/s (garbled) |
| GPT-2 float+OpenBLAS | cblas_sgemv | 15 tok/s (matches PyTorch) |
| GPT-2 on ARMv7 Android | Cortex-A7 @ 1.4GHz | 9.7 s/token (0.1 tok/s) |

### Speed-quality tradeoff (amd64 2-core, GPT-2 base, 20 tokens)

```
Quality
  Best │  float (7 tok/s) ─────────────────────────
       │  float+BLAS (15 tok/s) ───────────────────
  Good │  ★ Q8 default (34 tok/s) ─────────────────  ← BEST
       │  mixed-8 (15 tok/s) ──────────────────────
  Low  │  int8 (22 tok/s) ─────────────────────────
  Poor │  BWN (36 tok/s) ── BNN (47 tok/s) ────────
       └────────────────────────────────────────────
       0    10    20    30    40    50    60 tok/s
```

### Benchmark vs competitors (same hardware, GPT-2 base, 20 tokens)

| Implementation | Speed | Quality | Memory |
|---|---|---|---|
| PyTorch float32 | 13.2 tok/s | ✅ Coherent | 498 MB |
| PyTorch int8 (dynamic) | 20.0 tok/s | ❌ Degraded (repetitive) | 124 MB |
| llama.cpp Q8_0 (isolated matmul) | 0.131 ms | — | 576 KB/matrix |
| **LAL Q8 (default)** | **34 tok/s** | **✅ Coherent** | **27 MB** |
| LAL Q8 (isolated matmul) | **0.089 ms** | — | 579 KB/matrix |

## Testing

```bash
# Unit tests
gcc -O2 -o tests/test_bwn_patch tests/test_bwn_patch.c runtime/lal_runtime.c -lm
./tests/test_bwn_patch          # BWN forward/backward correctness

gcc -O2 -o tests/test_attention tests/test_attention.c runtime/lal_runtime.c -lm
./tests/test_attention          # Causal attention + KV cache

gcc -O3 -mavx2 -mfma -o tests/test_q8_simd tests/test_q8_simd.c -lm
./tests/test_q8_simd            # Q8 SIMD: 24x faster than scalar

gcc -O3 -mavx2 -mfma -o tests/test_q8_mistral tests/test_q8_mistral_style.c -lm
./tests/test_q8_mistral         # Q8 mistral.rs style: 4x faster than single-output

gcc -O3 -mavx2 -mfma -o tests/bench_llama_q8 tests/bench_llama_style_q8.c -lm
./tests/bench_llama_q8          # LAL Q8 vs llama.cpp Q8_0 head-to-head

# Quality baseline
bash scripts/quality_test.sh    # multi-config generation comparison

# PyTorch baseline (for speed comparison)
python3 bench_pytorch.py        # float32 benchmark
python3 bench_pytorch_int8.py   # int8 dynamic quantization benchmark
```

## License

MIT
