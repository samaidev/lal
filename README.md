# LAL — Logic-Assembly Language

A logic-native language that compiles logic programs to specialized C with bit-level parallelism (XNOR+popcount), plus a high-performance binary/float inference runtime optimized with OpenBLAS techniques.

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
│   │   ├── gpt2_server.c    # AVX2 SIMD + OpenBLAS + binary inference
│   │   ├── frontend.html    # Browser UI
│   │   └── README.md        # Inference modes (binary vs. float)
│   ├── arm/                 # ARMv7 / Termux / Android deployment
│   ├── export_*.py          # Weight export from PyTorch checkpoints
│   └── verify.py            # End-to-end correctness check
├── tests/                   # Unit tests (BWN, attention, int8, BLAS correctness)
├── scripts/                 # Quality baseline, logic analysis, corpus tools
├── prebuilt/                # Pre-built binaries (open out of the box!)
└── docs/                    # Design docs
```

## Quick start

```bash
# GPT-2 training (no PyTorch, pure C, 3.4ms/step)
./prebuilt/gpt2_train 10000 0.05            # 10000 steps in 35 seconds
./prebuilt/gpt2_train 500 0.002 --ste --logic --real-attention  # logic-guided STE

# GPT-2 web server — multiple inference modes
./prebuilt/gpt2_server                       # float (default), opens http://localhost:8080
./prebuilt/gpt2_server --bwn                 # BWN (27.5 tok/s, 3.5MB weights)
./prebuilt/gpt2_server --binary              # BNN (full binary, XNOR+popcount)
./prebuilt/gpt2_server --int8                # Int8 activation quantization
./prebuilt/gpt2_server --mixed-int8 8        # Mixed precision (recommended)
./prebuilt/gpt2_server --lm-head-int8        # LM head int8 quantization

# OpenBLAS-accelerated float (2x faster than SIMD)
make server-blas    # requires libopenblas-dev
./prebuilt/gpt2_server_blas                  # float with cblas_sgemv

# Compile a LAL program
python3 compiler/lal.py demos/basic/demo.lal classify demo.c
gcc -O3 -o demo demo.c -lm

# Build from source
make train         # builds prebuilt/gpt2_train
make server        # builds prebuilt/gpt2_server
make server-blas   # builds prebuilt/gpt2_server_blas (OpenBLAS)
make demos         # builds demo binaries
```

## Inference modes

| Mode | Flag | Speed | Quality | Use case |
|------|------|-------|---------|----------|
| Float | (default) | 7-8 tok/s | Best (coherent + factual) | Quality-critical, reference baseline |
| Float + OpenBLAS | `--blas`* | 15 tok/s | Best | Match PyTorch speed |
| BWN | `--bwn` | 27.5 tok/s | Good (coherent, not factual) | 1-bit weights, 3.5MB, 30x less mem |
| BNN | `--binary` | 47 tok/s | Poor (garbled) | Max speed, research only |
| Int8 | `--int8` | 22 tok/s | Low (common words only) | Speed over quality |
| Mixed-8 | `--mixed-int8 8` | 15 tok/s | Good (BWN-level) | **Recommended balance** |

*`--blas` via `make server-blas` build with `-DUSE_OPENBLAS`

## Key optimizations

### Binary Weight Network (BWN) — wbits + LUT sign-flip
- **Compact wbits** (1-bit packed, 3.5MB total, 30x less than float)
- **LUT sign-flip**: 256-entry × 8 floats (0.0/-0.0), XOR x to apply ±1 (no multiply)
- **ADD accumulate** (sign=±1 means multiply is just conditional negate)
- Result: BWN 8.5 → 27.5 tok/s (**3.2x speedup**), memory 108MB → 3.5MB (-97%)

### OpenBLAS integration (float mode)
- `cblas_sgemv` for large matmul (c_attn, LM head)
- Hybrid dispatch: BLAS for >1M elements, SIMD for smaller
- Result: float 8 → 15 tok/s (**1.9x**), matches PyTorch 14.2 tok/s

### Logic-guided binarization (CORE/BINARY/PRUNE)
- Per-output norm-based mask: top 20% CORE (float), bottom 10% PRUNE (zero), middle 70% BINARY
- `--logic` flag for gpt2_train, GB2L2 binary format for server
- Preserves core-logic weights at full precision

### Real causal attention
- Multi-head QK softmax + KV cache (replaces degenerate V-copy)
- Head-major KV cache layout for cache utilization
- SIMD vectorized (v8f wrapper, AVX2/NEON auto)

### Level-2 fusion: sampling filter (.lal logic constraints)
- `ban_last` / `ban_repeat(n)` / `ban_token(id)` — token-id level
- `ban_relation("trigger", "allow1", ...)` — concept-level (PHONE's "logic parsing mapping")
- Compiled via `lalc --filter-only`, linked into server sampling path

### Int8 activation quantization
- Per-tensor symmetric quantization, int8 × int8 → int32 accumulate
- `--int8` (pure) or `--mixed-int8 N` (first N layers int8, rest BWN)
- Mixed-8 is the sweet spot: 15 tok/s, BWN-level quality

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
| vs PyTorch | speedup | 250x |
| Weight compression | binary (sign+alpha) | 44x (498MB → 11.3MB) |
| **GPT-2 float (OpenBLAS)** | inference, x86_64 2-core | **15 tok/s** (matches PyTorch 14.2) |
| **GPT-2 BWN (wbits+LUT)** | binary weights, x86_64 2-core | **27.5 tok/s** (3.8x float, 3.5MB) |
| **GPT-2 mixed-8** | mixed int8+BWN, aarch64 | **15 tok/s** (3.6x float) |
| GPT-2 BNN (full binary) | XNOR+popcount | 47 tok/s (garbled quality) |
| GPT-2 on ARMv7 Android tablet | inference, Cortex-A7 @ 1.4GHz | 9.7 s/token (0.1 tok/s) |
| ARM RAM footprint | 1.9 GB total, 498 MB weights | works where PyTorch cannot run at all |

### Speed-quality tradeoff (amd64 2-core, GPT-2 base, 20 tokens)

```
Quality
  Best │  float (7-8 tok/s) ─────────────────────
       │  float+BLAS (15 tok/s) ─────────────────
  Good │  BWN (27.5 tok/s) ──────────────────────
       │  mixed-8 (15 tok/s) ────────────────────
  Low  │  int8 (22 tok/s) ───────────────────────
  Poor │  BNN (47 tok/s) ────────────────────────
       └─────────────────────────────────────────
       0    10    20    30    40    50    60 tok/s
```

## Testing

```bash
# Unit tests
gcc -O2 -o tests/test_bwn_patch tests/test_bwn_patch.c runtime/lal_runtime.c -lm
./tests/test_bwn_patch          # BWN forward/backward correctness

gcc -O2 -o tests/test_attention tests/test_attention.c runtime/lal_runtime.c -lm
./tests/test_attention          # Causal attention + KV cache

gcc -O3 -mavx2 -mfma -o tests/test_bwn_opt tests/test_bwn_optimized.c -lm
./tests/test_bwn_opt            # OpenBLAS-style BWN: 2.28x speedup

gcc -O2 -DUSE_OPENBLAS -o tests/test_blas tests/test_blas_correctness.c -lopenblas -lm
./tests/test_blas               # cblas_sgemv correctness

# Quality baseline
bash scripts/quality_test.sh    # multi-config generation comparison

# PyTorch baseline (for speed comparison)
python3 bench_pytorch.py
```

## License

MIT
