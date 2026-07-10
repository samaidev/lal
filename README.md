# LAL — Logic-Assembly Language

A high-performance CPU LLM inference runtime with Q8 per-row quantization (mistral.rs-inspired SIMD) as default.

**42 tok/s on amd64, 38 tok/s on arm64, 2.7x faster than PyTorch, coherent quality, 27MB memory.**

## Quick start

```bash
# Build (auto-detects OpenBLAS on x86, SDOT on arm64)
make server

# Run (optimal config is DEFAULT — no flags needed)
./prebuilt/gpt2_server                        # Q8: 42 tok/s, coherent, 27MB

# Optional modes (lower performance, kept for compatibility)
./prebuilt/gpt2_server --q8-mixed             # Layer 0+11 float: better quality, 21 tok/s
./prebuilt/gpt2_server --no-q8                # float: 6 tok/s, best quality, 108MB
./prebuilt/gpt2_server --q4                   # Q4: 49 tok/s, slight quality loss, 14MB
```

## Default optimizations (no flags needed)

The default mode automatically enables all optimal optimizations:

| Optimization | Description | Benefit |
|---|---|---|
| **Q8 per-row quantization** | 8-bit per-row scale (correlation 0.99994) | 27MB memory, 6x faster than float |
| **Real causal attention** | Multi-head QK softmax + KV cache | Coherent text (no repetition loops) |
| **int8 LM head** | VPMADDUBSW (same as Q8 matmul) + float rerank | 4x bandwidth reduction on bottleneck |
| **Multi-thread LM head** | 4 threads (arm64) / 2 threads (amd64) | 1.5x LM head speedup |
| **Multi-thread Q8 matmul** | Auto: 4+ cores, large matrices only | Parallel mlp_fc/proj on arm64 |
| **mistral.rs 8-output parallel** | 8 accumulators in registers, x shared | 4.12x vs single-output |
| **Pre-computed w_sums** | Zero-point offset at load time | Eliminates per-call overhead |
| **2-pass fused layer norm** | mean+var in one pass | 33% fewer memory reads |
| **Head-major KV cache** | Sequential memory access for attention | Better L1/L2 utilization |

## Inference modes

| Mode | Flag | Speed (amd64) | Speed (arm64) | Quality | Memory |
|------|------|---------------|---------------|---------|--------|
| **Q8 (DEFAULT)** | (none) | **42 tok/s** | **38 tok/s** | ✅ Coherent | 27 MB |
| Q8-mixed | `--q8-mixed` | 21 tok/s | 13 tok/s | ✅ Better | 35 MB |
| Q4 | `--q4` | 49 tok/s | 35 tok/s | ⚠️ Slight loss | 14 MB |
| Float | `--no-q8` | 6 tok/s | 6 tok/s | ✅ Best | 108 MB |

### Deprecated modes (kept in code for reference)
- `--bwn` / `--binary` / `--rsign`: 1-bit quantization, garbled output
- `--int8` / `--mixed-int8`: activation quantization, degraded quality
- `--turboquant`: KV cache int8 quantization, output garbled on long context
- `--vcopy`: degenerate V-copy attention (replaced by real causal attention)

## Key optimizations

### Q8 per-row quantization (DEFAULT)
- **8-bit per-row scale** (correlation 0.99994 vs float, near-lossless)
- **AVX2 VPMADDUBSW** (x86): uint8 × int8 → int16 → int32 (maddubs + madd)
- **NEON SDOT** (arm64): vdotq_s32, 16 int8 × 16 int8 → 4 int32 per instruction
- **NEON vmull_s8** (ARMv7): vpadalq_s16 + manual hadd (vpadd_s32)
- **8-output parallel** (mistral.rs register blocking): 8 accumulators in registers
- **Pre-computed w_sums**: zero-point offset at load time
- **Multi-threaded**: auto-parallel on 4+ cores for large matmuls

### int8 LM head (auto-enabled)
- **VPMADDUBSW kernel** (same as Q8 matmul): uint8 × int8 → int16
- **Two-pass design**: int8 candidate selection + float rerank (top-512)
- **Multi-threaded**: 4 threads (arm64) / 2 threads (amd64)
- **Pre-computed w_sums**: zero-point removal at load time
- 4x bandwidth reduction (154MB float → 37MB int8)

### Real causal attention
- Multi-head QK softmax + KV cache (replaces degenerate V-copy)
- Head-major KV cache layout (sequential memory access)
- SIMD vectorized (v8f wrapper, AVX2/NEON auto)
- dflash mode: Q register blocking (q0-q7 preloaded), 3-pass fused softmax

### 2-pass fused layer norm
- `var = E[x²] - (E[x])²` (saves one full pass over 768 floats)
- Fuse normalize: `out = fma(xn, w, b)` in one instruction
- 24 layer_norms per token, measurable speedup

## Benchmark

### End-to-end (GPT-2 base, 100 tokens)

| Implementation | amd64 2-core | arm64 8-core | ARMv7 3-core | Quality | Memory |
|---|---|---|---|---|---|
| PyTorch float32 | 16.3 tok/s | — | — | Coherent | 498 MB |
| PyTorch int8 | 20.7 tok/s | — | — | Degraded | 124 MB |
| **LAL Q8 (default)** | **42 tok/s** | **38 tok/s** | **2.2 tok/s** | **Coherent** | **27 MB** |
| LAL Q8-mixed | 21 tok/s | 13 tok/s | — | Better | 35 MB |
| LAL Q4 | 49 tok/s | 35 tok/s | — | Slight loss | 14 MB |
| LAL float | 6 tok/s | 6 tok/s | — | Best | 108 MB |

### Component profiling (amd64, per token)

| Component | Time | % | Status |
|---|---|---|---|
| Q8 matmul (4×12 layers) | 4.6 ms | 19% | Near bandwidth limit (17-22 GB/s) |
| LM head int8 | 3.7 ms | 16% | maddubs optimized (1.5x) |
| Attention + KV cache | ~1 ms | 4% | v8f SIMD |
| Layer norm (×24) | ~0.5 ms | 2% | 2-pass fused |
| Sampling (top-k=40) | ~0.5 ms | 2% | Cache-friendly linear scan |
| Other (prefill, overhead) | ~13 ms | 57% | — |
| **Total** | **23.8 ms** | 100% | **42 tok/s** |

### Isolated matmul (768×768, 500 trials)

| Implementation | amd64 (AVX2) | arm64 (SDOT) |
|---|---|---|
| LAL Q8 8-output | 0.023 ms | 0.025 ms |
| llama.cpp Q8_0 | 0.131 ms | — |
| **Speedup vs llama.cpp** | **1.48x** | — |

### Speed-quality tradeoff

```
Quality
  Best │  float (6 tok/s) ───────────────
  Better│  Q8-mixed (21 tok/s) ──────────
  Good │  ★ Q8 default (42 tok/s) ──────  ← BEST BALANCE
       │  Q4 (49 tok/s) ─────────────────
       └──────────────────────────────────
       0    10    20    30    40    50 tok/s
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

# ARMv7 (with NEON, important: -march=armv7-a required)
gcc -O3 -march=armv7-a -mfpu=neon -mfloat-abi=softfp -I. \
    -Wno-unused-function -Wno-unused-variable \
    tools/server/gpt2_server.c runtime/lal_runtime.c \
    -lm -lpthread -o gpt2_server

# Training (no PyTorch, pure C)
make train
./prebuilt/gpt2_train 10000 0.05 --ste --logic --real-attention
```

## API

```bash
# Generate text
curl -X POST http://localhost:18080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","n_tokens":50}'

# With sampling parameters
curl -X POST http://localhost:18080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","n_tokens":50,"temperature":0.5,"top_k":10,"rep_penalty":1.15}'

# Response
{"text":"Once upon a time...","time":"1.2","n_tokens":50,"tokens_per_sec":"42.0"}
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

## Recent fixes

- **Q8 NEON zero-point fix**: vsubq_s8 removes zero-point (was garbled on ARM)
- **ARMv7 NEON path**: added vmull_s8 + manual hadd (was using AVX2 stubs = zeros)
- **LM head multi-thread crash fix**: `n_embd` field was never set in job struct
- **LM head int8 maddubs**: VPMADDUBSW (uint8×int8) instead of cvtepi8+madd
- **Layer norm 2-pass fused**: mean+var in one pass
- **Multi-thread Q8 matmul**: auto-parallel on 4+ cores

## License

MIT
