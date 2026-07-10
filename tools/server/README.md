# LAL GPT-2 Server

A self-contained HTTP server that runs GPT-2 inference in pure C, with an embedded HTML frontend.

## Why this exists

The LAL project ships two inference paths:

| Path | File | Mode | Speed | Use case |
|------|------|------|-------|----------|
| Training / fine-tune | `models/gpt2.c` | Binary (XNOR+popcount) | ~3 ms/step | Train GPT-2 in 36s |
| Web demo | `tools/server/gpt2_server.c` | Float (AVX2 SIMD) | ~50 ms/token | Browser frontend |

The binary path is what gets the "几十倍提速" — it binarizes weights and uses XNOR+popcount for 64× reduced compute. But binarization needs ~3 GB peak memory (4× weights for transposed bits + alpha), which doesn't fit alongside the 498 MB weight file in our 4 GB sandbox.

So the server uses the **float path** with **AVX2+FMA SIMD** to recover as much speed as possible without binarizing. The previous server was a naive scalar loop (490 ms/token); this version is ~10× faster.

## Optimizations vs. the original `models/gpt2_server.c`

1. **AVX2+FMA SIMD matmul** — 8 outputs/cycle via `_mm256_fmadd_ps`. The original was scalar.
2. **AVX2 SIMD LM head** — `logits = wte @ x` for all 50,257 vocab entries in batched 8-wide form. The original did a scalar `for v in 50257: for i in 768: ...` loop — the single biggest hotspot.
3. **Hash-table tokenizer** — FNV-1a + open-addressing hash. The original scanned all 50,257 vocab tokens at each position (O(N × 50257)). Now O(N × 16) longest-match.
4. **Removed per-step clipping** — `clip_array` was a no-op for inference (residuals never blow up without training noise).
5. **Greedy longest-match in 16-char window** — replaces the O(vocab) scan with a hash probe.

## Build & run

```bash
# from repo root — using the Makefile (auto-detects OpenBLAS)
make server

# or, build directly
gcc -O3 -mavx2 -mfma -o prebuilt/gpt2_server \
    tools/server/gpt2_server.c runtime/lal_runtime.c -lm -lpthread

./prebuilt/gpt2_server          # serves on :8080
./prebuilt/gpt2_server 9000     # custom port
```

Open `http://localhost:8080`.

### Compatibility notes

- **GLIBC**: the prebuilt binary in `prebuilt/gpt2_server` requires GLIBC ≥ 2.38.
  On older systems (e.g. Debian 12 ships 2.36), rebuild from source — see the
  top-level [README troubleshooting section](../../README.md#troubleshooting).
- **Port conflicts**: if port 8080 is taken (common in cloud containers that
  run other agents on 8080/8090), pass a custom port as the first arg.
- **Weights**: the server loads `prebuilt/gpt2_weights.bin` (474 MB) at startup.
  This file is **not** in the repo — see [Setup](../../README.md#setup) for
  download instructions.
- **OpenBLAS**: optional but recommended (~2-3x speedup on float ops). The
  Makefile auto-detects it via `pkg-config`. Without it, the server uses
  hand-written AVX2 SIMD.

## API

> The server listens on `localhost:8080` by default (override with the first
> CLI argument). All endpoints accept JSON and return JSON.

| Route | Method | Body | Response |
|-------|--------|------|----------|
| `/` | GET | — | HTML frontend |
| `/generate` | POST | `{"prompt": "...", "n_tokens": 20}` | `{"text": "...", "time": "0.5", "n_tokens": 20, "tokens_per_sec": "40.0"}` |

Example:
```bash
curl -X POST http://localhost:8080/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Hello, how are","n_tokens":20}'
```

## Architecture note (quality vs. speed)

The server implements **real causal self-attention** with a multi-head QK softmax
and a head-major KV cache (72 MB for 1024 ctx × 12 layers × 2 × 768). This
replaced an earlier `attn_out = V` simplification that caused repetition loops
on long prompts. Real attention is enabled by default; the legacy V-copy path
is kept behind `--vcopy` for benchmark comparison only.

Residual repetition under `temperature=0` is now intrinsic to GPT-2 124M (not
a runtime artifact) — see [Known limitations](../../README.md#known-limitations).

## Files

```
tools/server/
├── gpt2_server.c    # HTTP server + SIMD inference (single file, ~600 lines)
├── frontend.html    # Browser UI (loaded at runtime, falls back to embedded)
└── README.md        # this file
```
