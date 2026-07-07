# LAL on ARM / Termux / Android

LAL runs GPT-2 on 32-bit ARMv7 Android via Termux. This is notable because:

- **PyTorch Android requires ARMv8 (64-bit) + 4 GB RAM.** LAL runs on ARMv7 (32-bit) + 1.9 GB RAM.
- The 498 MB GPT-2 weights load into 1.9 GB RAM with ~700 MB to spare for activations.
- The same `gpt2_server.c` source compiles on x86_64 (AVX2) and ARMv7 (NEON) without changes — the portable `v8f` wrapper picks the right intrinsics at compile time.

## Tested hardware

| Device | CPU | RAM | OS | Inference speed |
|--------|-----|-----|----|-----------------|
| Qualcomm APQ8009 (4-core Cortex-A7 @ 1.4 GHz) | ARMv7 + NEON | 1.9 GB | Android + Termux | 9.7 s/token (0.1 tok/s) |

That's ~100× slower than the x86_64 sandbox (96 ms/token), but it **works** — PyTorch cannot run GPT-2 124M on this device at all.

## Why so slow

1. **Cortex-A7 is a low-power 2011-era core** — single-core integer throughput is ~1/10 of a Xeon.
2. **ARMv7 NEON is 4-wide** vs AVX2's 8-wide. The LM head's 50257 × 768 matrix-vector product takes twice as many SIMD instructions.
3. **LM head is memory-bound** — 154 MB of weight reads per token. The tablet's LPDDR3 has ~3 GB/s bandwidth vs Xeon's ~20 GB/s.
4. **No FMA on ARMv7 NEON** — the v8f wrapper falls back to `vmulq + vaddq` (clang may fuse them, but it's not guaranteed).

For a 4-core A15 or A72 class CPU with NEON+FMA, expect 5-10× speedup over A7.

## Setup on Termux

```bash
# 1. Install build tools
pkg install clang make python git curl

# 2. Get the LAL source (use a token URL for private repo)
curl -L -H "Authorization: token <github_token>" \
  -o lal.tar.gz https://github.com/samaidev/lal/archive/refs/heads/main.tar.gz
tar xzf lal.tar.gz && mv lal-main lal && cd lal

# 3. Download GPT-2 weights from the GitHub release
curl -L -H "Authorization: token <github_token>" \
  -H "Accept: application/octet-stream" \
  -o prebuilt/gpt2_weights.bin \
  https://api.github.com/repos/samaidev/lal/releases/assets/<asset_id>
# Repeat for gpt2_tokenizer.bin

# 4. Compile
make server CC=clang

# 5. Run (use launcher.sh to detach from the parent shell)
bash tools/arm/launcher.sh 8080
```

## Verify

```bash
curl -s -X POST http://localhost:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_tokens":5}'
# → {"text":"Hello!!!!!","time":"48.357","n_tokens":5,"tokens_per_sec":"0.1"}
```

## Known issues

- **Termux kills background processes** when the parent shell exits. The `launcher.sh` script uses `setsid + nohup + stdio→/dev/null` to detach properly. Plain `./gpt2_server &` will not survive.
- **No `aligned_alloc` on Android API < 21** — `gpt2_server.c` uses `posix_memalign` instead.
- **`/tmp` doesn't exist on Termux** — use `~/` or `$PREFIX/tmp`. The `gpt2.c` weight path is overridable via `$LAL_WEIGHTS` env var.

## Files

- `tools/arm/launcher.sh` — daemon launcher with proper Termux detachment
