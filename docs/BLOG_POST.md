# I wrote a CPU LLM inference engine in 2000 lines of C — it reaches 84% of llama.cpp's speed

## The story of optimizing Q4_K from 0.1 to 1.4 tok/s on a 2-vCPU Xeon

---

I spent the last few weeks building [LAL](https://github.com/samaidev/lal) — a minimal CPU LLM inference engine. Not to replace llama.cpp (that's impossible for one person), but to understand: **how fast can a single human write a Q4_K kernel, starting from scratch?**

The answer: 84% of llama.cpp's speed, in about 2000 lines of C, with zero dependencies.

This is the story of the 14× optimization journey — from 0.1 tok/s to 1.4 tok/s — and the specific insights that got me there.

---

## The setup

- **Hardware**: 2-vCPU Intel Xeon Platinum @ 2.5GHz, 15GB RAM, no GPU
- **Model**: Qwen2.5-7B-Instruct
- **Format**: Q4_K (4-bit quantization with per-sub-block scales, same as llama.cpp)
- **Baseline**: llama.cpp Q4_K_M at 1.67 tok/s

The goal was simple: write a Q4_K matmul kernel from scratch, optimize it, and see how close I could get.

---

## The journey: 0.1 → 1.4 tok/s

### 0.1 tok/s — Scalar baseline

Just a naive C loop: `for each weight, multiply and accumulate`. No SIMD, no parallelism. This is the "reference implementation" — correct but useless.

### 0.3 tok/s — SIMD maddubs

The first real optimization: use `_mm_maddubs_epi16` to do 16 Q4×Q8 multiply-adds in one instruction. This is the workhorse of all Q4 quantized inference.

**Key insight**: Q4 values are unsigned (0-15), Q8 values are signed (-128 to 127). `maddubs` takes unsigned×signed in one instruction — perfect for Q4×Q8.

### 0.4 tok/s — Precompute scales

Q4_K has 6-bit scales packed into 12 bytes. I was unpacking them per-element in the hot loop. Moving the unpack to a precompute phase saved 30% of compute.

### 0.5 tok/s — 4-row parallel

Instead of computing one output row at a time, process 4 rows in parallel. This gives the compiler 4 independent accumulators for instruction-level parallelism (ILP).

### 0.6 tok/s — 8-row parallel + int16 scales

Pushed to 8 parallel rows (matching AVX2's 16 YMM registers). Also switched from float scales to int16 scales, enabling `_mm_madd_epi16` (multiply two int16 vectors and horizontal-add to int32) instead of separate multiply + add.

### 0.8 tok/s — ADJACENT packing (the key insight)

This was the breakthrough. I had been using INTERLEAVED packing: `byte[i] = q[i] | (q[i+16]<<4)`. It seemed natural — split each sub-block into first half and second half.

**The problem**: With INTERLEAVED, 256-bit maddubs requires `vpermute2x128` to rearrange xq. The permute is 3-cycle latency, port-5 only. **256-bit was slower than 128-bit.**

**The insight**: Switch to ADJACENT packing: `byte[i] = q[2i] | (q[2i+1]<<4)` (adjacent pairs, like llama.cpp). Now after splitting nibbles:
- `q4l` (32 bytes) = `[sub_a_even(16), sub_b_even(16)]`
- `xq_even` (32 bytes) = `[sub_a_even_xq(16), sub_b_even_xq(16)]`
- These are **naturally aligned** — `maddubs(q4l, xq_even)` gives correct results with **zero permute instructions**.

**Lesson**: Packing format determines SIMD efficiency. Design packing around the data path, not around "what looks natural".

### 0.9 tok/s — True 256-bit maddubs

With ADJACENT packing, 256-bit maddubs finally worked without permutes. The hot loop went from 4× 128-bit maddubs to 2× 256-bit maddubs — 50% fewer instructions, no extracti128 overhead.

### 1.0 tok/s — Micro-optimizations

- **Unrolled `unpack_scales_6bit`**: Better instruction-level parallelism
- **`extract_epi32` instead of `hadd_epi32`**: `hadd` is 2-cycle latency, port-5 only. `extract` is 1-cycle, uses ports 0/1/5 in parallel. 3× faster for min correction.

### 1.2 tok/s — Prefetch + madvise

- `_mm_prefetch(qs+144, T0)`: Prefetch the next superblock while processing the current one. Hides DRAM latency.
- `madvise(MADV_HUGEPAGE)`: Use 2MB transparent huge pages for the 7.5GB weight file. Reduces TLB entries from 1.9 million to 3,750.

### 1.4 tok/s — cvtepi8_epi16 + march=native

- `_mm_cvtepi8_epi16` instead of `_mm_set_epi16(8 args)`: 2 instructions instead of 7+ for building the min correction vector.
- `-march=native`: Compiler tuning for the specific CPU.

---

## What didn't work (the failures are more interesting)

### AVX512_BW 512-bit maddubs — 33% SLOWER

The hardware has AVX512_BW. I wrote a 512-bit maddubs kernel. It was correct but slower.

**Why**: Skylake-X downclocks ~12% for heavy 512-bit operations (2.5GHz → 2.2GHz). The 2× instruction throughput was negated by lower clock speed.

**Lesson**: Wider SIMD ≠ faster. Always check for downclocking.

### 256-bit maddubs with INTERLEAVED packing — 33% SLOWER

Before switching to ADJACENT, I tried 256-bit with INTERLEAVED. The `vpermute2x128` overhead killed performance.

**Lesson**: You can't just "widen the SIMD". The data layout must match.

### Quantizing embed/lm_head to Q4_K — NO SPEEDUP

I quantized the embedding table and LM head from F32 to Q4_K. Weight file shrank from 7.5GB to 4.0GB (smaller than llama.cpp's 4.36GB!). But speed didn't change.

**Why**: 
- embed_tokens is read once per token (14KB memcpy) — not a bottleneck
- lm_head was already int8 (520MB) — quantizing to Q4_K (290MB) saves only 6% of total bandwidth, but the Q4_K kernel is slower than the int8 sign-trick kernel

**Lesson**: Bandwidth savings only matter if the kernel is bandwidth-bound. LM head is compute-bound.

### Fused Binary Logic MLP — 1.8 tok/s but garbled

My most ambitious experiment: quantize MLP weights to 1 bit (sign only), fuse SiLU+elementwise multiply into in-register computation, and do XNOR+popcount for the down projection.

**Result**: 1.8 tok/s (faster than llama.cpp!) but output was garbage. Binary weights lose too much information.

**Lesson**: You can't beat information theory. 1-bit weights need STE training to work, not just sign extraction.

---

## The final numbers

| Engine | Format | tok/s (2 threads) | Weight size | Code lines |
|--------|--------|-------------------|-------------|------------|
| llama.cpp | Q4_K_M | 1.67 | 4.36 GB | 100,000+ |
| **LAL** | **Q4_K** | **1.4** | **7.5 GB** | **~2,000** |
| PyTorch (transformers) | F32 | 0.3 | 28 GB | millions |

**LAL is 84% of llama.cpp's speed, with 2% of the code.**

---

## What I learned

1. **SIMD optimization is a search problem, not a design problem.** I tried ~15 optimizations. About half worked. You can't predict which ones without benchmarking.

2. **The biggest wins come from data layout, not instruction choice.** ADJACENT vs INTERLEAVED packing was worth 2×. All the instruction-level micro-optimizations together were worth maybe 1.3×.

3. **Bandwidth is usually NOT the bottleneck people think it is.** I expected quantizing embed/lm_head to help (less bandwidth). It didn't, because those weren't the bottleneck. The bottleneck was the MLP matmul kernels being compute-bound at 5-7 GB/s out of 11 GB/s peak DRAM.

4. **AVX512 is a trap on Skylake-X.** The downclocking negates the wider SIMD. AVX2 is the safe default.

5. **The 16% gap to llama.cpp is structural.** They have imatrix-guided quantization (allows Q3_K/Q2_K), AVX512-VNNI support, and 5 years of kernel tuning. These aren't fixable by "one more optimization."

---

## Why open-source this?

LAL is not going to replace llama.cpp. That was never the goal.

The goal was to build something **understandable**. llama.cpp has 100,000+ lines and a ggml abstraction layer. Nobody can fully understand it. LAL has 2,000 lines — a single person can read it all in an afternoon.

If you want to learn how LLM inference engines work — how Q4_K packing works, how to write a SIMD matmul kernel, how attention and RoPE work — LAL is a starting point that doesn't require a PhD in ggml.

**The code is here**: https://github.com/samaidev/lal

**The docs explain why every decision was made**: [ARCHITECTURE.md](https://github.com/samaidev/lal/blob/main/docs/ARCHITECTURE.md), [PITFALLS.md](https://github.com/samaidev/lal/blob/main/docs/PITFALLS.md)

Fork it, learn from it, ignore it. That's fine.

---

*If you found this interesting, the [GitHub repo](https://github.com/samaidev/lal) has the full code, benchmarks, and documentation. Star it if you want to follow — though I make no promises about continued maintenance.*
