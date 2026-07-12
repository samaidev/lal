# Promotion Materials — for submitting LAL to awesome lists and forums

This file contains ready-to-use text for promoting LAL. Copy-paste as needed.

---

## 1. GitHub PR for awesome lists

### For `xlite-dev/Awesome-LLM-Inference` (5385★)

**PR title**: Add LAL — minimal CPU LLM inference engine (84% of llama.cpp, 2000 lines C)

**PR body**:
```
## What is LAL?

LAL (Logic-Assembly Language) is a minimal CPU LLM inference engine written in ~2000 lines of C with zero external dependencies. It implements Q4_K quantization (same format as llama.cpp) and achieves 84% of llama.cpp's speed on equivalent hardware.

## Why add it?

- **Educational value**: The entire codebase is small enough for one person to fully understand in an afternoon. Unlike llama.cpp (100,000+ lines, ggml abstraction), LAL has no abstraction layers.
- **Real performance**: 1.4 tok/s on Qwen2.5-7B with 2 vCPU Xeon (llama.cpp: 1.67 tok/s)
- **5 quantization formats**: Q8, Q4_0, Q8_0, Q4_0A, Q4_K
- **Hand-tuned AVX2 kernels**: 256-bit maddubs, 8-row parallelism, prefetch optimization
- **Detailed documentation**: ARCHITECTURE.md explains every design decision, PITFALLS.md documents 12 real bugs and their fixes

## Links

- Repo: https://github.com/samaidev/lal
- Architecture: https://github.com/samaidev/lal/blob/main/docs/ARCHITECTURE.md
- Benchmark: https://github.com/samaidev/lal/blob/main/docs/HARDWARE_TEST_REPORT.md

## Category suggestion

"Open Source Projects" or "Inference Frameworks" → "CPU Inference"
```

---

### For `pprp/Awesome-LLM-Quantization` (431★)

**PR title**: Add LAL — Q4_K implementation with detailed optimization docs

**PR body**:
```
## LAL — minimal Q4_K CPU inference engine

LAL implements Q4_K quantization (llama.cpp-compatible format) in ~2000 lines of C. The focus is on **readable, well-documented kernels** — every design decision is explained in [ARCHITECTURE.md](https://github.com/samaidev/lal/blob/main/docs/ARCHITECTURE.md).

### What makes it interesting for quantization research:

- **5 formats implemented**: Q8 (per-row scale), Q4_0, Q8_0, Q4_0A, Q4_K
- **ADJACENT vs INTERLEAVED packing**: Documents why ADJACENT enables 256-bit maddubs without vpermute (with benchmarks)
- **12 documented pitfalls**: uint64 overflow in 6-bit packing, AVX512 downclocking, hadd vs extract, prefetch distance, etc.
- **Q4_K kernel optimization journey**: 0.1 → 1.4 tok/s (14× improvement, each step explained)

### Links

- Repo: https://github.com/samaidev/lal
- Pitfalls: https://github.com/samaidev/lal/blob/main/docs/PITFALLS.md
- Q4_K kernel: https://github.com/samaidev/lal/blob/main/runtime/lal_q4k_kernel.h
```

---

## 2. Forum posts

### V2EX (中文)

**标题**: 我用 2000 行 C 写了个 CPU LLM 推理引擎，达到 llama.cpp 84% 的速度

**正文**:
```
花了几周时间做了个开源项目 LAL (Logic-Assembly Language)：
https://github.com/samaidev/lal

不是要替代 llama.cpp（一个人打不过一个社区），而是想验证：从零手写 Q4_K kernel，能跑到多快？

答案：llama.cpp 84% 的速度，2000 行 C，零依赖。

核心是一个 Q4_K matmul kernel，从 0.1 tok/s 优化到 1.4 tok/s，14 倍提升。每一步都有具体洞察：

1. ADJACENT packing（关键！）：INTERLEAVED 需要 vpermute，ADJACENT 天然对齐，256-bit maddubs 零开销
2. AVX512 反而更慢：Skylake-X 对 512-bit 操作降频 12%
3. hadd_epi32 是陷阱：extract_epi32 + 标量加法快 3 倍（端口压力不同）
4. prefetch 1-ahead 最优：2-ahead 反而更慢（cache 被驱逐）
5. 量化 embed/lm_head 没用：它们不是瓶颈

试过但失败的：
- AVX512_BW 512-bit maddubs（降频抵消优势）
- 256-bit + INTERLEAVED packing（vpermute 开销）
- Fused Binary Logic MLP（1-bit 权重质量太差）
- Sparse Logic Selector（dense 权重编码所有 selector）

详细的优化历程和踩坑记录都在 ARCHITECTURE.md 和 PITFALLS.md 里。

代码完全开源，MIT 协议。适合学习 LLM 推理引擎底层原理。不保证持续维护，欢迎 fork。
```

### 知乎

**标题**: 从 0.1 到 1.4 tok/s：我如何用 2000 行 C 写出 llama.cpp 84% 速度的 LLM 推理引擎

**正文**: (同 V2EX，可适当展开技术细节)

### Hacker News (English)

**Title**: Show HN: I wrote a CPU LLM inference engine in 2000 lines of C (84% of llama.cpp speed)

**Body**:
```
I built LAL (Logic-Assembly Language) — a minimal CPU LLM inference engine that reaches 84% of llama.cpp's speed with 2% of the code.

The core is a hand-tuned Q4_K matmul kernel. I optimized it from 0.1 to 1.4 tok/s over several iterations, each with a specific insight:

- ADJACENT vs INTERLEAVED packing: the layout determines whether 256-bit maddubs needs a vpermute (it shouldn't)
- AVX512_BW 512-bit maddubs was 33% SLOWER due to Skylake-X downclocking
- extract_epi32 beats hadd_epi32 (port pressure)
- Prefetch distance must match DRAM latency ÷ processing time

Full optimization story with benchmarks: https://github.com/samaidev/lal/blob/main/docs/ARCHITECTURE.md

12 documented pitfalls (uint64 overflow, packing traps, etc.): https://github.com/samaidev/lal/blob/main/docs/PITFALLS.md

The 16% gap to llama.cpp is structural (imatrix quantization, AVX512-VNNI, 5 years of tuning). LAL's value isn't being faster — it's being understandable.

Repo: https://github.com/samaidev/lal
```

### Reddit r/MachineLearning

**Title**: [D] I wrote a minimal CPU LLM inference engine (2000 lines C, 84% of llama.cpp speed) — full optimization writeup inside

**Body**: (similar to HN, but add a question to spark discussion)
```
... (same as HN) ...

I'm curious what the community thinks:
1. Is there value in a "minimal reference implementation" alongside production engines like llama.cpp?
2. What's the next bottleneck to tackle — KV cache quantization, or a better quantization scheme?
3. Has anyone else tried AVX512 and hit the downclocking wall?
```

---

## 3. Twitter/X thread

```
1/ I spent weeks building a CPU LLM inference engine from scratch.

Result: 84% of llama.cpp's speed, in 2000 lines of C, with zero dependencies.

Here's the optimization journey — from 0.1 to 1.4 tok/s (14×). 🧵

2/ The setup: 2-vCPU Xeon, no GPU, Qwen2.5-7B, Q4_K quantization.
Baseline: llama.cpp at 1.67 tok/s.
Goal: how fast can ONE person write a Q4_K kernel?

3/ Step 1 (0.1→0.3): SIMD maddubs. Q4×Q8 in one instruction. The workhorse of all Q4 inference.

4/ Step 2 (0.3→0.6): 8-row parallel + int16 scales. 8 independent accumulators = ILP. madd_epi16 instead of mul+add.

5/ Step 3 (0.6→0.9): THE KEY INSIGHT. Switch from INTERLEAVED to ADJACENT packing.

With ADJACENT, 256-bit maddubs needs ZERO permutes. Data is naturally aligned. 50% fewer instructions.

6/ I tried AVX512_BW (512-bit maddubs). It was 33% SLOWER.

Why? Skylake-X downclocks 12% for heavy 512-bit ops. 2× throughput negated by 1.12× lower clock.

Wider SIMD ≠ faster. Always benchmark.

7/ I tried quantizing embed/lm_head to Q4_K. Weight file shrank from 7.5GB to 4.0GB.

Speed didn't change. Because those weren't the bottleneck. Bandwidth savings only matter if the kernel is bandwidth-bound.

8/ The biggest lesson: data layout > instruction choice.

ADJACENT vs INTERLEAVED = 2× speedup.
All instruction micro-opts together = ~1.3×.

9/ The 16% gap to llama.cpp is structural: they have imatrix (allows Q3_K), AVX512-VNNI, 5 years of tuning.

Not fixable by "one more optimization." And that's okay.

10/ LAL is open source: https://github.com/samaidev/lal

- 2000 lines of C
- Zero dependencies
- ARCHITECTURE.md explains every decision
- PITFALLS.md documents 12 real bugs

For learning, forking, or ignoring. That's fine.
```

---

## 4. Short descriptions for link aggregators

### For `awesome-c` type lists
```
LAL — Minimal CPU LLM inference engine in ~2000 lines of C. Zero dependencies, hand-tuned AVX2 SIMD, Q4_K quantization. 84% of llama.cpp speed.
```

### For `awesome-simd` type lists
```
LAL — Real-world AVX2 optimization case study: Q4_K matmul kernel from 0.1 to 1.4 tok/s. Documents 12 SIMD pitfalls including AVX512 downclocking, hadd vs extract, prefetch distance.
```

### For `edge-ai` type lists
```
LAL — Run Qwen2.5-7B on 2-vCPU cloud instances with no GPU. 200KB binary, <100ms cold start, zero dependencies. CPU-only by design.
```
