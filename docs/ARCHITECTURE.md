# Architecture — Why LAL is written this way

This document explains the **design decisions** behind LAL's kernels. Not to brag, but so you don't repeat our mistakes.

If you're modifying a kernel and wondering "why didn't they just do X?", the answer is probably here.

---

## The 14× optimization journey

LAL's Q4_K kernel went from 0.1 tok/s to 1.4 tok/s over several iterations. Each step had a specific insight:

| Version | tok/s | What changed | Key insight |
|---------|-------|-------------|-------------|
| 0.1 | scalar baseline | — | — |
| 0.3 | SIMD maddubs | `_mm_maddubs_epi16` | Q4×Q8 in one instruction |
| 0.4 | precompute scales | unpack once per block | Scale setup was 30% of compute |
| 0.5 | 4-row parallel | process 4 outputs at once | ILP hides instruction latency |
| 0.6 | int16 scale + bsums | madd instead of mul+add | 8-row parallel + min correction |
| 0.8 | ADJACENT packing | `byte[i] = q[2i] \| (q[2i+1]<<4)` | Enables true 256-bit maddubs |
| 0.9 | 256-bit maddubs | no vpermute needed | ADJACENT = data already aligned |
| 1.0 | unrolled unpack + extract_epi32 | eliminate hadd | hadd is port-5 only (2-cycle) |
| 1.2 | prefetch + madvise | `_mm_prefetch(qs+144, T0)` | Hide DRAM latency |
| 1.4 | cvtepi8_epi16 + march=native | 2 instr vs 7+ for set_epi16 | Compiler tuning |

---

## Key design decisions

### 1. Why ADJACENT packing, not INTERLEAVED?

**Background:** Q4_K packs 32 4-bit values into 16 bytes. There are two ways to arrange them:

- **INTERLEAVED:** `byte[sub*16+i] = q[sub*32+i] | (q[sub*32+i+16] << 4)` — first half and second half of sub-block
- **ADJACENT:** `byte[sub*16+i] = q[sub*32+2i] | (q[sub*32+2i+1] << 4)` — adjacent pairs (llama.cpp style)

**We started with INTERLEAVED.** It seemed natural — split each sub-block into "low half" and "high half".

**The problem:** With INTERLEAVED, 256-bit maddubs requires `vpermute2x128` to rearrange xq for correct pairing. The permute adds 3-cycle latency and port-5 pressure. **256-bit maddubs was SLOWER than 128-bit.**

**The insight:** With ADJACENT packing, after splitting nibbles:
- `q4l` (32 bytes) = `[sub_a_even(16), sub_b_even(16)]`
- `xq_even` (32 bytes) = `[sub_a_even_xq(16), sub_b_even_xq(16)]`
- These are **naturally aligned** — `maddubs(q4l, xq_even)` gives correct results directly!

**Result:** Switching to ADJACENT + pre-arranging xq into even/odd halves enabled true 256-bit maddubs with **zero permute instructions**. 50% fewer instructions in the hot loop.

**Lesson:** Packing format determines what SIMD operations are efficient. Always design packing around the SIMD data path, not around "what looks natural".

---

### 2. Why lm_head uses int8, not Q4_K?

**The experiment:** We quantized lm_head to Q4_K (290MB) instead of int8 (520MB). Weight file shrank from 7.5GB to 4.0GB. We expected ~6% speedup from less bandwidth.

**The result:** No speedup. Actually slightly slower.

**Why:**
1. **lm_head is only 14% of per-token bandwidth** (520MB / 3.8GB total). Saving 230MB = 6% total. Negligible.
2. **int8 kernel uses sign-trick** (no xq pre-arrange needed). Q4_K kernel recomputes `xq_arr` per call — expensive for 152K rows.
3. **Double quantization (Q4_K → F32 → int8) loses accuracy.** Worse quality, no speed gain.

**Lesson:** Bandwidth savings only matter if the kernel is bandwidth-bound. lm_head is compute-bound (152K rows × 3584 dot products). The int8 sign-trick kernel is more compute-efficient than Q4_K for this shape.

---

### 3. Why 8-row parallel, not 4 or 16?

**4-row:** Not enough ILP to hide instruction latency. ACCUMULATORS underutilized.

**8-row:** Sweet spot. 8 independent accumulators (acc0-acc7), each updated per superblock. Compiler can schedule them in parallel. Register pressure is manageable (8 × `__m256` = 16 YMM registers, leaves 8 for temporals).

**16-row:** Too much register pressure. Compiler spills to stack, killing performance. Also, 16 rows × 144 bytes = 2304 bytes per superblock, exceeding L1 cache associativity.

**Lesson:** Parallel row count = min(ILP need, register budget, cache friendliness). 8 is the sweet spot for AVX2.

---

### 4. Why AVX2, not AVX512?

**We tested AVX512_BW** (512-bit maddubs). The kernel was correct. But **slower** (0.6 tok/s vs 0.9 tok/s).

**Why:** Skylake-X processors **downclock ~12%** for 512-bit operations (2.5GHz → 2.2GHz). The 2× instruction throughput advantage is negated by lower clock + same memory bandwidth bottleneck.

**When AVX512 wins:** On Ice Lake+ (no downclocking) or with AVX512-VNNI (`vpdpbusd` — single-instruction Q4×Q8). Our hardware has neither.

**Lesson:** Wider SIMD ≠ faster. Check for downclocking and port pressure. AVX2 is the safe default for 2017-2020 Xeons.

---

### 5. Why prefetch 1 superblock ahead, not 2 or 3?

**Tested distances:**
- 0 (no prefetch): 5.8 GB/s on gate matmul
- 1 ahead: **6.0 GB/s** ← best
- 2 ahead: 4.6 GB/s (worse!)
- 3 ahead: 3.2 GB/s (much worse)

**Why:** DRAM latency is ~100ns ≈ 250 cycles at 2.5GHz. One superblock takes ~15µs to process. So data prefetched 2+ superblocks ahead sits in L2/L1 for 30+ µs, risking eviction. 1 ahead matches the latency window perfectly.

**Lesson:** Prefetch distance must match latency ÷ processing time. Too far = cache eviction. Too near = no overlap.

---

### 6. Why `extract_epi32` instead of `hadd_epi32`?

Min correction needs to sum 4 int32 values from a `__m128i`.

**hadd approach:**
```c
mp = _mm_hadd_epi32(mp, mp);  // 2-cycle latency, port 5 only
mp = _mm_hadd_epi32(mp, mp);  // another 2-cycle
result = _mm_cvtsi128_si32(mp);
// Total: 4 cycles, serial, port-5 bottleneck
```

**extract approach:**
```c
result = _mm_cvtsi128_si32(mp)      // 1 cycle
       + _mm_extract_epi32(mp, 1)   // 1 cycle, parallel
       + _mm_extract_epi32(mp, 2)   // 1 cycle, parallel
       + _mm_extract_epi32(mp, 3);  // 1 cycle, parallel
// Total: 1 cycle latency (all parallel), uses ports 0/1/5
```

**Result:** ~3× faster for min correction.

**Lesson:** `hadd` is almost never the right answer. Use `extract` + scalar add, or `madd` with ones vector.

---

### 7. Why `madvise(MADV_HUGEPAGE)` on the mmap?

The 7.5GB weight file uses 4KB pages = **1.9 million page table entries**. Each TLB miss costs ~20 cycles.

With `MADV_HUGEPAGE`, the kernel uses 2MB transparent huge pages = **3,750 entries** (500× fewer). TLB miss rate drops dramatically.

**Result:** Measurable improvement on large matmuls (gate/up/down), no effect on small ones (q/k/v/o).

**Lesson:** For large mmap'd weight files, always hint huge pages. It's free performance.

---

## What we tried that didn't work

Documenting failures so you don't repeat them:

| Optimization | Expected | Actual | Why it failed |
|-------------|----------|--------|--------------|
| 256-bit maddubs with INTERLEAVED packing | 2× faster | 33% slower | vpermute overhead |
| AVX512_BW 512-bit maddubs | 2× faster | 33% slower | Downclocking |
| shuffle_epi8 for scale broadcast | Fewer instructions | Slower | Port-5 contention |
| Pre-build 4 scale vectors before loop | Better scheduling | Slower | Register pressure |
| Quantize embed_tokens to Q4_K | Less bandwidth | No change | Not a bottleneck |
| Quantize lm_head to Q4_K | Less bandwidth | No change | int8 kernel is faster |
| Batch prefetch 16 lines at loop top | More overlap | Slower | Prefetch instruction overhead |
| prefetch 2+ superblocks ahead | More overlap | Slower | Cache eviction |
| Fused Binary Logic MLP (1-bit) | 8× less bandwidth | 1.8 tok/s but garbled | Binary weights lose too much |
| Sparse Logic Selector MLP | Skip inactive selectors | Quality collapse | Dense weights encode all selectors |

---

## The fundamental bottleneck

At 1.4 tok/s, we're at **84% of llama.cpp**. The remaining 16% gap is:

1. **llama.cpp uses imatrix-guided quantization** — allows Q3_K/Q2_K (smaller weights, less bandwidth). We use simple min-max, can't go below Q4_K without quality collapse.
2. **llama.cpp has AVX512-VNNI support** — `vpdpbusd` is single-instruction Q4×Q8. Our hardware doesn't have it.
3. **llama.cpp's kernel has 5 years of tuning** — we have a few weeks.

These are not fixable by "one more optimization". They're structural advantages of a mature project.

**LAL's value isn't being faster. It's being understandable.**
