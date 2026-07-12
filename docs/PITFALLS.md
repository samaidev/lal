# Pitfalls — Bugs we hit and how to avoid them

This is the most valuable document in LAL. Every entry below is a real bug that cost us hours. **Read this before modifying any kernel.**

---

## 1. `uint64_t` overflow in 6-bit scale packing

### The bug
Q4_K packs 16 × 6-bit scales into 12 bytes (96 bits total). Our C test quantizer used:
```c
uint64_t bits = 0;
for (int j = 0; j < 16; j++)
    bits |= (combined[j] & 0x3F) << (j * 6);  // BUG: overflows at j=11
```

16 × 6 = 96 bits, but `uint64_t` only holds 64 bits. Scales 11-15 silently corrupted.

### The symptom
Kernel produced garbage on multi-superblock tests, but single-superblock tests passed (because the overflow only affects scales 11+, which are the "mins" for sub-blocks 3-7).

### The fix
```c
__uint128_t bits = 0;  // 128-bit, holds all 96 bits
for (int j = 0; j < 16; j++)
    bits |= ((__uint128_t)(combined[j] & 0x3F)) << (j * 6);
```

### Why Python didn't have this bug
Python uses arbitrary-precision integers. The C port introduced it.

### Lesson
**When packing N-bit fields, always check: N × count ≤ sizeof(integer) × 8.** Use `__uint128_t` for >64-bit packing.

---

## 2. INTERLEAVED vs ADJACENT packing — the 256-bit maddubs trap

### The bug
We tried to upgrade from 128-bit to 256-bit maddubs. It was **33% slower**.

### Why
With INTERLEAVED packing (`byte[i] = q[i] | (q[i+16]<<4)`), the q4 nibbles and xq values aren't aligned for 256-bit maddubs. You need `vpermute2x128` to rearrange xq — 3-cycle latency, port-5 only.

### The fix
Switch to ADJACENT packing (`byte[i] = q[2i] | (q[2i+1]<<4)`). Now q4l and xq_even are naturally aligned:
- `q4l` = [sub_a_even, sub_b_even]
- `xq_even` = [sub_a_even_xq, sub_b_even_xq]
- `maddubs(q4l, xq_even)` → correct, no permute needed

### Lesson
**Packing format determines SIMD efficiency.** Design packing around the data path, not around "what looks natural". See [ARCHITECTURE.md §1](ARCHITECTURE.md#1-why-adjacent-packing-not-interleaved).

---

## 3. AVX512_BW downclocking

### The bug
We had AVX512_BW available. We wrote a 512-bit maddubs kernel. It was correct but **33% slower** than AVX2.

### Why
Skylake-X processors **reduce clock speed** for heavy AVX512 instructions:
- Light AVX512 (no 512-bit ops): no penalty
- Heavy AVX512 (512-bit maddubs, FMA): ~12% downclock (2.5GHz → 2.2GHz)
- The downclock persists for ~1ms after the AVX512 code ends

The 2× instruction throughput was negated by 1.12× lower clock + same memory bandwidth.

### The fix
Stick with AVX2. The AVX512 kernel source is kept in `lal_q4k_kernel_avx512.h` for future Ice Lake+ CPUs (no downclocking).

### How to detect
```bash
cat /proc/cpuinfo | grep flags | tr ' ' '\n' | grep avx512
# If you see avx512f avx512bw but NOT avx512_vnni → Skylake-X, will downclock
```

### Lesson
**Wider SIMD ≠ faster.** Always benchmark. Check for downclocking on Skylake-X / Cascade Lake.

---

## 4. `hadd_epi32` is a trap

### The bug
Min correction needed to sum 4 int32 lanes. We used:
```c
mp = _mm_hadd_epi32(mp, mp);
mp = _mm_hadd_epi32(mp, mp);
result = _mm_cvtsi128_si32(mp);
```

This was ~3× slower than necessary.

### Why
`hadd_epi32` on Haswell/Skylake:
- 2-cycle latency
- Port 5 only (throughput 1)
- Two serial hadds = 4 cycles minimum

### The fix
```c
result = _mm_cvtsi128_si32(mp)      // lane 0
       + _mm_extract_epi32(mp, 1)   // lane 1, parallel
       + _mm_extract_epi32(mp, 2)   // lane 2, parallel
       + _mm_extract_epi32(mp, 3);  // lane 3, parallel
// 1 cycle latency, ports 0/1/5
```

### Lesson
**Never use `hadd` for horizontal sum.** Use `extract` + scalar add, or `madd_epi16` with a ones vector.

---

## 5. `shuffle_epi8` for scale broadcast — slower than `set1_epi16`

### The bug
We tried to replace 8× `_mm_set1_epi16` with 1× `broadcastsi128` + 4× `shuffle_epi8`. It was slower.

### Why
- `vpbroadcastw` (set1_epi16): ports 0/1/5, throughput 0.33-0.5
- `vpshufb` (shuffle_epi8): port 5 only, throughput 1

The shuffle approach concentrates pressure on port 5.

### The fix
Keep `_mm_set1_epi16`. The compiler optimizes it to `vpbroadcastw` which is faster.

### Lesson
**Don't assume "fewer instructions = faster".** Check port pressure. `vpbroadcastw` is a dedicated broadcast instruction — use it.

---

## 6. Prefetch distance matters

### The bug
Prefetching 2+ superblocks ahead was **worse** than 1 ahead.

### Why
- DRAM latency: ~100ns = 250 cycles at 2.5GHz
- One superblock processing time: ~15µs
- Prefetch 2 ahead: data sits in L1/L2 for 30+ µs → cache eviction
- Prefetch 1 ahead: data arrives just in time

### The fix
```c
_mm_prefetch((const char*)(qs+144), _MM_HINT_T0);       // 1 superblock ahead
_mm_prefetch((const char*)(qs+144+64), _MM_HINT_T0);
```

### Lesson
**Prefetch distance = DRAM_latency ÷ processing_time_per_block.** Too far = eviction. Too near = no overlap.

---

## 7. `MADV_HUGEPAGE` only helps for large files

### The bug
Added `madvise(MADV_HUGEPAGE)` — no measurable improvement on small matmuls.

### Why
Small matmuls (q/k/v/o_proj) fit in L2/L3. TLB misses don't matter. Only large matmuls (gate/up/down with 18944 rows) benefit.

### The fix
Keep `madvise` — it's free and helps the big matmuls (67% of layer time).

### Lesson
**Optimize for the bottleneck, but take free wins elsewhere.**

---

## 8. EOS token inflates benchmark numbers

### The bug
We reported "1.3 tok/s" based on `--temp 0.8` runs. The actual speed was 1.1 tok/s.

### Why
With `--temp 0.8`, the model sometimes generates EOS after 12-25 tokens. The output `[*] 25 tokens in 23s (1.1 tok/s)` was misread as "30 tokens in 23s would be 1.3 tok/s". But the model only generated 25 tokens, so real speed was 1.1.

### The fix
Always benchmark with deterministic sampling:
```bash
--temp 0.01 --top-k 1
```
This forces the model to generate exactly N tokens, giving accurate throughput.

### Lesson
**Benchmark methodology matters as much as optimization.** Verify token count matches `--n`.

---

## 9. Wrong dequant formula when changing packing

### The bug
After switching from INTERLEAVED to ADJACENT packing, the unit test's dequant check still used the INTERLEAVED formula. Tests "passed" but with 20% error (should be 9.8%).

### Why
INTERLEAVED dequant: `w[sub*32+i] = ascale * (byte & 0xF) - amin` for i=0..15, then `w[sub*32+i+16]` for high nibble.

ADJACENT dequant: `w[sub*32+2*i] = ascale * (byte & 0xF) - amin`, `w[sub*32+2*i+1] = ascale * (byte>>4) - amin`.

### The fix
Update ALL dequant code when changing packing:
- Kernel dequant (in `unpack_scales_6bit` callers)
- Test quantizer
- Test dequant checker
- Python converter

### Lesson
**Packing format is a cross-cutting concern.** Changing it requires updating 4 places. Use a single `#define PACKING_ADJACENT` and branch on it, or just don't change packing.

---

## 10. `lm_head` Q4_K is slower than int8

### The bug
Quantizing lm_head to Q4_K (290MB) instead of int8 (520MB) gave no speedup, despite 44% less bandwidth.

### Why
1. lm_head is only 14% of per-token bandwidth. Saving 230MB = 6% total — negligible.
2. int8 kernel uses sign-trick (no xq pre-arrange). Q4_K kernel recomputes `xq_arr` per call — expensive for 152K rows.
3. Q4_K→F32→int8 double quantization loses accuracy.

### The fix
Keep lm_head as F32 in the file, quantize to int8 at startup. F32 is read once (startup), int8 is read per-token (fast kernel).

### Lesson
**Bandwidth savings only matter if the kernel is bandwidth-bound.** lm_head is compute-bound. See [ARCHITECTURE.md §2](ARCHITECTURE.md#2-why-lm_head-uses-int8-not-q4_k).

---

## 11. Makefile tab vs spaces

### The bug
After editing the Makefile, build failed with `Makefile:34: *** missing separator`.

### Why
The Edit tool converted tabs to 8 spaces. Make requires tabs for recipe lines.

### The fix
```bash
python3 -c "
import re
with open('Makefile','r') as f: content = f.read()
content = re.sub(r'^        ', '\t', content, flags=re.MULTILINE)
with open('Makefile','w') as f: f.write(content)
"
```

### Lesson
**Never let a text editor auto-convert tabs in Makefiles.** Use `.editorconfig` with `Makefile` = `indent_style = tab`.

---

## 12. Qwen2.5-Instruct has UNTIED embeddings

### The bug
First run produced garbage output. The model was using `embed_tokens.weight` for the LM head.

### Why
Some models (like GPT-2) have **tied** embeddings — `embed_tokens` and `lm_head` share the same weight matrix. Qwen2.5-Instruct has **untied** — they're separate.

### The fix
Load `lm_head.weight` separately, don't reuse `embed_tokens.weight`.

### Lesson
**Never assume model architecture.** Check `config.json` for `tie_word_embeddings`. Qwen2.5-Instruct: `false`. GPT-2: `true`.
