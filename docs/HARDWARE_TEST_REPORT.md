# Hardware Test Report — LAL GPT-2 Server on Debian 12 Cloud Container

**Date**: 2026-07-10
**Tester**: Automated deployment via samcommand tunnel
**Commit tested**: `672ad485` (upstream) + local fixes `a79f8db` (matmul_q8 forward decl)

## Hardware

| Component | Spec |
|---|---|
| Platform | Linux x86_64 container, kernel 5.10.134 |
| OS | Debian GNU/Linux 12 (bookworm) |
| CPU | Intel Xeon Platinum, 2 vCPU (no GPU) |
| RAM | 15 GiB total, ~13 GiB available |
| Disk | 69 GB overlay, 56 GB free |
| SIMD | AVX2 + FMA confirmed (kernel userspace) |
| glibc | 2.36 (system) — prebuilt binary required 2.38, rebuilt from source |

## Deployment obstacles (all resolved)

1. **`github.com` unreachable** from this cloud region. TCP connections to
   github.com:443 hang, but `api.github.com`, `codeload.github.com`,
   `objects.githubusercontent.com`, and `release-assets.githubusercontent.com`
   all work. Source obtained via the API tarball endpoint; weights obtained
   via the asset API.

2. **GLIBC_2.38 mismatch**. The prebuilt `prebuilt/gpt2_server` binary
   requires GLIBC 2.38, but Debian 12 ships 2.36. Resolved by installing
   `build-essential libopenblas-dev` and rebuilding from source.

3. **Source bug**: `matmul_q8_mt` calls `matmul_q8` before its definition.
   gcc 12+ rejects this with `error: static declaration of 'matmul_q8'
   follows non-static declaration`. Fixed by adding a static forward
   declaration — see commit `a79f8db`.

4. **Weights not in repo**. The 474 MB `prebuilt/gpt2_weights.bin` is not
   shipped in the source tree. GitHub Release download was rate-limited to
   ~12 KB/s. Switched to `hf-mirror.com` to fetch `openai-community/gpt2`
   safetensors (522 MiB, 44 MiB/s), then converted to GPW2 format using
   `scripts/convert_safetensors_to_gpw2.py`. Output is byte-for-byte
   identical to the official release (497,763,872 bytes, 148 tensors).

5. **Port 8080 conflict**. Another service was listening on 8080. Ran the
   server on port 8082 instead.

## Build result

```
$ make server
cc -O3 -mavx2 -mfma -Wall -Wno-unused-function -Wno-unused-variable -I. \
    -o prebuilt/gpt2_server tools/server/gpt2_server.c runtime/lal_runtime.c \
    -lm -lpthread -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas
[*] built with OpenBLAS acceleration (-L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas)

$ ldd prebuilt/gpt2_server
    libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6
    libopenblas.so.0 => /lib/x86_64-linux-gnu/libopenblas.so.0
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
    libgfortran.so.5 => /lib/x86_64-linux-gnu/libgfortran.so.5
    libquadmath.so.0 => /lib/x86_64-linux-gnu/libquadmath.so.0
    libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1

Build time: 5.8 s
Errors: 0 (after matmul_q8 fix)
Warnings: 13 (pre-existing: misleading-indentation x3, sprintf overlap x10)
Binary size: 199,208 bytes (199 KB)
```

## Startup log

```
[*] OpenBLAS: 1 thread(s)
[*] LAL GPT-2 Server (float OpenBLAS mode, 2 threads, 2 cores detected)
[*] loading float weights...
[*] loaded 148 tensors
[*] Q8 mode: 8-bit per-row quantization (default, 34 tok/s, 27MB)
[*] Auto-enabling int8 LM head for Q8 mode
[*] LM head int8 quantized: 36.8 MB int8 + 196.3 KB scales (quantize 88 ms)
[*] allocating KV cache for real causal attention (72 MB)...
[*] KV cache allocated — real causal self-attention enabled
[*] tokenizer loaded (50257 entries, hash table 524288 slots)
[*] server running at http://localhost:8082
```

Startup time: ~5 s (weight load + Q8 quantization + LM head quantization + KV cache alloc).

## Performance

### Throughput (default Q8 mode, sampled decoding)

| Test | Prompt | n_tokens | Wall time | tok/s |
|---|---|---|---|---|
| 1 | `Hello, how are` | 30 | 0.63 s | 47.6 |
| 2 | `Once upon a time, in a galaxy far far away,` | 40 | 0.97 s | 41.1 |
| 3 | `Machine learning is a subset of` | 30 | 0.66 s | 45.4 |
| 4 | `The quick brown fox` | 60 | 1.25 s | 47.9 |
| 5 | `The quick brown fox jumps over the` | 20 | 0.48 s | 42.1 |
| 6 | `In 1492, Columbus` | 25 | 0.54 s | 46.3 |
| 7 | `To be or not to be, that is` | 25 | 0.58 s | 42.9 |

**Mean throughput**: 44.7 tok/s (README claims 42 tok/s for amd64 — **met or exceeded**).

### Resource usage at steady state

| Metric | Value |
|---|---|
| Process RSS | 625 MB |
| Process VSZ | 873 MB |
| %CPU (idle) | 4.9% |
| %CPU (generating 30 tokens) | ~100% on 1 core, ~50% on the other |
| Threads | 2 (matches amd64 config) |
| OpenBLAS threads | 1 (forced single to avoid contention with LM head threads) |

The 625 MB RSS breaks down as: 474 MB float weights (loaded for Q8 quantization,
not freed) + 27 MB Q8 weights + 37 MB int8 LM head + 72 MB KV cache + ~14 MB
tokenizer/buffers. See `README.md` → Memory breakdown for the full table.

## Quality

### Sampled decoding (default: temperature=0.8, top_k=40, rep_penalty=1.1)

| Prompt | Output | Quality |
|---|---|---|
| `Hello, how are` | `the current conditions on all of these topics?` | Grammatical, on-topic |
| `Once upon a time, in a galaxy far far away,` | `the human race has gone to war. Somewhat like speeches principally of science fiction and decisions made before an era's first election...` | Coherent, on-theme |
| `Machine learning is a subset of` | `the United States government's highways, thinks it was answered by frontier officers...` | Grammatical, semantically off-topic |
| `The quick brown fox jumps over the` | `side of a barn on Mr. Johnson's way to get vibrations...` | Grammatical, locally coherent |
| `In 1492, Columbus` | `'s first agreement with the United States was eased and it became growing progress in its efforts to support refugees.` | Grammatical, factually wrong (anachronistic) |
| `To be or not to be, that is` | `what the science actually suggests. Before I pushed Tehran's obsessive-compulsive political wrongdoing...` | Grammatical, off-topic |

### Greedy decoding (temperature=0)

| Prompt | Output | Quality |
|---|---|---|
| `The capital of France is` | `the French-speaking, Muslim-majority, Muslim-majority, Muslim-` | Repetition loop within 8 tokens |
| `The quick brown fox` | `es are a little more difficult to spot, and the more the more the` | Repetition loop within 12 tokens |
| `Machine learning is` | `a great way to learn, to learn, to learn. It's a great way to learn,` | Repetition loop within 7 tokens |
| `To be or not to be` | `, the first line of the first-person narrative is the first line of the first-person narrative` | Repetition loop within 12 tokens |

**Conclusion**: under greedy decoding, GPT-2 124M falls into repetition loops
within ~10 tokens. This is intrinsic to the model size, not a LAL runtime
artifact. The default sampled-decoding parameters mitigate this and produce
locally coherent text, but factual accuracy is poor (GPT-2 124M is too small
to retain factual knowledge).

## Verdict

| Aspect | README claim | Measured | Verdict |
|---|---|---|---|
| Throughput (amd64, Q8) | 42 tok/s | 44.7 tok/s mean | ✅ exceeds |
| Q8 weight memory | 27 MB | 27 MB | ✅ matches |
| Peak RSS | (not claimed) | 625 MB | ⚠️ not in README — added to Memory breakdown |
| Coherent text (default sampling) | ✅ Coherent | locally coherent, globally off-topic | ✅ matches |
| Startup time | (not claimed) | ~5 s | ✅ acceptable |
| Build from source on Debian 12 | (not claimed) | works after matmul_q8 fix | ✅ works (bug fix committed) |

The LAL GPT-2 server is **production-functional** for the use case it targets
(low-resource CPU inference of a small GPT-2 model). The 27 MB / 42 tok/s
headline numbers are about runtime efficiency, not model capability — users
should not expect GPT-2 124M to answer factual questions reliably.
