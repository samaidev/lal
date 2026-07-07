#!/usr/bin/env python3
"""Calculate memory usage of binary mode vs float mode for GPT-2 124M.

GPT-2 layer structure (per layer, n=768, m=3072):
  - attn.c_attn: [n, 3n] = 768x2304 = 1.69M params
  - attn.c_proj: [n, n]   = 768x768  = 0.59M params
  - mlp.c_fc:    [n, m]   = 768x3072 = 2.36M params
  - mlp.c_proj:  [m, n]   = 3072x768 = 2.36M params
  Per layer total: 7.0M params

  12 layers: 84M params
  + wte [50257, 768] = 38.6M
  + wpe [1024, 768]  = 0.79M
  + ln_f              = 1.5K
  Total: ~124M params

Float mode (current server):
  124M * 4 bytes = 496 MB

Binary mode (current bin_layer_init):
  For each weight matrix [in, out]:
    - wbits:    out * ceil(in/64) * 8 bytes    (forward)
    - wbits_T:  in * ceil(out/64) * 8 bytes    (backward — TRAINING ONLY)
    - alpha:    out * 4 bytes
    - bias:     out * 4 bytes

  Per matrix: wbits + wbits_T + alpha + bias

Let's compute for each layer.
"""
import math

def bin_mem(in_dim, out_dim, include_transpose=True):
    """Memory for one binary weight matrix."""
    n_words = (in_dim + 63) // 64
    n_words_T = (out_dim + 63) // 64
    wbits = out_dim * n_words * 8
    wbits_T = in_dim * n_words_T * 8 if include_transpose else 0
    alpha = out_dim * 4
    bias = out_dim * 4
    return wbits + wbits_T + alpha + bias

def float_mem(in_dim, out_dim):
    return in_dim * out_dim * 4

n = 768
m = 3072
n_layer = 12
vocab = 50257
n_ctx = 1024

print("=== Per-layer binary memory (with wbits_T for training) ===")
layer_matrices = [
    ("attn.c_attn", n, 3*n),
    ("attn.c_proj", n, n),
    ("mlp.c_fc",    n, m),
    ("mlp.c_proj",  m, n),
]

total_binary_train = 0
total_binary_infer = 0
total_float = 0

for name, in_d, out_d in layer_matrices:
    bm_train = bin_mem(in_d, out_d, include_transpose=True)
    bm_infer = bin_mem(in_d, out_d, include_transpose=False)
    fm = float_mem(in_d, out_d)
    total_binary_train += bm_train
    total_binary_infer += bm_infer
    total_float += fm
    print(f"  {name:15s} [{in_d}x{out_d}]: "
          f"train={bm_train/1e6:6.2f}MB  infer={bm_infer/1e6:6.2f}MB  "
          f"float={fm/1e6:6.2f}MB  "
          f"train_vs_float={bm_train/fm:.2f}x  infer_vs_float={bm_infer/fm:.2f}x")

print(f"\n  Per-layer total: train={total_binary_train/1e6:.1f}MB  "
      f"infer={total_binary_infer/1e6:.1f}MB  float={total_float/1e6:.1f}MB")

print(f"\n=== All 12 layers ===")
print(f"  train: {total_binary_train*n_layer/1e6:.1f}MB")
print(f"  infer: {total_binary_infer*n_layer/1e6:.1f}MB")
print(f"  float: {total_float*n_layer/1e6:.1f}MB")

print(f"\n=== Embeddings (wte + wpe) ===")
# wte and wpe are NOT binarized in current code — they stay float!
wte_float = float_mem(vocab, n)
wpe_float = float_mem(n_ctx, n)
print(f"  wte [{vocab}x{n}]: {wte_float/1e6:.1f}MB (stays float — not binarized)")
print(f"  wpe [{n_ctx}x{n}]: {wpe_float/1e6:.1f}MB (stays float — not binarized)")

print(f"\n=== TOTAL MEMORY ===")
total_train = total_binary_train * n_layer + wte_float + wpe_float
total_infer = total_binary_infer * n_layer + wte_float + wpe_float
total_float_all = total_float * n_layer + wte_float + wpe_float

print(f"  Binary (train mode, with wbits_T):  {total_train/1e6:.1f}MB")
print(f"  Binary (infer mode, no wbits_T):    {total_infer/1e6:.1f}MB")
print(f"  Float (current server):             {total_float_all/1e6:.1f}MB")
print(f"  Weight file size (binary):          ~11.3MB (just wbits+alpha, no wbits_T)")

print(f"\n=== ROOT CAUSE ===")
print(f"The 11.3MB weight FILE only contains wbits + alpha (forward-only).")
print(f"But bin_layer_init() at LOAD TIME allocates BOTH wbits AND wbits_T,")
print(f"plus it needs the original float weights in memory to compute sign().")
print(f"")
print(f"So the actual runtime memory is:")
print(f"  1. Float weights loaded from file:  {total_float_all/1e6:.1f}MB")
print(f"  2. wbits (forward bits):            {total_binary_infer*12/1e6:.1f}MB")
print(f"  3. wbits_T (backward bits, TRAIN):  {(total_binary_train-total_binary_infer)*12/1e6:.1f}MB")
print(f"  4. alpha + bias:                    ~2MB")
print(f"  5. Activations:                     ~10MB")
print(f"  TOTAL (train mode):                 ~{(total_float_all + total_binary_train*12 + 10)/1e6:.0f}MB")
print(f"")
print(f"For INFERENCE ONLY (skip wbits_T):")
print(f"  1. Float weights:                   {total_float_all/1e6:.1f}MB  ← STILL LOADED!")
print(f"  2. wbits (forward bits):            {total_binary_infer*12/1e6:.1f}MB")
print(f"  3. alpha + bias:                    ~2MB")
print(f"  4. Activations:                     ~10MB")
print(f"  TOTAL (infer):                      ~{(total_float_all + total_binary_infer*12 + 10)/1e6:.0f}MB")
print(f"")
print(f"=== THE FIX ===")
print(f"If we load the 11.3MB binary weight FILE directly (skip float entirely):")
print(f"  1. wbits (forward bits):            {total_binary_infer*12/1e6:.1f}MB")
print(f"  2. alpha + bias:                    ~2MB")
print(f"  3. wte + wpe (still float):         {(wte_float+wpe_float)/1e6:.1f}MB")
print(f"  4. Activations:                     ~10MB")
print(f"  TOTAL (binary file, infer):         ~{(total_binary_infer*12 + 2 + wte_float + wpe_float + 10)/1e6:.0f}MB")
print(f"  → {total_float_all/1e6:.0f}MB → {(total_binary_infer*12 + 2 + wte_float + wpe_float + 10)/1e6:.0f}MB "
      f"({total_float_all/1e6 / ((total_binary_infer*12 + 2 + wte_float + wpe_float + 10)/1e6):.1f}x smaller)")
