#!/usr/bin/env python3
"""Analyze GPT-2 weight sparsity at various thresholds.

Loads each weight tensor from gpt2_weights.bin and reports:
  - For each threshold t in [0.005, 0.01, 0.02, 0.05, 0.1]:
    - % of weights with |w| < t  (would be pruned)
    - % of total |w| mass lost   (accuracy impact proxy)

This tells us if sparse matmul is worth implementing.
"""
import struct, sys, os
import numpy as np

WEIGHTS = "/home/z/my-project/prebuilt/gpt2_weights.bin"
THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]

def load_tensors(path):
    """Load GPW2 format: magic(4) + n_tensors(4) + per-tensor blocks."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GPW2", f"bad magic: {magic}"
        n = struct.unpack("I", f.read(4))[0]
        tensors = {}
        for _ in range(n):
            klen = struct.unpack("I", f.read(4))[0]
            key = f.read(klen).decode("utf-8")
            ndim = struct.unpack("I", f.read(4))[0]
            shape = [struct.unpack("I", f.read(4))[0] for _ in range(ndim)]
            n_elem = 1
            for s in shape: n_elem *= s
            data = np.frombuffer(f.read(n_elem * 4), dtype=np.float32)
            tensors[key] = data.reshape(shape)
        return tensors

print(f"Loading {WEIGHTS}...")
tensors = load_tensors(WEIGHTS)
print(f"Loaded {len(tensors)} tensors\n")

# Aggregate per-layer-type stats
categories = {
    "embedding (wte/wpe)": [],
    "ln (weight/bias)": [],
    "attn.c_attn (weight/bias)": [],
    "attn.c_proj (weight/bias)": [],
    "mlp.c_fc (weight/bias)": [],
    "mlp.c_proj (weight/bias)": [],
    "ln_f (weight/bias)": [],
}

def categorize(key):
    if key.startswith("wte") or key.startswith("wpe"): return "embedding (wte/wpe)"
    if ".ln_1." in key or ".ln_2." in key: return "ln (weight/bias)"
    if ".attn.c_attn." in key: return "attn.c_attn (weight/bias)"
    if ".attn.c_proj." in key: return "attn.c_proj (weight/bias)"
    if ".mlp.c_fc." in key: return "mlp.c_fc (weight/bias)"
    if ".mlp.c_proj." in key: return "mlp.c_proj (weight/bias)"
    if "ln_f" in key: return "ln_f (weight/bias)"
    return None

for key, arr in tensors.items():
    cat = categorize(key)
    if cat:
        categories[cat].append((key, arr))

# Print per-category stats at each threshold
print(f"{'Category':<30} {'shape':<20} {'mean|w|':<10}", end="")
for t in THRESHOLDS:
    print(f"  pruned@{t:<5}", end="")
print("  mass_lost@0.02")
print("-" * 130)

total_params = 0
total_pruned = {t: 0 for t in THRESHOLDS}

for cat, items in categories.items():
    for key, arr in items:
        flat = np.abs(arr.flatten())
        n = len(flat)
        total_params += n
        mean_abs = flat.mean()
        # Pruned fraction at each threshold
        pruned_fracs = []
        for t in THRESHOLDS:
            p = (flat < t).sum() / n
            pruned_fracs.append(p)
            total_pruned[t] += (flat < t).sum()
        # Mass lost at threshold 0.02
        mask = flat < 0.02
        mass_lost = flat[mask].sum() / flat.sum()
        shape_str = "x".join(str(s) for s in arr.shape)
        print(f"{key:<30} {shape_str:<20} {mean_abs:<10.4f}", end="")
        for p in pruned_fracs:
            print(f"  {p*100:>7.1f}%", end="")
        print(f"  {mass_lost*100:>7.2f}%")

# Grand total
print("-" * 130)
print(f"\n{'TOTAL':<30} {total_params:>10} params")
print(f"\n=== Sparsity at each threshold (whole model) ===")
print(f"{'threshold':<12} {'pruned':<12} {'kept':<12} {'speedup_if_sparse':<20}")
for t in THRESHOLDS:
    p = total_pruned[t] / total_params
    kept = 1 - p
    # Speedup estimate: sparse matmul does kept*n_ops instead of n_ops
    # But sparse overhead (~3x per non-zero due to indirect indexing)
    # Net speedup ≈ 1 / (kept * 3) if kept*3 < 1, else no benefit
    if kept > 0:
        sparse_speedup = max(0.3, 1.0 / (kept * 3 + 0.1))
    else:
        sparse_speedup = 0
    print(f"  {t:<10.3f} {p*100:>8.1f}%   {kept*100:>8.1f}%   ~{sparse_speedup:.2f}x")

# What about LM head specifically (the bottleneck)?
print(f"\n=== LM head (wte) — the 30%+ bottleneck ===")
wte = tensors["wte.weight"]
flat = np.abs(wte.flatten())
print(f"shape: {wte.shape}, total params: {flat.size}")
for t in THRESHOLDS:
    p = (flat < t).sum() / flat.size
    mask = flat < t
    mass_lost = flat[mask].sum() / flat.sum()
    print(f"  threshold={t}: pruned {p*100:.1f}%, mass lost {mass_lost*100:.2f}%")
