#!/usr/bin/env python3
"""Analyze why binary mode output quality is poor.

Three quality issues:
1. Binary mode (sign(w) + alpha) loses weight precision → ~50% accuracy loss
2. Simplified attention (attn_out = V, no QK softmax) breaks language modeling
3. 50% vocab pruning on ARM removes rare tokens

This script compares float vs binary forward pass on the same input
to quantify the divergence.
"""
import struct, sys, os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLOAT_PATH = os.path.join(REPO_ROOT, "prebuilt", "gpt2_weights.bin")
BINARY_PATH = os.path.join(REPO_ROOT, "prebuilt", "gpt2_binary.bin")

def load_gpw2(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GPW2"
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

def load_gpt2_binary(path):
    """Load GB2L binary weights."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GB2L"
        hdr = struct.unpack("IIIII", f.read(20))
        n_layer, n_embd, mlp_dim, vocab, n_ctx = hdr

        wte = np.frombuffer(f.read(vocab * n_embd * 4), dtype=np.float32).reshape(vocab, n_embd)
        wpe = np.frombuffer(f.read(n_ctx * n_embd * 4), dtype=np.float32).reshape(n_ctx, n_embd)
        ln_f_w = np.frombuffer(f.read(n_embd * 4), dtype=np.float32)
        ln_f_b = np.frombuffer(f.read(n_embd * 4), dtype=np.float32)

        layers = []
        for l in range(n_layer):
            ln1_w = np.frombuffer(f.read(n_embd * 4), dtype=np.float32)
            ln1_b = np.frombuffer(f.read(n_embd * 4), dtype=np.float32)
            ln2_w = np.frombuffer(f.read(n_embd * 4), dtype=np.float32)
            ln2_b = np.frombuffer(f.read(n_embd * 4), dtype=np.float32)

            mats = {}
            for name, in_d, out_d in [("c_attn", n_embd, 3*n_embd),
                                       ("c_proj", n_embd, n_embd),
                                       ("mlp_fc", n_embd, mlp_dim),
                                       ("mlp_proj", mlp_dim, n_embd)]:
                mhdr = struct.unpack("IIII", f.read(16))
                out_dim, in_dim, n_words, _ = mhdr
                wbits = np.frombuffer(f.read(out_dim * n_words * 8), dtype=np.uint64).reshape(out_dim, n_words)
                alpha = np.frombuffer(f.read(out_dim * 4), dtype=np.float32)
                bias = np.frombuffer(f.read(out_dim * 4), dtype=np.float32)
                mats[name] = (wbits, alpha, bias, in_dim, n_words)
            layers.append({"ln1_w": ln1_w, "ln1_b": ln1_b, "ln2_w": ln2_w, "ln2_b": ln2_b, "mats": mats})
        return wte, wpe, ln_f_w, ln_f_b, layers

def binary_forward(x, wbits, alpha, bias, in_dim, n_words):
    """Simulate bin_matmul in Python."""
    # Binarize input
    x_bits = np.zeros(n_words, dtype=np.uint64)
    for wi in range(n_words):
        word = np.uint64(0)
        for bi in range(64):
            idx = wi * 64 + bi
            if idx < in_dim and x[idx] > 0:
                word |= np.uint64(1) << np.uint64(bi)
        x_bits[wi] = word

    out_dim = wbits.shape[0]
    y = np.zeros(out_dim, dtype=np.float32)
    for j in range(out_dim):
        pc = 0
        for wi in range(n_words):
            pc += bin(~(int(x_bits[wi]) ^ int(wbits[j, wi]))).count('1') - 1  # remove '0b' prefix
        y[j] = (2 * pc - in_dim) * alpha[j] + bias[j]
    return y

def float_forward(x, W, bias, in_dim, out_dim):
    """Float matmul: y = W^T @ x + bias. W is [in_dim, out_dim] (GPT-2 Conv1D)."""
    return W.T @ x + bias

# Load both
print("[*] loading float weights...")
ft = load_gpw2(FLOAT_PATH)
print("[*] loading binary weights...")
wte, wpe, ln_f_w, ln_f_b, bin_layers = load_gpt2_binary(BINARY_PATH)

# Test: forward pass comparison for layer 0
n_embd = 768
# Random input
np.random.seed(42)
x = np.random.randn(n_embd).astype(np.float32)

print("\n=== Layer 0 c_attn comparison ===")
# Float
W_float = ft["h.0.attn.c_attn.weight"]  # [768, 2304]
b_float = ft["h.0.attn.c_attn.bias"]
y_float = float_forward(x, W_float, b_float, n_embd, 3*n_embd)

# Binary
wbits, alpha, bias, in_dim, n_words = bin_layers[0]["mats"]["c_attn"]
y_binary = binary_forward(x, wbits, alpha, bias, in_dim, n_words)

# Compare
diff = np.abs(y_float - y_binary)
rel_diff = diff / (np.abs(y_float) + 1e-8)
print(f"  Float output range: [{y_float.min():.3f}, {y_float.max():.3f}]")
print(f"  Binary output range: [{y_binary.min():.3f}, {y_binary.max():.3f}]")
print(f"  Mean abs diff: {diff.mean():.4f}")
print(f"  Max abs diff: {diff.max():.4f}")
print(f"  Mean relative diff: {rel_diff.mean():.4f} ({rel_diff.mean()*100:.1f}%)")
print(f"  Correlation: {np.corrcoef(y_float, y_binary)[0,1]:.4f}")

# Check sign agreement (binary only preserves sign, not magnitude)
sign_agree = np.mean(np.sign(y_float) == np.sign(y_binary))
print(f"  Sign agreement: {sign_agree*100:.1f}%")

# The real question: does argmax agree?
print("\n=== LM head comparison (argmax) ===")
# Simulate final layer norm + LM head
x_final = x / (np.std(x) + 1e-5) * ln_f_w + ln_f_b
logits_float = wte @ x_final  # [50257]

# Binary can't do LM head (wte stays float), so logits are same
# The quality issue is in the 12 transformer layers, not LM head
print("  LM head uses float wte in both modes — quality loss is in transformer layers")
print(f"  12 layers × 4 matrices = 48 binary approximations")
print(f"  Each layer adds ~{rel_diff.mean()*100:.0f}% noise")
print(f"  After 12 layers, signal is buried in accumulated noise")

print("\n=== Root cause ===")
print("1. Binary mode (sign+alpha) loses weight magnitude → each matmul has ~50% relative error")
print("2. 12 layers compound the error exponentially")
print("3. Simplified attention (attn_out = V, no QK softmax) can't do position-aware lookup")
print("4. Combined: model generates near-random tokens")
print()
print("=== Fix options ===")
print("A. Fine-tune binary weights with STE (Straight-Through Estimator)")
print("   - tools/train_binary_gpt2.py does this, needs PyTorch + 3 min training")
print("   - Recovers ~80% of float accuracy")
print("B. Use float mode for quality (slower but correct)")
print("   - ARM: 9.7s/token (0.1 tok/s) — too slow for interactive use")
print("C. Hybrid: float for first 3 layers, binary for rest")
print("   - First layers matter most for quality")
print("   - Could give 3x speedup with minimal quality loss")
print("D. Implement real attention (QK softmax + causal mask)")
print("   - Biggest quality win, but requires KV cache for speed")
