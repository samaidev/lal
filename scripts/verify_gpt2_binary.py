#!/usr/bin/env python3
"""verify_gpt2_binary.py — Verify that prebuilt/gpt2_binary.bin (GB2L) correctly
binarizes the original GPT-2 weights.

For each matrix in a sample of layers, check:
  - sign(W_orig[i,j])  ==  sign reconstructed from packed wbits
  - alpha[j]            ==  mean(|W_orig[:,j]|)
  - bias[j]             ==  W_orig_bias[j]
  - wte/wpe/ln_f match the original float tensors byte-for-byte

If all match, the GB2L file is correctly formatted and any quality loss in BWN
output is the genuine behavior of 1-bit binarization, NOT a file-generation bug.
"""
import json
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE = os.path.join(REPO, "build", "gpt2.safetensors")
BIN = os.path.join(REPO, "prebuilt", "gpt2_binary.bin")

N_LAYER, N_EMBD, MLP_DIM, VOCAB, N_CTX = 12, 768, 3072, 50257, 1024


def load_safetensors(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode())
        data_start = 8 + hlen
    tensors = {}
    with open(path, "rb") as f:
        f.seek(data_start)
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            off0, off1 = info["data_offsets"]
            f.seek(data_start + off0)
            raw = f.read(off1 - off0)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy()
            tensors[name] = arr
    return tensors


def read_gpt2_binary(path):
    """Parse GB2L into a dict of numpy arrays + per-matrix sign-packed arrays."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GB2L", f"bad magic {magic}"
        n_layer, n_embd, mlp_dim, vocab, n_ctx = struct.unpack("<IIIII", f.read(20))
        assert (n_layer, n_embd, mlp_dim, vocab, n_ctx) == (N_LAYER, N_EMBD, MLP_DIM, VOCAB, N_CTX)

        out = {}
        out["wte"] = np.frombuffer(f.read(vocab * n_embd * 4), dtype=np.float32).reshape(vocab, n_embd).copy()
        out["wpe"] = np.frombuffer(f.read(n_ctx * n_embd * 4), dtype=np.float32).reshape(n_ctx, n_embd).copy()
        out["ln_f_w"] = np.frombuffer(f.read(n_embd * 4), dtype=np.float32).copy()
        out["ln_f_b"] = np.frombuffer(f.read(n_embd * 4), dtype=np.float32).copy()

        out["layers"] = []
        for l in range(n_layer):
            L = {}
            for key in ["ln_1_w", "ln_1_b", "ln_2_w", "ln_2_b"]:
                L[key] = np.frombuffer(f.read(n_embd * 4), dtype=np.float32).copy()
            L["mats"] = {}
            for name, in_dim, out_dim in [
                ("c_attn", n_embd, 3 * n_embd),
                ("c_proj", n_embd, n_embd),
                ("mlp_fc", n_embd, mlp_dim),
                ("mlp_proj", mlp_dim, n_embd),
            ]:
                mhdr = struct.unpack("<IIII", f.read(16))
                od, id_, n_words, _ = mhdr
                assert od == out_dim and id_ == in_dim, f"layer {l} {name}: hdr {mhdr}"
                wbits = np.frombuffer(
                    f.read(out_dim * n_words * 8), dtype=np.uint64
                ).reshape(out_dim, n_words).copy()
                alpha = np.frombuffer(f.read(out_dim * 4), dtype=np.float32).copy()
                bias = np.frombuffer(f.read(out_dim * 4), dtype=np.float32).copy()
                L["mats"][name] = (wbits, alpha, bias, in_dim, out_dim, n_words)
            out["layers"].append(L)
    return out


def unpack_signs(wbits, in_dim, out_dim, n_words):
    """Reconstruct [out_dim, in_dim] sign array (+1/-1) from packed wbits."""
    signs = np.full((out_dim, in_dim), -1.0, dtype=np.float32)
    for j in range(out_dim):
        for wi in range(n_words):
            w = int(wbits[j, wi])
            base = wi * 64
            for bi in range(64):
                idx = base + bi
                if idx >= in_dim:
                    break
                if (w >> bi) & 1:
                    signs[j, idx] = 1.0
    return signs


def main():
    print(f"[*] loading safetensors: {SAFE}")
    orig = load_safetensors(SAFE)
    print(f"[*] loading GB2L: {BIN}")
    binf = read_gpt2_binary(BIN)

    fails = 0
    checks = 0

    # Embeddings + LN (must match float exactly)
    for key, oval in [("wte", orig["wte.weight"]),
                      ("wpe", orig["wpe.weight"]),
                      ("ln_f_w", orig["ln_f.weight"]),
                      ("ln_f_b", orig["ln_f.bias"])]:
        bval = binf[key]
        ok = np.array_equal(oval, bval)
        checks += 1
        if ok:
            print(f"  ok  : {key:8} float tensor matches ({oval.shape})")
        else:
            print(f"  FAIL: {key} mismatch (max diff {np.abs(oval-bval).max():.4e})")
            fails += 1

    # Per-layer matrices
    orig_key = {
        "c_attn":    "h.{l}.attn.c_attn.weight",
        "c_proj":    "h.{l}.attn.c_proj.weight",
        "mlp_fc":    "h.{l}.mlp.c_fc.weight",
        "mlp_proj":  "h.{l}.mlp.c_proj.weight",
    }
    bias_key = {
        "c_attn":    "h.{l}.attn.c_attn.bias",
        "c_proj":    "h.{l}.attn.c_proj.bias",
        "mlp_fc":    "h.{l}.mlp.c_fc.bias",
        "mlp_proj":  "h.{l}.mlp.c_proj.bias",
    }

    for l in [0, 6, 11]:  # sample first/middle/last layer
        for name in orig_key:
            wbits, alpha, bias, in_dim, out_dim, n_words = binf["layers"][l]["mats"][name]
            W = orig[orig_key[name].format(l=l)]   # [in_dim, out_dim] Conv1D
            B = orig[bias_key[name].format(l=l)]   # [out_dim]

            # Reconstruct signs [out, in]
            signs_packed = unpack_signs(wbits, in_dim, out_dim, n_words)
            signs_orig = np.where(W.T > 0, 1.0, -1.0).astype(np.float32)
            sign_mismatch = int((signs_packed != signs_orig).sum())

            # alpha = mean(|W[:,j]|) per output j
            alpha_orig = np.abs(W.T).mean(axis=1).astype(np.float32)
            alpha_max_err = float(np.abs(alpha - alpha_orig).max())

            bias_max_err = float(np.abs(bias - B).max())

            checks += 3
            label = f"L{l:2d}.{name:9}"
            s_ok = sign_mismatch == 0
            a_ok = alpha_max_err < 1e-5
            b_ok = bias_max_err < 1e-5
            if s_ok:
                print(f"  ok  : {label} sign bits   ({in_dim}x{out_dim}) all match")
            else:
                print(f"  FAIL: {label} sign bits   {sign_mismatch} mismatches")
                fails += 1
            if a_ok:
                print(f"  ok  : {label} alpha       max_err={alpha_max_err:.2e}")
            else:
                print(f"  FAIL: {label} alpha       max_err={alpha_max_err:.2e}")
                fails += 1
            if b_ok:
                print(f"  ok  : {label} bias        max_err={bias_max_err:.2e}")
            else:
                print(f"  FAIL: {label} bias        max_err={bias_max_err:.2e}")
                fails += 1

    print(f"\n=== {checks} checks, {fails} failures ===")
    # Also compute the binarization error: how much does sign(W)*alpha differ from W?
    print("\n[*] binarization approximation error (|W - sign(W)*alpha| / |W|) per layer-0 matrix:")
    l = 0
    for name in orig_key:
        wbits, alpha, bias, in_dim, out_dim, n_words = binf["layers"][l]["mats"][name]
        W = orig[orig_key[name].format(l=l)].astype(np.float32)  # [in, out]
        signs = np.where(W.T > 0, 1.0, -1.0).astype(np.float32)   # [out, in]
        W_approx = (signs * alpha[:, None]).T                      # [in, out]
        rel = np.linalg.norm(W - W_approx) / np.linalg.norm(W)
        print(f"    L0.{name:9} rel_err={rel:.4f}  ({rel*100:.1f}% energy lost to 1-bit quant)")

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
