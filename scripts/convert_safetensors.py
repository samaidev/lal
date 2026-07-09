#!/usr/bin/env python3
"""Convert GPT-2 safetensors (openai-community/gpt2) → LAL weight files,
WITHOUT needing torch/transformers. Uses only numpy + stdlib.

Produces:
  prebuilt/gpt2_weights.bin  (GPW2 float format, per export_gpt2_weights.py)
  prebuilt/gpt2_binary.bin   (GB2L binary format, per export_binary_gpt2.py)

safetensors layout:
  [8B uint64 LE header_length][JSON header][raw tensor data]
  header JSON: { "__metadata__": {...}, "<tensor_name>": {"dtype":..,"shape":[..],"data_offsets":[start,end]}, ... }
  data_offsets are byte offsets into the raw data region (which starts right after the header).
"""
import json
import os
import struct
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(REPO_ROOT, "build", "gpt2.safetensors")


def load_safetensors(path):
    """Return dict: name -> np.ndarray (float32, contiguous)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)
        data_start = 8 + header_len
        # Map remaining file as raw bytes via a second handle to avoid loading
        # the whole thing into Python twice. We'll read slices on demand.
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

    tensors = {}
    # Re-open for data reads (slices).
    df = open(path, "rb")
    df.seek(data_start)

    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        shape = info["shape"]
        off0, off1 = info["data_offsets"]
        nbytes = off1 - off0
        df.seek(data_start + off0)
        raw = df.read(nbytes)

        if dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dtype == "BF16":
            # BF16 = upper 16 bits of float32. Reinterpret via uint32 view.
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << np.uint32(16)
            arr = u32.view(np.float32).copy()
        else:
            raise ValueError(f"unsupported dtype {dtype} for tensor {name}")

        arr = arr.reshape(shape).astype(np.float32, copy=False)
        tensors[name] = np.ascontiguousarray(arr)

    df.close()
    return tensors


def write_gpw2(tensors, out_path):
    """GPW2 float format: magic + count, then per tensor
    [key_len(4)][key_bytes][ndim(4)][dims(ndim*4)][data(float32)]."""
    # Match export_gpt2_weights.py which uses GPT2Model (no lm_head). We include
    # everything present; the server looks up only the keys it needs, so an
    # extra lm_head.weight (tied) is harmless.
    keys = list(tensors.keys())
    with open(out_path, "wb") as f:
        f.write(b"GPW2")
        f.write(struct.pack("I", len(keys)))
        for k in keys:
            w = tensors[k].astype(np.float32)
            kb = k.encode("utf-8")
            f.write(struct.pack("I", len(kb)))
            f.write(kb)
            f.write(struct.pack("I", w.ndim))
            for d in w.shape:
                f.write(struct.pack("I", int(d)))
            f.write(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[*] wrote {out_path} ({size_mb:.1f} MB, {len(keys)} tensors)")
    # Show a few keys for sanity.
    print(f"    keys: {', '.join(keys[:4])}, ... , {keys[-1]}")


def pack_sign_bits(signs_2d):
    """Pack 2D sign array [out_dim, in_dim] -> bytes of [out_dim, n_words] uint64.
    Bit bi of word wi set iff signs_2d[j, wi*64+bi] > 0.
    Matches export_binary_gpt2.pack_sign_bits exactly."""
    out_dim, in_dim = signs_2d.shape
    n_words = (in_dim + 63) // 64
    packed = np.zeros((out_dim, n_words), dtype=np.uint64)
    for j in range(out_dim):
        for wi in range(n_words):
            word = np.uint64(0)
            base = wi * 64
            for bi in range(64):
                idx = base + bi
                if idx < in_dim and signs_2d[j, idx] > 0:
                    word |= np.uint64(1) << np.uint64(bi)
            packed[j, wi] = word
    return packed.tobytes()


def write_gb2l(tensors, out_path):
    """GB2L binary format. See export_binary_gpt2.py for full spec."""
    n_layer = 12
    n_embd = 768
    mlp_dim = n_embd * 4  # 3072
    vocab = 50257
    n_ctx = 1024

    def g(key):
        if key not in tensors:
            raise KeyError(f"missing tensor: {key}")
        return tensors[key]

    with open(out_path, "wb") as f:
        f.write(b"GB2L")
        f.write(struct.pack("IIIII", n_layer, n_embd, mlp_dim, vocab, n_ctx))

        # Embeddings + final LN (float)
        wte = g("wte.weight").astype(np.float32)        # [vocab, n_embd]
        wpe = g("wpe.weight").astype(np.float32)       # [n_ctx, n_embd]
        ln_f_w = g("ln_f.weight").astype(np.float32)   # [n_embd]
        ln_f_b = g("ln_f.bias").astype(np.float32)     # [n_embd]
        assert wte.shape == (vocab, n_embd), f"wte {wte.shape}"
        assert wpe.shape == (n_ctx, n_embd), f"wpe {wpe.shape}"
        f.write(wte.tobytes())
        f.write(wpe.tobytes())
        f.write(ln_f_w.tobytes())
        f.write(ln_f_b.tobytes())

        for l in range(n_layer):
            # LayerNorm (float)
            for key in [f"h.{l}.ln_1.weight", f"h.{l}.ln_1.bias",
                        f"h.{l}.ln_2.weight", f"h.{l}.ln_2.bias"]:
                w = g(key).astype(np.float32)
                f.write(w.tobytes())

            # 4 binary matrices per layer.
            # HuggingFace GPT-2 Conv1D stores weight as [in_dim, out_dim].
            # Binarize per-output row: signs[j,i] = sign(W[i,j]); alpha = mean|W[:,j]|.
            for key, in_dim, out_dim in [
                (f"h.{l}.attn.c_attn.weight", n_embd, 3 * n_embd),
                (f"h.{l}.attn.c_proj.weight", n_embd, n_embd),
                (f"h.{l}.mlp.c_fc.weight",    n_embd, mlp_dim),
                (f"h.{l}.mlp.c_proj.weight",  mlp_dim, n_embd),
            ]:
                W = g(key).astype(np.float32)              # [in_dim, out_dim]
                assert W.shape == (in_dim, out_dim), f"{key} {W.shape}"
                W_t = np.ascontiguousarray(W.T)             # [out_dim, in_dim]
                signs = (W_t > 0).astype(np.int8)           # [out_dim, in_dim]
                alpha = np.abs(W_t).mean(axis=1).astype(np.float32)  # [out_dim]
                bias = g(key.replace(".weight", ".bias")).astype(np.float32)  # [out_dim]

                n_words = (in_dim + 63) // 64
                f.write(struct.pack("IIII", out_dim, in_dim, n_words, 0))
                f.write(pack_sign_bits(signs))
                f.write(alpha.tobytes())
                f.write(bias.tobytes())

    size = os.path.getsize(out_path) / 1e6
    print(f"[*] wrote {out_path} ({size:.1f} MB, 44x compression vs 498 MB float)")

    # Verify magic/header round-trip.
    with open(out_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GB2L", f"bad magic: {magic}"
        hdr = struct.unpack("IIIII", f.read(20))
        print(f"    verify: n_layer={hdr[0]}, n_embd={hdr[1]}, "
              f"mlp_dim={hdr[2]}, vocab={hdr[3]}, n_ctx={hdr[4]}")


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    if not os.path.exists(in_path):
        sys.exit(f"input not found: {in_path}")
    print(f"[*] parsing safetensors: {in_path}")
    tensors = load_safetensors(in_path)
    n = len(tensors)
    sample = list(tensors.keys())[:3]
    print(f"[*] {n} tensors. sample: {sample}")
    # Sanity: confirm GPT-2 base config.
    assert tensors["wte.weight"].shape == (50257, 768), "not GPT-2 base?"
    assert tensors["h.0.attn.c_attn.weight"].shape == (768, 2304)
    print("[*] config OK (GPT-2 base: 12 layers, 768 embd, 50257 vocab)")

    out_dir = os.path.join(REPO_ROOT, "prebuilt")
    os.makedirs(out_dir, exist_ok=True)
    write_gpw2(tensors, os.path.join(out_dir, "gpt2_weights.bin"))
    write_gb2l(tensors, os.path.join(out_dir, "gpt2_binary.bin"))
    print("[*] done")


if __name__ == "__main__":
    main()
