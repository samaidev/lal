#!/usr/bin/env python3
"""Export Qwen2.5-0.5B weights to GPW2 format — no PyTorch needed.

Reads safetensors directly (handles bfloat16 without numpy/torch support).
"""
import struct, os, sys, json, numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

def bf16_to_f32(u16_arr):
    """Convert bfloat16 (uint16) to float32."""
    u32 = u16_arr.astype(np.uint32) << 16
    return u32.view(np.float32)

def read_safetensor(path):
    """Read all tensors from a safetensors file, handling bfloat16."""
    f = open(path, 'rb')
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_size))
    tensors = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dtype = meta["dtype"]
        shape = meta["shape"]
        start, end = meta["data_offsets"]
        f.seek(8 + header_size + start)
        raw = f.read(end - start)

        n_elem = 1
        for s in shape:
            n_elem *= s

        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).copy()
            arr = bf16_to_f32(u16).reshape(shape)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
        elif dtype == "F16":
            f16 = np.frombuffer(raw, dtype=np.float16).copy()
            arr = f16.astype(np.float32).reshape(shape)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        tensors[key] = arr
        print(f"  {key} {list(shape)} {dtype} -> float32")

    f.close()
    return tensors

def main():
    out_path = os.path.join(REPO_ROOT, "prebuilt", "qwen_weights.bin")
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]

    print(f"[*] downloading {MODEL_NAME}...")
    files = list_repo_files(MODEL_NAME)
    st_files = [f for f in files if f.endswith(".safetensors")]

    local_paths = []
    for fname in st_files:
        p = hf_hub_download(MODEL_NAME, fname)
        local_paths.append(p)
        print(f"  downloaded {fname}")

    # Download tokenizer files
    tokenizer_dir = os.path.join(REPO_ROOT, "prebuilt", "qwen_tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    for fname in ["tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "config.json"]:
        try:
            hf_hub_download(MODEL_NAME, fname, local_dir=tokenizer_dir)
            print(f"  tokenizer: {fname}")
        except Exception:
            pass

    # Read tensors
    all_tensors = {}
    for lp in local_paths:
        print(f"\n[*] reading {lp}...")
        tensors = read_safetensor(lp)
        all_tensors.update(tensors)

    print(f"\n[*] total tensors: {len(all_tensors)}")

    # Write GPW2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"GPW2")
        f.write(struct.pack("I", len(all_tensors)))
        for i, (key, arr) in enumerate(all_tensors.items()):
            kb = key.encode("utf-8")
            f.write(struct.pack("I", len(kb)))
            f.write(kb)
            f.write(struct.pack("I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("I", d))
            f.write(arr.astype(np.float32).tobytes())
            if i % 30 == 0:
                print(f"  [{i}/{len(all_tensors)}] {key} {list(arr.shape)}", flush=True)

    size_mb = os.path.getsize(out_path) / 1e6
    total_params = sum(a.size for a in all_tensors.values())
    print(f"\n[*] wrote {out_path} ({size_mb:.1f} MB)")
    print(f"[*] params: {total_params/1e6:.1f}M, tensors: {len(all_tensors)}")

if __name__ == "__main__":
    main()