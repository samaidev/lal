#!/usr/bin/env python3
"""Export GPT-2 weights to a simple raw binary format for LAL compilation.

Each weight tensor is stored as: [key_len(4B)][key_bytes][ndim(4B)][dims(4B each)][data(float32)]
The file is a flat sequence of these tensors, preceded by a header with the count.

This lets LAL load specific weights by key at compile time, prune them by threshold,
and compile them into sparse specialized C code.
"""
import struct
import numpy as np

def main():
    from transformers import GPT2Model
    print("[*] loading gpt2...", flush=True)
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    sd = model.state_dict()

    out_path = "/home/z/my-project/prebuilt/gpt2_weights.bin"
    with open(out_path, "wb") as f:
        # Header: magic + count
        f.write(b"GPW2")
        n_tensors = len(sd)
        f.write(struct.pack("I", n_tensors))

        for key, tensor in sd.items():
            w = tensor.numpy().astype(np.float32)
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("I", len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack("I", w.ndim))
            for d in w.shape:
                f.write(struct.pack("I", d))
            f.write(w.tobytes())

    import os
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[*] wrote {out_path} ({size_mb:.1f} MB, {n_tensors} tensors)")
    print(f"[*] tensors: {', '.join(list(sd.keys())[:5])}, ...")

if __name__ == "__main__":
    main()
