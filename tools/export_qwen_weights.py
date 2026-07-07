#!/usr/bin/env python3
"""Export Qwen2.5-0.5B weights to GPW2 format for LAL runtime."""
import struct
import numpy as np

def main():
    from transformers import AutoModelForCausalLM
    import torch
    print("[*] loading Qwen2.5-0.5B (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)
    model.eval()
    sd = model.state_dict()

    out_path = "/home/z/my-project/prebuilt/qwen_weights.bin"
    # Write in chunks to avoid memory spike
    with open(out_path, "wb") as f:
        f.write(b"GPW2")
        f.write(struct.pack("I", len(sd)))
        for i, (key, tensor) in enumerate(sd.items()):
            # Convert to float32 one tensor at a time (saves memory)
            w = tensor.float().numpy()
            kb = key.encode("utf-8")
            f.write(struct.pack("I", len(kb)))
            f.write(kb)
            f.write(struct.pack("I", w.ndim))
            for d in w.shape:
                f.write(struct.pack("I", d))
            f.write(w.tobytes())
            if i % 50 == 0:
                print(f"  [{i}/{len(sd)}] {key}", flush=True)
            del w  # free immediately

    import os
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"[*] params: {n_params/1e6:.1f}M, tensors: {len(sd)}")

if __name__ == "__main__":
    main()
