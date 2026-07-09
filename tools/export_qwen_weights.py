#!/usr/bin/env python3
"""Export Qwen2.5-0.5B weights to GPW2 format for LAL qwen_server.

Usage:
    python3 tools/export_qwen_weights.py [--model Qwen/Qwen2.5-0.5B] [--output prebuilt/qwen_weights.bin]

The output GPW2 file can be loaded by:
    ./qwen_server --weights prebuilt/qwen_weights.bin
"""
import struct
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    from transformers import AutoModelForCausalLM
    import torch

    model_name = "Qwen/Qwen2.5-0.5B"
    out_path = os.path.join(REPO_ROOT, "prebuilt", "qwen_weights.bin")

    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
        elif arg == "--output" and i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]

    print(f"[*] loading {model_name} (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()
    sd = model.state_dict()

    # Show expected tensor keys for qwen_server
    expected_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    print("[*] verifying tensor keys...")
    for k in expected_keys:
        if k in sd:
            print(f"  OK  {k} {list(sd[k].shape)}")
        else:
            # Try alternate key names
            alt = k.replace("gate_proj", "gate").replace("up_proj", "up").replace("down_proj", "down")
            if alt in sd:
                print(f"  ALT {k} -> {alt} {list(sd[alt].shape)}")
            else:
                print(f"  MISSING {k}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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
