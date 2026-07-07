#!/usr/bin/env python3
"""Export GPT-2 weights + BPE tokenizer to a portable binary format for C inference.

Output: gpt2_model.bin
Layout:
  - Header: magic (4B "GPT2"), n_layers (4B), dim (4B), n_ctx (4B), vocab (4B), n_merges (4B)
  - Token embeddings: [vocab, dim] float32
  - Position embeddings: [n_ctx, dim] float32
  - Per-layer weights (12 layers):
    - ln_1.weight [dim], ln_1.bias [dim]
    - attn.c_attn.weight [dim, 2304], attn.c_attn.bias [2304]
    - attn.c_proj.weight [dim, dim], attn.c_proj.bias [dim]
    - ln_2.weight [dim], ln_2.bias [dim]
    - mlp.c_fc.weight [dim, 3072], mlp.c_fc.bias [3072]
    - mlp.c_proj.weight [3072, dim], mlp.c_proj.bias [dim]
  - Final ln_f.weight [dim], ln_f.bias [dim]
  - BPE merges: n_merges pairs of (len, bytes, len, bytes)
  - BPE encoder: vocab entries, each (token_id, len, bytes)

Also exports the BPE vocab as a text file for decoding.
"""
import struct
import json
import numpy as np

def main():
    from transformers import GPT2Model, GPT2Tokenizer
    import torch

    print("[*] loading gpt2...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    sd = model.state_dict()
    n_layers = model.config.n_layer      # 12
    dim = model.config.n_embd            # 768
    n_ctx = model.config.n_ctx           # 1024
    vocab = model.config.vocab_size      # 50257

    # GPT-2 uses byte-level BPE. Get the merges and encoder from the tokenizer.
    # The tokenizer's byte_decoder maps unicode chars to bytes.
    encoder = tokenizer.encoder  # dict: token_str -> token_id
    merges = tokenizer.bpe_ranks # dict: (a, b) -> rank

    # Build the merges list in rank order
    merge_list = sorted(merges.keys(), key=lambda p: merges[p])
    n_merges = len(merge_list)
    print(f"[*] layers={n_layers} dim={dim} ctx={n_ctx} vocab={vocab} merges={n_merges}")

    out_path = "/home/z/my-project/scripts/lal/gpt2_model.bin"
    with open(out_path, "wb") as f:
        # Header
        f.write(b"GPT2")
        f.write(struct.pack("IIIII", n_layers, dim, n_ctx, vocab, n_merges))

        # Token embeddings [vocab, dim]
        wte = sd["wte.weight"].numpy().astype(np.float32)
        f.write(wte.tobytes())

        # Position embeddings [n_ctx, dim]
        wpe = sd["wpe.weight"].numpy().astype(np.float32)
        f.write(wpe.tobytes())

        # Per-layer weights
        for i in range(n_layers):
            for name in ["ln_1.weight", "ln_1.bias",
                         "attn.c_attn.weight", "attn.c_attn.bias",
                         "attn.c_proj.weight", "attn.c_proj.bias",
                         "ln_2.weight", "ln_2.bias",
                         "mlp.c_fc.weight", "mlp.c_fc.bias",
                         "mlp.c_proj.weight", "mlp.c_proj.bias"]:
                key = f"h.{i}.{name}"
                w = sd[key].numpy().astype(np.float32)
                f.write(w.tobytes())

        # Final layer norm
        f.write(sd["ln_f.weight"].numpy().astype(np.float32).tobytes())
        f.write(sd["ln_f.bias"].numpy().astype(np.float32).tobytes())

        # BPE merges
        for a, b in merge_list:
            fa = a.encode("utf-8")
            fb = b.encode("utf-8")
            f.write(struct.pack("I", len(fa))); f.write(fa)
            f.write(struct.pack("I", len(fb))); f.write(fb)

        # BPE encoder: token_str -> token_id
        # GPT-2 tokens are unicode strings that map to bytes via byte_decoder.
        # We store the raw token string (unicode, utf-8 encoded) + token_id.
        f.write(struct.pack("I", len(encoder)))
        byte_decoder = tokenizer.byte_decoder  # dict: unicode_char -> byte_val
        for token_str, token_id in encoder.items():
            # Convert token_str to bytes via byte_decoder
            token_bytes = bytes(byte_decoder[c] for c in token_str)
            f.write(struct.pack("II", token_id, len(token_bytes)))
            f.write(token_bytes)

    import os
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[*] wrote {out_path} ({size_mb:.1f} MB)")
    print(f"[*] fp32 model size: {size_mb:.1f} MB (can be quantized to ~31 MB int4)")

if __name__ == "__main__":
    main()
