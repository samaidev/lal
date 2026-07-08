#!/usr/bin/env python3
"""Export GPT-2 weights in LAL binary format (GBIN2).

Format:
  Magic: "GB2L" (4 bytes)
  Header: n_layer(4), n_embd(4), mlp_dim(4), vocab(4), n_ctx(4)
  
  Embeddings (float, NOT binarized — they're lookup tables):
    wte: [vocab, n_embd] float32
    wpe: [n_ctx, n_embd] float32
    ln_f.weight: [n_embd] float32
    ln_f.bias:   [n_embd] float32
  
  Per layer (12 layers), 4 matrices each:
    For each [in_dim, out_dim] matrix:
      Header: out_dim(4), in_dim(4), n_words(4), reserved(4)
      wbits:  out_dim * n_words uint64s (sign bits, row-major per output)
      alpha:  out_dim float32s (per-output scale = mean|w|)
      bias:   out_dim float32s
  
  LayerNorm weights (float, NOT binarized):
    ln_1.weight, ln_1.bias, ln_2.weight, ln_2.bias — each [n_embd] float32

Total size: ~13 MB (vs 498 MB float).
  - 12 layers × 4 matrices × (wbits + alpha + bias) ≈ 13 MB
  - wte [50257, 768] float = 154 MB  ← still float, lookup table
  - wpe [1024, 768] float = 3 MB
  - ln weights ≈ 0.1 MB

The wte is the elephant in the room — 154 MB for the embedding lookup.
Binarizing it would hurt accuracy too much (it's used as both input
embedding AND output projection / LM head). Keep it float.
"""
import struct, sys, os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def pack_sign_bits(signs_2d):
    """Pack a 2D sign array [out_dim, in_dim] into uint64s.
    Returns bytes of shape [out_dim * n_words, 8]."""
    out_dim, in_dim = signs_2d.shape
    n_words = (in_dim + 63) // 64
    packed = np.zeros((out_dim, n_words), dtype=np.uint64)
    for j in range(out_dim):
        for wi in range(n_words):
            word = np.uint64(0)
            for bi in range(64):
                idx = wi * 64 + bi
                if idx < in_dim and signs_2d[j, idx] > 0:
                    word |= np.uint64(1) << np.uint64(bi)
            packed[j, wi] = word
    return packed.tobytes()

def main():
    from transformers import GPT2Model
    import torch

    print("[*] loading gpt2...", flush=True)
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    sd = model.state_dict()

    n_layer = model.config.n_layer      # 12
    n_embd = model.config.n_embd        # 768
    mlp_dim = model.config.n_embd * 4   # 3072
    vocab = model.config.vocab_size     # 50257
    n_ctx = model.config.n_ctx          # 1024

    out_path = os.path.join(REPO_ROOT, "prebuilt", "gpt2_binary.bin")
    
    with open(out_path, "wb") as f:
        # Magic + header
        f.write(b"GB2L")
        f.write(struct.pack("IIIII", n_layer, n_embd, mlp_dim, vocab, n_ctx))
        
        # === Embeddings (float) ===
        wte = sd["wte.weight"].numpy().astype(np.float32)  # [vocab, n_embd]
        wpe = sd["wpe.weight"].numpy().astype(np.float32)  # [n_ctx, n_embd]
        ln_f_w = sd["ln_f.weight"].numpy().astype(np.float32)
        ln_f_b = sd["ln_f.bias"].numpy().astype(np.float32)
        
        f.write(wte.tobytes())
        f.write(wpe.tobytes())
        f.write(ln_f_w.tobytes())
        f.write(ln_f_b.tobytes())
        
        # === Per-layer binary weights ===
        for l in range(n_layer):
            # LayerNorm weights (float)
            for key in [f"h.{l}.ln_1.weight", f"h.{l}.ln_1.bias",
                        f"h.{l}.ln_2.weight", f"h.{l}.ln_2.bias"]:
                w = sd[key].numpy().astype(np.float32)
                f.write(w.tobytes())
            
            # Binary matrices
            for key, in_dim, out_dim in [
                (f"h.{l}.attn.c_attn.weight", n_embd, 3*n_embd),
                (f"h.{l}.attn.c_proj.weight", n_embd, n_embd),
                (f"h.{l}.mlp.c_fc.weight",    n_embd, mlp_dim),
                (f"h.{l}.mlp.c_proj.weight",  mlp_dim, n_embd),
            ]:
                W = sd[key].numpy().astype(np.float32)  # [in_dim, out_dim] (GPT-2 Conv1D)
                W_t = W.T  # [out_dim, in_dim] — we want row-major per output
                
                # Binarize: sign(w) + alpha (per-output mean|w|)
                signs = (W_t > 0).astype(np.int8)  # [out_dim, in_dim]
                alpha = np.abs(W_t).mean(axis=1).astype(np.float32)  # [out_dim]
                
                # Bias
                bias_key = key.replace(".weight", ".bias")
                bias = sd[bias_key].numpy().astype(np.float32)  # [out_dim]
                
                n_words = (in_dim + 63) // 64
                f.write(struct.pack("IIII", out_dim, in_dim, n_words, 0))
                f.write(pack_sign_bits(signs))
                f.write(alpha.tobytes())
                f.write(bias.tobytes())
    
    size = os.path.getsize(out_path) / 1e6
    print(f"[*] wrote {out_path} ({size:.1f} MB)")
    
    # Verify
    with open(out_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GB2L", f"bad magic: {magic}"
        hdr = struct.unpack("IIIII", f.read(20))
        print(f"[*] verify: n_layer={hdr[0]}, n_embd={hdr[1]}, mlp_dim={hdr[2]}, vocab={hdr[3]}, n_ctx={hdr[4]}")
    
    print(f"[*] compression: 498 MB → {size:.1f} MB ({498/size:.1f}x)")

if __name__ == "__main__":
    main()
