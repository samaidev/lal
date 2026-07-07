#!/usr/bin/env python3
"""Export GPT-2 binary weights from existing float GPW2 file.

Reads prebuilt/gpt2_weights.bin (498 MB float) and produces
prebuilt/gpt2_binary.bin (~13 MB binary: wbits + alpha + bias).

Format (GB2L):
  Magic: "GB2L" (4 bytes)
  Header: n_layer(4), n_embd(4), mlp_dim(4), vocab(4), n_ctx(4)
  
  Embeddings (float, NOT binarized):
    wte: [vocab, n_embd] float32
    wpe: [n_ctx, n_embd] float32
    ln_f.weight: [n_embd] float32
    ln_f.bias:   [n_embd] float32
  
  Per layer (12 layers):
    LayerNorm weights (float):
      ln_1.weight, ln_1.bias, ln_2.weight, ln_2.bias — each [n_embd] float32
    Binary matrices (4 per layer):
      For each [in_dim, out_dim] matrix:
        Header: out_dim(4), in_dim(4), n_words(4), reserved(4)
        wbits:  out_dim * n_words uint64s
        alpha:  out_dim float32s
        bias:   out_dim float32s
"""
import struct, os, sys
import numpy as np

FLOAT_PATH = "/home/z/my-project/prebuilt/gpt2_weights.bin"
OUT_PATH = "/home/z/my-project/prebuilt/gpt2_binary.bin"

N_LAYER = 12
N_EMBD = 768
MLP_DIM = 3072
VOCAB = 50257
N_CTX = 1024

def load_gpw2(path):
    """Load all tensors from a GPW2 file into a dict."""
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

def pack_sign_bits_fast(signs_2d):
    """Pack [out_dim, in_dim] sign array into uint64s, vectorized."""
    out_dim, in_dim = signs_2d.shape
    n_words = (in_dim + 63) // 64
    # Pad to multiple of 64
    padded = np.zeros((out_dim, n_words * 64), dtype=np.uint8)
    padded[:, :in_dim] = (signs_2d > 0).astype(np.uint8)
    # Reshape to [out_dim, n_words, 64] and pack each 64-bit word
    packed = np.zeros((out_dim, n_words), dtype=np.uint64)
    for j in range(out_dim):
        for wi in range(n_words):
            word = np.uint64(0)
            for bi in range(64):
                if padded[j, wi * 64 + bi]:
                    word |= np.uint64(1) << np.uint64(bi)
            packed[j, wi] = word
    return packed.tobytes()

def main():
    print(f"[*] loading {FLOAT_PATH}...")
    tensors = load_gpw2(FLOAT_PATH)
    print(f"[*] loaded {len(tensors)} tensors")
    
    with open(OUT_PATH, "wb") as f:
        f.write(b"GB2L")
        f.write(struct.pack("IIIII", N_LAYER, N_EMBD, MLP_DIM, VOCAB, N_CTX))
        
        # Embeddings (float)
        print("[*] writing embeddings (float)...")
        wte = tensors["wte.weight"]
        wpe = tensors["wpe.weight"]
        ln_f_w = tensors["ln_f.weight"]
        ln_f_b = tensors["ln_f.bias"]
        f.write(wte.tobytes())
        f.write(wpe.tobytes())
        f.write(ln_f_w.tobytes())
        f.write(ln_f_b.tobytes())
        
        # Per-layer
        for l in range(N_LAYER):
            print(f"[*] layer {l}...", end=" ", flush=True)
            # LayerNorm (float)
            for key in [f"h.{l}.ln_1.weight", f"h.{l}.ln_1.bias",
                        f"h.{l}.ln_2.weight", f"h.{l}.ln_2.bias"]:
                w = tensors[key]
                f.write(w.tobytes())
            
            # Binary matrices
            for key, in_dim, out_dim in [
                (f"h.{l}.attn.c_attn.weight", N_EMBD, 3*N_EMBD),
                (f"h.{l}.attn.c_proj.weight", N_EMBD, N_EMBD),
                (f"h.{l}.mlp.c_fc.weight",    N_EMBD, MLP_DIM),
                (f"h.{l}.mlp.c_proj.weight",  MLP_DIM, N_EMBD),
            ]:
                W = tensors[key]  # [in_dim, out_dim] (GPT-2 Conv1D)
                W_t = W.T  # [out_dim, in_dim] — row-major per output
                
                signs = (W_t > 0).astype(np.int8)
                alpha = np.abs(W_t).mean(axis=1).astype(np.float32)
                
                bias_key = key.replace(".weight", ".bias")
                bias = tensors[bias_key]
                
                n_words = (in_dim + 63) // 64
                f.write(struct.pack("IIII", out_dim, in_dim, n_words, 0))
                f.write(pack_sign_bits_fast(signs))
                f.write(alpha.tobytes())
                f.write(bias.tobytes())
            print("done")
    
    size = os.path.getsize(OUT_PATH) / 1e6
    print(f"\n[*] wrote {OUT_PATH} ({size:.1f} MB)")
    print(f"[*] compression: 498 MB → {size:.1f} MB ({498/size:.1f}x)")
    
    # Verify
    with open(OUT_PATH, "rb") as f:
        magic = f.read(4)
        assert magic == b"GB2L"
        hdr = struct.unpack("IIIII", f.read(20))
        print(f"[*] verify header: n_layer={hdr[0]}, n_embd={hdr[1]}, mlp_dim={hdr[2]}, vocab={hdr[3]}, n_ctx={hdr[4]}")

if __name__ == "__main__":
    main()
