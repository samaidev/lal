#!/usr/bin/env python3
"""Export GPT-2 binary weights with logic-guided binarization (GB2L v2).

Extends the GB2L format with per-matrix logic_mask:
  - For each binary matrix, adds a logic_mask section after alpha+bias
  - logic_mask: out_dim bytes, each 0(CORE)/1(BINARY)/2(PRUNE)
  - CORE weights: stored as float after the mask (for w_core)
  - BINARY weights: sign bits in wbits (as before)
  - PRUNE weights: omitted (zero)

Format (GB2L2):
  Magic: "GB2L2" (5 bytes, was "GB2L" 4 bytes)
  Header: n_layer, n_embd, mlp_dim, vocab, n_ctx (5 × 4B)
  Embeddings (float): wte + wpe + ln_f
  Per layer:
    LayerNorm (float): ln_1_w/b, ln_2_w/b
    Per matrix (4 per layer):
      Header: out_dim, in_dim, n_words, n_core (4 × 4B)
      logic_mask: out_dim bytes
      wbits: (out_dim - n_core) × n_words uint64s (BINARY only, PRUNE excluded)
      alpha: (out_dim - n_core - n_prune) floats (BINARY only)
      bias: out_dim floats (all, PRUNE bias=0)
      w_core: n_core × in_dim floats (CORE only)
"""
import struct, sys, os
import numpy as np

FLOAT_PATH = "/home/z/my-project/prebuilt/gpt2_weights.bin"
OUT_PATH = "/home/z/my-project/prebuilt/gpt2_binary_logic.bin"

N_LAYER = 12
N_EMBD = 768
MLP_DIM = 3072
VOCAB = 50257
N_CTX = 1024

# Logic mask thresholds (based on analysis)
CORE_FRAC = 0.20   # top 20% norm → CORE (keep float)
PRUNE_FRAC = 0.10  # bottom 10% norm → PRUNE (zero)
# Middle 70% → BINARY (sign+alpha)

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

def compute_logic_mask(W_t, out_dim):
    """Compute per-output logic mask based on weight norms.
    W_t: [out_dim, in_dim] (transposed, per-output rows)
    Returns: uint8 array [out_dim], 0=CORE, 1=BINARY, 2=PRUNE
    """
    norms = np.linalg.norm(W_t, axis=1)  # [out_dim]
    sorted_norms = np.sort(norms)
    core_threshold = sorted_norms[int(out_dim * (1 - CORE_FRAC))]
    prune_threshold = sorted_norms[int(out_dim * PRUNE_FRAC)]
    
    mask = np.ones(out_dim, dtype=np.uint8)  # default BINARY
    mask[norms >= core_threshold] = 0  # CORE
    mask[norms <= prune_threshold] = 2  # PRUNE
    return mask

def pack_sign_bits(signs_2d):
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
    return packed

def main():
    print(f"[*] loading {FLOAT_PATH}...")
    tensors = load_gpw2(FLOAT_PATH)
    
    with open(OUT_PATH, "wb") as f:
        # Magic + header (GB2L2 = logic-guided)
        f.write(b"GB2L2")
        f.write(struct.pack("IIIII", N_LAYER, N_EMBD, MLP_DIM, VOCAB, N_CTX))
        
        # Embeddings (float)
        wte = tensors["wte.weight"]
        wpe = tensors["wpe.weight"]
        f.write(wte.tobytes())
        f.write(wpe.tobytes())
        f.write(tensors["ln_f.weight"].tobytes())
        f.write(tensors["ln_f.bias"].tobytes())
        
        total_core = 0
        total_binary = 0
        total_prune = 0
        
        for l in range(N_LAYER):
            # LayerNorm (float)
            for key in [f"h.{l}.ln_1.weight", f"h.{l}.ln_1.bias",
                        f"h.{l}.ln_2.weight", f"h.{l}.ln_2.bias"]:
                f.write(tensors[key].tobytes())
            
            # 4 binary matrices per layer
            for key, in_dim, out_dim in [
                (f"h.{l}.attn.c_attn.weight", N_EMBD, 3*N_EMBD),
                (f"h.{l}.attn.c_proj.weight", N_EMBD, N_EMBD),
                (f"h.{l}.mlp.c_fc.weight",    N_EMBD, MLP_DIM),
                (f"h.{l}.mlp.c_proj.weight",  MLP_DIM, N_EMBD),
            ]:
                W = tensors[key]  # [in_dim, out_dim] (GPT-2 Conv1D)
                W_t = W.T  # [out_dim, in_dim]
                bias = tensors[key.replace(".weight", ".bias")]
                
                # Compute logic mask
                mask = compute_logic_mask(W_t, out_dim)
                n_core = int(np.sum(mask == 0))
                n_binary = int(np.sum(mask == 1))
                n_prune = int(np.sum(mask == 2))
                total_core += n_core
                total_binary += n_binary
                total_prune += n_prune
                
                n_words = (in_dim + 63) // 64
                
                # Header
                f.write(struct.pack("IIII", out_dim, in_dim, n_words, n_core))
                
                # logic_mask
                f.write(mask.tobytes())
                
                # wbits: only BINARY outputs
                binary_signs = np.zeros((n_binary, in_dim), dtype=np.float32)
                binary_idx = 0
                for j in range(out_dim):
                    if mask[j] == 1:  # BINARY
                        binary_signs[binary_idx] = (W_t[j] > 0).astype(np.float32)
                        binary_idx += 1
                packed = pack_sign_bits(binary_signs)
                f.write(packed.tobytes())
                
                # alpha: only BINARY outputs
                for j in range(out_dim):
                    if mask[j] == 1:
                        alpha = np.abs(W_t[j]).mean()
                        f.write(struct.pack("f", float(alpha)))
                
                # bias: all outputs (PRUNE bias=0)
                for j in range(out_dim):
                    if mask[j] == 2:
                        f.write(struct.pack("f", 0.0))
                    else:
                        f.write(struct.pack("f", float(bias[j])))
                
                # w_core: CORE outputs (float weights)
                for j in range(out_dim):
                    if mask[j] == 0:
                        f.write(W_t[j].tobytes())
            
            if l < 3:
                print(f"  layer {l}: core={total_core} binary={total_binary} prune={total_prune}")
        
        size = os.path.getsize(OUT_PATH) / 1e6
        print(f"\n[*] wrote {OUT_PATH} ({size:.1f} MB)")
        print(f"[*] total: CORE={total_core} BINARY={total_binary} PRUNE={total_prune}")
        print(f"[*] compression: 498 MB → {size:.1f} MB ({498/size:.1f}x)")

if __name__ == "__main__":
    main()
