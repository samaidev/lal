#!/usr/bin/env python3
"""Export complete GPT-2 (weights + BPE tokenizer) for C inference.

Outputs two files:
  gpt2_full.bin    — all weight tensors (GPW2 format, same as before)
  gpt2_tokenizer.bin — BPE tokenizer (vocab + merges + byte mapping)

The tokenizer file format:
  Header: magic "GBT2", vocab_size(4B), n_merges(4B), n_ctx(4B), n_layer(4B), n_embd(4B)
  Byte-to-unicode mapping: 256 entries, each (byte_val(1B) + unicode_str_len(2B) + str)
  Merges: n_merges entries, each (rank(4B), len_a(2B), a_bytes, len_b(2B), b_bytes)
    where a/b are the unicode-string representations of the merge tokens
  Vocab: vocab_size entries, each (token_id(4B), len(2B), token_bytes)
    where token_bytes are the actual byte representation (via byte_decoder)
"""
import struct
import json
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def bytes_to_unicode():
    """GPT-2's byte-to-unicode mapping. Returns dict: byte_val -> unicode_char."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

def main():
    from transformers import GPT2Model, GPT2Tokenizer
    import torch

    print("[*] loading gpt2...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    sd = model.state_dict()

    n_layer = model.config.n_layer      # 12
    n_embd = model.config.n_embd        # 768
    n_ctx = model.config.n_ctx           # 1024
    vocab = model.config.vocab_size      # 50257

    # === Export weights (GPW2 format, same as before) ===
    weight_path = os.path.join(REPO_ROOT, "prebuilt", "gpt2_weights.bin")
    with open(weight_path, "wb") as f:
        f.write(b"GPW2")
        f.write(struct.pack("I", len(sd)))
        for key, tensor in sd.items():
            w = tensor.numpy().astype(np.float32)
            kb = key.encode("utf-8")
            f.write(struct.pack("I", len(kb)))
            f.write(kb)
            f.write(struct.pack("I", w.ndim))
            for d in w.shape:
                f.write(struct.pack("I", d))
            f.write(w.tobytes())
    import os
    print(f"[*] weights: {weight_path} ({os.path.getsize(weight_path)/1e6:.1f} MB)")

    # === Export tokenizer ===
    tok_path = os.path.join(REPO_ROOT, "prebuilt", "gpt2_tokenizer.bin")
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    # Get merges in rank order
    merges = list(tokenizer._merges)  # list of (a, b) tuples, ranked
    n_merges = len(merges)

    # Get vocab: token_str -> token_id
    vocab_dict = tokenizer._vocab  # dict: str -> id

    with open(tok_path, "wb") as f:
        f.write(b"GBT2")
        f.write(struct.pack("IIIII", vocab, n_merges, n_ctx, n_layer, n_embd))

        # Byte-to-unicode mapping (256 entries)
        for byte_val in range(256):
            uni_char = b2u[byte_val]
            uni_bytes = uni_char.encode("utf-8")
            f.write(struct.pack("B", byte_val))
            f.write(struct.pack("H", len(uni_bytes)))
            f.write(uni_bytes)

        # Merges (rank order)
        for rank, (a, b) in enumerate(merges):
            a_bytes = a.encode("utf-8")
            b_bytes = b.encode("utf-8")
            f.write(struct.pack("I", rank))
            f.write(struct.pack("H", len(a_bytes)))
            f.write(a_bytes)
            f.write(struct.pack("H", len(b_bytes)))
            f.write(b_bytes)

        # Vocab: token_id -> byte representation
        # Build reverse: id -> token_str
        id_to_token = {v: k for k, v in vocab_dict.items()}
        for token_id in range(vocab):
            if token_id in id_to_token:
                token_str = id_to_token[token_id]
                # Convert unicode string to actual bytes via u2b
                try:
                    token_bytes = bytes(u2b[c] for c in token_str)
                except KeyError:
                    token_bytes = token_str.encode("utf-8", errors="replace")
            else:
                token_bytes = b""
            f.write(struct.pack("I", token_id))
            f.write(struct.pack("H", len(token_bytes)))
            f.write(token_bytes)

    print(f"[*] tokenizer: {tok_path} ({os.path.getsize(tok_path)/1e6:.1f} MB)")
    print(f"[*] vocab={vocab} merges={n_merges} ctx={n_ctx} layers={n_layer} embd={n_embd}")

if __name__ == "__main__":
    main()
