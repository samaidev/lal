#!/usr/bin/env python3
"""Export real GPT-2 token embeddings to word2vec text format for LAL.

Loads gpt2 (124M parameters, 768-dim embeddings) and exports the input
embeddings for a curated word list. These are REAL learned embeddings from
GPT-2's pretraining on WebText (~40GB of internet text).

The resulting file can be loaded by LAL via:
    concept king = load_word2vec("gpt2_embeddings.txt", "king")

This lets LAL-compiled C code do real NLP using GPT-2's semantic knowledge.
"""
import sys
import math

def main():
    from transformers import GPT2Tokenizer, GPT2Model
    import torch

    print("[*] loading gpt2 (124M params, 768-dim)...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    # GPT-2's token embedding matrix: [vocab_size, 768]
    # vocab_size = 50257
    embed_weights = model.wte.weight  # [50257, 768]
    print(f"[*] embedding matrix: {embed_weights.shape}", flush=True)

    # Curated word list for demos: capitals, countries, analogies, sentiment
    WORDS = [
        # Royalty / gender (for king - man + woman = queen analogy)
        "king", "queen", "man", "woman", "prince", "princess", "royal",
        # Geography
        "France", "Paris", "Italy", "Rome", "Japan", "Tokyo", "Germany", "Berlin",
        "Spain", "Madrid", "Russia", "Moscow", "China", "Beijing",
        # Animals
        "cat", "dog", "horse", "bird", "fish", "lion", "tiger", "elephant",
        # Vehicles
        "car", "truck", "bus", "train", "plane", "boat", "ship", "bicycle",
        # Food
        "apple", "bread", "rice", "meat", "milk", "water", "coffee", "banana",
        # Tools
        "hammer", "knife", "wrench", "saw", "drill", "scissors",
        # Sentiment
        "good", "great", "excellent", "bad", "terrible", "awful", "happy", "sad",
        "love", "hate", "perfect", "horrible", "wonderful", "disappointing",
        # Common words for similarity tests
        "the", "is", "are", "was", "were", "big", "small", "fast", "slow",
        "hot", "cold", "up", "down", "left", "right", "yes", "no",
    ]

    # GPT-2 uses BPE tokenization. For single common words, the word is usually
    # a single token (with a leading space). We look up each word's token.
    DIM = 768
    out_path = "/home/z/my-project/prebuilt/gpt2_embeddings.txt"

    exported = []
    for word in WORDS:
        # GPT-2 tokenizes " king" (with leading space) as the word token.
        # Try both with and without leading space.
        for variant in [" " + word, word]:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if len(tokens) == 1:
                token_id = tokens[0]
                vec = embed_weights[token_id].tolist()
                # Normalize to unit length for cosine similarity
                norm = math.sqrt(sum(x*x for x in vec))
                if norm > 1e-9:
                    vec = [x / norm for x in vec]
                exported.append((word, vec))
                break
        else:
            print(f"  [skip] {word!r} — not a single GPT-2 token", flush=True)

    with open(out_path, "w") as f:
        f.write(f"{len(exported)} {DIM}\n")
        for word, vec in exported:
            f.write(word + " " + " ".join(f"{v:.6f}" for v in vec) + "\n")
    print(f"\n[*] wrote {out_path} ({len(exported)} words × {DIM} dims)")
    print(f"[*] These are REAL GPT-2 learned embeddings (not synthetic).")

    # Quick sanity check: king should be more similar to queen than to car
    import numpy as np
    emap = {w: np.array(v) for w, v in exported}
    king = emap["king"]
    print("\n[*] sanity check — cosine similarity to 'king':")
    for w in ["queen", "prince", "man", "car", "apple"]:
        sim = float(king @ emap[w])
        print(f"    king · {w:10s} = {sim:.4f}")

if __name__ == "__main__":
    main()
