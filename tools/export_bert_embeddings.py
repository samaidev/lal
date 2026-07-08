#!/usr/bin/env python3
"""Export real BERT embeddings to word2vec text format for LAL.

This script loads a real BERT model (bert-base-uncased, 768-dim) and exports
embeddings for a list of words to word2vec text format. The resulting file can
be loaded by LAL via:

    concept word = load_word2vec("bert_embeddings.txt", "word")

Requirements:
    pip install transformers torch

Usage:
    python3 export_bert_embeddings.py
    # → writes bert_embeddings.txt (768-dim, word2vec text format)

Note: BERT is contextual, so "bank" (river) and "bank" (money) get different
vectors depending on context. For single-word embedding, we use the [CLS] token
embedding of "[word]" as a static approximation. This is a known limitation;
for production use, consider sentence-transformers or BERT's averaged token
embeddings.
"""
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from transformers import BertTokenizer, BertModel
        import torch
    except ImportError:
        print("[!] This script requires `transformers` and `torch`.")
        print("    Install:  pip install transformers torch")
        print("    Then re-run.")
        print()
        print("Falling back to synthetic 768-dim embeddings (run gen_embeddings.py).")
        sys.exit(1)

    WORDS = [
        "cat", "dog", "horse", "elephant", "lion", "tiger", "bird", "fish",
        "car", "truck", "bus", "bicycle", "train", "airplane", "boat", "ship",
        "apple", "banana", "bread", "rice", "meat", "milk", "water", "coffee",
        "hammer", "screwdriver", "wrench", "saw", "drill", "knife", "scissors", "axe",
    ]

    print("[*] loading bert-base-uncased...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    DIM = 768
    out_path = os.path.join(REPO_ROOT, "scripts", "lal", "bert_embeddings.txt")

    with open(out_path, "w") as f:
        f.write(f"{len(WORDS)} {DIM}\n")
        with torch.no_grad():
            for word in WORDS:
                # Tokenize the single word and take the [CLS] embedding.
                inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
                outputs = model(**inputs)
                # Use mean-pooled last hidden state as the word embedding.
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()
                # Normalize to unit length
                import math
                norm = math.sqrt(sum(x*x for x in emb))
                emb = [x / norm for x in emb]
                f.write(word + " " + " ".join(f"{v:.6f}" for v in emb) + "\n")
                print(f"  [{word}] ok")

    print(f"\n[*] wrote {out_path} ({len(WORDS)} words × {DIM} dims)")

if __name__ == "__main__":
    main()
