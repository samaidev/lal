#!/usr/bin/env python3
"""Generate synthetic word2vec-format embedding files at multiple dimensions.

Generates:
  - embeddings_300d.txt  (word2vec scale, 300-dim)
  - embeddings_768d.txt  (BERT-base scale, 768-dim)

The vectors are synthetic but cluster-structured: words in the same semantic
cluster have higher similarity than words in different clusters. This is enough
to demonstrate the full LAL pipeline (SIMD, quantization, embedding loading) at
real embedding scale.

To load REAL BERT embeddings instead, see export_bert_embeddings.py (requires
the `transformers` library and internet to download the model).
"""
import random
import math

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORDS = {
    # Animals cluster
    "cat": "animal", "dog": "animal", "horse": "animal", "elephant": "animal",
    "lion": "animal", "tiger": "animal", "bird": "animal", "fish": "animal",
    # Vehicles cluster
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "bicycle": "vehicle",
    "train": "vehicle", "airplane": "vehicle", "boat": "vehicle", "ship": "vehicle",
    # Food cluster
    "apple": "food", "banana": "food", "bread": "food", "rice": "food",
    "meat": "food", "milk": "food", "water": "food", "coffee": "food",
    # Tools cluster
    "hammer": "tool", "screwdriver": "tool", "wrench": "tool", "saw": "tool",
    "drill": "tool", "knife": "tool", "scissors": "tool", "axe": "tool",
}


def generate(dim, path):
    random.seed(dim)  # different seed per dim for variety
    clusters = {}
    for cluster in set(WORDS.values()):
        centroid = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x*x for x in centroid))
        centroid = [x / norm for x in centroid]
        clusters[cluster] = centroid

    with open(path, "w") as f:
        f.write(f"{len(WORDS)} {dim}\n")
        for word, cluster in sorted(WORDS.items()):
            centroid = clusters[cluster]
            noise = [random.gauss(0, 0.3) for _ in range(dim)]
            vec = [centroid[i] + noise[i] for i in range(dim)]
            norm = math.sqrt(sum(x*x for x in vec))
            vec = [x / norm for x in vec]
            f.write(word + " " + " ".join(f"{v:.6f}" for v in vec) + "\n")
    print(f"[*] generated {path} with {len(WORDS)} words × {dim} dims")


if __name__ == "__main__":
    generate(300, os.path.join(REPO_ROOT, "prebuilt", "embeddings_300d.txt"))
    generate(768, os.path.join(REPO_ROOT, "prebuilt", "embeddings_768d.txt"))
