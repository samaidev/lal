#!/usr/bin/env python3
"""Generate a synthetic 300-dim word2vec-format embedding file.

We can't download real word2vec (1.5GB), so we generate deterministic pseudo-random
vectors that mimic the structure: words in the same semantic cluster have higher
similarity than words in different clusters. This is enough to demonstrate the
300-dim pipeline (SIMD, quantization, embedding loading) at real scale.
"""
import random
import math

WORDS = {
    # Animals cluster
    "cat":      "animal",
    "dog":      "animal",
    "horse":    "animal",
    "elephant": "animal",
    "lion":     "animal",
    "tiger":    "animal",
    "bird":     "animal",
    "fish":     "animal",
    # Vehicles cluster
    "car":      "vehicle",
    "truck":    "vehicle",
    "bus":      "vehicle",
    "bicycle":  "vehicle",
    "train":    "vehicle",
    "airplane": "vehicle",
    "boat":     "vehicle",
    "ship":     "vehicle",
    # Food cluster
    "apple":    "food",
    "banana":   "food",
    "bread":    "food",
    "rice":     "food",
    "meat":     "food",
    "milk":     "food",
    "water":    "food",
    "coffee":   "food",
    # Tools cluster
    "hammer":   "tool",
    "screwdriver": "tool",
    "wrench":   "tool",
    "saw":      "tool",
    "drill":    "tool",
    "knife":    "tool",
    "scissors": "tool",
    "axe":      "tool",
}

DIM = 300

def main():
    random.seed(42)
    # Generate a cluster centroid for each cluster
    clusters = {}
    for cluster in set(WORDS.values()):
        centroid = [random.gauss(0, 1) for _ in range(DIM)]
        # Normalize
        norm = math.sqrt(sum(x*x for x in centroid))
        centroid = [x / norm for x in centroid]
        clusters[cluster] = centroid

    # Generate word vectors: centroid + noise
    with open("/home/z/my-project/scripts/lal/embeddings_300d.txt", "w") as f:
        f.write(f"{len(WORDS)} {DIM}\n")
        for word, cluster in sorted(WORDS.items()):
            centroid = clusters[cluster]
            noise = [random.gauss(0, 0.3) for _ in range(DIM)]
            vec = [centroid[i] + noise[i] for i in range(DIM)]
            # Normalize to unit length
            norm = math.sqrt(sum(x*x for x in vec))
            vec = [x / norm for x in vec]
            f.write(word + " " + " ".join(f"{v:.6f}" for v in vec) + "\n")
    print(f"[*] generated embeddings_300d.txt with {len(WORDS)} words × {DIM} dims")

if __name__ == "__main__":
    main()
