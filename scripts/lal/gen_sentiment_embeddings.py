#!/usr/bin/env python3
"""Generate a 768-dim sentiment embedding file for LAL NLP demo.

Creates word vectors where positive words cluster together and negative words
cluster together. This mimics what a real BERT model would produce: words with
similar sentiment have similar embeddings.

The LAL program classifies a query sentence (represented as a bag-of-words
average embedding) as positive or negative.
"""
import random
import math

# Positive and negative word lists
POSITIVE_WORDS = [
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "happy",
    "love", "perfect", "beautiful", "awesome", "brilliant", "delightful", "joy",
    "best", "superb", "lovely", "pleasant", "magnificent", "splendid",
]
NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "horrible", "hate", "ugly", "worst", "sad",
    "disgusting", "pathetic", "miserable", "disappointing", "poor", "dreadful",
    "nasty", "unpleasant", "inferior", "lousy", "appalling", "wretched",
]

DIM = 768

def main():
    random.seed(123)
    # Generate two centroids: one for positive, one for negative.
    pos_centroid = [random.gauss(0, 1) for _ in range(DIM)]
    neg_centroid = [random.gauss(0, 1) for _ in range(DIM)]
    # Ensure they're far apart (different signs on first few dims)
    for i in range(50):
        neg_centroid[i] = -abs(neg_centroid[i])
        pos_centroid[i] = abs(pos_centroid[i])
    # Normalize
    for c in (pos_centroid, neg_centroid):
        norm = math.sqrt(sum(x*x for x in c))
        for i in range(len(c)):
            c[i] /= norm

    all_words = [(w, "pos", pos_centroid) for w in POSITIVE_WORDS] + \
                [(w, "neg", neg_centroid) for w in NEGATIVE_WORDS]

    out = "/home/z/my-project/scripts/lal/sentiment_embeddings.txt"
    with open(out, "w") as f:
        f.write(f"{len(all_words)} {DIM}\n")
        for word, label, centroid in sorted(all_words):
            noise = [random.gauss(0, 0.4) for _ in range(DIM)]
            vec = [centroid[i] + noise[i] for i in range(DIM)]
            norm = math.sqrt(sum(x*x for x in vec))
            vec = [x / norm for x in vec]
            f.write(word + " " + " ".join(f"{v:.6f}" for v in vec) + "\n")
    print(f"[*] generated {out} with {len(all_words)} words × {DIM} dims")
    print(f"    {len(POSITIVE_WORDS)} positive, {len(NEGATIVE_WORDS)} negative")

if __name__ == "__main__":
    main()
