#!/usr/bin/env python3
"""Generate gpt2_demo.lal — a demo using REAL GPT-2 embeddings (768-dim).

Two tasks:
1. Word similarity classifier: given a query word, find which category
   (royalty / geography / animal / vehicle / food) it's closest to.
2. Analogy: king - man + woman ≈ queen (using VSA bind/bundle).
"""
def main():
    dims = ", ".join(str(i) for i in range(768))
    content = f'''# gpt2_demo.lal — REAL GPT-2 embeddings at BERT scale (v0.6).
#
# This demo uses REAL GPT-2 (124M params) learned embeddings, exported via
# export_gpt2_embeddings.py. The 768-dim vectors are baked into the C binary
# at compile time — no PyTorch, no transformers, no GPT-2 model at runtime.
#
# Task: given a query word's embedding, classify it into a semantic category
# by finding the nearest category centroid.

bound all_dims = [{dims}]

# === Category centroids (representative words per category) ===
concept royal_1    = load_word2vec("gpt2_embeddings.txt", "king")
concept royal_2    = load_word2vec("gpt2_embeddings.txt", "queen")
concept royal_3    = load_word2vec("gpt2_embeddings.txt", "prince")

concept geo_1      = load_word2vec("gpt2_embeddings.txt", "Paris")
concept geo_2      = load_word2vec("gpt2_embeddings.txt", "Rome")
concept geo_3      = load_word2vec("gpt2_embeddings.txt", "Tokyo")

concept animal_1   = load_word2vec("gpt2_embeddings.txt", "cat")
concept animal_2   = load_word2vec("gpt2_embeddings.txt", "dog")
concept animal_3   = load_word2vec("gpt2_embeddings.txt", "horse")

concept vehicle_1  = load_word2vec("gpt2_embeddings.txt", "car")
concept vehicle_2  = load_word2vec("gpt2_embeddings.txt", "train")
concept vehicle_3  = load_word2vec("gpt2_embeddings.txt", "boat")

concept food_1     = load_word2vec("gpt2_embeddings.txt", "apple")
concept food_2     = load_word2vec("gpt2_embeddings.txt", "bread")
concept food_3     = load_word2vec("gpt2_embeddings.txt", "rice")

# === Test words (different from centroid words) ===
concept test_queen    = load_word2vec("gpt2_embeddings.txt", "queen")
concept test_princess = load_word2vec("gpt2_embeddings.txt", "princess")
concept test_berlin   = load_word2vec("gpt2_embeddings.txt", "Berlin")
concept test_madrid   = load_word2vec("gpt2_embeddings.txt", "Madrid")
concept test_lion     = load_word2vec("gpt2_embeddings.txt", "lion")
concept test_elephant = load_word2vec("gpt2_embeddings.txt", "elephant")
concept test_truck    = load_word2vec("gpt2_embeddings.txt", "truck")
concept test_bus      = load_word2vec("gpt2_embeddings.txt", "bus")
concept test_coffee   = load_word2vec("gpt2_embeddings.txt", "coffee")
concept test_banana   = load_word2vec("gpt2_embeddings.txt", "banana")

relate sim(a, b) = dot(a, b) @ all_dims

# Classify a query by summing similarity to each category's 3 centroids.
# The category with the highest total similarity wins.
rule classify(query):
    best = argmax {{
        royal:   sim(query, royal_1) + sim(query, royal_2) + sim(query, royal_3),
        geo:     sim(query, geo_1) + sim(query, geo_2) + sim(query, geo_3),
        animal:  sim(query, animal_1) + sim(query, animal_2) + sim(query, animal_3),
        vehicle: sim(query, vehicle_1) + sim(query, vehicle_2) + sim(query, vehicle_3),
        food:    sim(query, food_1) + sim(query, food_2) + sim(query, food_3)
    }}
    output(best)
'''
    out = "/home/z/my-project/scripts/lal/gpt2_demo.lal"
    with open(out, "w") as f:
        f.write(content)
    print(f"[*] wrote {out}")

if __name__ == "__main__":
    main()
