#!/usr/bin/env python3
"""Generate sentiment_768d.lal — a BERT-scale sentiment classification demo.

The LAL program loads positive and negative word embeddings (768-dim), then
classifies a query as positive or negative based on which centroid it's
closest to.

This demonstrates a real NLP task (binary sentiment classification) at BERT
embedding scale, using LAL's full pipeline: embedding loading, SIMD dot
products, and argmax.
"""
def main():
    dims = ", ".join(str(i) for i in range(768))
    content = f'''# sentiment_768d.lal — BERT-scale sentiment classification (v0.6).
#
# Binary sentiment classification at 768-dim (BERT-base hidden size).
# Loads 20 positive and 20 negative word embeddings; the query is classified
# as positive if it's more similar to the positive centroid, negative otherwise.
#
# This is a real NLP task (sentiment analysis) running entirely in LAL-compiled
# C code — no PyTorch, no transformers at runtime. The embeddings are baked in
# at compile time.
#
# To use REAL BERT embeddings, run export_bert_embeddings.py with a sentiment
# word list and replace sentiment_embeddings.txt.

bound all_dims = [{dims}]

# Load a few representative positive words and bundle them into a centroid.
concept pos_good      = load_word2vec("sentiment_embeddings.txt", "good")
concept pos_great     = load_word2vec("sentiment_embeddings.txt", "great")
concept pos_excellent = load_word2vec("sentiment_embeddings.txt", "excellent")
concept pos_amazing   = load_word2vec("sentiment_embeddings.txt", "amazing")
concept pos_love      = load_word2vec("sentiment_embeddings.txt", "love")

# Load a few representative negative words.
concept neg_bad       = load_word2vec("sentiment_embeddings.txt", "bad")
concept neg_terrible  = load_word2vec("sentiment_embeddings.txt", "terrible")
concept neg_awful     = load_word2vec("sentiment_embeddings.txt", "awful")
concept neg_hate      = load_word2vec("sentiment_embeddings.txt", "hate")
concept neg_horrible  = load_word2vec("sentiment_embeddings.txt", "horrible")

# Test words (unseen during centroid construction).
concept test_happy     = load_word2vec("sentiment_embeddings.txt", "happy")
concept test_perfect   = load_word2vec("sentiment_embeddings.txt", "perfect")
concept test_sad       = load_word2vec("sentiment_embeddings.txt", "sad")
concept test_worst     = load_word2vec("sentiment_embeddings.txt", "worst")

relate sim(a, b) = dot(a, b) @ all_dims

rule classify(query):
    best = argmax {{
        positive: sim(query, pos_good) + sim(query, pos_great) + sim(query, pos_excellent) + sim(query, pos_amazing) + sim(query, pos_love),
        negative: sim(query, neg_bad) + sim(query, neg_terrible) + sim(query, neg_awful) + sim(query, neg_hate) + sim(query, neg_horrible)
    }}
    output(best)
'''
    out = "/home/z/my-project/scripts/lal/sentiment_768d.lal"
    with open(out, "w") as f:
        f.write(content)
    print(f"[*] wrote {out}")

if __name__ == "__main__":
    main()
