#!/usr/bin/env python3
"""Generate embed_768d.lal — a 768-dim BERT-scale LAL demo.

Writes the .lal file with a `bound all_dims = [0, 1, 2, ..., 767]` line.
"""
def main():
    dims = ", ".join(str(i) for i in range(768))
    content = f'''# embed_768d.lal — 768-dim BERT-scale demo (v0.5).
#
# Loads 768-dimensional vectors (BERT-base hidden size) from embeddings_768d.txt.
# This is the same scale as a real BERT embedding. Compile with:
#   python3 lal.py embed_768d.lal classify embed_768d.c
#   python3 lal.py embed_768d.lal classify embed_768d_q8.c --quantize int8
#   python3 lal.py embed_768d.lal classify embed_768d_q4.c --quantize int4
#
# To use REAL BERT embeddings, run export_bert_embeddings.py first (requires
# the `transformers` library) and change the file path below.

bound all_dims = [{dims}]

concept animal  = load_word2vec("embeddings_768d.txt", "cat")
concept vehicle = load_word2vec("embeddings_768d.txt", "car")
concept food    = load_word2vec("embeddings_768d.txt", "apple")
concept tool    = load_word2vec("embeddings_768d.txt", "hammer")

concept test_dog    = load_word2vec("embeddings_768d.txt", "dog")
concept test_truck  = load_word2vec("embeddings_768d.txt", "truck")
concept test_banana = load_word2vec("embeddings_768d.txt", "banana")
concept test_knife  = load_word2vec("embeddings_768d.txt", "knife")

relate sim(a, b) = dot(a, b) @ all_dims

rule classify(query):
    best = argmax {{
        animal:  sim(query, animal),
        vehicle: sim(query, vehicle),
        food:    sim(query, food),
        tool:    sim(query, tool)
    }}
    output(best)
'''
    out = "/home/z/my-project/scripts/lal/embed_768d.lal"
    with open(out, "w") as f:
        f.write(content)
    print(f"[*] wrote {out}")

if __name__ == "__main__":
    main()
