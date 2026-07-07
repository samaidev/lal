#!/usr/bin/env python3
"""Generate gpt2_analogy.lal — the classic king-man+woman=queen analogy
using REAL GPT-2 embeddings, compiled to C by LAL.

Uses vadd/vsub (vector element-wise add/subtract) to compute:
    analogy = king - man + woman
then finds which candidate word is closest.
"""
def main():
    dims = ", ".join(str(i) for i in range(768))
    content = f'''# gpt2_analogy.lal — king - man + woman = queen, with REAL GPT-2 (v0.6).
#
# Uses real GPT-2 (124M) learned embeddings. The analogy vector is computed
# at runtime via vsub/vadd (element-wise vector ops), then compared to
# candidate words via dot product.
#
# This proves LAL can do real semantic reasoning with GPT-2's knowledge,
# compiled to standalone C with no runtime dependencies.

bound all_dims = [{dims}]

# Source words for the analogy
concept king   = load_word2vec("gpt2_embeddings.txt", "king")
concept man    = load_word2vec("gpt2_embeddings.txt", "man")
concept woman  = load_word2vec("gpt2_embeddings.txt", "woman")

# Candidate answers
concept queen     = load_word2vec("gpt2_embeddings.txt", "queen")
concept prince    = load_word2vec("gpt2_embeddings.txt", "prince")
concept princess  = load_word2vec("gpt2_embeddings.txt", "princess")
concept royal     = load_word2vec("gpt2_embeddings.txt", "royal")
concept car       = load_word2vec("gpt2_embeddings.txt", "car")
concept apple     = load_word2vec("gpt2_embeddings.txt", "apple")

# Vector arithmetic: king - man + woman
relate sub_man(a, b)  = vsub(a, b)
relate add_woman(a, b) = vadd(a, b)
relate sim(a, b) = dot(a, b) @ all_dims

# Compute the analogy vector, then find nearest candidate.
# Input: the king vector (we treat it as the query; man/woman are fixed).
rule solve(query):
    # query is king. Compute king - man.
    km = sub_man(query, man)
    # Compute (king - man) + woman.
    kmw = add_woman(km, woman)
    # Find which candidate is closest.
    best = argmax {{
        queen:    sim(kmw, queen),
        prince:   sim(kmw, prince),
        princess: sim(kmw, princess),
        royal:    sim(kmw, royal),
        car:      sim(kmw, car),
        apple:    sim(kmw, apple)
    }}
    output(best)
'''
    out = "/home/z/my-project/scripts/lal/gpt2_analogy.lal"
    with open(out, "w") as f:
        f.write(content)
    print(f"[*] wrote {out}")

if __name__ == "__main__":
    main()
