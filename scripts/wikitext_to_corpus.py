#!/usr/bin/env python3
"""wikitext_to_corpus.py — extract wikitext-2 parquet -> corpus.txt for
build_corpus.c (one sentence per line, 8-25 words so it tokenizes to <=32 BPE tokens).
"""
import os, re, sys
import pyarrow.parquet as pq

IN = sys.argv[1] if len(sys.argv) > 1 else "/workspace/lal/build/wikitext2.parquet"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/lal/build/corpus_wiki.txt"

t = pq.read_table(IN)
col = t.column("text").to_pylist()
print(f"[*] {len(col)} rows in parquet", file=sys.stderr)

seen = set()
out = []
for para in col:
    if not para:
        continue
    # wikitext-raw paragraphs are separated by \n; split on sentence boundaries.
    # Replace newlines with spaces, then split on . ! ? followed by space+capital.
    text = para.replace("\n", " ").strip()
    # Simple sentence split: keep punctuation, split on ". " etc.
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    for s in parts:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        # 8-25 words -> ~10-35 BPE tokens (most <= 32)
        if len(words) < 8 or len(words) > 25:
            continue
        # Dedup; wikitext has many repeated headers
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        # Skip section headers (= Title =), list items, etc.
        if s.startswith("=") or s.startswith("*") or s.startswith("#"):
            continue
        if "==" in s or s.count("=") > 2:
            continue
        out.append(s)

print(f"[*] {len(out)} sentences (8-25 words) after filter", file=sys.stderr)
# Cap at 2000 sentences for training time (300-1000 steps)
if len(out) > 2000:
    out = out[:2000]
    print(f"[*] capped to 2000", file=sys.stderr)

with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"[*] wrote {len(out)} sentences to {OUT} ({os.path.getsize(OUT)} bytes)", file=sys.stderr)
