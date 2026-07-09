#!/usr/bin/env python3
"""Benchmark PyTorch GPT-2 float32 CPU inference for comparison."""
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.set_num_threads(2)  # match samcommand 2-core

print("[*] loading gpt2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompts = [
    "The capital of France is",
    "Once upon a time",
    "Hello, how are",
    "Machine learning is",
]
n_tokens = 20

print(f"[*] torch {torch.__version__}, threads={torch.get_num_threads()}")
print(f"[*] generating {n_tokens} tokens per prompt\n")

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=n_tokens, do_sample=False, pad_token_id=50256)
    dt = time.time() - t0
    tps = n_tokens / dt
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  {prompt}: {tps:.1f} tok/s | {text[len(prompt):len(prompt)+60]}")

print("\n[*] done")
