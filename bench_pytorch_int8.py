#!/usr/bin/env python3
"""bench_pytorch_int8.py — PyTorch dynamic int8 quantization benchmark.
This approximates llama.cpp Q8 performance (dynamic per-channel int8)."""
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.set_num_threads(2)

print("[*] loading gpt2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Float32 baseline
print("[*] float32 benchmark:")
prompts = ["The capital of France is", "Once upon a time", "Hello, how are"]
for prompt in prompts:
    ids = tokenizer.encode(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=50256)
    dt = time.time() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  {prompt}: {20/dt:.1f} tok/s | {text[len(prompt):len(prompt)+60]}")

# Dynamic int8 quantization (per-channel weight, similar to llama.cpp Q8)
print("\n[*] dynamic int8 quantization...")
quantized_model = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_model.eval()
print(f"  quantized params: {sum(p.numel() for p in quantized_model.parameters())/1e6:.1f}M")

print("[*] int8 benchmark:")
for prompt in prompts:
    ids = tokenizer.encode(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = quantized_model.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=50256)
    dt = time.time() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  {prompt}: {20/dt:.1f} tok/s | {text[len(prompt):len(prompt)+60]}")

print("\n[*] done")
