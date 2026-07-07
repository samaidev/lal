#!/usr/bin/env python3
"""Quantization-aware fine-tuning of GPT-2 with binary weights (STE).

Uses the Straight-Through Estimator (STE):
  - Forward:  w_bin = sign(w) * alpha  (binary weights)
  - Backward: gradient passes through as if w were continuous

This recovers accuracy lost by binarization. After training, the binary
weights (signs + alpha) are exported for the C runtime.

CPU training: ~10-30s per step on a 64-token sequence. 200 steps ≈ 1-2 hours.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer
import math
import time
import os

class BinaryLinear(nn.Module):
    """Linear layer with binary weights via STE.
    
    Forward:  y = x @ (sign(W) * alpha) + b
    Backward: gradient flows through sign() as identity (STE)
    
    alpha = mean(|W|) per output, updated during training.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store full-precision weights (will be binarized in forward)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # alpha (scale factor) per output — learned
        self.alpha = nn.Parameter(torch.ones(out_features) * 0.1)
        
    def forward(self, x):
        # STE: forward uses sign(w), backward uses gradient of identity
        w_bin = self.weight.sign() * self.alpha.unsqueeze(1)
        # STE trick: w_bin = w + (w_bin - w).detach()
        w_ste = self.weight + (w_bin - self.weight).detach()
        return F.linear(x, w_ste, self.bias)
    
    def get_binary_weights(self):
        """Return (signs [out, in], alpha [out], bias [out]) for C export."""
        signs = (self.weight.data > 0).float() * 2 - 1  # {-1, +1}
        return signs, self.alpha.data, self.bias.data if self.bias is not None else None


def replace_linear_with_binary(model, layer_indices=None):
    """Replace all Linear layers in GPT-2 with BinaryLinear (STE).
    
    GPT-2 uses Conv1D (not Linear), so we need to handle the weight layout.
    GPT-2 Conv1D stores weight as [in, out] (transposed vs nn.Linear).
    """
    for layer_idx in range(12):
        layer = model.h[layer_idx]
        
        # Attention c_attn [768→2304]
        old = layer.attn.c_attn
        bl = BinaryLinear(768, 2304, bias=True)
        # Conv1D weight is [in, out], BinaryLinear weight is [out, in]
        bl.weight.data = old.weight.data.T.clone()
        bl.bias.data = old.bias.data.clone()
        bl.alpha.data = old.weight.data.abs().mean(dim=0).clone()
        layer.attn.c_attn = bl
        
        # Attention c_proj [768→768]
        old = layer.attn.c_proj
        bl = BinaryLinear(768, 768, bias=True)
        bl.weight.data = old.weight.data.T.clone()
        bl.bias.data = old.bias.data.clone()
        bl.alpha.data = old.weight.data.abs().mean(dim=0).clone()
        layer.attn.c_proj = bl
        
        # MLP c_fc [768→3072]
        old = layer.mlp.c_fc
        bl = BinaryLinear(768, 3072, bias=True)
        bl.weight.data = old.weight.data.T.clone()
        bl.bias.data = old.bias.data.clone()
        bl.alpha.data = old.weight.data.abs().mean(dim=0).clone()
        layer.mlp.c_fc = bl
        
        # MLP c_proj [3072→768]
        old = layer.mlp.c_proj
        bl = BinaryLinear(3072, 768, bias=True)
        bl.weight.data = old.weight.data.T.clone()
        bl.bias.data = old.bias.data.clone()
        bl.alpha.data = old.weight.data.abs().mean(dim=0).clone()
        layer.mlp.c_proj = bl
    
    return model


# Patch GPT-2 attention to work with BinaryLinear (which returns [batch, seq, out])
# GPT-2's Conv1D returns [batch, seq, out] but nn.Linear returns [*, in] -> [*, out]
# We need to make the attention forward use F.linear instead of Conv1D's matmul.
# The easiest fix: monkey-patch the Conv1D forward in BinaryLinear.

class BinaryConv1D(BinaryLinear):
    """GPT-2-compatible Conv1D replacement that uses binary weights."""
    def forward(self, x):
        # x: [batch, seq, in] → [batch, seq, out]
        w_bin = self.weight.sign() * self.alpha.unsqueeze(1)
        w_ste = self.weight + (w_bin - self.weight).detach()
        # BinaryLinear weight is [out, in], F.linear does x @ w.T
        # But GPT-2 Conv1D does x @ w (w is [in, out])
        # So we need x @ w_ste.T = F.linear(x, w_ste)
        return F.linear(x, w_ste, self.bias)


def replace_conv1d_with_binary(model):
    """Replace GPT-2 Conv1D layers with BinaryConv1D."""
    for layer_idx in range(12):
        layer = model.h[layer_idx]
        
        for name, (in_f, out_f) in [
            ("c_attn", (768, 2304)),
            ("c_proj", (768, 768)),
        ]:
            old = getattr(layer.attn, name)
            bl = BinaryConv1D(in_f, out_f, bias=True)
            bl.weight.data = old.weight.data.T.clone()  # [in,out] → [out,in]
            bl.bias.data = old.bias.data.clone()
            bl.alpha.data = old.weight.data.abs().mean(dim=0).clone()
            setattr(layer.attn, name, bl)
        
        for name, (in_f, out_f) in [
            ("c_fc", (768, 3072)),
            ("c_proj", (3072, 768)),
        ]:
            old = getattr(layer.mlp, name)
            bl = BinaryConv1D(in_f, out_f, bias=True)
            bl.weight.data = old.weight.data.T.clone()
            bl.bias.data = old.bias.data.clone()
            bl.alpha.data = old.weight.data.abs().mean(dim=0).clone()
            setattr(layer.mlp, name, bl)
    
    return model


# Simple training data: sentences for language modeling
TRAIN_TEXTS = [
    "The capital of France is Paris.",
    "The capital of England is London.",
    "The capital of Japan is Tokyo.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome.",
    "The capital of Spain is Madrid.",
    "The capital of Russia is Moscow.",
    "The capital of China is Beijing.",
    "The capital of Korea is Seoul.",
    "The capital of India is New Delhi.",
    "Hello, how are you doing today?",
    "Hello, how are you doing?",
    "I am fine, thank you for asking.",
    "Once upon a time, there was a great kingdom.",
    "Once upon a time, there was a beautiful princess.",
    "Once upon a time, in a land far away.",
    "The weather today is sunny and warm.",
    "The weather today is cold and rainy.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, there was nothing but darkness.",
    "The world is a place of great wonder and beauty.",
    "Science is the pursuit of knowledge and understanding.",
    "Mathematics is the language of the universe.",
    "The sun rises in the east and sets in the west.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth revolves around the Sun.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The speed of light is approximately 300000 kilometers per second.",
    "I think, therefore I am.",
    "To be, or not to be, that is the question.",
    "All animals are equal, but some animals are more equal than others.",
    "The only thing we have to fear is fear itself.",
    "Ask not what your country can do for you.",
    "I have a dream that one day this nation will rise up.",
    "The unexamined life is not worth living.",
    "Knowledge is power.",
    "A journey of a thousand miles begins with a single step.",
    "The best way to predict the future is to create it.",
] * 10  # repeat to get more data


def main():
    device = torch.device("cpu")
    print(f"[*] device: {device}")
    
    print("[*] loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").to(device)
    
    # Add LM head (weight tying with wte)
    lm_head = nn.Linear(768, 50257, bias=False)
    lm_head.weight = model.wte.weight  # weight tying
    
    # Replace Conv1D with binary versions
    print("[*] replacing weights with binary (STE)...")
    model = replace_conv1d_with_binary(model)
    model = model.to(device)
    lm_head = lm_head.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in lm_head.parameters())
    print(f"[*] parameters: {n_params/1e6:.1f}M")
    
    # Freeze embeddings (only train binary weights + alpha)
    model.wte.weight.requires_grad = False
    model.wpe.weight.requires_grad = False
    for layer in model.h:
        layer.ln_1.weight.requires_grad = False
        layer.ln_1.bias.requires_grad = False
        layer.ln_2.weight.requires_grad = False
        layer.ln_2.bias.requires_grad = False
    model.ln_f.weight.requires_grad = False
    model.ln_f.bias.requires_grad = False
    
    # Only train the binary linear weights + alpha + biases
    trainable = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append(p)
    for name, p in lm_head.named_parameters():
        if p.requires_grad:
            trainable.append(p)
    print(f"[*] trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    
    optimizer = torch.optim.Adam(trainable, lr=1e-4)
    
    # Tokenize training data
    print("[*] tokenizing training data...")
    all_tokens = []
    for text in TRAIN_TEXTS:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    print(f"[*] total tokens: {len(all_tokens)}")
    
    # Training loop
    SEQ_LEN = 32
    BATCH_SIZE = 1
    N_STEPS = 300
    
    print(f"[*] training: {N_STEPS} steps, seq_len={SEQ_LEN}, batch={BATCH_SIZE}")
    print(f"[*] est. time: ~{N_STEPS * 15}s = {N_STEPS * 15 / 60:.0f} min")
    
    model.train()
    losses = []
    
    for step in range(N_STEPS):
        # Sample a random sequence
        start = torch.randint(0, max(1, len(all_tokens) - SEQ_LEN - 1), (1,)).item()
        input_ids = torch.tensor([all_tokens[start:start+SEQ_LEN]], device=device)
        labels = torch.tensor([all_tokens[start+1:start+SEQ_LEN+1]], device=device)
        
        t0 = time.time()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids)
        hidden = outputs.last_hidden_state  # [1, seq_len, 768]
        logits = lm_head(hidden)  # [1, seq_len, 50257]
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits.view(-1, 50257), labels.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        dt = time.time() - t0
        losses.append(loss.item())
        
        if step % 10 == 0:
            avg_loss = sum(losses[-10:]) / max(1, len(losses[-10:]))
            print(f"  step {step:4d}  loss={loss.item():.4f}  avg={avg_loss:.4f}  {dt:.1f}s/step", flush=True)
    
    # Save fine-tuned binary weights
    print("\n[*] exporting binary weights...")
    export_binary_weights(model, "/home/z/my-project/scripts/lal/gpt2_binary_finetuned.bin")
    
    # Test generation with fine-tuned model
    print("\n[*] testing generation...")
    model.eval()
    test_generate(model, lm_head, tokenizer, "The capital of France is", device)
    test_generate(model, lm_head, tokenizer, "Hello, how are", device)
    
    print("\n[*] done!")
    print("[*] fine-tuned binary weights saved to gpt2_binary_finetuned.bin")


def export_binary_weights(model, path):
    """Export binary weights (signs + alpha + bias) to a file.
    
    Format per layer per matrix:
      [out_dim, in_dim] sign bits (packed as uint64s)
      [out_dim] alpha (float32)
      [out_dim] bias (float32)
    """
    import struct
    
    with open(path, "wb") as f:
        f.write(b"GBIN")  # magic
        f.write(struct.pack("II", 12, 768))  # n_layer, n_embd
        
        for layer_idx in range(12):
            layer = model.h[layer_idx]
            for name, module in [
                ("attn.c_attn", layer.attn.c_attn),
                ("attn.c_proj", layer.attn.c_proj),
                ("mlp.c_fc", layer.mlp.c_fc),
                ("mlp.c_proj", layer.mlp.c_proj),
            ]:
                signs, alpha, bias = module.get_binary_weights()
                out_dim, in_dim = signs.shape
                n_words = (in_dim + 63) // 64
                
                f.write(struct.pack("IIII", out_dim, in_dim, n_words, 0))  # header
                
                # Pack signs into uint64s
                sign_bits = (signs > 0).int()  # [out, in] of 0/1
                for j in range(out_dim):
                    for wi in range(n_words):
                        word = 0
                        for bi in range(64):
                            idx = wi * 64 + bi
                            if idx < in_dim and sign_bits[j, idx]:
                                word |= (1 << bi)
                        f.write(struct.pack("Q", word))
                
                # Alpha and bias
                f.write(alpha.cpu().numpy().astype('float32').tobytes())
                f.write(bias.cpu().numpy().astype('float32').tobytes())
    
    size = os.path.getsize(path) / 1e6
    print(f"  wrote {path} ({size:.1f} MB)")


def test_generate(model, lm_head, tokenizer, prompt, device, n_tokens=15):
    """Generate text with the fine-tuned model."""
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    
    print(f"  >>> {prompt}", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(n_tokens):
            outputs = model(input_ids=input_ids)
            hidden = outputs.last_hidden_state[:, -1:, :]  # last token
            logits = lm_head(hidden)  # [1, 1, 50257]
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            input_ids = torch.tensor([tokens], device=device)
    
    generated = tokenizer.decode(tokens)
    print(generated[len(prompt):])


if __name__ == "__main__":
    main()
