#!/usr/bin/env python3
"""
mini_train.py — train a TINY real transformer locally (no internet needed) so we
can actually exercise the LAL three-layer fusion LEVEL 1 (per-layer read +
activation steering) on a real neural net.

Why this exists: this sandbox has no internet and no pretrained LLM weights, so
the official qwen_server / gpt2_server (which need Qwen2.5-0.5B / GPT-2 weights)
cannot run here. Instead we train a small char-level transformer on a simple
grammar and reuse the EXACT same integration contract as qwen_server:

    forward loop:  for l in layers: x = layer(l, x); lal_layer_hook(l, x, D)

The C server (tools/mini_server.c) calls lal_layer_hook(layer, hidden, D) the same
way qwen_server.c does, and can dlopen a .lal-compiled .so via --lal-steer.

We also compute a REAL steering direction from the trained model: the mean
residual after a chosen layer for POSITIVE seed prompts minus the mean for
NEGATIVE seed prompts. Adding this direction into the residual at that layer
should push generation toward positive mood words — a measurable, real effect.

Outputs:
  prebuilt/mini_model.bin   weights (custom MINI format, C-loadable)
  prebuilt/mini_model.json  config + vocab
  demos/mini_steer.lal       .lal with the real direction (compiled to .so by build script)
"""
import json, math, os, struct, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, L, H, T = 32, 2, 2, 24          # d_model, n_layers, n_heads, context
DEVICE = "cpu"

# --------------------------------------------------------------------------
# Corpus: a trivially learnable grammar so a tiny model converges fast.
#   S -> "I feel" MOOD TIME "."
#   POSITIVE MOODS: happy, calm, glad     NEGATIVE: sad, angry, blue
#   TIME: today, now, lately, tonight
# --------------------------------------------------------------------------
MOODS_POS = ["happy", "calm", "glad"]
MOODS_NEG = ["sad", "angry", "blue"]
MOODS = MOODS_POS + MOODS_NEG
TIMES = ["today", "now", "lately", "tonight"]

import random
random.seed(1234)

def gen_sentences(n):
    out = []
    for _ in range(n):
        m = random.choice(MOODS)
        t = random.choice(TIMES)
        out.append(f"I feel {m} {t}.")
    return out

SENTENCES = gen_sentences(20000)
CORPUS = "\n".join(SENTENCES) + "\n"

CHARS = sorted(set(CORPUS))
V = len(CHARS)
char2i = {c: i for i, c in enumerate(CHARS)}
i2char = {i: c for c, i in char2i.items()}
print(f"[*] vocab size = {V}, corpus chars = {len(CORPUS)}")

# Build training sequences of length T by sliding over the corpus with step=T
# (non-overlapping) — the grammar is trivial, so far fewer sequences suffice and
# training stays well under a minute on CPU.
data = np.array([char2i[c] for c in CORPUS], dtype=np.int64)
seqs = []
for i in range(0, len(data) - T, T):       # step T -> ~10k non-overlapping seqs
    seqs.append(data[i:i+T+1])
seqs = np.array(seqs, dtype=np.int64)
print(f"[*] training sequences: {len(seqs)}")

# --------------------------------------------------------------------------
# Model (explicit weights, forward written manually so C can match exactly)
# --------------------------------------------------------------------------
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Parameter(torch.randn(V, D) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(T, D) * 0.02)
        self.ln1_w = nn.ParameterList([nn.Parameter(torch.ones(D)) for _ in range(L)])
        self.ln1_b = nn.ParameterList([nn.Parameter(torch.zeros(D)) for _ in range(L)])
        self.attn_q = nn.ParameterList([nn.Parameter(torch.randn(D, D)*0.02) for _ in range(L)])
        self.attn_k = nn.ParameterList([nn.Parameter(torch.randn(D, D)*0.02) for _ in range(L)])
        self.attn_v = nn.ParameterList([nn.Parameter(torch.randn(D, D)*0.02) for _ in range(L)])
        self.attn_o = nn.ParameterList([nn.Parameter(torch.randn(D, D)*0.02) for _ in range(L)])
        self.ln2_w = nn.ParameterList([nn.Parameter(torch.ones(D)) for _ in range(L)])
        self.ln2_b = nn.ParameterList([nn.Parameter(torch.zeros(D)) for _ in range(L)])
        self.mlp_fc_w = nn.ParameterList([nn.Parameter(torch.randn(4*D, D)*0.02) for _ in range(L)])
        self.mlp_fc_b = nn.ParameterList([nn.Parameter(torch.zeros(4*D)) for _ in range(L)])
        self.mlp_proj_w = nn.ParameterList([nn.Parameter(torch.randn(D, 4*D)*0.02) for _ in range(L)])
        self.mlp_proj_b = nn.ParameterList([nn.Parameter(torch.zeros(D)) for _ in range(L)])
        self.ln_f_w = nn.Parameter(torch.ones(D))
        self.ln_f_b = nn.Parameter(torch.zeros(D))
        # lm_head tied to tok_emb

    def layer_norm(self, x, w, b, eps=1e-5):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + eps) * w + b

    def forward(self, idx):
        B, Tt = idx.shape
        x = self.tok_emb[idx] + self.pos_emb[:Tt]           # [B,T,D]
        hdim = D // H
        per_layer_hidden = []
        for l in range(L):
            h = self.layer_norm(x, self.ln1_w[l], self.ln1_b[l])
            q = h @ self.attn_q[l].t()
            k = h @ self.attn_k[l].t()
            v = h @ self.attn_v[l].t()
            # multi-head
            q = q.view(B, Tt, H, hdim).transpose(1, 2)      # [B,H,T,hdim]
            k = k.view(B, Tt, H, hdim).transpose(1, 2)
            v = v.view(B, Tt, H, hdim).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(hdim)
            mask = torch.triu(torch.ones(Tt, Tt, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float("-inf"))
            att = F.softmax(scores, -1) @ v
            att = att.transpose(1, 2).reshape(B, Tt, D)
            att = att @ self.attn_o[l].t()
            x = x + att
            h2 = self.layer_norm(x, self.ln2_w[l], self.ln2_b[l])
            m = F.gelu(h2 @ self.mlp_fc_w[l].t() + self.mlp_fc_b[l])
            m = m @ self.mlp_proj_w[l].t() + self.mlp_proj_b[l]
            x = x + m
            per_layer_hidden.append(x.clone())
        x = self.layer_norm(x, self.ln_f_w, self.ln_f_b)
        logits = x @ self.tok_emb.t()                       # tied head [B,T,V]
        return logits, per_layer_hidden


torch.set_num_threads(2)
model = MiniTransformer().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)

# Training
BATCH = 256
n = len(seqs)
src = torch.tensor(seqs[:, :T], dtype=torch.long)
tgt = torch.tensor(seqs[:, 1:T+1], dtype=torch.long)
EPOCHS = 10
for ep in range(EPOCHS):
    perm = torch.randperm(n)
    tot = 0.0; cnt = 0
    for s in range(0, n, BATCH):
        idx = perm[s:s+BATCH]
        if len(idx) == 0: break
        opt.zero_grad()
        logits, _ = model(src[idx])
        loss = F.cross_entropy(logits.reshape(-1, V), tgt[idx].reshape(-1))
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx); cnt += len(idx)
    if ep % 5 == 0 or ep == EPOCHS-1:
        print(f"  epoch {ep:3d}  loss={tot/cnt:.4f}")

# --------------------------------------------------------------------------
# Compute REAL steering direction: mean residual after layer STEER_LAYER for
# positive prompts minus negative prompts (last position).
# --------------------------------------------------------------------------
STEER_LAYER = 1
def hidden_for_prompt(text):
    ids = torch.tensor([[char2i[c] for c in text if c in char2i]], dtype=torch.long)
    with torch.no_grad():
        _, hl = model(ids)
    return hl[STEER_LAYER][0, -1].numpy()        # residual after layer STEER_LAYER, last pos

pos_seeds = ["I feel happy", "I feel calm", "I feel glad"]
neg_seeds = ["I feel sad", "I feel angry", "I feel blue"]
pos_h = np.mean([hidden_for_prompt(s) for s in pos_seeds], axis=0)
neg_h = np.mean([hidden_for_prompt(s) for s in neg_seeds], axis=0)
direction = pos_h - neg_h
direction = direction / (np.linalg.norm(direction) + 1e-8)
print(f"[*] steering direction (layer {STEER_LAYER}) norm after norm = {np.linalg.norm(direction):.3f}")

# --------------------------------------------------------------------------
# Save weights in a C-loadable MINI format (mirrors GPW2 layout the servers use)
# --------------------------------------------------------------------------
def save_mini(path):
    tensors = {}
    tensors["tok_emb"] = model.tok_emb.detach().numpy().astype(np.float32)
    tensors["pos_emb"] = model.pos_emb.detach().numpy().astype(np.float32)
    tensors["ln_f_w"] = model.ln_f_w.detach().numpy().astype(np.float32)
    tensors["ln_f_b"] = model.ln_f_b.detach().numpy().astype(np.float32)
    for l in range(L):
        tensors[f"h.{l}.ln1_w"] = model.ln1_w[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.ln1_b"] = model.ln1_b[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.attn_q"] = model.attn_q[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.attn_k"] = model.attn_k[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.attn_v"] = model.attn_v[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.attn_o"] = model.attn_o[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.ln2_w"] = model.ln2_w[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.ln2_b"] = model.ln2_b[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.mlp_fc_w"] = model.mlp_fc_w[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.mlp_fc_b"] = model.mlp_fc_b[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.mlp_proj_w"] = model.mlp_proj_w[l].detach().numpy().astype(np.float32)
        tensors[f"h.{l}.mlp_proj_b"] = model.mlp_proj_b[l].detach().numpy().astype(np.float32)
    with open(path, "wb") as f:
        f.write(b"MINI")
        f.write(struct.pack("I", len(tensors)))
        for key, arr in tensors.items():
            kb = key.encode()
            f.write(struct.pack("I", len(kb))); f.write(kb)
            f.write(struct.pack("I", arr.ndim))
            for d in arr.shape: f.write(struct.pack("I", d))
            f.write(arr.astype(np.float32).tobytes())
    print(f"[*] wrote {path} ({len(tensors)} tensors)")

os.makedirs(os.path.join(REPO, "prebuilt"), exist_ok=True)
save_mini(os.path.join(REPO, "prebuilt", "mini_model.bin"))
with open(os.path.join(REPO, "prebuilt", "mini_model.json"), "w") as f:
    json.dump({"D": D, "L": L, "H": H, "T": T, "V": V,
               "chars": CHARS, "steer_layer": STEER_LAYER}, f, indent=2)

# Write the .lal with the REAL direction vector.
vec_str = ", ".join(f"{v:.6f}" for v in direction.tolist())
lal = (
    f"# mini_steer.lal — REAL steering direction from the trained mini transformer.\n"
    f"# Direction = mean(residual@layer{STEER_LAYER} | positive seeds)\n"
    f"#            - mean(residual@layer{STEER_LAYER} | negative seeds),\n"
    f"# then L2-normalized. Steering with +w should push generation toward positive mood.\n"
    f"concept pos = [{vec_str}]\n"
    f"concept steered = runtime_context(dim={D}, layer={STEER_LAYER}, add=pos, weight=4.0)\n"
)
with open(os.path.join(REPO, "demos", "mini_steer.lal"), "w") as f:
    f.write(lal)
print(f"[*] wrote demos/mini_steer.lal (dim={D}, layer={STEER_LAYER}, weight=4.0)")

# --------------------------------------------------------------------------
# Sanity: greedy generation in Python (baseline) for the neutral prompt.
# --------------------------------------------------------------------------
def generate(text, steps=16, steer=None, w=0.0):
    ids = [char2i[c] for c in text if c in char2i]
    for _ in range(steps):
        xin = ids[-T:]
        tin = torch.tensor([xin], dtype=torch.long)
        with torch.no_grad():
            logits, hl = model(tin)
        if steer is not None:
            hl[STEER_LAYER][0, -1] += w * torch.tensor(steer, dtype=torch.float32)
            # recompute logits from steered residual (re-run final norm+head)
            x = hl[STEER_LAYER]
            mu = x.mean(-1, keepdim=True); var = x.var(-1, keepdim=True, unbiased=False)
            x = (x - mu) / torch.sqrt(var + 1e-5) * model.ln_f_w + model.ln_f_b
            logits = (x @ model.tok_emb.t())
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        if i2char[nxt] == ".":
            break
    return "".join(i2char[i] for i in ids)

print("\n[*] PYTHON sanity (ground truth on trained model):")
print("  baseline :", generate("I feel", steps=16))
print("  steered  :", generate("I feel", steps=16, steer=direction, w=4.0))
print("\n[*] done. Now run: bash scripts/build_lal_mini_steer.sh && make mini-server")
