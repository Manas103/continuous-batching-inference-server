"""Reference GPT-style decoder-only transformer, used as the correctness
oracle and weight source for the C++/CUDA inference server. This is the
only place the model's numerics are defined "by spec" -- the C++ server's
job is to reproduce this forward pass on the GPU via hand-written CUDA/
cuBLAS code and be checked against it.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 1000
D_MODEL = 256
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS
N_LAYERS = 4
MLP_HIDDEN = 4 * D_MODEL
MAX_SEQ_LEN = 32768  # must mirror cfg::kMaxSeqLen in include/config.hpp

assert HEAD_DIM * N_HEADS == D_MODEL


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.wq = nn.Linear(D_MODEL, D_MODEL)
        self.wk = nn.Linear(D_MODEL, D_MODEL)
        self.wv = nn.Linear(D_MODEL, D_MODEL)
        self.wo = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, MLP_HIDDEN)
        self.fc2 = nn.Linear(MLP_HIDDEN, D_MODEL)

    def attn(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.wk(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        out = probs @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        return self.wo(out)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def build_model(seed=1234):
    torch.manual_seed(seed)
    model = TinyGPT()
    model.eval()
    return model
