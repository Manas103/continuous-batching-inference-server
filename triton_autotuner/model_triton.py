"""The same TinyGPT reference model (python/model.py), with its attention
replaced by the Triton fused kernel instead of separate matmul/softmax/matmul
ops. Everything else (layernorm, linear layers, GELU, embeddings) is bit-for-
bit the same code as the CUDA build's own correctness oracle, so a passing
diff against reference.bin is a statement about the attention kernel
specifically, not about a reimplemented model.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import torch.nn.functional as F

import model as ref_model
from attention_triton import fused_attention

N_HEADS = ref_model.N_HEADS
HEAD_DIM = ref_model.HEAD_DIM
D_MODEL = ref_model.D_MODEL


def triton_attn(block, x, triton_cfg=None):
    """Same math as model.Block.attn, with the q@k/softmax/@v replaced."""
    B, T, _ = x.shape
    q = block.wq(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    k = block.wk(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    v = block.wv(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    cfg = triton_cfg or {}
    out = fused_attention(
        q, k, v, sm_scale=1.0 / math.sqrt(HEAD_DIM),
        block_m=cfg.get("BLOCK_M", 64), block_n=cfg.get("BLOCK_N", 64),
        num_warps=cfg.get("num_warps", 4), num_stages=cfg.get("num_stages", 2),
    )
    out = out.transpose(1, 2).contiguous().view(B, T, D_MODEL)
    return block.wo(out)


def unfused_attn(block, x):
    """The 'unfused baseline': the same math as three separate eager torch
    ops (matmul, softmax, matmul), with no kernel fusion at all -- what the
    README's speedup claim is measured against."""
    B, T, _ = x.shape
    q = block.wq(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
    k = block.wk(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
    v = block.wv(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    out = probs @ v
    out = out.transpose(1, 2).contiguous().view(B, T, D_MODEL)
    return block.wo(out)


def forward_with_attn(model, idx, attn_fn, triton_cfg=None):
    """model.TinyGPT.forward, but calling attn_fn instead of block.attn."""
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device).unsqueeze(0)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    for blk in model.blocks:
        ln1 = blk.ln1(x)
        if attn_fn is triton_attn:
            a = attn_fn(blk, ln1, triton_cfg)
        else:
            a = attn_fn(blk, ln1)
        x = x + a
        x = x + blk.fc2(F.gelu(blk.fc1(blk.ln2(x))))
    x = model.ln_f(x)
    return model.lm_head(x)
