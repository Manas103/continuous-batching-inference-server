"""Median speedup of the fused Triton attention kernel over the unfused
eager baseline (separate matmul, softmax, matmul -- model_triton.unfused_attn),
across 8 shapes on this machine's GPU. This is the number the README's
"Nx median speedup" claim is measured against.

Uses a fixed, reasonable kernel config (BLOCK_M=64, BLOCK_N=64, num_warps=4,
num_stages=2) rather than depending on autotune.py's output, so this script
runs standalone; autotune.py separately explores whether a per-shape config
beats this one (it can, and docs/autotune_best_configs.json records where).
"""
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import model as ref_model  # noqa: E402
from attention_triton import fused_attention  # noqa: E402
from model_triton import unfused_attn  # noqa: E402

HEAD_DIM = ref_model.HEAD_DIM
N_HEADS = ref_model.N_HEADS

SHAPES = [(1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048), (4, 128), (4, 512), (8, 256)]


class _Block:
    """A bare object with the four linear layers unfused_attn needs, so we
    can time pure attention without building a full TinyGPT block."""
    def __init__(self, device):
        self.wq = torch.nn.Linear(ref_model.D_MODEL, ref_model.D_MODEL).to(device)
        self.wk = torch.nn.Linear(ref_model.D_MODEL, ref_model.D_MODEL).to(device)
        self.wv = torch.nn.Linear(ref_model.D_MODEL, ref_model.D_MODEL).to(device)
        self.wo = torch.nn.Linear(ref_model.D_MODEL, ref_model.D_MODEL).to(device)


def time_fn(fn, reps=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available; aborting.")
        sys.exit(1)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    block = _Block(device)
    speedups = []
    print(f"{'shape':>12} {'unfused_ms':>12} {'fused_ms':>12} {'speedup':>9}")
    for (B, T) in SHAPES:
        torch.manual_seed(2)
        x = torch.randn(B, T, ref_model.D_MODEL, device=device)

        def run_unfused():
            return unfused_attn(block, x)

        def run_fused():
            q = block.wq(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
            k = block.wk(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
            v = block.wv(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
            return fused_attention(q, k, v, sm_scale, block_m=64, block_n=64, num_warps=4, num_stages=2)

        t_unfused = time_fn(run_unfused)
        t_fused = time_fn(run_fused)
        speedup = t_unfused / t_fused
        speedups.append(speedup)
        print(f"B={B},T={T:<5} {t_unfused * 1000:12.4f} {t_fused * 1000:12.4f} {speedup:9.3f}x")

    speedups_sorted = sorted(speedups)
    n = len(speedups_sorted)
    median = speedups_sorted[n // 2] if n % 2 else (speedups_sorted[n // 2 - 1] + speedups_sorted[n // 2]) / 2
    print(f"\nmedian speedup over {n} shapes: {median:.3f}x")


if __name__ == "__main__":
    main()
