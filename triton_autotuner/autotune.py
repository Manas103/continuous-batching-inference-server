"""Prunes and searches the tile-size/warp-count space for the fused
attention kernel, per shape.

Pruning happens in two honest stages, each counted and reported:
  1. static pruning: configs Triton itself cannot compile/launch for this
     kernel (BLOCK_N bigger than head_dim's padded tile wastes a dot-product
     dimension pointlessly but is not invalid, so the real static prune
     here is BLOCK_M/BLOCK_N/num_warps combinations the kernel raises on --
     out-of-shared-memory or illegal num_warps for the block shape).
  2. correctness pruning: configs that compile and run but fail the
     correctness check (diff vs. the unfused eager baseline on a small
     canary input) are dropped before they are ever benchmarked.

What survives both is benchmarked on every shape; the best latency per
shape is kept. This script prints exactly how many configs started, how
many survived each stage, and how many were actually benchmarked -- those
counts are measured outputs, not the resume's targeted numbers.
"""
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import model as ref_model  # noqa: E402
from attention_triton import autotune_configs, fused_attention  # noqa: E402

HEAD_DIM = ref_model.HEAD_DIM
N_HEADS = ref_model.N_HEADS


def canary_inputs(device):
    torch.manual_seed(0)
    B, H, T, D = 2, N_HEADS, 96, HEAD_DIM
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)
    return q, k, v


def eager_reference(q, k, v, sm_scale):
    scores = (q @ k.transpose(-2, -1)) * sm_scale
    T = q.shape[-2]
    causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def is_valid_launch_config(cfg):
    """Static prune: Triton requires num_warps * 32 threads to be able to
    cover a BLOCK_M x BLOCK_D tile without more warps than rows, and
    BLOCK_N must be a multiple of 16 for tl.dot's shape requirements (true
    here by construction since all BLOCK_N values are powers of two >=16).
    The one real static-reject case for this kernel: num_warps bigger than
    BLOCK_M wastes warps on rows that do not exist and Triton raises on a
    handful of such combinations for this head_dim at compile time."""
    return cfg["num_warps"] <= cfg["BLOCK_M"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available; aborting.")
        sys.exit(1)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    q, k, v = canary_inputs(device)
    ref_out = eager_reference(q, k, v, sm_scale)

    all_configs = autotune_configs()
    print(f"raw configuration space: {len(all_configs)}")

    statically_valid = [c for c in all_configs if is_valid_launch_config(c)]
    print(f"after static pruning (illegal warp/tile combinations): {len(statically_valid)}")

    correct_configs = []
    compile_failures = 0
    correctness_failures = 0
    t_start = time.time()
    for i, cfg in enumerate(statically_valid):
        if i % 10 == 0:
            print(f"  correctness pass: {i}/{len(statically_valid)} "
                  f"({time.time() - t_start:.1f}s elapsed)", flush=True)
        try:
            out = fused_attention(q, k, v, sm_scale, block_m=cfg["BLOCK_M"], block_n=cfg["BLOCK_N"],
                                   num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        except Exception:
            compile_failures += 1
            continue
        max_diff = (out - ref_out).abs().max().item()
        if max_diff < 5e-3:
            correct_configs.append(cfg)
        else:
            correctness_failures += 1
    print(f"compile/launch failures on the canary input: {compile_failures}", flush=True)
    print(f"ran but failed the correctness canary (diff >= 5e-3): {correctness_failures}", flush=True)
    print(f"configurations actually benchmarked: {len(correct_configs)}", flush=True)

    shapes = [(1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048), (4, 128), (4, 512), (8, 256)]
    best_per_shape = {}
    for (B, T) in shapes:
        torch.manual_seed(1)
        qs = torch.randn(B, N_HEADS, T, HEAD_DIM, device=device)
        ks = torch.randn(B, N_HEADS, T, HEAD_DIM, device=device)
        vs = torch.randn(B, N_HEADS, T, HEAD_DIM, device=device)
        best_time, best_cfg = None, None
        for j, cfg in enumerate(correct_configs):
            try:
                for _ in range(2):
                    fused_attention(qs, ks, vs, sm_scale, block_m=cfg["BLOCK_M"], block_n=cfg["BLOCK_N"],
                                     num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
                torch.cuda.synchronize()
                start = time.perf_counter()
                reps = 10
                for _ in range(reps):
                    fused_attention(qs, ks, vs, sm_scale, block_m=cfg["BLOCK_M"], block_n=cfg["BLOCK_N"],
                                     num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / reps
            except Exception:
                continue
            if best_time is None or elapsed < best_time:
                best_time, best_cfg = elapsed, cfg
        best_per_shape[(B, T)] = (best_time, best_cfg)
        print(f"shape B={B} T={T}: best_latency={best_time * 1000:.4f} ms, config={best_cfg}", flush=True)

    total_search = len(correct_configs) * len(shapes)
    print(f"\ntotal (config, shape) latency measurements taken: {total_search}")
    print(f"search went from {len(all_configs)} raw configurations to {len(correct_configs)} benchmarked")

    out_path = os.path.join(os.path.dirname(__file__), "docs", "autotune_best_configs.json")
    serializable = {
        f"{B}x{T}": {"latency_ms": best_time * 1000, "config": cfg}
        for (B, T), (best_time, cfg) in best_per_shape.items()
    }
    serializable["_search_space"] = {
        "raw": len(all_configs),
        "after_static_prune": len(statically_valid),
        "benchmarked": len(correct_configs),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
