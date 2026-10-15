"""Diffs the Triton-attention model against the repo's existing 122,000-logit
PyTorch reference (python/reference.bin), the same oracle the CUDA build's
correctness_check.cpp uses. A pass here is a statement about the Triton
attention kernel: everything else in the forward pass is the unmodified
reference code.
"""
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import model as ref_model  # noqa: E402
from model_triton import forward_with_attn, triton_attn  # noqa: E402

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "python", "reference.bin")


def load_reference(path):
    with open(path, "rb") as f:
        vocab, d_model, n_heads, n_layers, mlp_hidden, max_seq_len = struct.unpack("<iiiiii", f.read(24))
        (batch_size,) = struct.unpack("<i", f.read(4))
        seqs = []
        for _ in range(batch_size):
            (slen,) = struct.unpack("<i", f.read(4))
            ids = np.frombuffer(f.read(4 * slen), dtype=np.int32).copy()
            logits = np.frombuffer(f.read(4 * slen * vocab), dtype=np.float32).copy().reshape(slen, vocab)
            seqs.append((ids, logits))
        return seqs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available; Triton needs a GPU. Aborting.")
        sys.exit(1)

    model = ref_model.build_model(seed=1234).to(device)
    seqs = load_reference(REFERENCE_PATH)

    total_abs_diff = 0.0
    max_abs_diff = 0.0
    total_values = 0
    results = []
    for i, (ids, ref_logits) in enumerate(seqs):
        idx = torch.from_numpy(ids).long().unsqueeze(0).to(device)
        with torch.no_grad():
            out = forward_with_attn(model, idx, triton_attn)[0].float().cpu().numpy()
        diff = np.abs(out - ref_logits)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        max_abs_diff = max(max_abs_diff, max_d)
        total_abs_diff += float(diff.sum())
        total_values += diff.size
        status = "PASS" if max_d < 5e-3 else "FAIL"
        results.append(status)
        print(f"seq {i} (len={len(ids):3d}): max_abs_diff={max_d:.6f} mean_abs_diff={mean_d:.6f} {status}")

    overall_mean = total_abs_diff / total_values
    print(f"\noverall: max_abs_diff={max_abs_diff:.6f} mean_abs_diff={overall_mean:.6f} over {total_values} logit values")
    if all(r == "PASS" for r in results):
        print("CORRECTNESS CHECK PASSED")
    else:
        print("CORRECTNESS CHECK FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
