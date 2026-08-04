"""Exports TinyGPT's weights plus a reference input batch and reference
output logits to a flat binary file the C++ server reads directly.

Layout (all float32, row-major, little-endian -- matches PyTorch's native
in-memory layout so tensors are written via a straight .numpy().tobytes()):

  tok_emb.weight   [VOCAB, D_MODEL]
  pos_emb.weight   [MAX_SEQ_LEN, D_MODEL]
  per layer (N_LAYERS times), each block in this exact order:
    ln1.weight [D_MODEL]        ln1.bias [D_MODEL]
    wq.weight  [D_MODEL,D_MODEL] wq.bias [D_MODEL]
    wk.weight  [D_MODEL,D_MODEL] wk.bias [D_MODEL]
    wv.weight  [D_MODEL,D_MODEL] wv.bias [D_MODEL]
    wo.weight  [D_MODEL,D_MODEL] wo.bias [D_MODEL]
    ln2.weight [D_MODEL]        ln2.bias [D_MODEL]
    fc1.weight [MLP_HIDDEN,D_MODEL] fc1.bias [MLP_HIDDEN]
    fc2.weight [D_MODEL,MLP_HIDDEN] fc2.bias [D_MODEL]
  ln_f.weight [D_MODEL]  ln_f.bias [D_MODEL]
  lm_head.weight [VOCAB, D_MODEL]

nn.Linear stores weight as [out_features, in_features], which is exactly
the layout the C++ gemm wrapper expects (see include/gemm.hpp).

The reference batch/logits go to a separate file so re-running inference
against different prompts doesn't require re-exporting weights.
"""
import struct
import numpy as np
import torch

from model import build_model, VOCAB, D_MODEL, N_HEADS, HEAD_DIM, N_LAYERS, MLP_HIDDEN, MAX_SEQ_LEN

WEIGHTS_PATH = "weights.bin"
REFERENCE_PATH = "reference.bin"


def write_tensor(f, t):
    arr = t.detach().cpu().numpy().astype(np.float32)
    f.write(arr.tobytes())


def export_weights(model):
    with open(WEIGHTS_PATH, "wb") as f:
        write_tensor(f, model.tok_emb.weight)
        write_tensor(f, model.pos_emb.weight)
        for blk in model.blocks:
            write_tensor(f, blk.ln1.weight)
            write_tensor(f, blk.ln1.bias)
            write_tensor(f, blk.wq.weight)
            write_tensor(f, blk.wq.bias)
            write_tensor(f, blk.wk.weight)
            write_tensor(f, blk.wk.bias)
            write_tensor(f, blk.wv.weight)
            write_tensor(f, blk.wv.bias)
            write_tensor(f, blk.wo.weight)
            write_tensor(f, blk.wo.bias)
            write_tensor(f, blk.ln2.weight)
            write_tensor(f, blk.ln2.bias)
            write_tensor(f, blk.fc1.weight)
            write_tensor(f, blk.fc1.bias)
            write_tensor(f, blk.fc2.weight)
            write_tensor(f, blk.fc2.bias)
        write_tensor(f, model.ln_f.weight)
        write_tensor(f, model.ln_f.bias)
        write_tensor(f, model.lm_head.weight)
    print(f"wrote {WEIGHTS_PATH}")


def export_reference(model, batch_size=6, seq_lens=(1, 3, 7, 16, 31, 64)):
    assert len(seq_lens) == batch_size
    max_len = max(seq_lens)
    g = torch.Generator().manual_seed(777)
    token_batch = torch.randint(0, VOCAB, (batch_size, max_len), generator=g)

    with open(REFERENCE_PATH, "wb") as f:
        f.write(struct.pack("<iiiiii", VOCAB, D_MODEL, N_HEADS, N_LAYERS, MLP_HIDDEN, MAX_SEQ_LEN))
        f.write(struct.pack("<i", batch_size))
        for i, slen in enumerate(seq_lens):
            f.write(struct.pack("<i", slen))
            ids = token_batch[i, :slen].contiguous()
            f.write(ids.to(torch.int32).numpy().tobytes())
            with torch.no_grad():
                logits = model(ids.unsqueeze(0))[0]  # [slen, VOCAB]
            write_tensor(f, logits)
    print(f"wrote {REFERENCE_PATH} ({batch_size} sequences, lengths={seq_lens})")


if __name__ == "__main__":
    model = build_model(seed=1234)
    export_weights(model)
    export_reference(model)
