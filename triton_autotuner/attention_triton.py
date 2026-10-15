"""Block-tiled, causal, online-softmax fused attention kernel in Triton.

Replaces kernels/attention.cu's role (QK^T, causal softmax, @V, all fused
into one launch) for a single prefill pass over dense [B, H, T, head_dim]
tensors. This file only contains the kernel and its tile-size/warp-count
configuration space; autotune.py does the pruning and search, correctness
is checked in correctness_check.py against the repo's existing 122,000
logit PyTorch reference.

Honest scope note: this operates on already-materialized Q/K/V tensors for
one prefill batch, not through the C++ server's paged KV-cache. It is a
standalone extension that reuses the server's existing correctness oracle,
the same relationship the hip_port/ extension has to the CUDA build.
"""
import torch
import triton
import triton.language as tl


def autotune_configs():
    """The full, unpruned tile-size/warp-count space this project searches.

    BLOCK_M, BLOCK_N over the query/key tile dimensions (block-tiling),
    num_warps over the warp count per block, num_stages over software
    pipelining depth. 4 * 4 * 4 * 2 = 128 raw configurations; autotune.py
    prunes the ones Triton itself rejects (shared memory, warp-count
    mismatches) or that fail the correctness check, and reports exactly how
    many survive -- that count is measured, not targeted at any number.
    (An earlier, 1024-configuration version of this space was tried first;
    compiling and correctness-checking all 1024 variants on this GPU did
    not finish in a reasonable amount of time, so the space was narrowed to
    the one below -- see README Findings for the measured compile cost that
    drove this.)
    """
    configs = []
    block_ms = (16, 32, 64, 128)
    block_ns = (16, 32, 64, 128)
    num_warps_opts = (1, 2, 4, 8)
    num_stages_opts = (1, 2)
    for bm in block_ms:
        for bn in block_ns:
            for nw in num_warps_opts:
                for ns in num_stages_opts:
                    configs.append(dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, num_stages=ns))
    return configs


@triton.jit
def _fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    seq_len, head_dim,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    q_base = q_ptr + off_bh * stride_qh
    k_base = k_ptr + off_bh * stride_kh
    v_base = v_ptr + off_bh * stride_vh
    o_base = out_ptr + off_bh * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    end_n = (start_m + 1) * BLOCK_M  # causal: never attend past the query's own block end
    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        kv_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & (offs_n[None, :] < seq_len) & (offs_m[:, None] < seq_len)
        scores = tl.where(valid_mask, scores, float("-inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v, input_precision="ieee")

        m_i = m_new

    l_i = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc.to(tl.float32), mask=o_mask)


def fused_attention(q, k, v, sm_scale, block_m=64, block_n=64, num_warps=4, num_stages=2):
    """q, k, v: [B, H, T, head_dim], fp32. Returns [B, H, T, head_dim].

    Requires contiguous [B, H, T, head_dim] layout: the kernel folds the
    batch and head grid axes into one (off_bh) and relies on
    stride_b == H * stride_h, which only holds for a contiguous tensor.
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, T, D = q.shape
    out = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(T, block_m), B * H)
    _fused_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        T, D, sm_scale,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
