#pragma once
#include <hip/hip_runtime.h>

// Copies K_proj/V_proj (this step's freshly computed [total_rows, d_model]
// projections) into the paged cache at each row's designated physical
// (page, offset) slot. write_page/write_offset are device int arrays of
// length total_rows.
void scatter_kv_to_cache(const float* K_proj, const float* V_proj,
                          float* k_pages, float* v_pages,
                          const int* write_page, const int* write_offset,
                          int total_rows, int d_model, int page_size,
                          int n_heads, int head_dim, hipStream_t stream);

// Fused causal attention: for every row r (query position row_query_pos[r]
// in its sequence) and every head h, computes
//   softmax_j( Q_r,h . K_j,h / sqrt(head_dim) )  for j in [0, row_query_pos[r]]
//   attn_out[r,h] = sum_j softmax_j * V_j,h
// gathering K/V for position j through row_page_table directly from the
// paged cache. See kernels/attention.hip.cpp for the HIP-side warp/wavefront
// note: the CUDA original assigned one warp (32 lanes) per (row, head); see
// docs/port_failure_taxonomy.md for how that assumption was handled in the
// port.
void fused_paged_causal_attention(const float* Q_proj, const float* k_pages,
                                   const float* v_pages,
                                   const int* row_page_table,
                                   const int* row_query_pos, float* attn_out,
                                   int total_rows, int n_heads, int head_dim,
                                   int page_size, int pages_per_seq,
                                   hipStream_t stream);
