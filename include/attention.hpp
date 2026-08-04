#pragma once
#include <cuda_runtime.h>

// Copies K_proj/V_proj (this step's freshly computed [total_rows, d_model]
// projections) into the paged cache at each row's designated physical
// (page, offset) slot. write_page/write_offset are device int arrays of
// length total_rows.
void scatter_kv_to_cache(const float* K_proj, const float* V_proj,
                          float* k_pages, float* v_pages,
                          const int* write_page, const int* write_offset,
                          int total_rows, int d_model, int page_size,
                          int n_heads, int head_dim, cudaStream_t stream);

// Fused causal attention: for every row r (query position row_query_pos[r]
// in its sequence) and every head h, computes
//   softmax_j( Q_r,h . K_j,h / sqrt(head_dim) )  for j in [0, row_query_pos[r]]
//   attn_out[r,h] = sum_j softmax_j * V_j,h
// gathering K/V for position j through row_page_table (physical page id per
// logical page index, kPagesPerSeq entries per row) directly from the paged
// cache -- QK^T, causal bound, softmax, and the P*V reduction all happen in
// a single kernel launch, one warp per (row, head) since head_dim==32==warp
// size lets the dot-product reduction map onto warp shuffles.
void fused_paged_causal_attention(const float* Q_proj, const float* k_pages,
                                   const float* v_pages,
                                   const int* row_page_table,
                                   const int* row_query_pos, float* attn_out,
                                   int total_rows, int n_heads, int head_dim,
                                   int page_size, int pages_per_seq,
                                   cudaStream_t stream);
