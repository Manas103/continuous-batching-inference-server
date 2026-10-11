#pragma once
#include <vector>
#include <hip/hip_runtime.h>
#include "model_weights.hpp"
#include "paged_kv_cache.hpp"

// Same role as the CUDA original's TransformerRunner. No hipblasHandle_t
// member: this port's linear_forward (kernels/linear.hip.cpp) is a
// hand-written HIP kernel, not a hipBLAS call, so there is no BLAS handle
// to own. See include/linear.hpp for why.
class TransformerRunner {
 public:
  TransformerRunner(const ModelWeights& weights, PagedKVCache& cache, int max_total_rows);
  ~TransformerRunner();

  std::vector<float> forward_step(const std::vector<int>& token_ids,
                                   const std::vector<int>& positions,
                                   const std::vector<int>& seq_ids);

 private:
  const ModelWeights& weights_;
  PagedKVCache& cache_;
  int max_total_rows_;
  hipStream_t stream_;

  int *d_token_ids_, *d_positions_, *d_write_page_, *d_write_offset_, *d_row_page_table_;
  float *d_x_, *d_norm_, *d_q_, *d_k_, *d_v_, *d_attn_out_, *d_proj_out_, *d_mlp_hidden_, *d_mlp_out_, *d_logits_;
};
