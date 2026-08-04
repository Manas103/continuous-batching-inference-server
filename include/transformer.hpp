#pragma once
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "model_weights.hpp"
#include "paged_kv_cache.hpp"

// Runs one "step" of the model over a flattened, continuous-batching-style
// token batch: each row is one token from one active sequence (either a
// single decode-phase token, or -- concatenated across several rows -- an
// entire prompt being prefilled this step). All linear/layernorm/gelu ops
// operate on the whole [total_rows, d_model] matrix in one shot regardless
// of which sequence each row came from; only the attention kernel needs to
// know per-row sequence identity, via the paged cache block table.
class TransformerRunner {
 public:
  TransformerRunner(const ModelWeights& weights, PagedKVCache& cache, int max_total_rows);
  ~TransformerRunner();

  // token_ids[i]/positions[i]/seq_ids[i] describe row i. The caller must
  // have already called cache.ensure_capacity(seq_ids[i], positions[i]) for
  // every row before calling this. Returns row-major [total_rows, kVocab]
  // logits.
  std::vector<float> forward_step(const std::vector<int>& token_ids,
                                   const std::vector<int>& positions,
                                   const std::vector<int>& seq_ids);

 private:
  const ModelWeights& weights_;
  PagedKVCache& cache_;
  int max_total_rows_;
  cublasHandle_t handle_;
  cudaStream_t stream_;

  int *d_token_ids_, *d_positions_, *d_write_page_, *d_write_offset_, *d_row_page_table_;
  float *d_x_, *d_norm_, *d_q_, *d_k_, *d_v_, *d_attn_out_, *d_proj_out_, *d_mlp_hidden_, *d_mlp_out_, *d_logits_;
};
