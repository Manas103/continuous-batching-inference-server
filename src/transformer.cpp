#include "transformer.hpp"
#include "config.hpp"
#include "cuda_check.hpp"
#include "linear.hpp"
#include "elementwise.hpp"
#include "attention.hpp"
#include <stdexcept>

using namespace cfg;

TransformerRunner::TransformerRunner(const ModelWeights& weights, PagedKVCache& cache, int max_total_rows)
    : weights_(weights), cache_(cache), max_total_rows_(max_total_rows) {
  CUBLAS_CHECK(cublasCreate(&handle_));
  CUDA_CHECK(cudaStreamCreate(&stream_));

  CUDA_CHECK(cudaMalloc(&d_token_ids_, max_total_rows * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_positions_, max_total_rows * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_write_page_, max_total_rows * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_write_offset_, max_total_rows * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_row_page_table_, (size_t)max_total_rows * kPagesPerSeq * sizeof(int)));

  CUDA_CHECK(cudaMalloc(&d_x_, (size_t)max_total_rows * kDModel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_norm_, (size_t)max_total_rows * kDModel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q_, (size_t)max_total_rows * kDModel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k_, (size_t)max_total_rows * kDModel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v_, (size_t)max_total_rows * kDModel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_out_, (size_t)max_total_rows * kDModel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_proj_out_, (size_t)max_total_rows * kDModel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mlp_hidden_, (size_t)max_total_rows * kMlpHidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mlp_out_, (size_t)max_total_rows * kDModel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_logits_, (size_t)max_total_rows * kVocab * sizeof(float)));
}

TransformerRunner::~TransformerRunner() {
  cudaFree(d_token_ids_); cudaFree(d_positions_); cudaFree(d_write_page_);
  cudaFree(d_write_offset_); cudaFree(d_row_page_table_);
  cudaFree(d_x_); cudaFree(d_norm_); cudaFree(d_q_); cudaFree(d_k_); cudaFree(d_v_);
  cudaFree(d_attn_out_); cudaFree(d_proj_out_); cudaFree(d_mlp_hidden_); cudaFree(d_mlp_out_);
  cudaFree(d_logits_);
  cublasDestroy(handle_);
  cudaStreamDestroy(stream_);
}

std::vector<float> TransformerRunner::forward_step(const std::vector<int>& token_ids,
                                                     const std::vector<int>& positions,
                                                     const std::vector<int>& seq_ids) {
  int total_rows = (int)token_ids.size();
  if (total_rows == 0) return {};
  if (total_rows > max_total_rows_) throw std::runtime_error("forward_step: total_rows exceeds max_total_rows");

  std::vector<int> write_page(total_rows), write_offset(total_rows);
  std::vector<int> row_page_table((size_t)total_rows * kPagesPerSeq);
  for (int i = 0; i < total_rows; ++i) {
    cache_.resolve_slot(seq_ids[i], positions[i], &write_page[i], &write_offset[i]);
    auto table = cache_.page_table(seq_ids[i]);
    for (int p = 0; p < kPagesPerSeq; ++p) row_page_table[(size_t)i * kPagesPerSeq + p] = table[p];
  }

  CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, token_ids.data(), total_rows * sizeof(int), cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_positions_, positions.data(), total_rows * sizeof(int), cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_write_page_, write_page.data(), total_rows * sizeof(int), cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_write_offset_, write_offset.data(), total_rows * sizeof(int), cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_row_page_table_, row_page_table.data(), row_page_table.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));

  embed_lookup_add(d_token_ids_, d_positions_, weights_.tok_emb, weights_.pos_emb, d_x_, total_rows, kDModel, stream_);

  for (int l = 0; l < kNLayers; ++l) {
    const BlockWeights& bw = weights_.blocks[l];

    layernorm_forward(d_x_, bw.ln1_w, bw.ln1_b, d_norm_, total_rows, kDModel, stream_);
    linear_forward(handle_, d_norm_, bw.wq_w, bw.wq_b, d_q_, total_rows, kDModel, kDModel, stream_);
    linear_forward(handle_, d_norm_, bw.wk_w, bw.wk_b, d_k_, total_rows, kDModel, kDModel, stream_);
    linear_forward(handle_, d_norm_, bw.wv_w, bw.wv_b, d_v_, total_rows, kDModel, kDModel, stream_);

    scatter_kv_to_cache(d_k_, d_v_, cache_.k_base(l), cache_.v_base(l), d_write_page_, d_write_offset_,
                         total_rows, kDModel, cache_.page_size(), kNHeads, kHeadDim, stream_);

    fused_paged_causal_attention(d_q_, cache_.k_base(l), cache_.v_base(l), d_row_page_table_, d_positions_,
                                  d_attn_out_, total_rows, kNHeads, kHeadDim, cache_.page_size(),
                                  kPagesPerSeq, stream_);

    linear_forward(handle_, d_attn_out_, bw.wo_w, bw.wo_b, d_proj_out_, total_rows, kDModel, kDModel, stream_);
    residual_add(d_x_, d_proj_out_, total_rows, kDModel, stream_);

    layernorm_forward(d_x_, bw.ln2_w, bw.ln2_b, d_norm_, total_rows, kDModel, stream_);
    linear_forward(handle_, d_norm_, bw.fc1_w, bw.fc1_b, d_mlp_hidden_, total_rows, kDModel, kMlpHidden, stream_);
    gelu_forward(d_mlp_hidden_, total_rows, kMlpHidden, stream_);
    linear_forward(handle_, d_mlp_hidden_, bw.fc2_w, bw.fc2_b, d_mlp_out_, total_rows, kMlpHidden, kDModel, stream_);
    residual_add(d_x_, d_mlp_out_, total_rows, kDModel, stream_);
  }

  layernorm_forward(d_x_, weights_.ln_f_w, weights_.ln_f_b, d_norm_, total_rows, kDModel, stream_);
  linear_forward(handle_, d_norm_, weights_.lm_head, nullptr, d_logits_, total_rows, kDModel, kVocab, stream_);

  std::vector<float> logits((size_t)total_rows * kVocab);
  CUDA_CHECK(cudaMemcpyAsync(logits.data(), d_logits_, logits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  return logits;
}
