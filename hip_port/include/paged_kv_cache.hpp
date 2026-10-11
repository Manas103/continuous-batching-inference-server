#pragma once
#include <vector>
#include <unordered_map>
#include <hip/hip_runtime.h>
#include "config.hpp"

// Identical design to the CUDA original's PagedKVCache (see that repo's
// include/paged_kv_cache.hpp for the full vLLM-style rationale); only the
// device-memory calls underneath are HIP, not CUDA.
class PagedKVCache {
 public:
  PagedKVCache(int num_layers, int max_pages, int page_size, int n_heads, int head_dim);
  ~PagedKVCache();

  int allocate_sequence();
  void free_sequence(int seq_id);
  bool ensure_capacity(int seq_id, int token_pos);
  void resolve_slot(int seq_id, int token_pos, int* physical_page, int* offset) const;
  std::vector<int> page_table(int seq_id) const;

  float* k_base(int layer) const { return k_pages_[layer]; }
  float* v_base(int layer) const { return v_pages_[layer]; }

  int num_free_pages() const { return (int)free_pages_.size(); }
  int max_pages() const { return max_pages_; }
  int page_size() const { return page_size_; }

 private:
  int num_layers_, max_pages_, page_size_, n_heads_, head_dim_;
  std::vector<float*> k_pages_, v_pages_;
  std::vector<int> free_pages_;
  std::unordered_map<int, std::vector<int>> seq_pages_;
  int next_seq_id_ = 0;
};
