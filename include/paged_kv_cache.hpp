#pragma once
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include "config.hpp"

// Block-based (paged) KV cache. Physical storage is one big device buffer
// per layer, shaped [max_pages, page_size, n_heads, head_dim]. A sequence
// owns a variable-length list of physical page indices (its "block table"),
// allocated page-by-page as it grows -- so a long sequence doesn't need a
// large contiguous allocation, and a finished sequence's pages go straight
// back into a free list for reuse by the next admitted request. This is the
// same idea vLLM's PagedAttention popularized: treat the KV cache like an
// OS treats virtual memory, instead of reserving max_seq_len worth of
// contiguous cache per request up front (which either wastes memory on
// short sequences or caps how many concurrent sequences fit).
class PagedKVCache {
 public:
  PagedKVCache(int num_layers, int max_pages, int page_size, int n_heads, int head_dim);
  ~PagedKVCache();

  int allocate_sequence();
  void free_sequence(int seq_id);

  // Grows the sequence's page list, if needed, so it can hold token index
  // `token_pos` (0-based). Returns false if the free list is exhausted.
  bool ensure_capacity(int seq_id, int token_pos);

  // Physical page id and in-page offset that token_pos maps to for seq_id.
  // Requires ensure_capacity(seq_id, token_pos) to have been called first.
  void resolve_slot(int seq_id, int token_pos, int* physical_page, int* offset) const;

  // Full logical->physical page table for seq_id, padded with 0 past the
  // sequence's current length (harmless: the attention kernel never reads
  // past row_query_pos, so padding entries are never dereferenced).
  std::vector<int> page_table(int seq_id) const;

  float* k_base(int layer) const { return k_pages_[layer]; }
  float* v_base(int layer) const { return v_pages_[layer]; }

  int num_free_pages() const { return (int)free_pages_.size(); }
  int max_pages() const { return max_pages_; }
  int page_size() const { return page_size_; }

 private:
  int num_layers_, max_pages_, page_size_, n_heads_, head_dim_;
  std::vector<float*> k_pages_, v_pages_;  // one device buffer per layer
  std::vector<int> free_pages_;
  std::unordered_map<int, std::vector<int>> seq_pages_;  // seq_id -> physical page ids
  int next_seq_id_ = 0;
};
