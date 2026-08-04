#include "paged_kv_cache.hpp"
#include "cuda_check.hpp"
#include <stdexcept>

PagedKVCache::PagedKVCache(int num_layers, int max_pages, int page_size, int n_heads, int head_dim)
    : num_layers_(num_layers), max_pages_(max_pages), page_size_(page_size),
      n_heads_(n_heads), head_dim_(head_dim) {
  size_t bytes_per_layer = (size_t)max_pages * page_size * n_heads * head_dim * sizeof(float);
  k_pages_.resize(num_layers);
  v_pages_.resize(num_layers);
  for (int l = 0; l < num_layers; ++l) {
    CUDA_CHECK(cudaMalloc(&k_pages_[l], bytes_per_layer));
    CUDA_CHECK(cudaMalloc(&v_pages_[l], bytes_per_layer));
  }
  free_pages_.reserve(max_pages);
  for (int p = max_pages - 1; p >= 0; --p) free_pages_.push_back(p);
}

PagedKVCache::~PagedKVCache() {
  for (auto* p : k_pages_) cudaFree(p);
  for (auto* p : v_pages_) cudaFree(p);
}

int PagedKVCache::allocate_sequence() {
  int id = next_seq_id_++;
  seq_pages_[id] = {};
  return id;
}

void PagedKVCache::free_sequence(int seq_id) {
  auto it = seq_pages_.find(seq_id);
  if (it == seq_pages_.end()) return;
  for (int page : it->second) free_pages_.push_back(page);
  seq_pages_.erase(it);
}

bool PagedKVCache::ensure_capacity(int seq_id, int token_pos) {
  auto& pages = seq_pages_.at(seq_id);
  int needed_pages = token_pos / page_size_ + 1;
  while ((int)pages.size() < needed_pages) {
    if (free_pages_.empty()) return false;
    pages.push_back(free_pages_.back());
    free_pages_.pop_back();
  }
  return true;
}

void PagedKVCache::resolve_slot(int seq_id, int token_pos, int* physical_page, int* offset) const {
  const auto& pages = seq_pages_.at(seq_id);
  int logical_page = token_pos / page_size_;
  *physical_page = pages.at(logical_page);
  *offset = token_pos % page_size_;
}

std::vector<int> PagedKVCache::page_table(int seq_id) const {
  const auto& pages = seq_pages_.at(seq_id);
  std::vector<int> table(cfg::kPagesPerSeq, 0);
  for (size_t i = 0; i < pages.size() && i < table.size(); ++i) table[i] = pages[i];
  return table;
}
