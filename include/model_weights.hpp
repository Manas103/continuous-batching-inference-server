#pragma once
#include <string>
#include <vector>

// All pointers are device pointers. Layout mirrors python/export_weights.py
// exactly -- see that file's docstring for the on-disk order.
struct BlockWeights {
  float *ln1_w, *ln1_b;
  float *wq_w, *wq_b, *wk_w, *wk_b, *wv_w, *wv_b, *wo_w, *wo_b;
  float *ln2_w, *ln2_b;
  float *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};

struct ModelWeights {
  float* tok_emb;
  float* pos_emb;
  std::vector<BlockWeights> blocks;
  float* ln_f_w;
  float* ln_f_b;
  float* lm_head;
  std::vector<float*> owned_;  // every device alloc, for cleanup

  static ModelWeights load(const std::string& path);
  void free_all();
};
