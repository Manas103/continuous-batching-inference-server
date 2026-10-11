// HIP-side twin of the CUDA original's src/correctness_check.cpp. Loads
// the exact same python/weights.bin and python/reference.bin (the
// pre-port reference) and re-runs the same sequences through this HIP
// pipeline, diffing logits position-by-position. No CUDA/HIP calls in
// this file itself, which is why hipify-clang reported 0 converted refs
// for it (hip_port/_hipify_raw/correctness_check.cpp.hip) -- it only
// needed the headers it includes to be ported.
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "config.hpp"
#include "model_weights.hpp"
#include "paged_kv_cache.hpp"
#include "transformer.hpp"

using namespace cfg;

int main(int argc, char** argv) {
  std::string weights_path = argc > 1 ? argv[1] : "../python/weights.bin";
  std::string reference_path = argc > 2 ? argv[2] : "../python/reference.bin";

  std::ifstream f(reference_path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", reference_path.c_str()); return 1; }

  int hdr[6];
  f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  if (hdr[0] != kVocab || hdr[1] != kDModel || hdr[2] != kNHeads || hdr[3] != kNLayers ||
      hdr[4] != kMlpHidden || hdr[5] != kMaxSeqLen) {
    std::fprintf(stderr, "reference.bin config mismatch with config.hpp\n");
    return 1;
  }
  int batch_size;
  f.read(reinterpret_cast<char*>(&batch_size), sizeof(batch_size));

  printf("loading weights from %s ...\n", weights_path.c_str());
  ModelWeights weights = ModelWeights::load(weights_path);

  const int kCheckMaxSeqLen = 128;
  const int kCheckPagesPerSeq = kCheckMaxSeqLen / kPageSize;
  PagedKVCache cache(kNLayers, /*max_pages=*/kCheckPagesPerSeq * batch_size + 8, kPageSize, kNHeads, kHeadDim);
  TransformerRunner runner(weights, cache, /*max_total_rows=*/kCheckMaxSeqLen);

  double overall_max_abs = 0.0, overall_sum_abs = 0.0;
  size_t overall_count = 0;
  bool all_ok = true;

  for (int s = 0; s < batch_size; ++s) {
    int seq_len;
    f.read(reinterpret_cast<char*>(&seq_len), sizeof(seq_len));
    std::vector<int32_t> ids32(seq_len);
    f.read(reinterpret_cast<char*>(ids32.data()), seq_len * sizeof(int32_t));
    std::vector<float> ref_logits((size_t)seq_len * kVocab);
    f.read(reinterpret_cast<char*>(ref_logits.data()), ref_logits.size() * sizeof(float));

    int seq_id = cache.allocate_sequence();
    std::vector<int> token_ids(seq_len), positions(seq_len), seq_ids(seq_len);
    for (int i = 0; i < seq_len; ++i) {
      token_ids[i] = ids32[i];
      positions[i] = i;
      seq_ids[i] = seq_id;
      if (!cache.ensure_capacity(seq_id, i)) {
        std::fprintf(stderr, "out of KV cache pages during correctness check\n");
        return 1;
      }
    }

    std::vector<float> logits = runner.forward_step(token_ids, positions, seq_ids);

    double max_abs = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
      double d = std::fabs((double)logits[i] - (double)ref_logits[i]);
      max_abs = std::max(max_abs, d);
      sum_abs += d;
    }
    double mean_abs = sum_abs / logits.size();
    overall_max_abs = std::max(overall_max_abs, max_abs);
    overall_sum_abs += sum_abs;
    overall_count += logits.size();

    bool ok = max_abs < 5e-3;
    all_ok = all_ok && ok;
    printf("seq %d (len=%3d): max_abs_diff=%.6f mean_abs_diff=%.6f %s\n", s, seq_len, max_abs,
           mean_abs, ok ? "PASS" : "FAIL");

    cache.free_sequence(seq_id);
  }

  printf("\noverall: max_abs_diff=%.6f mean_abs_diff=%.6f over %zu logit values\n", overall_max_abs,
         overall_sum_abs / overall_count, overall_count);
  printf("%s\n", all_ok ? "CORRECTNESS CHECK PASSED" : "CORRECTNESS CHECK FAILED");
  weights.free_all();
  return all_ok ? 0 : 1;
}
