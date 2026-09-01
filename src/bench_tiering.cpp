// Claim: "cutting time-to-first-token 3.1x versus full recompute on a
// 32K-token prefix-reuse workload".
//
// Two real forward-pass paths, same synthetic 32768-token prompt, same
// model, same GPU:
//
//   full recompute: chunked prefill (512 tokens/chunk, 64 chunks) of the
//   whole prompt through TransformerRunner, then one decode step to
//   produce the first new token. Timed start to finish.
//
//   prefix reuse: the SAME prefix's real KV blocks (already computed by
//   the recompute run above, not re-derived) are spilled through
//   TieredKVBlockManager's cost-aware eviction across GPU/DRAM/NVMe with
//   zlib compression, into a *second*, otherwise-empty PagedKVCache
//   simulating a new incoming request. Timed: restoring every block into
//   that second cache, then running the one decode step needed to produce
//   the first new token.
//
// The ratio of the two elapsed times is the claim.
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include "config.hpp"
#include "model_weights.hpp"
#include "paged_kv_cache.hpp"
#include "transformer.hpp"
#include "tiered_kv_manager.hpp"

using namespace cfg;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kPromptLen = 32768;         // exactly kMaxSeqLen
constexpr int kChunk = 512;               // chunked-prefill step size
constexpr int kGpuTierCapacityBlocks = 512;   // 8192 tokens stay GPU-resident
constexpr int kDramTierCapacityBlocks = 1024; // next 16384 tokens spill to DRAM
// remaining 512 blocks (8192 tokens, the oldest) spill to NVMe

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int argmax_row(const float* row, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) if (row[i] > row[best]) best = i;
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_path = argc > 1 ? argv[1] : "python/weights.bin";
  printf("loading weights from %s ...\n", weights_path.c_str());
  ModelWeights weights = ModelWeights::load(weights_path);

  static_assert(kPromptLen == kMaxSeqLen, "bench_tiering assumes a full-length 32768-token prefix");
  const int total_pages = kPromptLen / kPageSize;  // 2048

  std::mt19937 rng(9001);
  std::uniform_int_distribution<int> token_dist(0, kVocab - 1);
  std::vector<int> prompt(kPromptLen);
  for (auto& t : prompt) t = token_dist(rng);

  printf("workload: prompt_len=%d tokens (%d pages of %d), chunked prefill in %d-token steps\n\n",
         kPromptLen, total_pages, kPageSize, kChunk);

  // ---------------- Path A: full recompute ----------------
  PagedKVCache cache1(kNLayers, /*max_pages=*/total_pages + 4, kPageSize, kNHeads, kHeadDim);
  TransformerRunner runner1(weights, cache1, /*max_total_rows=*/kChunk);
  int seq1 = cache1.allocate_sequence();

  auto t0 = Clock::now();
  int last_token = 0;
  for (int start = 0; start < kPromptLen; start += kChunk) {
    int len = std::min(kChunk, kPromptLen - start);
    std::vector<int> token_ids(len), positions(len), seq_ids(len, seq1);
    for (int i = 0; i < len; ++i) {
      int pos = start + i;
      if (!cache1.ensure_capacity(seq1, pos)) { fprintf(stderr, "OOM pages (recompute prefill)\n"); return 1; }
      token_ids[i] = prompt[pos];
      positions[i] = pos;
    }
    std::vector<float> logits = runner1.forward_step(token_ids, positions, seq_ids);
    if (start + len == kPromptLen) last_token = argmax_row(&logits[(size_t)(len - 1) * kVocab], kVocab);
  }
  // the one additional decode step that produces the first new token
  cache1.ensure_capacity(seq1, kPromptLen);
  runner1.forward_step({last_token}, {kPromptLen}, {seq1});
  auto t1 = Clock::now();
  double ttft_full_recompute_ms = ms_between(t0, t1);
  printf("full recompute TTFT (prefill + 1 decode step): %.2f ms\n\n", ttft_full_recompute_ms);

  // ---------------- Spill this real prefix's blocks (untimed setup) ----------------
  // cache1's 2048 pages now hold the real, just-computed K/V for the whole
  // prefix. Register every page as a manager-tracked block and let the
  // cost-aware policy push the least-recently-touched ones down to DRAM,
  // then NVMe, as the (deliberately small, relative to the full prefix)
  // GPU-tier capacity is exceeded -- exactly the "GPU tier full, need a
  // new page" path the eviction policy exists for.
  TieredKVBlockManager mgr(kNLayers, kPageSize, kNHeads, kHeadDim, "bench_tiering_scratch.nvme",
                            kGpuTierCapacityBlocks, kDramTierCapacityBlocks);
  mgr.calibrate();

  std::vector<int> table1 = cache1.page_table(seq1);  // 2048 real physical page ids, in token order
  for (int j = 0; j < total_pages; ++j) {
    mgr.admit_gpu_resident((uint64_t)j, cache1, table1[j], /*now_step=*/j);
  }
  BlockStats spill_stats = mgr.stats();
  size_t spilled_raw_bytes = (spill_stats.dram_count + spill_stats.nvme_count) * mgr.raw_block_bytes();
  double compression_ratio = spill_stats.compressed_bytes_total > 0
                                  ? (double)spilled_raw_bytes / (double)spill_stats.compressed_bytes_total
                                  : 0.0;
  printf("after spill: %zu blocks GPU-resident, %zu DRAM, %zu NVMe (of %d total)\n",
         spill_stats.gpu_count, spill_stats.dram_count, spill_stats.nvme_count, total_pages);
  printf("compression: %zu raw bytes -> %zu compressed bytes (%.3fx)\n\n",
         spilled_raw_bytes, spill_stats.compressed_bytes_total, compression_ratio);

  // ---------------- Path B: prefix reuse ----------------
  PagedKVCache cache2(kNLayers, /*max_pages=*/total_pages + 4, kPageSize, kNHeads, kHeadDim);
  TransformerRunner runner2(weights, cache2, /*max_total_rows=*/4);
  int seq2 = cache2.allocate_sequence();

  auto r0 = Clock::now();
  for (int j = 0; j < total_pages; ++j) {
    int pos = j * kPageSize;
    if (!cache2.ensure_capacity(seq2, pos)) { fprintf(stderr, "OOM pages (reuse restore)\n"); return 1; }
    int phys_page, offset;
    cache2.resolve_slot(seq2, pos, &phys_page, &offset);
    mgr.restore((uint64_t)j, cache2, phys_page, /*now_step=*/1'000'000 + j);
  }
  // the one additional decode step (same shape/cost as path A's; the
  // specific token id fed here doesn't affect this step's latency, only
  // its generated content, which this benchmark isn't measuring)
  cache2.ensure_capacity(seq2, kPromptLen);
  runner2.forward_step({last_token}, {kPromptLen}, {seq2});
  auto r1 = Clock::now();
  double ttft_reuse_ms = ms_between(r0, r1);
  printf("prefix reuse TTFT (restore %d blocks + 1 decode step): %.2f ms\n\n", total_pages, ttft_reuse_ms);

  double speedup = ttft_full_recompute_ms / ttft_reuse_ms;
  printf("== result ==\n");
  printf("full recompute TTFT : %.2f ms\n", ttft_full_recompute_ms);
  printf("prefix reuse TTFT   : %.2f ms\n", ttft_reuse_ms);
  printf("speedup             : %.2fx\n", speedup);

  cache1.free_sequence(seq1);
  cache2.free_sequence(seq2);
  weights.free_all();
  return 0;
}
