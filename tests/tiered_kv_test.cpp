// Functional correctness tests for the three-tier KV-cache block manager:
// spill/restore round-trips through DRAM and NVMe reproduce the exact
// original bytes, and the cost-aware eviction policy picks the block the
// documented formula (include/tiered_kv_manager.hpp) says it should. Needs
// a GPU (uses real PagedKVCache device buffers), unlike
// tests/correctness_test.cpp's page-bookkeeping checks. The 100,000-block
// bit-identical restore claim has its own dedicated stress test:
// src/spill_restore_stress.cpp.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include "config.hpp"
#include "cuda_check.hpp"
#include "paged_kv_cache.hpp"
#include "tiered_kv_manager.hpp"

using namespace cfg;

static int g_checks = 0;
static int g_failures = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    ++g_checks;                                                         \
    if (!(cond)) {                                                      \
      ++g_failures;                                                     \
      std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__,     \
                   __LINE__, #cond);                                    \
    }                                                                   \
  } while (0)

namespace {

size_t block_bytes() {
  return (size_t)kNLayers * 2 * kPageSize * kNHeads * kHeadDim * sizeof(float);
}

// Fills physical_page in `cache` with deterministic pseudo-random K/V bytes
// and returns them (the "known original" restores are later compared
// against byte-for-byte).
std::vector<uint8_t> fill_page_with_random(PagedKVCache& cache, int physical_page, uint32_t seed) {
  size_t page_elems = (size_t)kPageSize * kNHeads * kHeadDim;
  size_t page_bytes = page_elems * sizeof(float);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-10.f, 10.f);

  std::vector<uint8_t> raw(block_bytes());
  std::vector<float> tmp(page_elems);
  size_t off = 0;
  for (int l = 0; l < kNLayers; ++l) {
    for (auto& v : tmp) v = dist(rng);
    std::memcpy(raw.data() + off, tmp.data(), page_bytes);
    CUDA_CHECK(cudaMemcpy(cache.k_base(l) + (size_t)physical_page * page_elems, tmp.data(),
                           page_bytes, cudaMemcpyHostToDevice));
    off += page_bytes;

    for (auto& v : tmp) v = dist(rng);
    std::memcpy(raw.data() + off, tmp.data(), page_bytes);
    CUDA_CHECK(cudaMemcpy(cache.v_base(l) + (size_t)physical_page * page_elems, tmp.data(),
                           page_bytes, cudaMemcpyHostToDevice));
    off += page_bytes;
  }
  return raw;
}

std::vector<uint8_t> read_page(PagedKVCache& cache, int physical_page) {
  size_t page_elems = (size_t)kPageSize * kNHeads * kHeadDim;
  size_t page_bytes = page_elems * sizeof(float);
  std::vector<uint8_t> raw(block_bytes());
  size_t off = 0;
  for (int l = 0; l < kNLayers; ++l) {
    CUDA_CHECK(cudaMemcpy(raw.data() + off, cache.k_base(l) + (size_t)physical_page * page_elems,
                           page_bytes, cudaMemcpyDeviceToHost));
    off += page_bytes;
    CUDA_CHECK(cudaMemcpy(raw.data() + off, cache.v_base(l) + (size_t)physical_page * page_elems,
                           page_bytes, cudaMemcpyDeviceToHost));
    off += page_bytes;
  }
  return raw;
}

void test_dram_roundtrip() {
  PagedKVCache cache(kNLayers, /*max_pages=*/4, kPageSize, kNHeads, kHeadDim);
  TieredKVBlockManager mgr(kNLayers, kPageSize, kNHeads, kHeadDim, "tiered_test_dram.nvme",
                            /*gpu_capacity_blocks=*/4, /*dram_capacity_blocks=*/4);

  auto original = fill_page_with_random(cache, /*physical_page=*/0, /*seed=*/1);
  mgr.admit_gpu_resident(/*block_id=*/100, cache, /*physical_page=*/0, /*now_step=*/0);
  CHECK(mgr.tier_of(100) == Tier::GPU);

  mgr.spill(100, Tier::DRAM, /*now_step=*/1);
  CHECK(mgr.tier_of(100) == Tier::DRAM);

  mgr.restore(100, cache, /*physical_page=*/2, /*now_step=*/2);
  CHECK(mgr.tier_of(100) == Tier::GPU);
  auto restored = read_page(cache, 2);
  CHECK(restored.size() == original.size());
  CHECK(std::memcmp(restored.data(), original.data(), original.size()) == 0);
}

void test_nvme_roundtrip() {
  PagedKVCache cache(kNLayers, /*max_pages=*/4, kPageSize, kNHeads, kHeadDim);
  TieredKVBlockManager mgr(kNLayers, kPageSize, kNHeads, kHeadDim, "tiered_test_nvme.nvme",
                            /*gpu_capacity_blocks=*/4, /*dram_capacity_blocks=*/4);

  auto original = fill_page_with_random(cache, 1, /*seed=*/2);
  mgr.admit_gpu_resident(200, cache, 1, 0);
  mgr.spill(200, Tier::NVME, 1);
  CHECK(mgr.tier_of(200) == Tier::NVME);

  mgr.restore(200, cache, 3, 2);
  auto restored = read_page(cache, 3);
  CHECK(std::memcmp(restored.data(), original.data(), original.size()) == 0);
}

void test_eviction_picks_most_stale() {
  PagedKVCache cache(kNLayers, /*max_pages=*/4, kPageSize, kNHeads, kHeadDim);
  TieredKVBlockManager mgr(kNLayers, kPageSize, kNHeads, kHeadDim, "tiered_test_evict.nvme",
                            /*gpu_capacity_blocks=*/2, /*dram_capacity_blocks=*/4);

  fill_page_with_random(cache, 0, 10);
  fill_page_with_random(cache, 1, 11);
  mgr.admit_gpu_resident(/*block_id=*/1, cache, 0, /*now_step=*/0);  // touched at step 0
  mgr.admit_gpu_resident(/*block_id=*/2, cache, 1, /*now_step=*/5);  // touched at step 5, fresher
  CHECK(mgr.tier_of(1) == Tier::GPU);
  CHECK(mgr.tier_of(2) == Tier::GPU);

  // Admitting a 3rd block at capacity 2 must evict the more stale one
  // (block 1, last touched at step 0) per score() = alpha*staleness -
  // beta*restore_cost: both blocks are the same size and would be evicted
  // to the same destination tier, so restore_cost is identical and
  // staleness alone decides.
  fill_page_with_random(cache, 2, 12);
  mgr.admit_gpu_resident(/*block_id=*/3, cache, 2, /*now_step=*/10);

  CHECK(mgr.tier_of(1) == Tier::DRAM);  // evicted
  CHECK(mgr.tier_of(2) == Tier::GPU);   // stays, was fresher
  CHECK(mgr.tier_of(3) == Tier::GPU);
}

void test_eviction_spills_to_nvme_once_dram_full() {
  PagedKVCache cache(kNLayers, /*max_pages=*/8, kPageSize, kNHeads, kHeadDim);
  TieredKVBlockManager mgr(kNLayers, kPageSize, kNHeads, kHeadDim, "tiered_test_dram_full.nvme",
                            /*gpu_capacity_blocks=*/1, /*dram_capacity_blocks=*/1);

  fill_page_with_random(cache, 0, 20);
  mgr.admit_gpu_resident(1, cache, 0, 0);  // resident
  fill_page_with_random(cache, 1, 21);
  mgr.admit_gpu_resident(2, cache, 1, 1);  // evicts block 1 -> DRAM (has room)
  CHECK(mgr.tier_of(1) == Tier::DRAM);
  fill_page_with_random(cache, 2, 22);
  mgr.admit_gpu_resident(3, cache, 2, 2);  // evicts block 2 -> DRAM now full (cap=1) -> NVMe
  CHECK(mgr.tier_of(2) == Tier::NVME);
}

void test_stats() {
  PagedKVCache cache(kNLayers, /*max_pages=*/4, kPageSize, kNHeads, kHeadDim);
  TieredKVBlockManager mgr(kNLayers, kPageSize, kNHeads, kHeadDim, "tiered_test_stats.nvme",
                            /*gpu_capacity_blocks=*/4, /*dram_capacity_blocks=*/4);
  fill_page_with_random(cache, 0, 30);
  mgr.admit_gpu_resident(1, cache, 0, 0);
  BlockStats s = mgr.stats();
  CHECK(s.gpu_count == 1);
  CHECK(s.raw_bytes_total == block_bytes());
}

}  // namespace

int main() {
  test_dram_roundtrip();
  test_nvme_roundtrip();
  test_eviction_picks_most_stale();
  test_eviction_spills_to_nvme_once_dram_full();
  test_stats();

  std::printf("%d/%d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
