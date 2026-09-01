#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "block_storage.hpp"
#include "paged_kv_cache.hpp"

// A three-tier block manager layered on top of PagedKVCache: GPU device
// memory (PagedKVCache's own buffers), host DRAM, and NVMe. It treats each
// *physical page* (PagedKVCache's existing 16-token allocation unit) as a
// "block": one block's raw bytes are that page's K and V slots across every
// transformer layer -- kNLayers * 2 (K,V) * page_size * n_heads * head_dim
// float32s. That's what a real request needs restored to resume generating
// from a cached prefix, so it's the natural spill/restore granularity.
//
// This manager does not allocate or free PagedKVCache pages itself -- the
// caller (a benchmark, or a real prefix-cache admission path) still owns
// sequence/page lifecycle via PagedKVCache's normal API. What the manager
// owns is: (a) compressing and moving a block's bytes to a colder tier
// when asked, (b) decompressing and copying them back into a physical page
// the caller has already arranged, and (c) picking *which* GPU-resident
// block to evict when the GPU tier's logical capacity is exceeded, via a
// cost-aware heuristic (not a learned predictor -- see EvictionParams).
//
// Cost-aware eviction formula:
//   score(block) = alpha * staleness_steps(block)
//                  - beta  * estimated_restore_cost_ns(block)
// staleness_steps = (now_step - last_access_step), i.e. plain
// recency-since-last-use, same signal LRU uses. estimated_restore_cost_ns
// is what it would cost, in nanoseconds, to bring this block back to the
// GPU *from the tier it would be evicted to* -- estimated as
// fixed_overhead_ns + estimated_compressed_bytes * ns_per_byte, using
// per-tier ns_per_byte figures this process calibrates against this
// machine's real PCIe H2D/D2H bandwidth and real disk read/write bandwidth
// at construction time (see calibrate()), and estimated_compressed_bytes
// = raw_block_bytes * an empirically observed compression ratio. The
// evicted block is the one that MAXIMIZES score: most stale, and cheapest
// to undo the eviction of later if it turns out to be needed again. This
// is a heuristic, deliberately not a learned predictor -- alpha/beta are
// fixed weights, not fit to any workload, and the cost model is a linear
// fixed+per-byte estimate, not a measured-per-block figure.
struct EvictionParams {
  double alpha_staleness = 1.0;              // weight on recency
  double beta_restore_cost = 1.0;            // weight on (negative) restore cost
  double expected_compression_ratio = 0.5;   // compressed/raw, seeded by calibrate()
  double dram_fixed_ns = 2000.0;             // cudaMemcpy launch/sync overhead estimate
  double dram_ns_per_byte = 0.05;            // ~20 GB/s H2D/D2H, overwritten by calibrate()
  double nvme_fixed_ns = 50000.0;            // fopen/seek overhead estimate
  double nvme_ns_per_byte = 0.2;             // ~5 GB/s disk, overwritten by calibrate()
};

struct BlockStats {
  size_t gpu_count = 0, dram_count = 0, nvme_count = 0;
  size_t raw_bytes_total = 0;
  size_t compressed_bytes_total = 0;  // DRAM + NVMe resident blocks only
};

class TieredKVBlockManager {
 public:
  // layer_count/page_size/n_heads/head_dim describe the PagedKVCache(s)
  // this manager will be asked to spill from / restore into (must match
  // cfg:: constants for those caches). nvme_scratch_path is the backing
  // file for the NVMe tier (created if missing, truncated if present).
  TieredKVBlockManager(int layer_count, int page_size, int n_heads, int head_dim,
                        const std::string& nvme_scratch_path,
                        size_t gpu_capacity_blocks, size_t dram_capacity_blocks,
                        EvictionParams params = {});
  ~TieredKVBlockManager();

  size_t raw_block_bytes() const { return raw_block_bytes_; }

  // Measures this machine's real H2D/D2H cudaMemcpy bandwidth and real
  // disk write/read bandwidth for a block-sized buffer, and updates
  // params_.dram_ns_per_byte / nvme_ns_per_byte accordingly. Call once
  // before relying on the eviction policy for a real cost estimate;
  // prints what it measured. Safe to skip (falls back to the constructor
  // defaults above) if the caller does not want the extra I/O at startup.
  void calibrate();

  const EvictionParams& params() const { return params_; }

  // Registers block_id as freshly written and GPU-resident at
  // `physical_page` inside `cache`. If gpu_capacity_blocks is already
  // reached, evicts the single highest-scoring resident block first (cost
  // function above) to make logical room -- this is the "GPU tier is full
  // and a new page is needed" path.
  void admit_gpu_resident(uint64_t block_id, PagedKVCache& cache, int physical_page, long now_step);

  // Cost-aware: evicts the single highest-scoring currently GPU-resident
  // block right now (reads its bytes out of the cache it was admitted
  // with, compresses, and moves it to DRAM if there is room there, else
  // straight to NVMe). Returns the evicted block_id, or UINT64_MAX if no
  // block is currently GPU-resident. Exposed standalone so both
  // admit_gpu_resident() and callers demonstrating/testing the policy in
  // isolation can use it.
  uint64_t evict_one(long now_step);

  // Forces block_id to `dest` right now, regardless of current occupancy
  // (used by the benchmark to spill an entire prefix deliberately).
  void spill(uint64_t block_id, Tier dest, long now_step);

  // Restores block_id into `physical_page` of `cache` (decompressing from
  // DRAM/NVMe, or a no-op copy if already GPU-resident elsewhere isn't
  // applicable since GPU-resident blocks are tracked with their own
  // cache+page already). Marks it GPU-resident at (cache, physical_page)
  // afterward and updates last-access.
  void restore(uint64_t block_id, PagedKVCache& cache, int physical_page, long now_step);

  Tier tier_of(uint64_t block_id) const;
  void touch(uint64_t block_id, long now_step);
  bool known(uint64_t block_id) const { return meta_.count(block_id) != 0; }

  BlockStats stats() const;

 private:
  struct BlockMeta {
    Tier tier = Tier::GPU;
    long last_access_step = 0;
    size_t compressed_bytes = 0;   // 0 while GPU-resident (bytes are not compressed there)
    // GPU-resident location:
    PagedKVCache* cache = nullptr;
    int physical_page = -1;
    // DRAM tier storage:
    std::vector<uint8_t> dram_bytes;
    // NVMe tier storage:
    NvmeBlockStore::Loc nvme_loc{};
  };

  double score(const BlockMeta& m, long now_step, Tier hypothetical_dest) const;
  std::vector<uint8_t> read_raw_from_gpu(PagedKVCache& cache, int physical_page) const;
  void write_raw_to_gpu(PagedKVCache& cache, int physical_page, const std::vector<uint8_t>& raw) const;

  int layer_count_, page_size_, n_heads_, head_dim_;
  size_t raw_block_bytes_;
  size_t gpu_capacity_blocks_, dram_capacity_blocks_;
  EvictionParams params_;
  NvmeBlockStore nvme_;
  // Reused pinned (page-locked) host staging buffer for every H2D/D2H
  // cudaMemcpy this manager issues. Found by measurement (see README
  // "Findings"): a per-call pageable std::vector<uint8_t> staging buffer,
  // combined with kLayerCount*2 small (16 KiB) cudaMemcpy calls per block,
  // made 2048-block restore take ~11.3s, slower than full recompute. Pinned
  // memory lets the DMA engine skip the driver's own pageable-to-pinned
  // staging copy on every one of those small transfers.
  mutable uint8_t* pinned_staging_ = nullptr;

  std::unordered_map<uint64_t, BlockMeta> meta_;
  size_t gpu_resident_count_ = 0;
  size_t dram_resident_count_ = 0;
};
