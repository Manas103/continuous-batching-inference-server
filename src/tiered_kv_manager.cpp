#include "tiered_kv_manager.hpp"
#include "cuda_check.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

TieredKVBlockManager::TieredKVBlockManager(int layer_count, int page_size, int n_heads, int head_dim,
                                            const std::string& nvme_scratch_path,
                                            size_t gpu_capacity_blocks, size_t dram_capacity_blocks,
                                            EvictionParams params)
    : layer_count_(layer_count), page_size_(page_size), n_heads_(n_heads), head_dim_(head_dim),
      raw_block_bytes_((size_t)layer_count * 2 * page_size * n_heads * head_dim * sizeof(float)),
      gpu_capacity_blocks_(gpu_capacity_blocks), dram_capacity_blocks_(dram_capacity_blocks),
      params_(params), nvme_(nvme_scratch_path) {
  CUDA_CHECK(cudaMallocHost((void**)&pinned_staging_, raw_block_bytes_));
}

TieredKVBlockManager::~TieredKVBlockManager() {
  if (pinned_staging_) cudaFreeHost(pinned_staging_);
}

void TieredKVBlockManager::calibrate() {
  const int kIters = 20;
  std::vector<uint8_t> host_buf(raw_block_bytes_, 0xAB);
  float* dev_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_buf, raw_block_bytes_));
  uint8_t* pinned = nullptr;
  CUDA_CHECK(cudaMallocHost((void**)&pinned, raw_block_bytes_));
  std::memcpy(pinned, host_buf.data(), raw_block_bytes_);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kIters; ++i) {
    CUDA_CHECK(cudaMemcpy(dev_buf, pinned, raw_block_bytes_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pinned, dev_buf, raw_block_bytes_, cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  params_.dram_ns_per_byte = (ms * 1e6) / (2.0 * kIters * (double)raw_block_bytes_);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dev_buf);
  cudaFreeHost(pinned);

  // Disk: a real write+read of a scratch block through the NVMe store's
  // backing file (not /dev/null, not O_DIRECT-bypassed -- whatever the OS
  // page cache does with it is exactly what the benchmark will also see).
  auto t0 = std::chrono::steady_clock::now();
  NvmeBlockStore::Loc loc = nvme_.write_block(host_buf.data(), (uint32_t)host_buf.size());
  auto t1 = std::chrono::steady_clock::now();
  std::vector<uint8_t> back = nvme_.read_block(loc);
  auto t2 = std::chrono::steady_clock::now();
  double write_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
  double read_ns = std::chrono::duration<double, std::nano>(t2 - t1).count();
  params_.nvme_ns_per_byte = (write_ns + read_ns) / (2.0 * (double)raw_block_bytes_);

  std::printf("[TieredKVBlockManager::calibrate] measured on this machine: "
              "H2D/D2H %.5f ns/byte (%.2f GB/s), disk write+read %.5f ns/byte (%.2f GB/s)\n",
              params_.dram_ns_per_byte, 1.0 / params_.dram_ns_per_byte,
              params_.nvme_ns_per_byte, 1.0 / params_.nvme_ns_per_byte);
}

std::vector<uint8_t> TieredKVBlockManager::read_raw_from_gpu(PagedKVCache& cache, int physical_page) const {
  size_t page_elems = (size_t)page_size_ * n_heads_ * head_dim_;
  size_t page_bytes = page_elems * sizeof(float);
  size_t off = 0;
  for (int l = 0; l < layer_count_; ++l) {
    CUDA_CHECK(cudaMemcpy(pinned_staging_ + off, cache.k_base(l) + (size_t)physical_page * page_elems,
                           page_bytes, cudaMemcpyDeviceToHost));
    off += page_bytes;
    CUDA_CHECK(cudaMemcpy(pinned_staging_ + off, cache.v_base(l) + (size_t)physical_page * page_elems,
                           page_bytes, cudaMemcpyDeviceToHost));
    off += page_bytes;
  }
  return std::vector<uint8_t>(pinned_staging_, pinned_staging_ + raw_block_bytes_);
}

void TieredKVBlockManager::write_raw_to_gpu(PagedKVCache& cache, int physical_page,
                                             const std::vector<uint8_t>& raw) const {
  std::memcpy(pinned_staging_, raw.data(), raw_block_bytes_);
  size_t page_elems = (size_t)page_size_ * n_heads_ * head_dim_;
  size_t page_bytes = page_elems * sizeof(float);
  size_t off = 0;
  for (int l = 0; l < layer_count_; ++l) {
    CUDA_CHECK(cudaMemcpy(cache.k_base(l) + (size_t)physical_page * page_elems, pinned_staging_ + off,
                           page_bytes, cudaMemcpyHostToDevice));
    off += page_bytes;
    CUDA_CHECK(cudaMemcpy(cache.v_base(l) + (size_t)physical_page * page_elems, pinned_staging_ + off,
                           page_bytes, cudaMemcpyHostToDevice));
    off += page_bytes;
  }
}

double TieredKVBlockManager::score(const BlockMeta& m, long now_step, Tier hypothetical_dest) const {
  double staleness = (double)(now_step - m.last_access_step);
  double est_compressed = (double)raw_block_bytes_ * params_.expected_compression_ratio;
  double restore_cost_ns = (hypothetical_dest == Tier::DRAM)
                                ? params_.dram_fixed_ns + est_compressed * params_.dram_ns_per_byte
                                : params_.nvme_fixed_ns + est_compressed * params_.nvme_ns_per_byte;
  return params_.alpha_staleness * staleness - params_.beta_restore_cost * restore_cost_ns;
}

void TieredKVBlockManager::admit_gpu_resident(uint64_t block_id, PagedKVCache& cache, int physical_page,
                                               long now_step) {
  if (meta_.count(block_id)) {
    throw std::runtime_error("admit_gpu_resident: block_id already tracked; use restore() instead");
  }
  if (gpu_resident_count_ >= gpu_capacity_blocks_) {
    evict_one(now_step);  // makes logical room; the "GPU tier full" path.
  }
  BlockMeta m;
  m.tier = Tier::GPU;
  m.last_access_step = now_step;
  m.cache = &cache;
  m.physical_page = physical_page;
  meta_.emplace(block_id, std::move(m));
  gpu_resident_count_++;
}

uint64_t TieredKVBlockManager::evict_one(long now_step) {
  if (gpu_resident_count_ == 0) return UINT64_MAX;
  Tier dest = (dram_resident_count_ < dram_capacity_blocks_) ? Tier::DRAM : Tier::NVME;

  uint64_t best_id = UINT64_MAX;
  double best_score = -1e300;
  for (auto& kv : meta_) {
    if (kv.second.tier != Tier::GPU) continue;
    double s = score(kv.second, now_step, dest);
    if (s > best_score) {
      best_score = s;
      best_id = kv.first;
    }
  }
  if (best_id == UINT64_MAX) return UINT64_MAX;
  spill(best_id, dest, now_step);
  return best_id;
}

void TieredKVBlockManager::spill(uint64_t block_id, Tier dest, long /*now_step*/) {
  auto it = meta_.find(block_id);
  if (it == meta_.end()) throw std::runtime_error("spill: unknown block_id");
  BlockMeta& m = it->second;
  if (m.tier != Tier::GPU) throw std::runtime_error("spill: block is not GPU-resident");

  std::vector<uint8_t> raw = read_raw_from_gpu(*m.cache, m.physical_page);
  std::vector<uint8_t> compressed = zlib_compress(raw.data(), raw.size());

  gpu_resident_count_--;
  m.cache = nullptr;
  m.physical_page = -1;
  m.compressed_bytes = compressed.size();

  if (dest == Tier::DRAM) {
    m.tier = Tier::DRAM;
    m.dram_bytes = std::move(compressed);
    dram_resident_count_++;
  } else {
    m.tier = Tier::NVME;
    m.nvme_loc = nvme_.write_block(compressed.data(), (uint32_t)compressed.size());
  }
}

void TieredKVBlockManager::restore(uint64_t block_id, PagedKVCache& cache, int physical_page, long now_step) {
  auto it = meta_.find(block_id);
  if (it == meta_.end()) throw std::runtime_error("restore: unknown block_id");
  BlockMeta& m = it->second;

  std::vector<uint8_t> raw;
  bool was_gpu = (m.tier == Tier::GPU);
  if (m.tier == Tier::DRAM) {
    raw.resize(raw_block_bytes_);
    zlib_decompress(m.dram_bytes.data(), m.dram_bytes.size(), raw.data(), raw.size());
    dram_resident_count_--;
    m.dram_bytes.clear();
    m.dram_bytes.shrink_to_fit();
  } else if (m.tier == Tier::NVME) {
    raw.resize(raw_block_bytes_);
    std::vector<uint8_t> compressed = nvme_.read_block(m.nvme_loc);
    zlib_decompress(compressed.data(), compressed.size(), raw.data(), raw.size());
  } else {
    raw = read_raw_from_gpu(*m.cache, m.physical_page);
  }

  write_raw_to_gpu(cache, physical_page, raw);

  if (!was_gpu && gpu_resident_count_ >= gpu_capacity_blocks_) {
    evict_one(now_step);  // this block is still marked non-GPU here, so it can't evict itself
  }

  m.tier = Tier::GPU;
  m.cache = &cache;
  m.physical_page = physical_page;
  m.compressed_bytes = 0;
  m.last_access_step = now_step;
  if (!was_gpu) gpu_resident_count_++;
}

Tier TieredKVBlockManager::tier_of(uint64_t block_id) const {
  auto it = meta_.find(block_id);
  if (it == meta_.end()) throw std::runtime_error("tier_of: unknown block_id");
  return it->second.tier;
}

void TieredKVBlockManager::touch(uint64_t block_id, long now_step) {
  auto it = meta_.find(block_id);
  if (it != meta_.end()) it->second.last_access_step = now_step;
}

BlockStats TieredKVBlockManager::stats() const {
  BlockStats s;
  for (auto& kv : meta_) {
    const BlockMeta& m = kv.second;
    s.raw_bytes_total += raw_block_bytes_;
    switch (m.tier) {
      case Tier::GPU:
        s.gpu_count++;
        break;
      case Tier::DRAM:
        s.dram_count++;
        s.compressed_bytes_total += m.compressed_bytes;
        break;
      case Tier::NVME:
        s.nvme_count++;
        s.compressed_bytes_total += m.compressed_bytes;
        break;
    }
  }
  return s;
}
