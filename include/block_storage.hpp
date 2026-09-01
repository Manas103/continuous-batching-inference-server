#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Host-only building blocks the tiered KV-cache manager (and its 100k-block
// stress test) use for the two colder tiers: lossless compression and an
// NVMe-backed block blob store. Neither of these touches CUDA at all, so
// they're usable (and unit-tested) without a GPU.

// Which of the three tiers a KV-cache block currently lives in.
enum class Tier { GPU, DRAM, NVME };

inline const char* tier_name(Tier t) {
  switch (t) {
    case Tier::GPU: return "GPU";
    case Tier::DRAM: return "DRAM";
    case Tier::NVME: return "NVME";
  }
  return "?";
}

// zlib deflate, level 6 (the library default -- a reasonable size/speed
// trade for float32 K/V bytes, not tuned further). Lossless: decompress()
// reproduces the exact input bytes, verified by the caller (the stress
// test does this 100,000+ times over exact byte comparison, not a hash
// alone, though it checks a SHA-256-style digest too as a second signal).
std::vector<uint8_t> zlib_compress(const void* data, size_t len);

// Decompresses into a caller-owned buffer of exactly `out_len` bytes
// (the original, known, raw length). Throws std::runtime_error if the
// decompressed size doesn't match or zlib reports an error.
void zlib_decompress(const uint8_t* comp, size_t comp_len, void* out, size_t out_len);

// A minimal append-only single-file blob store: write_block() appends
// compressed bytes and returns their (offset, length); read_block()
// pread()s them back. One growing file instead of one-file-per-block
// because a real NVMe prefix-cache tier wouldn't pay a filesystem-metadata
// open() per 32KB-128KB block either, and because the 100k-block stress
// test would otherwise create 100,000 directory entries for no benefit.
class NvmeBlockStore {
 public:
  struct Loc {
    uint64_t offset = 0;
    uint32_t length = 0;
  };

  explicit NvmeBlockStore(const std::string& path);
  ~NvmeBlockStore();
  NvmeBlockStore(const NvmeBlockStore&) = delete;
  NvmeBlockStore& operator=(const NvmeBlockStore&) = delete;

  Loc write_block(const uint8_t* data, uint32_t length);
  std::vector<uint8_t> read_block(Loc loc) const;

  uint64_t bytes_written() const { return next_offset_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  std::FILE* file_ = nullptr;
  uint64_t next_offset_ = 0;
};
