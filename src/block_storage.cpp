#include "block_storage.hpp"
#include <zlib.h>
#include <stdexcept>
#include <cerrno>
#include <cstring>

std::vector<uint8_t> zlib_compress(const void* data, size_t len) {
  uLongf bound = compressBound((uLong)len);
  std::vector<uint8_t> out(bound);
  uLongf out_len = bound;
  // Z_BEST_SPEED, not the zlib default: this repo's own measurement (see
  // README "Findings") is that K/V float32 bytes barely compress either way
  // (mantissas essentially never repeat), so the higher default level buys
  // negligible extra ratio for a real, meaningful slowdown at the 100,000
  // block/12.8GB scale this stress test runs at. Still real, still lossless
  // deflate; only the effort level changed.
  int rc = compress2(out.data(), &out_len, reinterpret_cast<const Bytef*>(data), (uLong)len,
                      /*level=*/Z_BEST_SPEED);
  if (rc != Z_OK) throw std::runtime_error("zlib_compress: compress2 failed, rc=" + std::to_string(rc));
  out.resize(out_len);
  return out;
}

void zlib_decompress(const uint8_t* comp, size_t comp_len, void* out, size_t out_len) {
  uLongf dest_len = (uLongf)out_len;
  int rc = uncompress(reinterpret_cast<Bytef*>(out), &dest_len, comp, (uLong)comp_len);
  if (rc != Z_OK) throw std::runtime_error("zlib_decompress: uncompress failed, rc=" + std::to_string(rc));
  if (dest_len != out_len) throw std::runtime_error("zlib_decompress: size mismatch after decompression");
}

NvmeBlockStore::NvmeBlockStore(const std::string& path) : path_(path) {
  file_ = std::fopen(path.c_str(), "w+b");
  if (!file_) throw std::runtime_error("NvmeBlockStore: cannot open " + path + ": " + std::strerror(errno));
}

NvmeBlockStore::~NvmeBlockStore() {
  if (file_) std::fclose(file_);
}

NvmeBlockStore::Loc NvmeBlockStore::write_block(const uint8_t* data, uint32_t length) {
  Loc loc{next_offset_, length};
#if defined(_WIN32)
  if (_fseeki64(file_, (long long)loc.offset, SEEK_SET) != 0)
#else
  if (fseeko(file_, (off_t)loc.offset, SEEK_SET) != 0)
#endif
    throw std::runtime_error("NvmeBlockStore::write_block: seek failed");
  size_t written = std::fwrite(data, 1, length, file_);
  if (written != length) throw std::runtime_error("NvmeBlockStore::write_block: short write");
  next_offset_ += length;
  return loc;
}

std::vector<uint8_t> NvmeBlockStore::read_block(Loc loc) const {
  std::vector<uint8_t> buf(loc.length);
#if defined(_WIN32)
  if (_fseeki64(file_, (long long)loc.offset, SEEK_SET) != 0)
#else
  if (fseeko(file_, (off_t)loc.offset, SEEK_SET) != 0)
#endif
    throw std::runtime_error("NvmeBlockStore::read_block: seek failed");
  size_t got = std::fread(buf.data(), 1, loc.length, file_);
  if (got != loc.length) throw std::runtime_error("NvmeBlockStore::read_block: short read");
  return buf;
}
