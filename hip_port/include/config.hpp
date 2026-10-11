#pragma once
// Mirrors the CUDA original's include/config.hpp exactly (same constants,
// same python/weights.bin and python/reference.bin), so this port's
// correctness check is diffed against the exact same pre-port reference
// the CUDA build was checked against, not a freshly regenerated one.
namespace cfg {
constexpr int kVocab = 1000;
constexpr int kDModel = 256;
constexpr int kNHeads = 8;
constexpr int kHeadDim = kDModel / kNHeads;  // 32
constexpr int kNLayers = 4;
constexpr int kMlpHidden = 4 * kDModel;
constexpr int kMaxSeqLen = 32768;
constexpr int kPageSize = 16;
constexpr int kPagesPerSeq = kMaxSeqLen / kPageSize;  // 2048

static_assert(kHeadDim * kNHeads == kDModel, "head dims must sum to d_model");
static_assert(kMaxSeqLen % kPageSize == 0, "seq len must be page-aligned");
}  // namespace cfg
