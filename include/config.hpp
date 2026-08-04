#pragma once
// Must mirror python/model.py exactly -- these are the fixed dimensions of
// the reference model whose weights.bin/reference.bin this server loads.
namespace cfg {
constexpr int kVocab = 1000;
constexpr int kDModel = 256;
constexpr int kNHeads = 8;
constexpr int kHeadDim = kDModel / kNHeads;  // 32, deliberately == warp size
constexpr int kNLayers = 4;
constexpr int kMlpHidden = 4 * kDModel;
constexpr int kMaxSeqLen = 128;

constexpr int kPageSize = 16;                      // tokens per KV cache page
constexpr int kPagesPerSeq = kMaxSeqLen / kPageSize;  // 8

static_assert(kHeadDim * kNHeads == kDModel, "head dims must sum to d_model");
static_assert(kMaxSeqLen % kPageSize == 0, "seq len must be page-aligned");
}  // namespace cfg
