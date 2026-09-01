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
// Raised from 128 -> 32768 to honestly cover the 32K-token prefix-reuse
// tiering workload (see include/tiered_kv_manager.hpp). This grows
// pos_emb from [128,256] to [32768,256] in weights.bin (the position
// table has to cover every position a real forward pass can address);
// it does not change correctness for short sequences, since unused
// page-table entries past a sequence's length are padding the attention
// kernel never dereferences (see PagedKVCache::page_table).
constexpr int kMaxSeqLen = 32768;

constexpr int kPageSize = 16;                      // tokens per KV cache page
constexpr int kPagesPerSeq = kMaxSeqLen / kPageSize;  // 2048

static_assert(kHeadDim * kNHeads == kDModel, "head dims must sum to d_model");
static_assert(kMaxSeqLen % kPageSize == 0, "seq len must be page-aligned");
}  // namespace cfg
