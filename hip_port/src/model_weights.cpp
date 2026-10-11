#include "model_weights.hpp"
#include "config.hpp"
#include "hip_check.hpp"
#include <fstream>
#include <stdexcept>

namespace {

class BinReader {
 public:
  explicit BinReader(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    buf_.resize(n);
    f.read(reinterpret_cast<char*>(buf_.data()), n);
    pos_ = 0;
  }

  float* next_tensor(size_t count, std::vector<float*>* owned) {
    if (pos_ + count * sizeof(float) > buf_.size()) {
      throw std::runtime_error("weights.bin truncated");
    }
    float* dptr;
    CUDA_CHECK(hipMalloc(&dptr, count * sizeof(float)));
    CUDA_CHECK(hipMemcpy(dptr, buf_.data() + pos_, count * sizeof(float), hipMemcpyHostToDevice));
    pos_ += count * sizeof(float);
    owned->push_back(dptr);
    return dptr;
  }

 private:
  std::vector<uint8_t> buf_;
  size_t pos_;
};

}  // namespace

ModelWeights ModelWeights::load(const std::string& path) {
  using namespace cfg;
  BinReader r(path);
  ModelWeights w;

  w.tok_emb = r.next_tensor((size_t)kVocab * kDModel, &w.owned_);
  w.pos_emb = r.next_tensor((size_t)kMaxSeqLen * kDModel, &w.owned_);

  w.blocks.resize(kNLayers);
  for (int l = 0; l < kNLayers; ++l) {
    BlockWeights& b = w.blocks[l];
    b.ln1_w = r.next_tensor(kDModel, &w.owned_);
    b.ln1_b = r.next_tensor(kDModel, &w.owned_);
    b.wq_w = r.next_tensor((size_t)kDModel * kDModel, &w.owned_);
    b.wq_b = r.next_tensor(kDModel, &w.owned_);
    b.wk_w = r.next_tensor((size_t)kDModel * kDModel, &w.owned_);
    b.wk_b = r.next_tensor(kDModel, &w.owned_);
    b.wv_w = r.next_tensor((size_t)kDModel * kDModel, &w.owned_);
    b.wv_b = r.next_tensor(kDModel, &w.owned_);
    b.wo_w = r.next_tensor((size_t)kDModel * kDModel, &w.owned_);
    b.wo_b = r.next_tensor(kDModel, &w.owned_);
    b.ln2_w = r.next_tensor(kDModel, &w.owned_);
    b.ln2_b = r.next_tensor(kDModel, &w.owned_);
    b.fc1_w = r.next_tensor((size_t)kMlpHidden * kDModel, &w.owned_);
    b.fc1_b = r.next_tensor(kMlpHidden, &w.owned_);
    b.fc2_w = r.next_tensor((size_t)kDModel * kMlpHidden, &w.owned_);
    b.fc2_b = r.next_tensor(kDModel, &w.owned_);
  }

  w.ln_f_w = r.next_tensor(kDModel, &w.owned_);
  w.ln_f_b = r.next_tensor(kDModel, &w.owned_);
  w.lm_head = r.next_tensor((size_t)kVocab * kDModel, &w.owned_);

  return w;
}

void ModelWeights::free_all() {
  for (auto* p : owned_) hipFree(p);
  owned_.clear();
}
