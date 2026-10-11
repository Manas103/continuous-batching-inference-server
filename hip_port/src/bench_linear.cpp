// Throughput of the hand-written tiled HIP GEMM kernel that replaces
// cuBLAS in this port (see include/linear.hpp, kernels/linear.hip.cpp,
// docs/port_failure_taxonomy.md cause 3). Not a claim on the resume; a
// secondary number showing honestly what the no-hipBLAS decision costs.
#include <cstdio>
#include <vector>
#include <chrono>
#include "linear.hpp"
#include "hip_check.hpp"

static double bench_one(int T, int in_features, int out_features, int iters) {
  std::vector<float> hX((size_t)T * in_features, 0.01f);
  std::vector<float> hW((size_t)out_features * in_features, 0.02f);
  std::vector<float> hB(out_features, 0.0f);

  float *dX, *dW, *dB, *dY;
  CUDA_CHECK(hipMalloc(&dX, hX.size() * sizeof(float)));
  CUDA_CHECK(hipMalloc(&dW, hW.size() * sizeof(float)));
  CUDA_CHECK(hipMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(hipMalloc(&dY, (size_t)T * out_features * sizeof(float)));
  CUDA_CHECK(hipMemcpy(dX, hX.data(), hX.size() * sizeof(float), hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(dW, hW.data(), hW.size() * sizeof(float), hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice));

  hipStream_t stream;
  CUDA_CHECK(hipStreamCreate(&stream));

  // warm-up
  for (int i = 0; i < 5; ++i) linear_forward(dX, dW, dB, dY, T, in_features, out_features, stream);
  CUDA_CHECK(hipStreamSynchronize(stream));

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) linear_forward(dX, dW, dB, dY, T, in_features, out_features, stream);
  CUDA_CHECK(hipStreamSynchronize(stream));
  auto t1 = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  double gflops = 2.0 * (double)T * in_features * out_features / (ms * 1e6);

  printf("T=%5d in=%5d out=%5d : %8.4f ms/call, %8.2f GFLOP/s\n", T, in_features, out_features, ms, gflops);

  hipFree(dX); hipFree(dW); hipFree(dB); hipFree(dY);
  hipStreamDestroy(stream);
  return ms;
}

int main() {
  printf("hand-written tiled HIP GEMM (16x16 tile, no tensor cores, no autotuning)\n");
  printf("shapes match this repo's TinyGPT forward pass (d_model=256, mlp_hidden=1024)\n\n");
  // T=64: one full-length reference sequence's worth of rows (see
  // python/export_weights.py's longest seq_len).
  bench_one(64, 256, 256, 200);     // wq/wk/wv/wo
  bench_one(64, 256, 1024, 200);    // fc1
  bench_one(64, 1024, 256, 200);    // fc2
  bench_one(64, 256, 1000, 200);    // lm_head
  return 0;
}
