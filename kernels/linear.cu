#include "linear.hpp"
#include "cuda_check.hpp"
#include <algorithm>

namespace {
__global__ void add_bias_rowmajor(float* Y, const float* b, int T, int out_features) {
  int row = blockIdx.x;
  int col = threadIdx.x + blockIdx.y * blockDim.x;
  if (row < T && col < out_features) {
    Y[row * out_features + col] += b[col];
  }
}
}  // namespace

void linear_forward(cublasHandle_t handle, const float* X, const float* W,
                     const float* b, float* Y, int T, int in_features,
                     int out_features, cudaStream_t stream) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  // See include/linear.hpp: derivation of the row-major-via-cuBLAS trick is
  // in docs/gemm_layout.md. W read as CUBLAS_OP_T of its (in,out) column-major
  // view (== W's own (out,in) row-major bytes) times X read as CUBLAS_OP_N of
  // its (in,T) column-major view (== X's own (T,in) row-major bytes) yields a
  // (out,T) column-major result whose bytes are exactly Y's (T,out) row-major.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, T,
                            in_features, &alpha, W, in_features, X,
                            in_features, &beta, Y, out_features));
  if (b != nullptr) {
    dim3 block(std::min(out_features, 256));
    dim3 grid(T, (out_features + block.x - 1) / block.x);
    add_bias_rowmajor<<<grid, block, 0, stream>>>(Y, b, T, out_features);
    CUDA_CHECK(cudaGetLastError());
  }
}
