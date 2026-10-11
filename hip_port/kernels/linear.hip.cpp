#include <hip/hip_runtime.h>
#include "linear.hpp"
#include "hip_check.hpp"
#include <algorithm>

// Port failure (cause 3, docs/port_failure_taxonomy.md): the CUDA original
// calls cublasSgemm for this layer. hipify-clang converts the surrounding
// types (cublasHandle_t -> hipblasHandle_t, CUBLAS_OP_T -> HIPBLAS_OP_T;
// see hip_port/_hipify_raw/linear.cu.hip) but leaves the call-site symbol
// itself as the bare name "cublasSgemm" (CUDA's <cublas_v2.h> aliases
// cublasSgemm to cublasSgemm_v2 via a preprocessor #define, and hipify's
// textual pass only matches the literal "cublasSgemm_v2" spelling -- see
// the tool's own warning, captured verbatim in
// hip_port/_hipify_raw/linear.cu.hip). Left as-is this does not compile:
// hipblasHandle_t/HIPBLAS_OP_T don't match the cublasSgemm declaration it
// would need to resolve against.
//
// Separately, and the bigger reason this file does not use hipBLAS even
// with the symbol fixed: hipBLAS's package dependency chain (hipblas-dev
// -> hipblas -> rocblas, rocsolver) pulls rocBLAS and rocSOLVER, whose
// AMD-target Tensile GEMM kernel libraries alone are a measured ~8.8 GiB
// installed (apt-cache show rocblas + rocsolver on this machine, Oct 2026
// ROCm 6.2.2: Installed-Size 5,844,706 KiB + 3,383,024 KiB) -- for a GEMM
// that, on this single consumer GPU running
// through HIP's CUDA backend, would just be forwarding to cuBLAS anyway.
// That is not justified for one port, so this file ships a hand-written
// tiled GEMM kernel instead. It is deliberately not a cuBLAS-class kernel
// (no tensor cores, no autotuning, a fixed 16x16 tile) -- see Measured
// results in the README for how much slower that makes it, stated
// honestly rather than hidden.

namespace {

constexpr int kTile = 16;

// Y[T,out] = X[T,in] @ W^T + b. W is row-major [out,in] (nn.Linear layout).
// Classic shared-memory-tiled GEMM: each block computes one kTile x kTile
// tile of Y, looping over kTile-wide slices of the K (=in_features)
// dimension, staging one tile of X and one tile of W^T into shared memory
// per iteration so each element is read from global memory once per tile
// instead of once per output element.
__global__ void tiled_linear_kernel(const float* X, const float* W, const float* b,
                                     float* Y, int T, int in_features, int out_features) {
  __shared__ float Xs[kTile][kTile];
  __shared__ float Ws[kTile][kTile];

  int row = blockIdx.y * kTile + threadIdx.y;  // row in [0, T)
  int col = blockIdx.x * kTile + threadIdx.x;  // col in [0, out_features)

  // Accumulate in double: a genuine attempt (see docs/port_failure_taxonomy.md
  // measurement note) to close the gap to cuBLAS's own internal accumulation
  // precision, since this GEMM's per-element sum over in_features terms is
  // exactly where fp32-vs-fp32-with-wider-internal-accumulation differences
  // between this kernel and cuBLAS's algorithm would show up.
  double acc = 0.0;
  int n_tiles = (in_features + kTile - 1) / kTile;
  for (int t = 0; t < n_tiles; ++t) {
    int k = t * kTile;

    Xs[threadIdx.y][threadIdx.x] = (row < T && k + threadIdx.x < in_features)
        ? X[(size_t)row * in_features + k + threadIdx.x] : 0.f;
    // W is [out_features, in_features]; we need W^T's (k, col) element,
    // which is W[col, k] -- i.e. row `col` of W, column `k`.
    Ws[threadIdx.y][threadIdx.x] = (col < out_features && k + threadIdx.y < in_features)
        ? W[(size_t)col * in_features + k + threadIdx.y] : 0.f;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTile; ++kk) {
      acc += (double)Xs[threadIdx.y][kk] * (double)Ws[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < T && col < out_features) {
    if (b != nullptr) acc += (double)b[col];
    Y[(size_t)row * out_features + col] = (float)acc;
  }
}

}  // namespace

void linear_forward(const float* X, const float* W, const float* b, float* Y,
                     int T, int in_features, int out_features, hipStream_t stream) {
  dim3 block(kTile, kTile);
  dim3 grid((out_features + kTile - 1) / kTile, (T + kTile - 1) / kTile);
  tiled_linear_kernel<<<grid, block, 0, stream>>>(X, W, b, Y, T, in_features, out_features);
  CUDA_CHECK(hipGetLastError());
}
