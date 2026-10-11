#pragma once
#include <hip/hip_runtime.h>

// Row-major Y[T,out] = X[T,in] @ W^T + b, where W is stored row-major
// [out,in] -- exactly nn.Linear's weight layout, so weights.bin can be
// loaded straight into device memory with no transpose step. b may be
// nullptr (lm_head has no bias).
//
// The CUDA original calls into cuBLAS (cublasSgemm). This port does not:
// hipBLAS's AMD backend (rocBLAS, plus rocSOLVER/rocSPARSE it depends on)
// is a multi-GB dependency that buys nothing on a single consumer GPU run
// through HIP's CUDA backend, where the matmul already lowers to a plain
// cublasSgemm underneath. See docs/port_failure_taxonomy.md, cause 3. This
// file instead ships a hand-written tiled HIP GEMM kernel; it is not a
// cuBLAS-class GEMM (no tensor cores, no autotuned tile sizes) and is not
// claimed to be fast, only numerically correct -- see Measured results.
void linear_forward(const float* X, const float* W, const float* b, float* Y,
                     int T, int in_features, int out_features,
                     hipStream_t stream);
