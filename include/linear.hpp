#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Row-major Y[T,out] = X[T,in] @ W^T + b, where W is stored row-major
// [out,in] -- exactly nn.Linear's weight layout, so weights.bin can be
// loaded straight into device memory with no transpose step. b may be
// nullptr (lm_head has no bias).
void linear_forward(cublasHandle_t handle, const float* X, const float* W,
                     const float* b, float* Y, int T, int in_features,
                     int out_features, cudaStream_t stream);
