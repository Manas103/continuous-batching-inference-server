#pragma once
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                    \
  do {                                                                      \
    cudaError_t err_ = (expr);                                              \
    if (err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, \
                   __LINE__, cudaGetErrorString(err_));                     \
      std::exit(1);                                                        \
    }                                                                       \
  } while (0)

#define CUBLAS_CHECK(expr)                                                   \
  do {                                                                       \
    cublasStatus_t st_ = (expr);                                             \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error %s at %s:%d: status=%d\n", #expr,   \
                   __FILE__, __LINE__, static_cast<int>(st_));               \
      std::exit(1);                                                         \
    }                                                                        \
  } while (0)
