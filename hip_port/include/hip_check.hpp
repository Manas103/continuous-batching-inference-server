#pragma once
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define CUDA_CHECK(expr)                                                    \
  do {                                                                      \
    hipError_t err_ = (expr);                                              \
    if (err_ != hipSuccess) {                                              \
      std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, \
                   __LINE__, hipGetErrorString(err_));                      \
      std::exit(1);                                                        \
    }                                                                        \
  } while (0)
