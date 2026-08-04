#include "elementwise.hpp"
#include "cuda_check.hpp"
#include <math.h>

namespace {

__global__ void embed_lookup_add_kernel(const int* token_ids, const int* positions,
                                         const float* tok_emb, const float* pos_emb,
                                         float* out, int d_model) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  int tok = token_ids[row];
  int pos = positions[row];
  for (; col < d_model; col += blockDim.x) {
    out[row * d_model + col] = tok_emb[tok * d_model + col] + pos_emb[pos * d_model + col];
  }
}

// One block per row. Two-pass block reduction for mean/variance, then
// normalize. d_model is small (256) so a single block comfortably covers it.
__global__ void layernorm_kernel(const float* X, const float* gamma, const float* beta,
                                  float* Y, int d_model) {
  extern __shared__ float shmem[];
  float* sum_buf = shmem;
  float* sumsq_buf = shmem + blockDim.x;

  int row = blockIdx.x;
  const float* x_row = X + row * d_model;
  float* y_row = Y + row * d_model;

  float local_sum = 0.f, local_sumsq = 0.f;
  for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
    float v = x_row[col];
    local_sum += v;
    local_sumsq += v * v;
  }
  sum_buf[threadIdx.x] = local_sum;
  sumsq_buf[threadIdx.x] = local_sumsq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_buf[threadIdx.x] += sum_buf[threadIdx.x + stride];
      sumsq_buf[threadIdx.x] += sumsq_buf[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float mean = sum_buf[0] / d_model;
  float var = sumsq_buf[0] / d_model - mean * mean;
  float inv_std = rsqrtf(var + 1e-5f);

  for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
    float normed = (x_row[col] - mean) * inv_std;
    y_row[col] = normed * gamma[col] + beta[col];
  }
}

__global__ void gelu_kernel(float* X, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = X[i];
    // Exact erf-based GELU, matching F.gelu's default approximate='none'.
    X[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
  }
}

__global__ void residual_add_kernel(float* X, const float* Y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    X[i] += Y[i];
  }
}

}  // namespace

void embed_lookup_add(const int* token_ids, const int* positions,
                       const float* tok_emb, const float* pos_emb, float* out,
                       int T, int d_model, cudaStream_t stream) {
  int threads = d_model < 256 ? d_model : 256;
  embed_lookup_add_kernel<<<T, threads, 0, stream>>>(token_ids, positions, tok_emb,
                                                       pos_emb, out, d_model);
  CUDA_CHECK(cudaGetLastError());
}

void layernorm_forward(const float* X, const float* gamma, const float* beta,
                        float* Y, int T, int d_model, cudaStream_t stream) {
  int threads = d_model < 256 ? d_model : 256;
  size_t shmem_bytes = 2 * threads * sizeof(float);
  layernorm_kernel<<<T, threads, shmem_bytes, stream>>>(X, gamma, beta, Y, d_model);
  CUDA_CHECK(cudaGetLastError());
}

void gelu_forward(float* X, int T, int width, cudaStream_t stream) {
  int n = T * width;
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  gelu_kernel<<<blocks, threads, 0, stream>>>(X, n);
  CUDA_CHECK(cudaGetLastError());
}

void residual_add(float* X, const float* Y, int T, int width, cudaStream_t stream) {
  int n = T * width;
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  residual_add_kernel<<<blocks, threads, 0, stream>>>(X, Y, n);
  CUDA_CHECK(cudaGetLastError());
}
