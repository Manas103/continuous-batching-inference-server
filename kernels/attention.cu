#include "attention.hpp"
#include "cuda_check.hpp"
#include <math.h>

namespace {

__global__ void scatter_kv_kernel(const float* K_proj, const float* V_proj,
                                   float* k_pages, float* v_pages,
                                   const int* write_page, const int* write_offset,
                                   int d_model, int page_size, int n_heads, int head_dim) {
  int row = blockIdx.x;
  int page = write_page[row];
  int offset = write_offset[row];
  size_t slot_base = (size_t)page * page_size * n_heads * head_dim + (size_t)offset * n_heads * head_dim;
  for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
    k_pages[slot_base + col] = K_proj[(size_t)row * d_model + col];
    v_pages[slot_base + col] = V_proj[(size_t)row * d_model + col];
  }
}

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffu, val, offset);
  }
  return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
  }
  return val;
}

// One warp per (row, head). blockDim.x == head_dim == 32.
__global__ void fused_attention_kernel(const float* Q_proj, const float* k_pages,
                                        const float* v_pages, const int* row_page_table,
                                        const int* row_query_pos, float* attn_out,
                                        int n_heads, int head_dim, int page_size,
                                        int pages_per_seq) {
  extern __shared__ float scores[];  // kv_len floats, kv_len <= kMaxSeqLen

  int row = blockIdx.x;
  int head = blockIdx.y;
  int lane = threadIdx.x;  // 0..head_dim-1
  int d_model = n_heads * head_dim;

  int qpos = row_query_pos[row];
  int kv_len = qpos + 1;

  float q_val = Q_proj[(size_t)row * d_model + head * head_dim + lane];
  const int* page_table_row = row_page_table + row * pages_per_seq;

  const float scale = rsqrtf((float)head_dim);
  for (int j = 0; j < kv_len; ++j) {
    int page = page_table_row[j / page_size];
    int offset = j % page_size;
    size_t slot = (size_t)page * page_size * n_heads * head_dim + (size_t)offset * n_heads * head_dim + head * head_dim;
    float k_val = k_pages[slot + lane];
    float dot = warp_reduce_sum(q_val * k_val);
    if (lane == 0) {
      scores[j] = dot * scale;
    }
  }
  __syncwarp();

  // Parallel max reduction over kv_len scores (each lane scans a strided slice).
  float local_max = -INFINITY;
  for (int j = lane; j < kv_len; j += warpSize) local_max = fmaxf(local_max, scores[j]);
  float row_max = warp_reduce_max(local_max);
  row_max = __shfl_sync(0xffffffffu, row_max, 0);

  float local_sum = 0.f;
  for (int j = lane; j < kv_len; j += warpSize) {
    float e = expf(scores[j] - row_max);
    scores[j] = e;
    local_sum += e;
  }
  float row_sum = warp_reduce_sum(local_sum);
  row_sum = __shfl_sync(0xffffffffu, row_sum, 0);
  __syncwarp();

  float inv_sum = 1.0f / row_sum;
  float acc = 0.f;
  for (int j = 0; j < kv_len; ++j) {
    int page = page_table_row[j / page_size];
    int offset = j % page_size;
    size_t slot = (size_t)page * page_size * n_heads * head_dim + (size_t)offset * n_heads * head_dim + head * head_dim;
    float v_val = v_pages[slot + lane];
    acc += (scores[j] * inv_sum) * v_val;
  }
  attn_out[(size_t)row * d_model + head * head_dim + lane] = acc;
}

}  // namespace

void scatter_kv_to_cache(const float* K_proj, const float* V_proj, float* k_pages,
                          float* v_pages, const int* write_page, const int* write_offset,
                          int total_rows, int d_model, int page_size, int n_heads,
                          int head_dim, cudaStream_t stream) {
  int threads = d_model < 256 ? d_model : 256;
  scatter_kv_kernel<<<total_rows, threads, 0, stream>>>(K_proj, V_proj, k_pages, v_pages,
                                                          write_page, write_offset, d_model,
                                                          page_size, n_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
}

void fused_paged_causal_attention(const float* Q_proj, const float* k_pages,
                                   const float* v_pages, const int* row_page_table,
                                   const int* row_query_pos, float* attn_out,
                                   int total_rows, int n_heads, int head_dim,
                                   int page_size, int pages_per_seq, cudaStream_t stream) {
  dim3 grid(total_rows, n_heads);
  dim3 block(head_dim);
  size_t shmem_bytes = pages_per_seq * page_size * sizeof(float);  // upper bound on kv_len
  fused_attention_kernel<<<grid, block, shmem_bytes, stream>>>(
      Q_proj, k_pages, v_pages, row_page_table, row_query_pos, attn_out, n_heads,
      head_dim, page_size, pages_per_seq);
  CUDA_CHECK(cudaGetLastError());
}
