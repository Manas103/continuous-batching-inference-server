#include <hip/hip_runtime.h>
#include "attention.hpp"
#include "hip_check.hpp"
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

// Port failure (cause 1, docs/port_failure_taxonomy.md): the CUDA original
// used __shfl_down_sync/__shfl_sync, which hipify-clang flags as
// "unsupported in HIP" (verified: hip_port/_hipify_raw/attention.cu.hip,
// 3 unconverted refs). The _sync intrinsics exist on NVIDIA because
// Volta+ warps have independent per-thread program counters and need an
// explicit mask of which of the warp's 32 lanes are actually
// participating; AMD's wavefronts (64 lanes on CDNA, gfx942 included)
// have no such per-lane divergence state, so HIP exposes only the
// unconditional, mask-free __shfl_down/__shfl -- which is also exactly
// warp_reduce_sum's real semantics here (every lane in the block always
// participates, there is no divergent subset). Fix: drop the mask and the
// _sync suffix. Functionally identical on this GPU since HIP's CUDA
// backend still lowers __shfl_down to the hardware's native full-warp
// shuffle either way.
__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset);
  }
  return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down(val, offset));
  }
  return val;
}

// One warp per (row, head). blockDim.x == head_dim == 32.
//
// Second warp-size-32 assumption, not a build failure but a scope note
// (docs/port_failure_taxonomy.md, cause 1 continued): this kernel's grid
// still launches blocks of exactly head_dim (32) threads, so on a real
// gfx942 part each block would occupy only half of one 64-lane wavefront,
// wasting the other 32 lanes rather than corrupting results (HIP/ROCm
// supports sub-wavefront block sizes; it is just not occupancy-efficient).
// Fixing that properly means doubling head_dim or packing two (row,head)
// pairs per wavefront, which is an occupancy redesign out of scope for a
// numerical-parity port -- recorded here, not silently absorbed.
__global__ void fused_attention_kernel(const float* Q_proj, const float* k_pages,
                                        const float* v_pages, const int* row_page_table,
                                        const int* row_query_pos, float* attn_out,
                                        int n_heads, int head_dim, int page_size,
                                        int pages_per_seq) {
  int row = blockIdx.x;
  int head = blockIdx.y;
  int lane = threadIdx.x;  // 0..head_dim-1
  int d_model = n_heads * head_dim;

  int qpos = row_query_pos[row];
  int kv_len = qpos + 1;

  float q_val = Q_proj[(size_t)row * d_model + head * head_dim + lane];
  const int* page_table_row = row_page_table + row * pages_per_seq;
  const float scale = rsqrtf((float)head_dim);

  float m = -INFINITY;
  float l = 0.f;
  float acc = 0.f;

  for (int j = 0; j < kv_len; ++j) {
    int page = page_table_row[j / page_size];
    int offset = j % page_size;
    size_t slot = (size_t)page * page_size * n_heads * head_dim + (size_t)offset * n_heads * head_dim + head * head_dim;

    float k_val = k_pages[slot + lane];
    float dot = warp_reduce_sum(q_val * k_val);
    float score = __shfl(dot, 0) * scale;

    float m_new = fmaxf(m, score);
    float correction = expf(m - m_new);
    float p = expf(score - m_new);

    float v_val = v_pages[slot + lane];
    acc = acc * correction + p * v_val;
    l = l * correction + p;
    m = m_new;
  }

  attn_out[(size_t)row * d_model + head * head_dim + lane] = acc / l;
}

}  // namespace

void scatter_kv_to_cache(const float* K_proj, const float* V_proj, float* k_pages,
                          float* v_pages, const int* write_page, const int* write_offset,
                          int total_rows, int d_model, int page_size, int n_heads,
                          int head_dim, hipStream_t stream) {
  int threads = d_model < 256 ? d_model : 256;
  scatter_kv_kernel<<<total_rows, threads, 0, stream>>>(K_proj, V_proj, k_pages, v_pages,
                                                          write_page, write_offset, d_model,
                                                          page_size, n_heads, head_dim);
  CUDA_CHECK(hipGetLastError());
}

void fused_paged_causal_attention(const float* Q_proj, const float* k_pages,
                                   const float* v_pages, const int* row_page_table,
                                   const int* row_query_pos, float* attn_out,
                                   int total_rows, int n_heads, int head_dim,
                                   int page_size, int pages_per_seq, hipStream_t stream) {
  dim3 grid(total_rows, n_heads);
  dim3 block(head_dim);
  fused_attention_kernel<<<grid, block, 0, stream>>>(
      Q_proj, k_pages, v_pages, row_page_table, row_query_pos, attn_out, n_heads,
      head_dim, page_size, pages_per_seq);
  CUDA_CHECK(hipGetLastError());
}
