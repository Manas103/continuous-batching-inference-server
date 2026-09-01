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
//
// Online (flash-attention-style) softmax: instead of materializing all
// kv_len scores in a shared-memory array (the original version of this
// kernel did exactly that, in an `extern __shared__ float scores[]` sized
// to kMaxSeqLen -- see README "what broke" section), each lane streams
// over key positions once, keeping a running max `m`, a running
// normalizer `l`, and a running weighted-V accumulator `acc` (one float
// per lane, since each lane already owns exactly one output dimension).
// On seeing a new score s_j:
//   m_new   = max(m, s_j)
//   correction = exp(m - m_new)          (rescales everything seen so far)
//   l       = l * correction + exp(s_j - m_new)
//   acc     = acc * correction + exp(s_j - m_new) * v_j[lane]
//   m       = m_new
// After the loop, attn_out = acc / l. This is mathematically identical to
// the two-pass max-then-sum softmax (same result to fp32 rounding) but
// needs O(1) extra state per lane instead of O(kv_len) shared memory, so
// kv_len -- and therefore sequence length -- is no longer bounded by the
// GPU's shared-memory budget.
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

  float m = -INFINITY;   // running max of scores seen so far
  float l = 0.f;         // running softmax normalizer
  float acc = 0.f;       // running weighted-V accumulator, this lane's output dim

  for (int j = 0; j < kv_len; ++j) {
    int page = page_table_row[j / page_size];
    int offset = j % page_size;
    size_t slot = (size_t)page * page_size * n_heads * head_dim + (size_t)offset * n_heads * head_dim + head * head_dim;

    float k_val = k_pages[slot + lane];
    float dot = warp_reduce_sum(q_val * k_val);       // whole warp now holds the same dot (lane 0 computed it)
    float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

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
  // No dynamic shared memory needed any more: the online-softmax rewrite
  // keeps only O(1) running state (m, l, acc) per lane in registers,
  // instead of an O(kv_len) shared-memory scores buffer.
  fused_attention_kernel<<<grid, block, 0, stream>>>(
      Q_proj, k_pages, v_pages, row_page_table, row_query_pos, attn_out, n_heads,
      head_dim, page_size, pages_per_seq);
  CUDA_CHECK(cudaGetLastError());
}
