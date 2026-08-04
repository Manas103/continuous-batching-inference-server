#pragma once
#include <cuda_runtime.h>

// out[row] = tok_emb[token_ids[row]] + pos_emb[positions[row]], each a
// d_model-length row-major slice. T rows total.
void embed_lookup_add(const int* token_ids, const int* positions,
                       const float* tok_emb, const float* pos_emb, float* out,
                       int T, int d_model, cudaStream_t stream);

// Y = LayerNorm(X) * gamma + beta, row-wise over d_model, eps=1e-5 (matches
// torch.nn.LayerNorm's default).
void layernorm_forward(const float* X, const float* gamma, const float* beta,
                        float* Y, int T, int d_model, cudaStream_t stream);

// In-place GELU (tanh approximation matches torch.nn.functional.gelu's
// default 'none' exact-erf form closely enough for fp32 tolerance here --
// we use the exact erf form to match PyTorch's default precisely).
void gelu_forward(float* X, int T, int width, cudaStream_t stream);

// X += Y, elementwise, T rows of width `width`.
void residual_add(float* X, const float* Y, int T, int width, cudaStream_t stream);
