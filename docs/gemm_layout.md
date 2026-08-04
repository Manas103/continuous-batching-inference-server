# Row-major linear layers via a column-major BLAS

cuBLAS is a Fortran-heritage, column-major API. Every tensor in this project
(weights loaded from `weights.bin`, activations produced by the CUDA
kernels) is row-major, matching PyTorch's native layout and `nn.Linear`'s
`[out_features, in_features]` weight shape. Rather than transpose anything
in memory, `linear_forward` (`kernels/linear.cu`) exploits the identity

```
row-major matrix of shape (m, n)  ==  column-major matrix of shape (n, m)
```

i.e. transposing a matrix is free -- it's the same bytes, just read with the
other convention.

We want, row-major throughout: `Y[T,out] = X[T,in] @ W^T[in,out]`, where `W`
is stored row-major as `[out,in]` (nn.Linear's layout).

Transpose both sides: `Y^T[out,T] = W[out,in] @ X^T[in,T]`.

- `Y^T` read column-major `(out,T)` has exactly `Y`'s row-major `(T,out)` bytes.
- `X`'s row-major `(T,in)` bytes, read column-major, are `X^T`'s `(in,T)` --
  so `X^T` needs `CUBLAS_OP_N` with `ld=in`.
- `W`'s row-major `(out,in)` bytes, read column-major, are `W^T`'s `(in,out)`
  -- transposing *that* (`CUBLAS_OP_T`) gives `W`'s `(out,in)`, `ld=in`.

So a single call

```cpp
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            out_features, T, in_features,
            &alpha, W, in_features, X, in_features,
            &beta, Y, out_features);
```

computes `W @ X^T` column-major into `Y`'s buffer, which -- read back
row-major -- is exactly `X @ W^T`. No transpose kernel, no extra memory
traffic; this is the standard trick for driving cuBLAS from row-major
tensors and it generalizes to any row-major `C[m,n] = A[m,k] @ B[k,n]` (see
any "cuBLAS row major gemm" writeup for the fully general form -- this
project only ever needs the `B = W^T` / nn.Linear special case above).

Bias addition is a separate elementwise kernel (`add_bias_rowmajor`) rather
than folded into the gemm, since cuBLAS's bias-fusion path
(`cublasSgemmEx` with an epilogue, or `cublasLtMatmul`) adds API surface
this project didn't need for its problem size.
