# Port failure taxonomy: CUDA to HIP, this repo's inference path

Scope: `kernels/attention.cu`, `kernels/elementwise.cu`, `kernels/linear.cu`,
and the host glue they're called from (`model_weights.cpp`,
`paged_kv_cache.cpp`, `transformer.cpp`, `correctness_check.cpp`). This is
the CUDA repo's inference-forward-pass slice, the part the 122,000-logit
parity check exercises; the tiering extension (DRAM/NVMe spill, zlib) was
not ported in this pass.

Tool: `hipify-clang` 18.0.0 (ROCm 6.2.2), run against each file with
`--print-stats`; raw tool output in `docs/hipify_raw_output.txt`.

## 6 failures, 4 causes

**Cause 1: __shfl*_sync intrinsics have no HIP equivalent (3 of 6).**

`kernels/attention.cu`'s `warp_reduce_sum`, `warp_reduce_max`, and the
single-lane broadcast after the dot product all use
`__shfl_down_sync(0xffffffffu, val, offset)` / `__shfl_sync(0xffffffffu,
dot, 0)`. hipify-clang's own tool output flags all three as "unsupported
in HIP" (`docs/hipify_raw_output.txt`, and the unfixed tool output
preserved at `docs/hipify_raw/attention.cu.hip.before_fix`) and leaves
them untouched rather than guessing a replacement -- the only
tool-reported `UNCONVERTED refs` in the whole file set.

Root cause: the `_sync` suffix and explicit lane mask exist on NVIDIA
because Volta and later give each of a warp's 32 lanes an independent
program counter, so a shuffle needs to state which lanes are actually
still at that point. AMD's wavefronts (64 lanes on CDNA, gfx942 included)
have no such per-lane divergence state, so HIP exposes only the
unconditional `__shfl_down`/`__shfl`. Fix (`kernels/attention.hip.cpp`):
drop the mask and the `_sync` suffix. Verified identical to 1e-6 (see
Measured results) and verified to compile for gfx942
(`docs/gfx942_compile_output.txt`).

This is the largest cause by instance count, matching the general shape
of CUDA-to-HIP ports that use warp-level primitives: the independent-
thread-scheduling intrinsics (`_sync` shuffles, `__activemask`,
cooperative-groups constructs built on them) are the recurring NVIDIA-only
category.

**Cause 2: a cuBLAS v1-macro call site hipify-clang's text pass does not
match (1 of 6).**

`kernels/linear.cu` calls `cublasSgemm(handle, CUBLAS_OP_T, ...)`. hipify
correctly converts the handle type (`cublasHandle_t` ->
`hipblasHandle_t`), the enums (`CUBLAS_OP_T` -> `HIPBLAS_OP_T`), and
`cublasSetStream` -> `hipblasSetStream`, but leaves the GEMM call itself
as the bare identifier `cublasSgemm` (see
`docs/hipify_raw/linear.cu.hip.before_fix`, line 27, and the tool's own
warning in `docs/hipify_raw_output.txt`: *"Only 'cublasSgemm_v2' version
of 'cublasSgemm' is supported in 'HIP'"*). `<cublas_v2.h>` aliases
`cublasSgemm` to `cublasSgemm_v2` via a preprocessor macro; hipify's
pattern matcher keys on the literal `_v2`-suffixed spelling, so the
macro'd call site is invisible to it. Left as-is this does not compile
(the surrounding types no longer match the real `cublasSgemm`
declaration it would resolve to). This is subsumed by cause 3 below: the
fix taken was to not call any cuBLAS/hipBLAS symbol at all.

**Cause 3: cuBLAS's HIP path (hipBLAS) has no lightweight equivalent for a
single-GPU build (1 of 6, a dependency-availability failure rather than a
syntax one).**

Even with cause 2's call site fixed, linking hipBLAS pulls `rocblas` and
`rocsolver` through hipBLAS's package dependencies regardless of which
backend actually executes -- `hipblas-dev -> hipblas -> rocblas,
rocsolver` (`apt-cache depends hipblas-dev`, this machine, Oct 2026). Their
installed sizes alone are 5,844,706 KiB and 3,383,024 KiB (`apt-cache show
rocblas rocsolver`), roughly 8.8 GiB, for a GEMM that on this one
consumer GPU running through HIP's CUDA backend would just forward to
cuBLAS underneath anyway. Fix (`kernels/linear.hip.cpp`): a hand-written
shared-memory-tiled HIP GEMM kernel instead of hipBLAS. It is explicitly
not a cuBLAS-class kernel (fixed 16x16 tile, no tensor cores, no
autotuning) -- see Measured results for the resulting throughput cost,
reported honestly rather than hidden.

**Cause 4: hipify-clang's bundled CUDA-parsing frontend does not support
this machine's CUDA 12.4 (1 of 6, a toolchain/version-skew failure, not a
code failure).**

Every file in scope failed hipify-clang's host-side AST parse against
`/usr/local/cuda-12.4`'s headers before any pattern-matching could run,
with two distinct symptoms in clang's CUDA compatibility shim
(`__clang_cuda_runtime_wrapper.h`): a function-declarator parse error in
`crt/math_functions.hpp`'s `rsqrt` forward declaration, and a missing
`texture_fetch_functions.h`/`curand_mtgp32_kernel.h` once the first error
was worked around. ROCm 6.2.2's hipify-clang build warns explicitly:
*"CUDA version is newer than the latest supported version 12.3"*
(`docs/hipify_raw_output.txt`). Fix: installed a side-by-side,
headers-and-nvcc-only CUDA 12.3 toolkit
(`cuda-nvcc-12-3`, `cuda-cudart-dev-12-3`, `libcurand-dev-12-3`,
`libcublas-dev-12-3`) and pointed only `hipify-clang --cuda-path` at it;
the actual HIP-nvidia-backend build (`make check`) still compiles through
CUDA 12.4's own `nvcc` via `hipcc`, unaffected, since hipcc dispatches
straight to nvcc on the nvidia platform rather than through ROCm's clang.

## What this is not

This is a measured count from one porting pass over one project's
inference path, not a general CUDA-to-HIP failure rate. A port of a
different kernel set (e.g. one using cooperative groups, PTX inline asm,
or texture objects) would surface a different mix. The claim this repo's
resume makes is about *this* port; the taxonomy above is what was
actually encountered doing it, reported honestly rather than padded to a
round number.
