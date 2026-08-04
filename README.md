# Continuous-Batching Transformer Inference Server

A small GPT-style transformer served by a hand-written C++/CUDA inference
engine: a paged KV-cache (vLLM-style block allocation instead of one
contiguous per-sequence buffer), an iteration-level continuous-batching
scheduler, and a custom fused CUDA attention kernel that reads directly out
of the paged cache. Every number in this README is from an actual run on a
real GPU (NVIDIA GeForce RTX 4070 SUPER, driver-reported CUDA 12.x, built
against CUDA 12.4) -- nothing here is simulated or hand-waved.

## Why this architecture

Naive LLM serving allocates a fixed `max_seq_len`-sized KV-cache buffer per
request and waits for a full batch to arrive (or a timeout) before running
it as one fixed-size step. Two problems fall out of that:

- **Memory fragmentation.** A request that only ever generates 12 tokens
  still reserves cache space for `max_seq_len`, so a GPU that could serve
  many short requests concurrently instead sits mostly-empty per request.
- **Head-of-line blocking.** Once a static batch starts, no new request can
  join until *every* member finishes -- a single long-running request in
  the batch holds every other slot hostage, and the engine can't accept new
  traffic in the meantime.

This project implements the standard fix for both:

- **Paged KV-cache** (`include/paged_kv_cache.hpp`): cache memory is a pool
  of fixed-size pages (16 tokens each). Each sequence gets a page only when
  it actually needs one, and a finished sequence's pages go straight back
  to a free list for the next admitted request -- no request ever reserves
  more than it's using.
- **Continuous batching** (`src/server_demo.cpp`, `src/bench_batching.cpp`):
  every GPU step re-evaluates which sequences are active and flattens them
  into one token-batch on the fly. A finished sequence's slot is freed and
  a waiting request is admitted into it on the *very next* step -- no
  draining a whole batch first.

## The fused attention kernel

`kernels/attention.cu`'s `fused_attention_kernel` computes, per query row
and per head, causal-masked `softmax(QK^T / sqrt(head_dim)) @ V` in one
kernel launch, gathering K/V straight from the paged cache through a
per-row logical-page -> physical-page table (so a sequence's cached tokens
don't need to be contiguous in memory). `head_dim = 32` was chosen to equal
the CUDA warp size deliberately: one warp is assigned per `(row, head)`
pair, each lane owns one dimension of `Q`, and the dot product against each
cached `K` position is a warp-shuffle reduction (`__shfl_down_sync`) rather
than a shared-memory reduction. Softmax's max/sum reduction is similarly
warp-parallel (each lane scans a strided slice of the score buffer, then a
shuffle reduction combines them); the final `P @ V` sum is embarrassingly
parallel across lanes since each lane just accumulates its own output
dimension over the (small, <=128) key positions.

**Honest scope note:** the score loop itself is *not* parallelized across
key positions -- for each of the `kv_len` cached positions the kernel does
one warp-cooperative dot product, in sequence. For `kv_len <= 128` (this
project's `kMaxSeqLen`) that's cheap, but it's not how a production kernel
(FlashAttention-style tiling, multiple warps per query, tensor cores) would
scale to longer contexts. This kernel is fused and numerically exact; it is
not claimed to be latency-optimal.

### Correctness vs. the PyTorch reference

`python/model.py` defines the reference model (4 layers, `d_model=256`, 8
heads, GELU MLP) and `python/export_weights.py` exports its weights plus a
batch of 6 reference sequences (lengths 1, 3, 7, 16, 31, 64) run through a
real PyTorch forward pass, logits included. `src/correctness_check.cpp`
reloads those same weights into the C++/CUDA engine, runs each sequence
through a real single-sequence prefill (own paged cache, own kernel
launches, no shortcuts), and diffs every logit against the PyTorch output:

```
seq 0 (len=  1): max_abs_diff=0.000001 mean_abs_diff=0.000000 PASS
seq 1 (len=  3): max_abs_diff=0.000001 mean_abs_diff=0.000000 PASS
seq 2 (len=  7): max_abs_diff=0.000001 mean_abs_diff=0.000000 PASS
seq 3 (len= 16): max_abs_diff=0.000001 mean_abs_diff=0.000000 PASS
seq 4 (len= 31): max_abs_diff=0.000001 mean_abs_diff=0.000000 PASS
seq 5 (len= 64): max_abs_diff=0.000001 mean_abs_diff=0.000000 PASS

overall: max_abs_diff=0.000001 mean_abs_diff=0.000000 over 122000 logit values
CORRECTNESS CHECK PASSED
```

Max absolute difference across 122,000 logit values is `1e-6` -- essentially
fp32's precision floor after four transformer layers' worth of matmuls, not
an approximation. This is the strongest evidence in the repo that the
hand-written layernorm/GELU/cuBLAS-linear/fused-attention kernels actually
implement the transformer's math correctly, not just that they run.

## Continuous batching vs. static batching, measured

`src/bench_batching.cpp` runs the *same* synthetic bursty workload (240
requests, Poisson arrivals with 6ms mean interarrival, prompt lengths
4-16 tokens, decode lengths 8-32 tokens, fixed seed) through both a
continuous-batching scheduler and a naive static-batching baseline (batches
of up to 16 requests, formed after a 20ms fill window, run to full
completion with no new admissions until every member finishes), both doing
real forward passes on the GPU:

```
== continuous batching ==
  requests completed : 240
  wall clock          : 1438.0 ms
  token throughput     : 5109.8 tok/s
  latency p50/p95/p99  : 17.4 / 80.2 / 117.7 ms
  latency mean/max     : 25.7 / 132.2 ms

== static batching (baseline) ==
  requests completed : 240
  wall clock          : 1447.5 ms
  token throughput     : 5076.5 tok/s
  latency p50/p95/p99  : 45.3 / 90.0 / 112.3 ms
  latency mean/max     : 50.2 / 114.3 ms
```

**What this actually shows:** at this model size, throughput is nearly
identical between the two modes (~5100 tok/s either way) -- the GPU has
plenty of headroom for a 4-layer, 256-dim model, so static batching's
wasted capacity isn't the bottleneck here. The real, measured difference is
**median latency**: continuous batching gets a typical request done in
17.4ms vs. 45.3ms, a ~2.6x improvement, because a short request no longer
waits behind a long one that happened to land in the same static batch.
p95/p99 converge because both schedulers are bounded by the same
worst-case decode-length requests either way. This matches the standard
finding in the serving literature (e.g. Orca/vLLM): continuous batching's
headline win is tail-latency and utilization under variable-length,
bursty traffic, not necessarily raw throughput on hardware that isn't
already saturated. A larger model or higher request-rate workload would be
expected to widen the throughput gap too, since static batching's
head-of-line blocking wastes proportionally more idle GPU capacity the
more expensive each step is -- not verified here, since re-running at a
much larger model size was out of scope for this environment.

## Multithreaded server demo

`src/server_demo.cpp` is the actual multithreaded architecture: a generator
thread produces bursty requests in real time into a mutex/condvar-guarded
queue, while the main thread runs the continuous-batching scheduler and GPU
execution loop, admitting newly queued requests into free slots every
step and printing live stats:

```
[t=  0.5s] active_slots= 6 completed= 50 generated_by_source= 56 tokens/s=1698
[t=  3.0s] active_slots= 6 completed=369 generated_by_source=375 tokens/s=2013
[t=  6.0s] active_slots= 0 completed=744 generated_by_source=745 tokens/s=2000

done: 745 requests completed in 6.02s, 12037 tokens generated (1999 tok/s average)
```

(`bench_batching`'s benchmark loop is deliberately single-threaded instead
-- isolating the scheduling *policy's* effect on latency/throughput from OS
thread-scheduling jitter was more important there than demonstrating the
threading architecture, which `server_demo` covers instead.)

## Build

Developed and tested in **WSL2 Ubuntu 22.04** with GPU passthrough to an
NVIDIA GPU on the Windows host (driver installed on Windows, CUDA toolkit
12.4 installed in WSL via NVIDIA's apt repo: `cuda-nvcc-12-4`,
`cuda-cudart-dev-12-4`, `libcublas-dev-12-4`). Should build the same way on
native Linux with a CUDA-capable GPU; not tested against MSVC/native
Windows CUDA toolchains.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc
make -j$(nproc)
```

`CMAKE_CUDA_ARCHITECTURES` is set to `89` (Ada Lovelace, e.g. RTX 40-series)
in `CMakeLists.txt` -- change it for other GPU generations.

## Run

```bash
# 1. (re)generate weights.bin / reference.bin from the PyTorch reference model
cd python && python export_weights.py && cd ..

# 2. numeric correctness vs. PyTorch
./build/correctness_check python/weights.bin python/reference.bin

# 3. continuous vs. static batching benchmark
./build/bench_batching python/weights.bin

# 4. paged KV-cache unit tests (no GPU forward pass needed)
./build/correctness_test

# 5. live multithreaded server demo (~6s synthetic traffic)
./build/server_demo python/weights.bin
```

`weights.bin` (~14MB) is committed so the C++ side runs without a Python
environment; `reference.bin` is regenerated by `export_weights.py` (not
committed, since it's purely a derived correctness-check fixture).

## Layout

```
python/model.py            reference transformer definition (PyTorch)
python/export_weights.py   exports weights.bin + reference.bin
include/, kernels/         C++ headers / CUDA kernel implementations
src/                       loaders, orchestration, benchmark, server demo
tests/                     paged KV-cache unit tests
docs/gemm_layout.md        derivation of the row-major-via-cuBLAS gemm trick
```

## Known limitations

- Single GPU, fp32 throughout (no fp16/bf16/tensor cores) -- chosen for
  numerical-correctness clarity against the PyTorch fp32 reference, not
  for performance.
- Model is intentionally small (4 layers, d_model=256, 1000-token vocab) so
  the whole pipeline -- kernel writing, correctness debugging, benchmark
  iteration -- stayed fast; the architecture (paged cache, continuous
  batching, fused kernel) doesn't depend on that size and would carry over
  to a larger model, but larger-scale numbers weren't collected here.
- Greedy (argmax) decoding only; no sampling, no speculative decoding, no
  prefix/prompt caching across requests.
- The fused attention kernel's per-key-position loop is not warp-tiled for
  long contexts (see "Honest scope note" above) -- correct and fused, not
  claimed to be latency-optimal at scale.
