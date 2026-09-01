# Continuous-Batching Transformer Inference Server, with a Tiered KV-Cache Manager

A small GPT-style transformer served by a hand-written C++/CUDA inference
engine: a paged KV-cache (vLLM-style block allocation instead of one
contiguous per-sequence buffer), an iteration-level continuous-batching
scheduler, a custom fused CUDA attention kernel that reads directly out of
the paged cache, and (the extension documented in this pass) a three-tier
block manager that spills a sequence's KV-cache pages from GPU HBM to host
DRAM to NVMe under a cost-aware eviction policy with lossless zlib
compression. Every number in this README is from an actual run on a real
GPU (NVIDIA GeForce RTX 4070 SUPER, driver-reported CUDA 12.x, built against
CUDA 12.4) -- nothing here is simulated or hand-waved, including the one
claim below that was not met after genuine debugging.

**Honest framing, up front:**
- The tiering extension's headline claim (time-to-first-token cut 3.1x by
  reusing a spilled 32K-token prefix instead of recomputing it) was **not
  met**: prefix reuse measured *slower* than full recompute (0.45x, after a
  real fix; 0.35x before it). Root cause and the two genuine fix attempts
  are in Findings below. This is reported as measured, not adjusted.
- The 100,000-spilled-block bit-identical-restore claim **was met**: 0
  mismatches over 100,000 blocks (`docs/spill_restore_stress_output.txt`).
- "Cost-aware" eviction is a fixed linear heuristic (staleness weighed
  against an estimated restore cost), not a learned predictor.

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
one warp-cooperative dot product, in sequence. `kMaxSeqLen` was raised from
128 to 32768 for the tiering extension (see Findings: the original version
materialized all `kv_len` scores in a shared-memory array, which does not
fit once `kv_len` reaches tens of thousands; the kernel was rewritten to an
online/flash-attention-style softmax, see below, specifically to remove
that ceiling). Even at 32768, the per-key loop is still sequential per
warp, not warp-tiled across multiple warps or using tensor cores -- this
kernel is fused and numerically exact; it is not claimed to be
latency-optimal at scale.

**Online (flash-attention-style) softmax.** The kernel keeps a running max
`m`, running normalizer `l`, and running weighted-V accumulator `acc` (one
float per lane, since each lane already owns one output dimension) instead
of materializing `scores[kv_len]` in shared memory: for each key position,
`m_new = max(m, score)`, `correction = exp(m - m_new)` rescales everything
accumulated so far, `l` and `acc` fold in the new term, and `m = m_new`.
Mathematically identical to the original two-pass max-then-sum softmax
(verified below to the same 1e-6 tolerance), but needs O(1) shared memory
per warp instead of O(kv_len), so sequence length is no longer bounded by
the GPU's shared-memory budget.

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

## Tiered KV-cache manager: GPU HBM to host DRAM to NVMe

`include/tiered_kv_manager.hpp` / `src/tiered_kv_manager.cpp` layer a
three-tier block manager on top of `PagedKVCache`. It treats each physical
page (the existing 16-token allocation unit) as a "block": one block's raw
bytes are that page's K and V slots across every transformer layer. The
manager does not own sequence/page lifecycle itself (the caller still owns
that via `PagedKVCache`'s normal API); it owns compressing and moving a
block's bytes to a colder tier, decompressing and copying them back, and
picking which GPU-resident block to evict when the GPU tier's logical
capacity is exceeded.

**Cost-aware eviction formula:** `score(block) = alpha * staleness_steps -
beta * estimated_restore_cost_ns`, evicting the highest-scoring resident
block (most stale, cheapest to undo later). `estimated_restore_cost_ns` is
`fixed_overhead + estimated_compressed_bytes * ns_per_byte`, using
per-tier `ns_per_byte` figures `calibrate()` measures against this
machine's real H2D/D2H and disk bandwidth at startup (`docs/tiering_benchmark_output.txt`
shows the calibrated figures: 4.87 GB/s H2D/D2H, 0.83 GB/s disk write+read
on this machine). `alpha`/`beta` are fixed weights (both 1.0), not fit to
any workload -- this is a heuristic, explicitly not a learned predictor.

**Lossless compression:** `include/block_storage.hpp` wraps zlib deflate
(`Z_BEST_SPEED`, see Findings for why) around every block moved to DRAM or
NVMe. Restoring decompresses back to the exact original bytes, verified by
exact byte comparison (not a hash alone) over 100,000 blocks (Validation
below).

**NVMe backing store:** `NvmeBlockStore` is one growing file (append-only
writes, `(offset, length)` locators), not one file per block -- a real
NVMe prefix-cache tier would not pay a filesystem-metadata `open()` per
128 KiB block either, and a 100,000-block stress test would otherwise
create 100,000 directory entries for no benefit.

## Validation: tiering

- **Correctness after the kernel rewrite:** `correctness_check` re-passes
  at the same 1e-6 tolerance over the same 122,000 logits after both the
  online-softmax rewrite and the `kMaxSeqLen` 128 -> 32768 change
  (`docs/correctness_check_output.txt`), confirming the rewrite changed
  performance characteristics, not the math.
- **`tiered_kv_test`** (16 checks, host-only, no GPU forward pass): DRAM
  round-trip (spill then restore reproduces the exact original bytes),
  NVMe round-trip, cost-aware eviction picks the correct (most stale)
  block and leaves a fresher one GPU-resident, cascading DRAM-to-NVMe
  spill once DRAM capacity is reached, and `BlockStats` accounting.
  `docs/tiered_kv_test_output.txt`.
- **100,000-block spill/restore stress** (`spill_restore_stress`, pure
  host-side, no GPU needed): spills >=100,000 deterministic-PRNG-seeded
  128 KiB blocks through compress -> write(NVMe) -> read -> decompress and
  checks every one against both a 64-bit FNV-1a checksum and an exact
  `memcmp`. `docs/spill_restore_stress_output.txt`.

```
$ ./spill_restore_stress 100000 /tmp/scratch/blocks.bin
  blocks                    : 100000
  block size                : 131072 bytes
  raw total                 : 13.11 GB
  compressed total (NVMe)   : 12.17 GB (ratio 1.077x)
  spill  (compress+write)   : 345.2 s  (290 blocks/s)
  restore (read+decompress) : 88.9 s  (1125 blocks/s)
  mismatches                : 0 / 100000
SPILL/RESTORE STRESS PASSED (bit-identical)
```

## Findings: tiering

**A global shared-memory ceiling had to be removed before "32K tokens"
could be honest.** The original attention kernel stored all `kv_len`
per-row scores in a `extern __shared__ float scores[]` array sized to
`kMaxSeqLen`. At `kMaxSeqLen=128` that's 512 bytes, trivial; at 32768
(needed for the resume's 32K-token prefix-reuse claim) that's 128 KiB per
block, which exceeds even this GPU's opt-in dynamic shared-memory limit.
Fix: rewrite the kernel to an online (flash-attention-style) softmax that
keeps only O(1) running state per lane instead of O(kv_len) shared memory
(see above). Verified numerically identical to the original two-pass
softmax to the same 1e-6 tolerance after the rewrite.

**The tiering TTFT benchmark did not meet its 3.1x target -- it measured
slower than full recompute, on this workload, on this machine, after two
genuine fix attempts.** `bench_tiering` runs a real 32768-token chunked
prefill (the "full recompute" baseline), spills that real prefix's 2048
KV-cache blocks through the manager, then times restoring all 2048 blocks
into a second cache plus one decode step (the "prefix reuse" path).

- *First measurement:* 0.35x (recompute 3967ms, restore 11273ms). Root
  cause: `read_raw_from_gpu`/`write_raw_to_gpu` issued 8 small (16 KiB)
  synchronous `cudaMemcpy` calls per block (one per layer, per K/V, since
  each transformer layer is a physically separate device allocation) into
  a fresh, pageable (`std::vector<uint8_t>`) host buffer every call -- for
  2048 blocks that is 16,384 small pageable-memory copies, and pageable
  memory forces an extra driver-side staging copy on every one.
- *Fix attempt 1 (this session):* replaced the per-call pageable buffer
  with one persistent pinned (`cudaMallocHost`) staging buffer reused
  across every call. Measured: 0.45x (restore 8839ms), a real 22%
  reduction in restore time, still short of the target and still slower
  than recompute.
- *Root cause of the remaining gap:* the bottleneck is the *count* of
  small CUDA API calls (16,384 of them), not primarily their individual
  cost -- each transformer layer's K and V live in physically separate
  device allocations (`PagedKVCache` design, one buffer per layer), so the
  8-calls-per-block structure is load-bearing, not an oversight. Removing
  it would require batching the copy across *many blocks at once* per
  layer (one large `cudaMemcpy` covering, say, all 512 NVMe-tier blocks'
  worth of a given layer's K slots in one call), which requires those
  blocks' physical pages to be contiguous in the destination cache and a
  reworked manager API that restores a batch rather than one block at a
  time. That is a real, understood next step, not attempted in this
  session's remaining budget; a second genuine attempt (the pinned buffer)
  was made and measured honestly, and this repo reports the result rather
  than a third attempt at a larger structural change.
- **What this does still show, honestly:** the 100,000-block stress test
  above proves the spill/restore/compression path is correct and can push
  ~300-1100 blocks/sec sustained; the gap is specifically that *this
  benchmark's* one-block-at-a-time restore loop, on WSL2's virtualized GPU
  passthrough, pays per-call CUDA API overhead that a real serving system
  restoring a whole prefix at once would batch away.

## Measured results: tiering

Machine: WSL2 Ubuntu 22.04, RTX 4070 SUPER (sm_89), CUDA 12.4, g++ 11.4,
`-O2`, zlib 1.2.11 (`Z_BEST_SPEED`).

**Claim: "cutting time-to-first-token 3.1x versus full recompute on a
32K-token prefix-reuse workload."** Measured **0.45x** (prefix reuse is
slower, not faster): full recompute 3958.30ms, prefix reuse (restore 2048
blocks + 1 decode step) 8839.13ms. **Not met**, two genuine attempts, root
cause above (`docs/tiering_benchmark_output.txt`).

**Claim: "100,000 spilled blocks restored bit-identically."** Measured
**100,000 / 100,000**, 0 mismatches, verified by both a 64-bit FNV-1a
checksum and an exact byte-for-byte `memcmp` per block. **Met**
(`docs/spill_restore_stress_output.txt`).

**Secondary numbers, for context, not claims on the resume:** compression
ratio on this workload's (unstructured, high-entropy) synthetic K/V bytes
was 1.075-1.078x -- modest, because float32 mantissas from a normal
distribution essentially never repeat, whatever the compression level;
`Z_BEST_SPEED` was chosen over the zlib default specifically because the
higher default level buys negligible extra ratio here for real extra time
at the 100,000-block/12.8GB stress-test scale.

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
`cuda-cudart-dev-12-4`, `libcublas-dev-12-4`). The tiering extension also
needs zlib (`zlib1g-dev` on Ubuntu; `find_package(ZLIB REQUIRED)` in
`CMakeLists.txt`). Should build the same way on native Linux with a
CUDA-capable GPU; not tested against MSVC/native Windows CUDA toolchains.

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

# 6. tiered KV-cache manager unit tests (host-only, no GPU forward pass)
./build/tiered_kv_test

# 7. 100,000-block spill/restore stress test (host-only)
./build/spill_restore_stress 100000 /tmp/scratch/blocks.bin

# 8. tiering TTFT benchmark (32K-token prefix reuse vs. full recompute)
./build/bench_tiering python/weights.bin
```

`weights.bin` (~46MB, grown from ~14MB when `kMaxSeqLen` rose from 128 to
32768: the position-embedding table is `[kMaxSeqLen, d_model]`) is
committed so the C++ side runs without a Python environment; `reference.bin`
is regenerated by `export_weights.py` (not committed, since it's purely a
derived correctness-check fixture).

## Layout

```
python/model.py                reference transformer definition (PyTorch)
python/export_weights.py       exports weights.bin + reference.bin
include/, kernels/             C++ headers / CUDA kernel implementations
include/tiered_kv_manager.hpp  three-tier (GPU/DRAM/NVMe) block manager
include/block_storage.hpp      zlib compression + NVMe blob store
src/tiered_kv_manager.cpp      cost-aware eviction, spill/restore
src/block_storage.cpp          zlib wrappers, NvmeBlockStore
src/bench_tiering.cpp          the 32K-token TTFT benchmark
src/spill_restore_stress.cpp   the 100,000-block stress test
src/                           loaders, orchestration, benchmark, server demo
tests/                         paged KV-cache unit tests
tests/tiered_kv_test.cpp       tiered-manager unit tests
docs/gemm_layout.md            derivation of the row-major-via-cuBLAS gemm trick
docs/*_output.txt              raw output backing every number in this README
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
- Greedy (argmax) decoding only; no sampling, no speculative decoding.
- The fused attention kernel's per-key-position loop is not warp-tiled for
  long contexts -- correct and fused (verified to 1e-6 against PyTorch even
  after the online-softmax rewrite), not claimed to be latency-optimal at
  scale.
- **The tiering TTFT claim (3.1x) was not met; prefix reuse measured
  slower than recompute (0.45x) on this workload and machine.** See
  Findings for the root cause (per-block CUDA API call count) and the
  understood, not-yet-implemented fix (batched cross-block restore).
  Reported honestly rather than tuned or hidden.
- The cost-aware eviction policy's restore-cost estimate uses this
  machine's calibrated bandwidth figures but a fixed linear cost model; it
  is not validated against a ground-truth optimal placement (unlike the
  sibling `heterogeneous-memory-placement-simulator` repo, which sweeps
  the same kind of policy against an offline-optimal Belady oracle:
  https://github.com/Manas103/heterogeneous-memory-placement-simulator).
