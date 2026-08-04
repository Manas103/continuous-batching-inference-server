// Head-to-head benchmark: continuous batching vs. naive static batching,
// under the same synthetic bursty (Poisson-arrival) workload, both driving
// the real CUDA forward pass on the GPU. This is a deliberately
// single-threaded, trace-driven event loop rather than the multithreaded
// server in src/server_demo.cpp -- that keeps the measured numbers free of
// OS thread-scheduling jitter, since what's being isolated here is the
// *scheduling policy's* effect on throughput/latency, not thread overhead.
#include <cstdio>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include "config.hpp"
#include "model_weights.hpp"
#include "paged_kv_cache.hpp"
#include "transformer.hpp"

using namespace cfg;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kMaxSlots = 16;
constexpr int kMaxPages = 128;
constexpr int kMaxTotalRows = 512;
constexpr int kNumRequests = 240;
constexpr double kMeanInterarrivalMs = 6.0;
constexpr int kPromptLenMin = 4, kPromptLenMax = 16;
constexpr int kDecodeLenMin = 8, kDecodeLenMax = 32;
constexpr double kStaticBatchTimeoutMs = 20.0;

struct Request {
  int id;
  double arrival_ms;
  int prompt_len;
  int decode_len;
  std::vector<int> prompt_tokens;
};

std::vector<Request> make_workload(int seed) {
  std::mt19937 rng(seed);
  std::exponential_distribution<double> interarrival(1.0 / kMeanInterarrivalMs);
  std::uniform_int_distribution<int> prompt_len_dist(kPromptLenMin, kPromptLenMax);
  std::uniform_int_distribution<int> decode_len_dist(kDecodeLenMin, kDecodeLenMax);
  std::uniform_int_distribution<int> token_dist(0, kVocab - 1);

  std::vector<Request> reqs;
  double t = 0.0;
  for (int i = 0; i < kNumRequests; ++i) {
    t += interarrival(rng);
    Request r;
    r.id = i;
    r.arrival_ms = t;
    r.prompt_len = prompt_len_dist(rng);
    r.decode_len = decode_len_dist(rng);
    r.prompt_tokens.resize(r.prompt_len);
    for (int& tok : r.prompt_tokens) tok = token_dist(rng);
    reqs.push_back(std::move(r));
  }
  return reqs;
}

int argmax_row(const float* logits_row, int n) {
  int best = 0;
  float best_val = logits_row[0];
  for (int i = 1; i < n; ++i) {
    if (logits_row[i] > best_val) { best_val = logits_row[i]; best = i; }
  }
  return best;
}

struct Result {
  int id;
  double arrival_ms;
  double completion_ms;
  double latency_ms() const { return completion_ms - arrival_ms; }
};

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// ---------------- Continuous batching ----------------

struct Slot {
  bool active = false;
  int seq_id = -1;
  const Request* req = nullptr;
  int next_pos = 0;      // absolute position of the next row to feed
  bool prefilled = false;
  int last_token = 0;    // token to feed on the next decode step
  int generated = 0;
};

std::vector<Result> run_continuous(TransformerRunner& runner, PagedKVCache& cache,
                                    const std::vector<Request>& workload) {
  std::vector<Result> results;
  results.reserve(workload.size());
  std::vector<Slot> slots(kMaxSlots);
  size_t next_arrival_idx = 0;
  int completed = 0;

  auto start = Clock::now();
  while (completed < (int)workload.size()) {
    double now_ms = elapsed_ms(start);

    // Admit any arrived requests into free slots.
    for (auto& s : slots) {
      if (s.active) continue;
      if (next_arrival_idx >= workload.size()) break;
      if (workload[next_arrival_idx].arrival_ms > now_ms) break;
      s.active = true;
      s.req = &workload[next_arrival_idx++];
      s.seq_id = cache.allocate_sequence();
      s.next_pos = 0;
      s.prefilled = false;
      s.generated = 0;
    }

    bool any_active = false;
    std::vector<int> token_ids, positions, seq_ids;
    std::vector<Slot*> row_owner_slot;      // which slot each *last* row of a prefill/decode step belongs to
    std::vector<bool> row_is_last_of_slot;

    for (auto& s : slots) {
      if (!s.active) continue;
      any_active = true;
      if (!s.prefilled) {
        for (int p = 0; p < s.req->prompt_len; ++p) {
          if (!cache.ensure_capacity(s.seq_id, p)) { std::fprintf(stderr, "OOM pages\n"); std::exit(1); }
          token_ids.push_back(s.req->prompt_tokens[p]);
          positions.push_back(p);
          seq_ids.push_back(s.seq_id);
          row_owner_slot.push_back(&s);
          row_is_last_of_slot.push_back(p == s.req->prompt_len - 1);
        }
        s.next_pos = s.req->prompt_len;
        s.prefilled = true;
      } else {
        if (!cache.ensure_capacity(s.seq_id, s.next_pos)) { std::fprintf(stderr, "OOM pages\n"); std::exit(1); }
        token_ids.push_back(s.last_token);
        positions.push_back(s.next_pos);
        seq_ids.push_back(s.seq_id);
        row_owner_slot.push_back(&s);
        row_is_last_of_slot.push_back(true);
        s.next_pos += 1;
      }
    }

    if (!any_active) {
      if (next_arrival_idx < workload.size()) {
        double wait_ms = workload[next_arrival_idx].arrival_ms - now_ms;
        if (wait_ms > 0) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(wait_ms));
      }
      continue;
    }

    std::vector<float> logits = runner.forward_step(token_ids, positions, seq_ids);
    double step_now_ms = elapsed_ms(start);

    for (size_t i = 0; i < token_ids.size(); ++i) {
      if (!row_is_last_of_slot[i]) continue;
      Slot* s = row_owner_slot[i];
      int next_tok = argmax_row(&logits[i * kVocab], kVocab);
      s->last_token = next_tok;
      s->generated += 1;
      if (s->generated >= s->req->decode_len) {
        results.push_back({s->req->id, s->req->arrival_ms, step_now_ms});
        cache.free_sequence(s->seq_id);
        s->active = false;
        completed += 1;
      }
    }
  }
  return results;
}

// ---------------- Static batching ----------------

std::vector<Result> run_static(TransformerRunner& runner, PagedKVCache& cache,
                                const std::vector<Request>& workload) {
  std::vector<Result> results;
  results.reserve(workload.size());
  size_t next_arrival_idx = 0;
  auto start = Clock::now();

  while (results.size() < workload.size()) {
    double now_ms = elapsed_ms(start);
    // Wait until at least one request has arrived.
    while (next_arrival_idx >= workload.size() ? false : workload[next_arrival_idx].arrival_ms > now_ms) {
      std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(
          workload[next_arrival_idx].arrival_ms - now_ms));
      now_ms = elapsed_ms(start);
    }

    // Fill the batch: grab up to kMaxSlots arrived requests, or wait up to
    // kStaticBatchTimeoutMs from the first one for more to arrive.
    double batch_form_deadline = now_ms + kStaticBatchTimeoutMs;
    std::vector<const Request*> batch;
    while ((int)batch.size() < kMaxSlots && next_arrival_idx < workload.size()) {
      now_ms = elapsed_ms(start);
      if (workload[next_arrival_idx].arrival_ms <= now_ms) {
        batch.push_back(&workload[next_arrival_idx++]);
      } else if (workload[next_arrival_idx].arrival_ms <= batch_form_deadline && !batch.empty()) {
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(
            std::min(1.0, workload[next_arrival_idx].arrival_ms - now_ms)));
      } else {
        break;
      }
    }
    if (batch.empty()) continue;

    struct BSlot { const Request* req; int seq_id; int next_pos; bool done; int last_token; int generated; };
    std::vector<BSlot> bslots;
    for (auto* r : batch) {
      int seq_id = cache.allocate_sequence();
      bslots.push_back({r, seq_id, 0, false, 0, 0});
    }

    // Prefill all members together.
    {
      std::vector<int> token_ids, positions, seq_ids;
      for (auto& bs : bslots) {
        for (int p = 0; p < bs.req->prompt_len; ++p) {
          cache.ensure_capacity(bs.seq_id, p);
          token_ids.push_back(bs.req->prompt_tokens[p]);
          positions.push_back(p);
          seq_ids.push_back(bs.seq_id);
        }
        bs.next_pos = bs.req->prompt_len;
      }
      std::vector<float> logits = runner.forward_step(token_ids, positions, seq_ids);
      int row = 0;
      for (auto& bs : bslots) {
        row += bs.req->prompt_len - 1;
        bs.last_token = argmax_row(&logits[(size_t)row * kVocab], kVocab);
        bs.generated = 1;
        row += 1;
      }
    }

    // Lockstep decode until every member reaches its own decode_len; a
    // member that finishes early still occupies its slot (no new admission)
    // until the whole batch drains -- this is the head-of-line blocking a
    // static batcher pays for.
    bool all_done = false;
    while (!all_done) {
      std::vector<int> token_ids, positions, seq_ids;
      std::vector<int> active_indices;
      for (size_t i = 0; i < bslots.size(); ++i) {
        auto& bs = bslots[i];
        if (bs.done) continue;
        cache.ensure_capacity(bs.seq_id, bs.next_pos);
        token_ids.push_back(bs.last_token);
        positions.push_back(bs.next_pos);
        seq_ids.push_back(bs.seq_id);
        active_indices.push_back((int)i);
        bs.next_pos += 1;
      }
      if (!token_ids.empty()) {
        std::vector<float> logits = runner.forward_step(token_ids, positions, seq_ids);
        for (size_t k = 0; k < active_indices.size(); ++k) {
          auto& bs = bslots[active_indices[k]];
          bs.last_token = argmax_row(&logits[k * kVocab], kVocab);
          bs.generated += 1;
          if (bs.generated >= bs.req->decode_len) bs.done = true;
        }
      }
      all_done = true;
      for (auto& bs : bslots) all_done = all_done && bs.done;
    }

    double batch_done_ms = elapsed_ms(start);
    for (auto& bs : bslots) {
      results.push_back({bs.req->id, bs.req->arrival_ms, batch_done_ms});
      cache.free_sequence(bs.seq_id);
    }
  }
  return results;
}

void report(const char* label, const std::vector<Result>& results, double total_tokens) {
  std::vector<double> lat;
  lat.reserve(results.size());
  double total_wall_ms = 0.0;
  for (auto& r : results) { lat.push_back(r.latency_ms()); total_wall_ms = std::max(total_wall_ms, r.completion_ms); }
  std::sort(lat.begin(), lat.end());
  auto pct = [&](double p) { return lat[std::min(lat.size() - 1, (size_t)(p * lat.size()))]; };
  double throughput = total_tokens / (total_wall_ms / 1000.0);
  printf("== %s ==\n", label);
  printf("  requests completed : %zu\n", results.size());
  printf("  wall clock          : %.1f ms\n", total_wall_ms);
  printf("  token throughput     : %.1f tok/s\n", throughput);
  printf("  latency p50/p95/p99  : %.1f / %.1f / %.1f ms\n", pct(0.50), pct(0.95), pct(0.99));
  printf("  latency mean/max     : %.1f / %.1f ms\n",
         std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size(), lat.back());
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_path = argc > 1 ? argv[1] : "python/weights.bin";
  printf("loading weights...\n");
  ModelWeights weights = ModelWeights::load(weights_path);

  auto workload = make_workload(/*seed=*/42);
  double total_tokens = 0;
  for (auto& r : workload) total_tokens += r.prompt_len + r.decode_len;
  printf("workload: %d requests, prompt_len in [%d,%d], decode_len in [%d,%d], mean interarrival %.1fms\n\n",
         kNumRequests, kPromptLenMin, kPromptLenMax, kDecodeLenMin, kDecodeLenMax, kMeanInterarrivalMs);

  {
    PagedKVCache cache(kNLayers, kMaxPages, kPageSize, kNHeads, kHeadDim);
    TransformerRunner runner(weights, cache, kMaxTotalRows);
    auto results = run_continuous(runner, cache, workload);
    report("continuous batching", results, total_tokens);
  }
  printf("\n");
  {
    PagedKVCache cache(kNLayers, kMaxPages, kPageSize, kNHeads, kHeadDim);
    TransformerRunner runner(weights, cache, kMaxTotalRows);
    auto results = run_static(runner, cache, workload);
    report("static batching (baseline)", results, total_tokens);
  }

  weights.free_all();
  return 0;
}
