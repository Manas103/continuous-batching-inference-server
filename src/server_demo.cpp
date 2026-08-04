// Live multithreaded demo: a generator thread produces bursty synthetic
// requests in real time into a mutex/condvar-guarded queue while the main
// thread runs the continuous-batching scheduler + GPU execution loop,
// admitting newly arrived requests into free slots as it goes and printing
// periodic live stats. src/bench_batching.cpp is the rigorous single
// threaded A/B measurement tool; this is the architecture the task asked
// for (generator thread + scheduler/executor) running against the same
// real GPU pipeline.
#include <cstdio>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <random>
#include <chrono>
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
constexpr double kMeanInterarrivalMs = 8.0;
constexpr double kRunSeconds = 6.0;

struct Request {
  int id;
  std::vector<int> prompt_tokens;
  int decode_len;
};

class RequestQueue {
 public:
  void push(Request r) {
    std::lock_guard<std::mutex> lk(m_);
    q_.push_back(std::move(r));
    cv_.notify_one();
  }
  bool try_pop(Request* out) {
    std::lock_guard<std::mutex> lk(m_);
    if (q_.empty()) return false;
    *out = std::move(q_.front());
    q_.pop_front();
    return true;
  }
  void close() {
    std::lock_guard<std::mutex> lk(m_);
    closed_ = true;
    cv_.notify_all();
  }
  bool closed_and_empty() {
    std::lock_guard<std::mutex> lk(m_);
    return closed_ && q_.empty();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Request> q_;
  bool closed_ = false;
};

int argmax_row(const float* row, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) if (row[i] > row[best]) best = i;
  return best;
}

void generator_thread_fn(RequestQueue* queue, std::atomic<int>* generated_count) {
  std::mt19937 rng(2024);
  std::exponential_distribution<double> interarrival(1.0 / kMeanInterarrivalMs);
  std::uniform_int_distribution<int> prompt_len_dist(4, 16);
  std::uniform_int_distribution<int> decode_len_dist(8, 24);
  std::uniform_int_distribution<int> token_dist(0, kVocab - 1);

  auto start = Clock::now();
  int id = 0;
  while (std::chrono::duration<double>(Clock::now() - start).count() < kRunSeconds) {
    double wait_ms = interarrival(rng);
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(wait_ms));
    Request r;
    r.id = id++;
    r.prompt_tokens.resize(prompt_len_dist(rng));
    for (int& t : r.prompt_tokens) t = token_dist(rng);
    r.decode_len = decode_len_dist(rng);
    queue->push(std::move(r));
    generated_count->fetch_add(1);
  }
  queue->close();
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_path = argc > 1 ? argv[1] : "python/weights.bin";
  printf("loading weights...\n");
  ModelWeights weights = ModelWeights::load(weights_path);
  PagedKVCache cache(kNLayers, kMaxPages, kPageSize, kNHeads, kHeadDim);
  TransformerRunner runner(weights, cache, kMaxTotalRows);

  RequestQueue queue;
  std::atomic<int> generated_count{0};
  std::atomic<int> completed_count{0};
  std::atomic<long long> tokens_generated{0};

  std::thread gen_thread(generator_thread_fn, &queue, &generated_count);

  struct Slot {
    bool active = false;
    int seq_id = -1;
    Request req;
    int next_pos = 0;
    bool prefilled = false;
    int last_token = 0;
    int generated = 0;
  };
  std::vector<Slot> slots(kMaxSlots);

  auto start = Clock::now();
  auto last_report = start;

  while (true) {
    // Admit newly arrived requests into free slots.
    for (auto& s : slots) {
      if (s.active) continue;
      Request r;
      if (!queue.try_pop(&r)) break;
      s.active = true;
      s.req = std::move(r);
      s.seq_id = cache.allocate_sequence();
      s.next_pos = 0;
      s.prefilled = false;
      s.generated = 0;
    }

    bool any_active = false;
    std::vector<int> token_ids, positions, seq_ids;
    std::vector<Slot*> owner;
    std::vector<bool> is_last;

    for (auto& s : slots) {
      if (!s.active) continue;
      any_active = true;
      if (!s.prefilled) {
        for (int p = 0; p < (int)s.req.prompt_tokens.size(); ++p) {
          cache.ensure_capacity(s.seq_id, p);
          token_ids.push_back(s.req.prompt_tokens[p]);
          positions.push_back(p);
          seq_ids.push_back(s.seq_id);
          owner.push_back(&s);
          is_last.push_back(p == (int)s.req.prompt_tokens.size() - 1);
        }
        s.next_pos = (int)s.req.prompt_tokens.size();
        s.prefilled = true;
      } else {
        cache.ensure_capacity(s.seq_id, s.next_pos);
        token_ids.push_back(s.last_token);
        positions.push_back(s.next_pos);
        seq_ids.push_back(s.seq_id);
        owner.push_back(&s);
        is_last.push_back(true);
        s.next_pos += 1;
      }
    }

    if (any_active) {
      std::vector<float> logits = runner.forward_step(token_ids, positions, seq_ids);
      for (size_t i = 0; i < token_ids.size(); ++i) {
        if (!is_last[i]) continue;
        Slot* s = owner[i];
        s->last_token = argmax_row(&logits[i * kVocab], kVocab);
        s->generated += 1;
        tokens_generated.fetch_add(1);
        if (s->generated >= s->req.decode_len) {
          cache.free_sequence(s->seq_id);
          s->active = false;
          completed_count.fetch_add(1);
        }
      }
    } else if (queue.closed_and_empty()) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto now = Clock::now();
    if (std::chrono::duration<double>(now - last_report).count() >= 0.5) {
      int active_slots = 0;
      for (auto& s : slots) if (s.active) active_slots++;
      double elapsed = std::chrono::duration<double>(now - start).count();
      printf("[t=%5.1fs] active_slots=%2d completed=%3d generated_by_source=%3d tokens/s=%.0f\n",
             elapsed, active_slots, completed_count.load(), generated_count.load(),
             tokens_generated.load() / elapsed);
      last_report = now;
    }
  }

  gen_thread.join();
  double total_elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  printf("\ndone: %d requests completed in %.2fs, %lld tokens generated (%.0f tok/s average)\n",
         completed_count.load(), total_elapsed, (long long)tokens_generated.load(),
         tokens_generated.load() / total_elapsed);

  weights.free_all();
  return 0;
}
