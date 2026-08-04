// Lightweight assert-based test harness (no framework dependency) covering
// the paged KV-cache's block allocation/free/reuse bookkeeping -- the part
// of the system that has nothing to do with numerics and is easy to get
// subtly wrong (off-by-one on page boundaries, stale pages handed to a new
// sequence, etc). End-to-end model correctness against the PyTorch
// reference is covered separately by src/correctness_check.cpp, which
// needs the GPU and exported weights and so isn't run here.
#include <cstdio>
#include <cstdlib>
#include "config.hpp"
#include "paged_kv_cache.hpp"

using namespace cfg;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                      \
  do {                                                                   \
    ++g_checks;                                                          \
    if (!(cond)) {                                                       \
      ++g_failures;                                                      \
      std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__,      \
                   __LINE__, #cond);                                     \
    }                                                                    \
  } while (0)

static void test_allocate_and_grow() {
  PagedKVCache cache(kNLayers, /*max_pages=*/8, kPageSize, kNHeads, kHeadDim);
  int seq = cache.allocate_sequence();
  CHECK(cache.num_free_pages() == 8);

  // Filling positions 0..page_size-1 should need exactly 1 page.
  for (int p = 0; p < kPageSize; ++p) CHECK(cache.ensure_capacity(seq, p));
  CHECK(cache.num_free_pages() == 7);

  // Crossing into the next page should allocate a 2nd page, and only then.
  CHECK(cache.ensure_capacity(seq, kPageSize));
  CHECK(cache.num_free_pages() == 6);
  CHECK(cache.ensure_capacity(seq, kPageSize + 1));
  CHECK(cache.num_free_pages() == 6);  // still within the 2nd page

  int page, offset;
  cache.resolve_slot(seq, 0, &page, &offset);
  CHECK(offset == 0);
  cache.resolve_slot(seq, kPageSize, &page, &offset);
  CHECK(offset == 0);  // first slot of the 2nd page
  cache.resolve_slot(seq, kPageSize + 3, &page, &offset);
  CHECK(offset == 3);

  cache.free_sequence(seq);
  CHECK(cache.num_free_pages() == 8);
}

static void test_page_reuse_across_sequences() {
  PagedKVCache cache(kNLayers, /*max_pages=*/2, kPageSize, kNHeads, kHeadDim);
  int seq_a = cache.allocate_sequence();
  CHECK(cache.ensure_capacity(seq_a, 0));
  CHECK(cache.ensure_capacity(seq_a, kPageSize));  // takes both pages
  CHECK(cache.num_free_pages() == 0);

  int seq_b = cache.allocate_sequence();
  CHECK(!cache.ensure_capacity(seq_b, 0));  // pool exhausted -> must fail, not crash

  cache.free_sequence(seq_a);
  CHECK(cache.num_free_pages() == 2);
  CHECK(cache.ensure_capacity(seq_b, 0));  // now it can proceed
}

static void test_page_table_padding() {
  PagedKVCache cache(kNLayers, /*max_pages=*/4, kPageSize, kNHeads, kHeadDim);
  int seq = cache.allocate_sequence();
  cache.ensure_capacity(seq, 0);
  auto table = cache.page_table(seq);
  CHECK((int)table.size() == kPagesPerSeq);
  // Only logical page 0 is real; the rest are zero-padding the attention
  // kernel is guaranteed to never dereference (bounded by row_query_pos).
  for (int i = 1; i < kPagesPerSeq; ++i) CHECK(table[i] == 0);
}

static void test_disjoint_sequences_get_disjoint_pages() {
  PagedKVCache cache(kNLayers, /*max_pages=*/4, kPageSize, kNHeads, kHeadDim);
  int a = cache.allocate_sequence();
  int b = cache.allocate_sequence();
  cache.ensure_capacity(a, 0);
  cache.ensure_capacity(b, 0);
  auto ta = cache.page_table(a);
  auto tb = cache.page_table(b);
  CHECK(ta[0] != tb[0]);
}

int main() {
  test_allocate_and_grow();
  test_page_reuse_across_sequences();
  test_page_table_padding();
  test_disjoint_sequences_get_disjoint_pages();

  std::printf("%d/%d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
