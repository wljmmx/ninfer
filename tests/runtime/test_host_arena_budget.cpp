// V10-75: the host expert tier's ask, without a Device.
//
// allocate_host_arena() has to turn a candidate size plus a commit budget into the bytes it asks
// cudaHostAlloc for. The three answers are not interchangeable, and only one of them used to be
// wrong: a measured ZERO budget was read as "no clamp" and the full candidate was asked for anyway.
// That branch cannot be reached on a healthy box (it needs the page file to be empty), so it is
// pinned here instead: the decision is pure, and this test is its whole contract.

#include "runtime/engine/host_arena_budget.h"

#include <cstdint>
#include <cstdio>
#include <optional>

namespace {

using ninfer::runtime::host_arena_ask;

constexpr std::uint64_t kMib = 1024ULL * 1024ULL;
constexpr std::uint64_t kGib = 1024ULL * kMib;

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) { return; }
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

// The tier asks 14550 MiB and the box has 21000 MiB of headroom: nothing to clamp.
void test_a_budget_above_the_ask_leaves_it_alone() {
    check(host_arena_ask(14550 * kMib, 21000 * kMib, false) == 14550 * kMib,
          "a budget above the ask does not move it");
}

// The tier asks more than the page file can back: the budget is the answer, so the tier is sized
// from what was granted rather than from what was wanted.
void test_a_budget_below_the_ask_clamps() {
    check(host_arena_ask(14550 * kMib, 3684 * kMib, false) == 3684 * kMib,
          "a budget below the ask is the ask");
    check(host_arena_ask(14550 * kMib, 1, false) == 1, "a one-byte budget is still a budget");
}

// The branch V10-75 changed: a measured zero budget asks for NOTHING, so allocate_host_arena()
// refuses with its own message instead of attempting a block the commit budget says cannot exist.
void test_a_measured_zero_budget_asks_for_nothing() {
    check(host_arena_ask(14550 * kMib, 0, false) == 0, "a measured zero budget asks for nothing");
    check(host_arena_ask(1, 0, false) == 0, "a measured zero budget asks for nothing at any size");
}

// ... and "the box cannot be measured" is NOT that answer: a platform without the query keeps its
// whole ask, because reading nothing must not be read as zero.
void test_an_unknown_budget_keeps_the_ask() {
    check(host_arena_ask(14550 * kMib, std::nullopt, false) == 14550 * kMib,
          "an unmeasurable budget leaves the ask as named");
}

// The explicit override (NINFER_EXPERT_HOST_ARENA_MB) is asked for as named at every budget,
// including zero: an experiment must not silently run at a different size, and a refusal there is
// the caller's error rather than the engine's.
void test_an_exact_ask_is_never_clamped() {
    check(host_arena_ask(14550 * kMib, 3684 * kMib, true) == 14550 * kMib,
          "an exact ask ignores a smaller budget");
    check(host_arena_ask(14550 * kMib, 0, true) == 14550 * kMib,
          "an exact ask ignores a zero budget");
    check(host_arena_ask(14550 * kMib, std::nullopt, true) == 14550 * kMib,
          "an exact ask ignores an unknown budget");
}

// The property the whole clamp exists for: the answer never exceeds the candidate, whatever the
// budget says, so a caller cannot be handed a tier larger than the one it sized its ledger for.
void test_the_answer_never_exceeds_the_candidate() {
    for (std::uint64_t budget = 0; budget <= 24 * kGib; budget += 512 * kMib) {
        const std::uint64_t ask = host_arena_ask(16 * kGib, budget, false);
        check(ask <= 16 * kGib, "the clamped ask never exceeds the candidate");
        check(ask == (budget < 16 * kGib ? budget : 16 * kGib), "the clamp is the minimum");
    }
}

} // namespace

int main() {
    test_a_budget_above_the_ask_leaves_it_alone();
    test_a_budget_below_the_ask_clamps();
    test_a_measured_zero_budget_asks_for_nothing();
    test_an_unknown_budget_keeps_the_ask();
    test_an_exact_ask_is_never_clamped();
    test_the_answer_never_exceeds_the_candidate();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
