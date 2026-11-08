// T042 (upstream d0ab3ba9 + 75be1384) and L10: prefix checkpoint selection.

#include "runtime/engine/context_cache/prefix_checkpoint.h"

#include <array>
#include <cstdio>

namespace {

using ninfer::runtime::plan_prefix_checkpoint;
using ninfer::runtime::PrefixBlockHasher;
using ninfer::runtime::PrefixCheckpointPolicy;
using ninfer::runtime::PrefixCheckpointReason;
using ninfer::runtime::PrefixCheckpointRequest;
using ninfer::runtime::shared_prefix_cap;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void test_hasher_is_incremental() {
    PrefixBlockHasher streamed(4);
    for (int token = 1; token <= 9; ++token) { streamed.push(token); }
    check(streamed.tokens() == 9, "the cursor counts every token");
    check(streamed.complete_blocks() == 2, "nine tokens close two blocks of four");
    const auto first = streamed.block_digest(0);

    PrefixBlockHasher reference(4);
    for (int token = 1; token <= 9; ++token) { reference.push(token); }
    check(reference.block_digest(0) == first, "the digest does not depend on call grouping");

    for (int token = 10; token <= 40; ++token) { streamed.push(token); }
    check(streamed.block_digest(0) == first, "an older block digest never changes");

    // lookahead(0) is the digest of block 0 mixed with token 5, the first token after it.
    PrefixBlockHasher five(4);
    for (int token = 1; token <= 5; ++token) { five.push(token); }
    const auto lookahead = five.lookahead_digest(0);
    check(lookahead.has_value(), "a fifth token completes the lookahead of block 0");
    check(*lookahead != five.block_digest(0), "lookahead covers the successor token");

    PrefixBlockHasher eight(4);
    for (int token = 1; token <= 8; ++token) { eight.push(token); }
    check(!eight.lookahead_digest(1).has_value(), "the newest block has no successor yet");

    streamed.reset();
    check(streamed.tokens() == 0 && streamed.complete_blocks() == 0, "reset clears the cursor");
}

void test_selection_spacing_and_boundaries() {
    PrefixCheckpointPolicy policy{.interval = 1000};
    PrefixCheckpointRequest request{
        .begin = 0, .end = 3000, .last_saved = 0, .next_semantic = std::nullopt, .semantic_boundaries = {}};
    const auto plain = plan_prefix_checkpoint(request, policy);
    check(plain.frontier.has_value() && *plain.frontier == 3000,
          "the point is the end of the execution window");
    check(plain.reason == PrefixCheckpointReason::Selected, "a plain point is selected");

    const std::array<std::uint32_t, 2> boundaries{980, 1600};
    request.semantic_boundaries = boundaries;
    const auto snapped          = plan_prefix_checkpoint(request, policy);
    check(*snapped.frontier == 980, "a proven boundary within a quarter interval wins");

    request.next_semantic = 3200;
    check(plan_prefix_checkpoint(request, policy).reason ==
              PrefixCheckpointReason::NearSemanticBoundary,
          "a near semantic capture suppresses the intermediate one");

    request.next_semantic = 9000;
    request.end           = 500;
    check(plan_prefix_checkpoint(request, policy).reason ==
              PrefixCheckpointReason::BelowInterval,
          "a window shorter than the interval is not worth a point");
}

void test_shared_prefix_cap_and_minimum() {
    PrefixCheckpointPolicy policy{.interval = 1000, .shared_prefix_limit = 900};
    PrefixCheckpointRequest request{.begin = 0, .end = 3000, .last_saved = 0, .semantic_boundaries = {}};
    const auto capped = plan_prefix_checkpoint(request, policy);
    check(*capped.frontier == 900, "the shared-prefix cap shortens the point");

    policy.shared_prefix_limit = 400;
    check(plan_prefix_checkpoint(request, policy).reason ==
              PrefixCheckpointReason::BelowInterval,
          "half an interval is the floor even at the cap");

    policy.shared_prefix_limit = 0;
    check(plan_prefix_checkpoint(request, policy).reason ==
              PrefixCheckpointReason::BeyondSharedPrefix,
          "no shared prefix means no shared checkpoint");

    policy.shared_prefix_limit = 3000;
    policy.minimum_frontier     = 1500;
    policy.shared_prefix_limit = 1200;
    check(plan_prefix_checkpoint(request, policy).reason ==
              PrefixCheckpointReason::BelowMinimumFrontier,
          "the cache cannot address a prefix under two chunks");

    check(shared_prefix_cap(1000, 640) == 640, "the first output position caps the prefix");
    check(shared_prefix_cap(0, 640) == 0, "an empty shared prefix stays empty");
}

void test_fail_closed_and_no_progress() {
    PrefixCheckpointRequest request{.begin = 0, .end = 3000, .last_saved = 0, .semantic_boundaries = {}};
    PrefixCheckpointPolicy policy{.interval = 1000};
    policy.speculative = true;
    check(plan_prefix_checkpoint(request, policy).reason ==
              PrefixCheckpointReason::DraftStateAbsent,
          "MTP without a draft span in the payload fails closed");

    policy.draft_state_in_payload = true;
    check(plan_prefix_checkpoint(request, policy).frontier.has_value(),
          "with a draft span the point is allowed again");

    PrefixCheckpointPolicy disabled{.interval = 0};
    check(plan_prefix_checkpoint(request, disabled).reason ==
              PrefixCheckpointReason::IntervalDisabled,
          "interval zero disables intermediate checkpoints");

    PrefixCheckpointRequest stale{.begin = 10, .end = 3000, .last_saved = 20, .semantic_boundaries = {}};
    check(plan_prefix_checkpoint(stale, PrefixCheckpointPolicy{.interval = 1000}).reason ==
              PrefixCheckpointReason::NoProgress,
          "a save past the cursor is stale, not a reason to save again");
}

} // namespace

int main() {
    test_hasher_is_incremental();
    test_selection_spacing_and_boundaries();
    test_shared_prefix_cap_and_minimum();
    test_fail_closed_and_no_progress();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
