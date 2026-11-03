// Q67/S1: the prefill class ladder and the merge decision.
//
// These are the arithmetic the batch container will run on: which class a set of queued prompts is
// padded up to, and whether padding them up at all is cheaper than running them one at a time. The
// numbers come from the 4090 measurements in the plan (26 tok = 1232 ms, 74 = 2245, 138 = 2596,
// 970 = 5131), so a test that disagrees with them is a test that would have shipped the losing
// 1024-wide merge.

#include "models/qwen3_5/execution/shape_class.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <span>

namespace execution = ninfer::models::qwen3_5::execution;

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) { return; }
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

void check_equal(std::uint64_t actual, std::uint64_t expected, const char* what) {
    if (actual == expected) { return; }
    std::fprintf(stderr, "FAIL: %s (got %llu, want %llu)\n", what,
                 static_cast<unsigned long long>(actual),
                 static_cast<unsigned long long>(expected));
    ++failures;
}

template <std::size_t N>
execution::PrefillBatchFit fit(const std::array<std::uint32_t, N>& remaining) {
    return execution::prefill_batch_fit(std::span<const std::uint32_t>(remaining.data(), N));
}

} // namespace

int main() {
    // The ladder itself: 16 is the floor (an sm_89 MMA tile is 16 rows), 1 is not in this set
    // because the single-request path is pinned by decode_class_for(1), not by padding.
    check_equal(execution::prefill_class_for(1), 16, "a one-token width still rounds to 16");
    check_equal(execution::prefill_class_for(16), 16, "16 is a class");
    check_equal(execution::prefill_class_for(17), 32, "17 rounds up to 32");
    check_equal(execution::prefill_class_for(104), 128, "104 rounds up to 128");
    check_equal(execution::prefill_class_for(5000), 5000, "above the ladder the width is unchanged");

    // Four short prompts: this is the case the old {1024,2048,4096} set turned into a 1024-row pass.
    {
        const auto result = fit(std::array<std::uint32_t, 4>{26, 26, 26, 26});
        check_equal(static_cast<std::uint64_t>(result.width), 32, "4x26 merges at 32 columns wide");
        check_equal(result.lanes, 4, "4x26 takes all four lanes");
        check(result.merged_ms < result.separate_ms, "4x26: merging is cheaper");
    }

    // Two one-token prompts: padding is 16x the useful width and still wins, because a pass costs
    // about a second before any row is useful. A padding-ratio rule would have refused this.
    {
        const auto result = fit(std::array<std::uint32_t, 2>{1, 1});
        check_equal(static_cast<std::uint64_t>(result.width), 16, "1+1 merges at 16");
        check_equal(result.lanes, 2, "1+1 takes both lanes");
    }

    // Two long prompts do not fit one class usefully: 1200 tokens would be padded to 2048, which
    // costs more than running both.
    {
        const auto result = fit(std::array<std::uint32_t, 2>{600, 600});
        check_equal(static_cast<std::uint64_t>(result.width), 0, "2x600 does not merge");
        check_equal(result.lanes, 0, "2x600 keeps the single path");
    }

    // Five 1000-token prompts: only the first four fit under 4096, and that prefix wins.
    {
        const auto result = fit(std::array<std::uint32_t, 5>{1000, 1000, 1000, 1000, 1000});
        check_equal(static_cast<std::uint64_t>(result.width), 1024, "4x1000 merges at 1024 columns wide");
        check_equal(result.lanes, 4, "4x1000 stops where the class stops");
    }

    // A prompt wider than the largest class is not merged at all; the caller splits it.
    {
        const auto result = fit(std::array<std::uint32_t, 1>{5000});
        check_equal(static_cast<std::uint64_t>(result.width), 0, "a 5000-token prompt does not merge");
        check_equal(result.lanes, 0, "a 5000-token prompt keeps its own pass");
    }

    // Eight one-chunk prompts: the case the plan quotes as 3.5x.
    {
        const auto result = fit(std::array<std::uint32_t, 8>{128, 128, 128, 128, 128, 128, 128, 128});
        check_equal(static_cast<std::uint64_t>(result.width), 128, "8x128 merges at 128 columns wide");
        check_equal(result.lanes, 8, "8x128 takes all eight lanes");
        // Conservative on purpose: the model charges 12 ms for every padded row, while the measured
        // marginal at ~1000 rows is ~2 ms, so the real ratio is 3.5x (18 s against 5.1 s) and the
        // model only promises 1.6x. A model that errs this way never merges a batch that would lose.
        check(result.separate_ms / result.merged_ms > 1.5, "8x128: merging wins by at least 1.5x");
    }

    // Whatever the mixture, a merged width is always a class and always covers the lanes it names.
    {
        const auto result = fit(std::array<std::uint32_t, 3>{37, 41, 5});
        check_equal(static_cast<std::uint64_t>(result.width), 64, "37+41+5 merges at 64 columns wide");
        check_equal(result.lanes, 3, "37+41+5 takes three lanes");
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("shape class: ladder and merge decision OK\n");
    return 0;
}
