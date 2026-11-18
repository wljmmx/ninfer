// V11-02: the cost-aware draft window (core/draft_window.h), after strata's DraftPolicy::best_rows
// and its chain rule. Pure arithmetic, so the whole contract is assertable without a Device.
//
// What is pinned here:
//   * the choice equals an independently enumerated argmax of E(d)/C(d) over random probability
//     sequences - the test does not repeat the implementation's loop, it recomputes E and C;
//   * a non-convex cost curve is handled (a high fixed cost favours a short window, and the
//     maximum can sit in the middle of the range, which a threshold rule cannot express);
//   * degenerate inputs (zero probability, zero cost, zero width) answer 0 rather than guessing.
#include "core/draft_window.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

int failures = 0;

void require(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Independent oracle: enumerate every width and take the same ratio, without sharing the loop.
ninfer::core::DraftWindowChoice enumerate(const std::vector<double>& p, double row, double fixed,
                                          int max_width) {
    ninfer::core::DraftWindowChoice best;
    for (int d = 1; d <= max_width && d <= static_cast<int>(p.size()); ++d) {
        double expected = 0.0, prefix = 1.0;
        for (int i = 0; i < d; ++i) {
            prefix *= std::clamp(p[static_cast<std::size_t>(i)], 0.0, 1.0);
            expected += prefix;
        }
        const double cost = std::max(0.0, fixed) + d * row;
        const double rate = cost > 0.0 ? expected / cost : 0.0;
        if (rate > best.tokens_per_cost) { best = {d, expected, cost, rate}; }
    }
    return best;
}

void random_sequences_match_enumeration() {
    std::mt19937 rng(20261118);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> cost(0.05, 4.0);
    for (int trial = 0; trial < 200; ++trial) {
        const int width = 1 + static_cast<int>(rng() % 16);
        std::vector<double> p(static_cast<std::size_t>(width));
        for (auto& value : p) value = prob(rng);
        const double row   = cost(rng);
        const double fixed = cost(rng) * (rng() % 3); // sometimes zero
        const auto got     = ninfer::core::choose_draft_window(p, row, fixed, width);
        const auto want    = enumerate(p, row, fixed, width);
        require(got.width == want.width, "random: width matches the enumerated argmax");
        require(std::abs(got.expected_tokens - want.expected_tokens) < 1e-12,
                "random: expected count matches");
        require(std::abs(got.expected_cost - want.expected_cost) < 1e-12, "random: cost matches");
    }
}

void full_acceptance_takes_the_whole_window() {
    // With a per-round fixed cost the rate grows with d; with none it is flat (1/row), and the
    // rule keeps the shorter window on a tie. Pin the fixed-cost case, which is the real one.
    const std::vector<double> p(8, 1.0);
    const auto choice = ninfer::core::choose_draft_window(p, 0.5, 0.2, 8);
    require(choice.width == 8, "p=1 with a fixed cost: the whole window pays");
    require(ninfer::core::choose_draft_window(p, 0.5, 0.0, 8).width == 1,
            "p=1 without a fixed cost: flat rate, the shorter window wins the tie");
}

void impossible_draft_stops_at_zero() {
    const std::vector<double> p{0.0, 0.9, 0.9};
    const auto choice = ninfer::core::choose_draft_window(p, 0.5, 0.0, 3);
    require(choice.width == 0, "p=0 on the first row: no speculation");
}

void row_cost_shortens_and_fixed_cost_widens_the_window() {
    // Two different levers, and they point opposite ways: a dearer ROW makes each extra draft pay
    // for itself less often, so the window shortens; a dearer FIXED cost is amortised by more rows,
    // so the window widens. A single threshold cannot express either.
    const std::vector<double> p{0.9, 0.8, 0.7, 0.6, 0.5, 0.4};
    const auto cheap_row = ninfer::core::choose_draft_window(p, 0.2, 0.1, 6);
    const auto dear_row  = ninfer::core::choose_draft_window(p, 1.0, 0.1, 6);
    require(dear_row.width < cheap_row.width, "a dearer row cost shortens the window");

    const auto dear_fixed = ninfer::core::choose_draft_window(p, 0.2, 4.0, 6);
    require(dear_fixed.width >= cheap_row.width, "a dearer fixed cost is amortised, not avoided");
}

void near_certain_draft_still_pays_by_the_marginal_row() {
    // p=0.99 per column: the first row is worth ~1 token for 0.1 ms; by the sixth the cumulative
    // probability (~0.94) no longer covers six rows of cost. The winner is short even though every
    // column on its own looks certain - the point of pricing rows instead of thresholding them.
    const std::vector<double> p{0.99, 0.99, 0.99, 0.99, 0.99, 0.99};
    const auto choice = ninfer::core::choose_draft_window(p, 0.1, 0.0, 6);
    require(choice.width >= 1 && choice.width <= 3,
            "near-certain draft: the marginal rows stop paying early");
}

void degenerate_inputs_answer_zero() {
    const std::vector<double> p{0.9, 0.8};
    require(ninfer::core::choose_draft_window(p, 0.0, 0.0, 2).width == 0,
            "zero row cost: no speculation (the caller mispriced it)");
    require(ninfer::core::choose_draft_window(p, 0.5, 0.0, 0).width == 0, "zero width");
    require(ninfer::core::choose_draft_window({}, 0.5, 0.0, 4).width == 0, "empty probabilities");
}

void scalar_helper_matches_the_geometric_sequence() {
    const double p = 0.75;
    std::vector<double> sequence(8, p);
    const auto direct = ninfer::core::choose_draft_window(sequence, 0.3, 0.2, 8).width;
    const int scalar  = ninfer::core::cost_aware_width(p, 0.3, 0.2, 8);
    require(direct == scalar, "cost_aware_width agrees with the expanded sequence");
}

} // namespace

int main() {
    random_sequences_match_enumeration();
    full_acceptance_takes_the_whole_window();
    impossible_draft_stops_at_zero();
    row_cost_shortens_and_fixed_cost_widens_the_window();
    near_certain_draft_still_pays_by_the_marginal_row();
    degenerate_inputs_answer_zero();
    scalar_helper_matches_the_geometric_sequence();
    if (failures != 0) {
        std::fprintf(stderr, "FAIL draft_window: %d assertion(s)\n", failures);
        return 1;
    }
    std::printf("ok draft_window\n");
    return 0;
}
