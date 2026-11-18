#pragma once
//
// port (sm_89, V11-02): the cost-aware draft window, after strata's DraftPolicy::best_rows
// (687810556) and its chain rule (d232721db).
//
// The draft width is not a fixed constant in principle: one more draft row costs one more verify
// row, and it is only worth it while the probability that the row is accepted covers that cost.
// strata scores candidate widths by
//
//     E(d) = sum_{i=1..d} prod_{j<=i} p_j        expected accepted tokens
//     C(d) = fixed_ms + d * row_ms                what the window costs
//     pick d that maximises E(d) / C(d)
//
// This header is the same arithmetic as a pure function, with no device calls and no policy: the
// caller supplies the per-column acceptance probabilities (in order) and the measured row cost.
// Keeping it here means the rule can be unit-tested without a GPU, and an engine that later gains
// a dynamic draft width can call it instead of re-deriving it.
//
// The rule is deliberately not "take every column above a threshold". Costs are not necessarily
// convex in d (a graph capture on the first width, a step at a tile size), so the maximum of E/C
// can sit in the middle; a threshold rule cannot express that.
#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>

namespace ninfer::core {

struct DraftWindowChoice {
    int width              = 0;   // rows to draft; 0 means "do not speculate this round"
    double expected_tokens = 0.0; // E(width)
    double expected_cost   = 0.0; // C(width) in the unit the caller passed (ms)
    double tokens_per_cost = 0.0; // E(width) / C(width), the value that was maximised
};

// `accept_prob[i]` is the probability that draft row i+1 is accepted given that rows 1..i were.
// A probability outside [0,1] is clamped, because a caller reading it off a device buffer should
// not be able to make the choice explode. Rows are considered in order and the scan stops at the
// first probability of zero (the expected count stops growing, and every later cost is dead
// weight). A non-positive row cost or max_width means no speculation.
[[nodiscard]] inline DraftWindowChoice choose_draft_window(std::span<const double> accept_prob,
                                                           double row_cost, double fixed_cost,
                                                           int max_width) {
    DraftWindowChoice best;
    if (max_width <= 0 || !(row_cost > 0.0)) { return best; }
    fixed_cost = std::max(0.0, fixed_cost);

    double expected = 0.0; // running E(d): sum of prefix products
    double prefix   = 1.0; // prod_{j<=d} p_j
    for (int d = 1; d <= max_width && static_cast<std::size_t>(d) <= accept_prob.size(); ++d) {
        const double p = std::clamp(accept_prob[static_cast<std::size_t>(d - 1)], 0.0, 1.0);
        prefix *= p;
        expected += prefix;
        const double cost = fixed_cost + static_cast<double>(d) * row_cost;
        const double rate = cost > 0.0 ? expected / cost : 0.0;
        // Strictly better only: on a tie the shorter window wins, because it costs less to undo.
        if (rate > best.tokens_per_cost) { best = DraftWindowChoice{d, expected, cost, rate}; }
        if (p == 0.0) { break; }
    }
    return best;
}

// Convenience for a caller that has one scalar confidence instead of a full sequence: the width
// whose marginal row still pays for itself, p^d * 1 row of value >= d rows of cost, i.e. the
// largest d with p^d > 0 and at least one expected acceptance per row. Returns 0 when even the
// first row does not pay.
[[nodiscard]] inline int cost_aware_width(double accept_probability, double row_cost,
                                          double fixed_cost, int max_width) {
    if (max_width <= 0 || !(row_cost > 0.0)) { return 0; }
    DraftWindowChoice choice;
    // Expand the scalar into the geometric sequence the chain rule assumes.
    const double p = std::clamp(accept_probability, 0.0, 1.0);
    double values[64];
    const int count = std::min(max_width, 64);
    for (int i = 0; i < count; ++i) { values[i] = p; }
    choice = choose_draft_window(std::span<const double>(values, static_cast<std::size_t>(count)),
                                 row_cost, fixed_cost, count);
    return choice.width;
}

} // namespace ninfer::core
