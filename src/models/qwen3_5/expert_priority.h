#pragma once

// V10-14 (upstream c996baa8, T161): the cold-start prior for the routed-expert cache.
//
// The stream has exactly one order today - the decayed route counts it observes while it runs
// (`ExpertStream::hot_score_`). A fresh process has no observation, so every seat decision falls
// back to the logical id order: the first layer of the first token seats experts 0..capacity-1.
// Upstream publishes a calibrated default order with the model instead, and this is the local form
// of that idea: a per-layer table that the seat ranking consults WHEN THE HEAT IS TIED.
//
// It is a prior, not a policy. `precedes()` only reads the table when two experts are equally hot,
// which is exactly the cold-start case (nothing observed yet, or two experts never observed), so an
// expert the router has actually used always outranks a table entry. An empty table reproduces the
// order the comparators produced before this file existed.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

// "This expert is not in the table", and therefore loses every tie to one that is. The value a rank
// takes for an expert the row does not name - including every expert of an empty row.
inline constexpr std::uint32_t kNoColdPriority = std::numeric_limits<std::uint32_t>::max();

// One row per text block, in the artifact's own form (the one upstream's calibration emits and the
// one an artifact's metadata or a conversion report carries): a permutation of every logical expert
// id of that block, highest cache priority first. An empty row - a dense block, or a block the
// calibration did not cover - means "no prior for this layer", not "every expert is equal": the row
// is skipped and that layer keeps the id order.
struct ExpertPriorityTable {
    std::vector<std::vector<std::uint32_t>> rows;

    [[nodiscard]] bool empty() const noexcept { return rows.empty(); }
    [[nodiscard]] std::size_t layers() const noexcept { return rows.size(); }

    [[nodiscard]] std::span<const std::uint32_t> row(std::size_t layer) const noexcept {
        return layer < rows.size() ? std::span<const std::uint32_t>(rows[layer])
                                   : std::span<const std::uint32_t>{};
    }

    // The row as ranks: `rank[expert]` is the expert's position in `row(layer)`, and kNoColdPriority
    // for an expert the row does not name. This is the form the seat comparator reads, because a
    // comparator must answer "who goes first" for any pair, including experts the table omits.
    [[nodiscard]] std::vector<std::uint32_t> rank(std::size_t layer, std::size_t experts) const {
        std::vector<std::uint32_t> out(experts, kNoColdPriority);
        const auto order = row(layer);
        for (std::size_t i = 0; i < order.size(); ++i) {
            if (order[i] < out.size()) { out[order[i]] = static_cast<std::uint32_t>(i); }
        }
        return out;
    }
};

// Row-level merge: every non-empty row of `override_rows` replaces the same row of `base`, and every
// empty one keeps the base's. "Explicit override wins over the table" therefore holds per layer,
// which is what lets a hand-written file name only the layers it was measured on.
[[nodiscard]] inline ExpertPriorityTable merge_expert_priority(
    const ExpertPriorityTable& base, const ExpertPriorityTable& override_rows) {
    if (override_rows.empty()) { return base; }
    ExpertPriorityTable out = base;
    if (out.rows.size() < override_rows.rows.size()) { out.rows.resize(override_rows.rows.size()); }
    for (std::size_t layer = 0; layer < override_rows.rows.size(); ++layer) {
        if (!override_rows.rows[layer].empty()) { out.rows[layer] = override_rows.rows[layer]; }
    }
    return out;
}

// The rank of one expert in a rank row, with the "not named" answer folded in so every caller reads
// the same value instead of inventing its own sentinel.
[[nodiscard]] inline std::uint32_t cold_priority_rank(std::span<const std::uint32_t> rank,
                                                      std::uint32_t expert) noexcept {
    return expert < rank.size() ? rank[expert] : kNoColdPriority;
}

// THE one ordering every seat decision uses: observed heat descending, the cold-start prior breaking
// ties, the expert id last. With an empty (or partial) `cold_rank` the tie falls through to the id,
// which is exactly what the previous comparators did - `hot_score_[a] > hot_score_[b]` over an
// ascending id sequence, and `... else a < b` where the tie-break was written out.
[[nodiscard]] inline bool precedes(std::uint32_t a, std::uint32_t b, std::span<const float> observed,
                                   std::span<const std::uint32_t> cold_rank) noexcept {
    if (observed[a] != observed[b]) { return observed[a] > observed[b]; }
    const auto rank_a = cold_priority_rank(cold_rank, a);
    const auto rank_b = cold_priority_rank(cold_rank, b);
    if (rank_a != rank_b) { return rank_a < rank_b; }
    return a < b;
}

// The whole ranking, for the placements that ask for an order rather than a comparator. One function
// so the "which experts get the seats" answer cannot disagree between the fused path, the split path
// and the freeze.
[[nodiscard]] inline std::vector<std::uint32_t> seat_ranking(
    std::span<const float> observed, std::span<const std::uint32_t> cold_rank) {
    std::vector<std::uint32_t> order(observed.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return precedes(a, b, observed, cold_rank);
    });
    return order;
}

} // namespace ninfer::models::qwen3_5
