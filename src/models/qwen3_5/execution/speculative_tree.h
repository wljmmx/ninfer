#pragma once

// Tree-shaped speculation: a verify pass whose columns are nodes of a tree, not steps of a chain.
//
// Why the shape has to be decided on the host, and why it is a tree at all: a verify row's cost is
// its COLUMN, not its token. speculative_round.cuh is row-major and every column of a row shares one
// recurrent state update, so a chain's columns are serial; measured on this box, a chain costs 0.093 s
// per column while four sibling rows cost 0.26 s together. A tree expressed in columns therefore
// pays for every distinct (depth, branch) node it visits, which is why the shape must be pruned by an
// accumulated probability rather than merely widened: the path probability is a product, so a tree
// deeper than two or three collapses to a single path at any usable threshold.
//
// This header owns the shape and the mask, nothing else. It is host-side and pure, so the two kernel
// steps that consume it - the verify inputs (positions and ids per column) and the attention mask
// (which columns a column may read) - can be brought up one at a time against a shape that is
// already proven correct on its own.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// One node per verify column. Column 0 is the anchor: the token the previous round emitted, which
// every path continues from. It has no parent and its probability is 1 by definition.
struct SpeculativeTree {
    // parent[c] = the column c continues from, or -1 for the anchor.
    std::vector<std::int32_t> parent;
    // The accumulated path probability the column was admitted on (the anchor is 1). Kept rather
    // than recomputed: a dump that says why a branch exists is worth more than the bytes it costs,
    // and the policy's threshold is only reproducible against this number.
    std::vector<float> path_probability;

    [[nodiscard]] std::int32_t columns() const noexcept {
        return static_cast<std::int32_t>(parent.size());
    }

    // The anchor is depth 0; a direct child of the anchor is depth 1.
    [[nodiscard]] std::int32_t depth(std::int32_t column) const noexcept {
        std::int32_t levels = 0;
        for (std::int32_t cursor = column; cursor > 0 && cursor < columns();) {
            cursor = parent[static_cast<std::size_t>(cursor)];
            if (cursor < 0) { break; }
            ++levels;
        }
        return levels;
    }

    // Is `ancestor` on the path from the anchor to `column`, the column itself included? This is the
    // relation the attention mask encodes, and the relation a verifier must respect: a sibling is a
    // token that was never generated after this one, so letting it be read would verify a sequence
    // that does not exist.
    [[nodiscard]] bool is_ancestor(std::int32_t ancestor, std::int32_t column) const noexcept {
        if (ancestor < 0 || column < 0 || ancestor >= columns() || column >= columns()) {
            return false;
        }
        for (std::int32_t cursor = column; cursor >= 0;) {
            if (cursor == ancestor) { return true; }
            if (cursor == 0) { break; }
            cursor = parent[static_cast<std::size_t>(cursor)];
        }
        return false;
    }

    // visible[c * columns + k] = 1 when k is an ancestor-or-self of c. Row-major by column, which is
    // the order a per-column kernel walks.
    void build_mask(std::vector<std::uint8_t>& visible) const {
        const auto n = static_cast<std::size_t>(columns());
        visible.assign(n * n, 0);
        for (std::size_t c = 0; c < n; ++c) {
            for (std::int32_t cursor = static_cast<std::int32_t>(c); cursor >= 0;) {
                visible[c * n + static_cast<std::size_t>(cursor)] = 1;
                if (cursor == 0) { break; }
                cursor = parent[static_cast<std::size_t>(cursor)];
            }
        }
    }

    // The verify pass's positions: base + depth(c). build_speculative_tree emits columns in BFS order,
    // so these are non-decreasing and positions[columns-1] is the deepest column - which is what keeps
    // the attention geometry's two derivations (base_pos = positions[0], window = positions[last] + 1)
    // exactly what they already are for a chain. Positions alone do not make a tree verifiable: a
    // triangle mask over these would let a column read its siblings, which is why the visibility below
    // has to travel with them.
    [[nodiscard]] std::vector<std::int32_t> depth_positions(std::int32_t base) const {
        std::vector<std::int32_t> out(static_cast<std::size_t>(columns()), base);
        for (std::int32_t c = 0; c < columns(); ++c) {
            out[static_cast<std::size_t>(c)] = base + depth(c);
        }
        return out;
    }

    // build_mask, packed one word per column: bit k of row c is set when column k is an ancestor-or-self
    // of column c. Eight verify columns is the small-T band, so the whole visibility is eight words - a
    // shape a decode kernel can stage in one small buffer, unlike an n*n byte matrix. Empty when the
    // tree outgrows one word per column (> 64), which the policy's max_columns never asks for.
    [[nodiscard]] std::vector<std::uint64_t> visibility_bits() const {
        const auto n = columns();
        if (n <= 0 || n > 64) { return {}; }
        std::vector<std::uint64_t> out(static_cast<std::size_t>(n), 0);
        for (std::int32_t c = 0; c < n; ++c) {
            for (std::int32_t cursor = c; cursor >= 0;) {
                out[static_cast<std::size_t>(c)] |= std::uint64_t{1} << static_cast<unsigned>(cursor);
                if (cursor == 0) { break; }
                cursor = parent[static_cast<std::size_t>(cursor)];
            }
        }
        return out;
    }

    // The number of leaves: the paths a verifier can accept, i.e. the tree's whole point. One leaf
    // means the "tree" is a chain and the extra columns bought nothing.
    [[nodiscard]] std::int32_t paths() const noexcept {
        const auto n = static_cast<std::size_t>(columns());
        std::vector<std::uint8_t> has_child(n, 0);
        for (std::size_t c = 1; c < n; ++c) {
            const auto p = parent[c];
            if (p >= 0 && static_cast<std::size_t>(p) < n) {
                has_child[static_cast<std::size_t>(p)] = 1;
            }
        }
        std::int32_t leaves = 0;
        for (std::size_t c = 0; c < n; ++c) {
            if (has_child[c] == 0) { ++leaves; }
        }
        return leaves;
    }

    // Columns on the longest path, anchor included. A chain of k drafts has k + 1.
    [[nodiscard]] std::int32_t longest_path() const noexcept {
        std::int32_t longest = 0;
        for (std::int32_t c = 0; c < columns(); ++c) {
            longest = std::max(longest, depth(c) + 1);
        }
        return longest;
    }
};

// How much of the draft's candidate list becomes columns. The draft half already produces sixteen
// candidates per column (kSparseSpeculativeCandidates); a tree only decides how many of them are
// worth a column of their own.
struct TreeBuildPolicy {
    std::int32_t max_depth   = 2;    // drafted layers below the anchor; depth 2 is the measured knee
    std::int32_t max_columns = 8;    // the small-T band the verify row is shaped for
    std::int32_t branch_width = 2;   // candidates kept per column
    float threshold          = 0.10F; // floor on the ACCUMULATED probability, not the column's own
};

namespace detail {

// The extent kernel's own reading of a column: a temperature-1 softmax over the sixteen candidate
// scores. Duplicated deliberately rather than shared: that kernel runs on the device over a
// different layout, and the two must be able to change independently without one silently moving
// the other's threshold semantics.
inline float candidate_probability(const float* scores, std::int32_t candidates,
                                   std::int32_t index) noexcept {
    float best = scores[0];
    for (std::int32_t c = 1; c < candidates; ++c) { best = std::max(best, scores[c]); }
    double mass = 0.0;
    for (std::int32_t c = 0; c < candidates; ++c) {
        mass += std::exp(static_cast<double>(scores[c] - best));
    }
    if (!(mass > 0.0)) { return 0.0F; }
    return static_cast<float>(std::exp(static_cast<double>(scores[index] - best)) / mass);
}

} // namespace detail

// Build the tree from the draft's per-column candidate scores.
//
// `scores` is [columns * candidates], column-major by step: scores[column * candidates + c] is the
// c-th candidate's score for the token that would follow `column`'s token. That is the layout the
// extent kernel already walks (row * steps + step) * candidates, so a dump can be fed in unchanged.
//
// A column is admitted only when its accumulated probability - its parent's times its own - clears
// the threshold. The product is the reason max_depth is small: at threshold 0.10 the second layer is
// already down to a few percent, so a depth-3 tree is a chain wearing extra columns.
[[nodiscard]] inline SpeculativeTree build_speculative_tree(const float* scores,
                                                            std::int32_t columns,
                                                            std::int32_t candidates,
                                                            const TreeBuildPolicy& policy) {
    SpeculativeTree tree;
    tree.parent.assign(1, -1);
    tree.path_probability.assign(1, 1.0F);
    if (scores == nullptr || columns <= 0 || candidates <= 0 || policy.max_depth <= 0 ||
        policy.max_columns <= 1) {
        return tree;
    }
    std::vector<std::int32_t> frontier{0};
    for (std::int32_t level = 0; level < policy.max_depth && !frontier.empty(); ++level) {
        std::vector<std::int32_t> next;
        for (const std::int32_t column : frontier) {
            if (column >= columns) { continue; }
            const float* row = scores + static_cast<std::size_t>(column) * candidates;
            // Rank the candidates once per column instead of per link, because the same column is
            // expanded once and its ordering decides which children exist at all.
            std::vector<std::int32_t> order(static_cast<std::size_t>(candidates));
            for (std::int32_t c = 0; c < candidates; ++c) { order[static_cast<std::size_t>(c)] = c; }
            std::stable_sort(order.begin(), order.end(), [row](std::int32_t a, std::int32_t b) {
                return row[a] > row[b];
            });
            std::int32_t kept = 0;
            for (const std::int32_t candidate : order) {
                if (kept >= policy.branch_width) { break; }
                if (tree.columns() >= policy.max_columns) { break; }
                const float own = detail::candidate_probability(row, candidates, candidate);
                const float accumulated =
                    tree.path_probability[static_cast<std::size_t>(column)] * own;
                if (!(accumulated >= policy.threshold)) { break; } // ranked: the rest is smaller
                tree.parent.push_back(column);
                tree.path_probability.push_back(accumulated);
                next.push_back(tree.columns() - 1);
                ++kept;
            }
        }
        frontier = std::move(next);
    }
    return tree;
}

} // namespace ninfer::models::qwen3_5::execution
