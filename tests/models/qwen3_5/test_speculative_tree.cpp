#include "models/qwen3_5/execution/speculative_tree.h"

#include <cstdint>
#include <iostream>
#include <vector>

// The tree's shape, on its own terms: no GPU, no model, no draft. What is asserted here is the part
// everything downstream depends on - which column continues from which, and which columns a column
// may read - because a wrong mask does not crash: it silently verifies a sequence that was never
// generated, and the acceptance rate drifts instead of the program failing.

namespace {

using namespace ninfer::models::qwen3_5::execution;

int failures = 0;

void require(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

// A column whose first `wide` candidates share the mass: logits 0 for those, deeply negative for the
// rest, so the softmax is 1/wide each and the arithmetic in the tests stays exact.
void fill_column(std::vector<float>& scores, int column, int candidates, int wide) {
    for (int c = 0; c < candidates; ++c) {
        scores[static_cast<std::size_t>(column) * candidates + c] = c < wide ? 0.0F : -60.0F;
    }
}

void chain_stays_a_chain() {
    constexpr int kColumns = 4;
    constexpr int kCandidates = 4;
    std::vector<float> scores(static_cast<std::size_t>(kColumns) * kCandidates, -60.0F);
    for (int c = 0; c < kColumns; ++c) { fill_column(scores, c, kCandidates, 1); }

    TreeBuildPolicy policy;
    policy.max_depth    = 3;
    policy.max_columns  = 8;
    policy.branch_width = 2; // one candidate has the mass, so a width of two still yields one
    policy.threshold    = 0.10F;

    const SpeculativeTree tree =
        build_speculative_tree(scores.data(), kColumns, kCandidates, policy);
    require(tree.columns() == 4, "a single-candidate column adds exactly one node per level");
    require(tree.paths() == 1, "a chain has one path");
    require(tree.longest_path() == 4, "a chain of four columns is four deep including the anchor");
    for (int c = 1; c < tree.columns(); ++c) {
        require(tree.parent[static_cast<std::size_t>(c)] == c - 1, "a chain links to its predecessor");
        require(tree.depth(c) == c, "a chain's depth equals its index");
    }
}

void branching_makes_siblings() {
    constexpr int kColumns = 8;
    constexpr int kCandidates = 4;
    std::vector<float> scores(static_cast<std::size_t>(kColumns) * kCandidates, -60.0F);
    for (int c = 0; c < kColumns; ++c) { fill_column(scores, c, kCandidates, 2); }

    TreeBuildPolicy policy;
    policy.max_depth    = 2;
    policy.max_columns  = 8;
    policy.branch_width = 2;
    // Each column's mass is split in two, so a depth-2 path accumulates 0.25 - above this floor.
    policy.threshold    = 0.10F;

    const SpeculativeTree tree =
        build_speculative_tree(scores.data(), kColumns, kCandidates, policy);
    require(tree.columns() == 1 + 2 + 4, "two wide candidates over two levels is seven nodes");
    // The anchor's two children are siblings; so are each child's two children.
    require(tree.parent[1] == 0 && tree.parent[2] == 0, "the two first-level nodes share the anchor");
    require(!tree.is_ancestor(1, 2), "first-level siblings are not ancestors of each other");
    require(!tree.is_ancestor(2, 1), "ancestry is directional");
    require(tree.is_ancestor(0, 2), "the anchor is an ancestor of every node");
    require(tree.paths() == 4, "four leaves");
    require(tree.longest_path() == 3, "the anchor plus two levels");
}

void mask_matches_ancestry() {
    constexpr int kColumns = 8;
    constexpr int kCandidates = 4;
    std::vector<float> scores(static_cast<std::size_t>(kColumns) * kCandidates, -60.0F);
    for (int c = 0; c < kColumns; ++c) { fill_column(scores, c, kCandidates, 2); }

    TreeBuildPolicy policy;
    policy.max_depth    = 2;
    policy.max_columns  = 8;
    policy.branch_width = 2;
    policy.threshold    = 0.10F;
    const SpeculativeTree tree =
        build_speculative_tree(scores.data(), kColumns, kCandidates, policy);

    std::vector<std::uint8_t> visible;
    tree.build_mask(visible);
    const auto n = static_cast<std::size_t>(tree.columns());
    for (std::size_t c = 0; c < n; ++c) {
        for (std::size_t k = 0; k < n; ++k) {
            const auto bit = visible[c * n + k];
            const auto expected =
                tree.is_ancestor(static_cast<std::int32_t>(k), static_cast<std::int32_t>(c)) ? 1 : 0;
            require(bit == static_cast<std::uint8_t>(expected),
                    "mask bit equals the ancestry relation");
        }
        require(visible[c * n + c] == 1, "a column always reads itself");
        // The causal property the mask exists for: no node reads a node that came after it, so the
        // mask is lower-triangular whatever the shape does.
        for (std::size_t k = c + 1; k < n; ++k) {
            require(visible[c * n + k] == 0, "nothing to the right of the diagonal is visible");
        }
    }
}

void threshold_prunes_the_product() {
    constexpr int kColumns = 4;
    constexpr int kCandidates = 4;
    std::vector<float> scores(static_cast<std::size_t>(kColumns) * kCandidates, -60.0F);
    for (int c = 0; c < kColumns; ++c) { fill_column(scores, c, kCandidates, 2); }

    TreeBuildPolicy policy;
    policy.max_depth    = 2;
    policy.max_columns  = 8;
    policy.branch_width = 2;
    // Each column splits its mass in two: level 1 is 0.5, level 2 is 0.25. A floor of 0.4 keeps the
    // first level and refuses the second - the product is what is compared, not the column's own.
    policy.threshold    = 0.40F;
    const SpeculativeTree shallow =
        build_speculative_tree(scores.data(), kColumns, kCandidates, policy);
    require(shallow.columns() == 3, "a floor of 0.40 admits the 0.5 level and refuses the 0.25 one");
    require(shallow.longest_path() == 2, "so the tree is the anchor and one level");

    policy.threshold = 0.60F;
    const SpeculativeTree stub =
        build_speculative_tree(scores.data(), kColumns, kCandidates, policy);
    require(stub.columns() == 1, "a floor above the first level leaves the anchor alone");
    require(stub.paths() == 1, "a one-node tree has the anchor as its only leaf");
}

void accumulation_is_the_product() {
    constexpr int kColumns = 4;
    constexpr int kCandidates = 4;
    std::vector<float> scores(static_cast<std::size_t>(kColumns) * kCandidates, -60.0F);
    for (int c = 0; c < kColumns; ++c) { fill_column(scores, c, kCandidates, 2); }

    TreeBuildPolicy policy;
    policy.max_depth    = 2;
    policy.max_columns  = 8;
    policy.branch_width = 1;
    policy.threshold    = 0.01F;
    const SpeculativeTree tree =
        build_speculative_tree(scores.data(), kColumns, kCandidates, policy);
    require(tree.columns() == 3, "one branch per level");
    require(tree.path_probability[0] == 1.0F, "the anchor's probability is one by definition");
    require(tree.path_probability[1] > 0.49F && tree.path_probability[1] < 0.51F,
            "the first level carries the column's own mass");
    require(tree.path_probability[2] > 0.24F && tree.path_probability[2] < 0.26F,
            "the second level is the product, not a restatement of the column");
}

void column_cap_truncates() {
    constexpr int kColumns = 16;
    constexpr int kCandidates = 4;
    std::vector<float> scores(static_cast<std::size_t>(kColumns) * kCandidates, -60.0F);
    for (int c = 0; c < kColumns; ++c) { fill_column(scores, c, kCandidates, 4); }

    TreeBuildPolicy policy;
    policy.max_depth    = 3;
    policy.max_columns  = 6;
    policy.branch_width = 4;
    policy.threshold    = 0.01F;
    const SpeculativeTree tree =
        build_speculative_tree(scores.data(), kColumns, kCandidates, policy);
    require(tree.columns() == 6, "the column cap is absolute");
    require(tree.columns() <= 40, "the small-T band is not exceeded by construction");
}

void unbounded_input_is_refused() {
    TreeBuildPolicy policy;
    const SpeculativeTree empty = build_speculative_tree(nullptr, 8, 16, policy);
    require(empty.columns() == 1, "no scores means the anchor alone, not a crash");

    const SpeculativeTree no_columns = build_speculative_tree(nullptr, 0, 16, policy);
    require(no_columns.columns() == 1, "no columns means the anchor alone");

    TreeBuildPolicy no_depth;
    no_depth.max_depth = 0;
    const std::vector<float> scores(64, 0.0F);
    const SpeculativeTree flat = build_speculative_tree(scores.data(), 4, 16, no_depth);
    require(flat.columns() == 1, "no depth means no drafted nodes");
}

// A chain is the shape every existing verify kernel already implements, so it is the shape the new
// positions and visibility must degrade into bit for bit: positions base + c, and a visibility that is
// exactly the lower triangle. If this ever fails, the tree path is not a generalisation of the chain -
// it is a different computation wearing the same name.
void chain_positions_and_bits_reduce_to_the_chain() {
    SpeculativeTree chain;
    constexpr int kColumns = 5;
    chain.parent.assign({-1, 0, 1, 2, 3});
    chain.path_probability.assign(kColumns, 1.0F);

    const auto positions = chain.depth_positions(7);
    require(static_cast<int>(positions.size()) == kColumns, "one position per column");
    for (int c = 0; c < kColumns; ++c) {
        require(positions[static_cast<std::size_t>(c)] == 7 + c,
                "a chain advances one position per column, as the chain kernel does");
    }

    const auto bits = chain.visibility_bits();
    require(static_cast<int>(bits.size()) == kColumns, "one word per column");
    for (int c = 0; c < kColumns; ++c) {
        const std::uint64_t expected = (std::uint64_t{1} << static_cast<unsigned>(c + 1)) - 1;
        require(bits[static_cast<std::size_t>(c)] == expected,
                "a chain's visibility is the lower triangle, exactly");
    }
}

// The property the mask exists for: a sibling was never generated after this token, so it must not be
// readable. A triangle over depth positions would make siblings visible; the bits must not.
void siblings_are_not_visible() {
    SpeculativeTree tree;
    // 0 -> {1, 2}, then 1 -> 3, 2 -> 4: two siblings at depth 1, one leaf each below them.
    tree.parent.assign({-1, 0, 0, 1, 2});
    tree.path_probability.assign(5, 1.0F);

    const auto positions = tree.depth_positions(100);
    require(positions[1] == 101 && positions[2] == 101, "siblings share one position");

    const auto bits = tree.visibility_bits();
    require((bits[1] & (std::uint64_t{1} << 2)) == 0, "column 1 must not read its sibling 2");
    require((bits[2] & (std::uint64_t{1} << 1)) == 0, "column 2 must not read its sibling 1");
    require((bits[3] & (std::uint64_t{1} << 2)) == 0, "column 3 must not read its uncle 2");
    require((bits[3] & (std::uint64_t{1} << 1)) != 0, "column 3 reads its parent 1");
    require((bits[3] & (std::uint64_t{1} << 0)) != 0, "column 3 reads the anchor");
    require((bits[4] & (std::uint64_t{1} << 2)) != 0, "column 4 reads its parent 2");
    require((bits[4] & (std::uint64_t{1} << 3)) == 0, "column 4 must not read its cousin 3");
}

// The build's own ordering is what keeps the attention geometry valid: base_pos = positions[0] and
// window = positions[last] + 1 both assume the last column is the deepest. Assert it rather than
// assume it, because a build that emitted a deep column early would silently shrink the window and
// blind every column after it.
void build_emits_columns_in_depth_order() {
    constexpr int kColumns = 8;
    constexpr int kCandidates = 4;
    std::vector<float> scores(static_cast<std::size_t>(kColumns) * kCandidates, -60.0F);
    for (int c = 0; c < kColumns; ++c) { fill_column(scores, c, kCandidates, 2); }

    TreeBuildPolicy policy;
    policy.max_depth    = 3;
    policy.max_columns  = 8;
    policy.branch_width = 2;
    policy.threshold    = 0.10F;
    const SpeculativeTree tree =
        build_speculative_tree(scores.data(), kColumns, kCandidates, policy);

    const auto positions = tree.depth_positions(1000);
    for (std::size_t c = 1; c < positions.size(); ++c) {
        require(positions[c] >= positions[c - 1], "columns are non-decreasing in depth");
    }
    require(positions.back() == 1000 + tree.longest_path() - 1,
            "the last column is one of the deepest, so window = positions[last] + 1 is the whole tree");
    require(tree.columns() <= 8, "the small-T band fits one visibility word per column");
}

} // namespace

int main() {
    chain_stays_a_chain();
    branching_makes_siblings();
    mask_matches_ancestry();
    threshold_prunes_the_product();
    accumulation_is_the_product();
    column_cap_truncates();
    unbounded_input_is_refused();
    chain_positions_and_bits_reduce_to_the_chain();
    siblings_are_not_visible();
    build_emits_columns_in_depth_order();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "speculative tree: shape, mask and the chain's own reduction ok\n";
    return 0;
}
