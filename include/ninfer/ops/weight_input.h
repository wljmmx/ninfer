#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/sparse_moe.h"

#include <array>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::ops {

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
};

struct PairedProjectionWeights {
    Weight first, second;
};

// One parent per logical projection: the artifact keeps each matrix in its own object, so the four
// inputs have no fused row order to concatenate. `rows` follows the caller's own order - query,
// key, gate, value for attention, query, key, value, z for GDN - and each row is executed on its
// own through Linear.
struct SeparateProjectionWeights {
    std::array<SingleProjectionWeight, 4> rows;
};

using ProjectionWeights =
    std::variant<SingleProjectionWeight, PairedProjectionWeights, SeparateProjectionWeights>;

// port (sm_89): address-independent matrix description used by execution planning. The bench
// tree keeps these in projection_weights.h; this tree keeps one header for both.
struct LinearProfile {
    QType qtype;
    std::int32_t n, k;
    LinearPolicy policy = LinearPolicy::A16Only;
    QuantLayout layout  = QuantLayout::Contiguous;

    friend bool operator==(const LinearProfile&, const LinearProfile&) = default;
};

struct ProjectionProfile {
    LinearProfile first;
    std::optional<LinearProfile> second;
    // One entry per parent when the artifact keeps each projection in its own object; empty for
    // the fused and paired forms.
    std::vector<LinearProfile> separate;

    friend bool operator==(const ProjectionProfile&, const ProjectionProfile&) = default;
};

[[nodiscard]] inline LinearProfile linear_profile(const SingleProjectionWeight& p) {
    return {p.weight.qtype, p.weight.n, p.weight.k, p.policy, p.weight.layout};
}

[[nodiscard]] inline LinearProfile linear_profile(LinearProfile p) { return p; }

[[nodiscard]] inline ProjectionProfile projection_profile(const ProjectionWeights& weights) {
    if (const auto* single = std::get_if<SingleProjectionWeight>(&weights)) {
        return {linear_profile(*single), {}, {}};
    }
    if (const auto* separate = std::get_if<SeparateProjectionWeights>(&weights)) {
        std::vector<LinearProfile> rows;
        rows.reserve(separate->rows.size());
        for (const auto& row : separate->rows) { rows.push_back(linear_profile(row)); }
        return {rows.front(), {}, std::move(rows)};
    }
    const auto& pair = std::get<PairedProjectionWeights>(weights);
    return {linear_profile({pair.first}), linear_profile({pair.second}), {}};
}

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                                     const WeightInput& key,
                                                                     const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query,
                                                                const WeightInput& key,
                                                                const WeightInput& gate,
                                                                const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query,
                                                               const WeightInput& key,
                                                               const WeightInput& value,
                                                               const WeightInput& z);
[[nodiscard]] ProjectionWeights prepare_gdn_control_proj_weights(const WeightInput& a,
                                                                 const WeightInput& b);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);
[[nodiscard]] SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down);

} // namespace ninfer::ops
