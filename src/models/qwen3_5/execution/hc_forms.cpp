// The one place that turns an HcParameters storage form into an Op call.
//
// Qwen4Exp's low-rank correction arrives in two shapes: a compact BF16 parent whose tail rows are
// the injection coefficients, or q5_g64_fp16 groupwise planes with a scale plane each. Both are
// production artifacts of this family (the packed GGUF one and the native 35B one), so the model
// layer dispatches here instead of every execution and planning site testing the storage itself.
#include "models/qwen3_5/execution/parameters.h"

#include "ninfer/ops/gated_residual.h"

#include <type_traits>

namespace ninfer::models::qwen3_5::execution {

void hc_mix(const Tensor& normalized, const HcParameters::BlockForm& weights,
            WorkspaceArena& workspace, Tensor& block_input, Tensor& injection,
            DeviceExecutionView execution) {
    std::visit(
        [&](const auto& form) {
            ops::hc_mix(normalized, form, workspace, block_input, injection, execution);
        },
        weights);
}

void hc_mix(const Tensor& normalized, const HcParameters::FinalForm& weights,
            WorkspaceArena& workspace, Tensor& sample_hidden, DeviceExecutionView execution) {
    std::visit(
        [&](const auto& form) {
            ops::hc_mix(normalized, form, workspace, sample_hidden, execution);
        },
        weights);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcParameters::BlockForm& weights,
                                            std::int32_t first, std::int32_t last,
                                            DeviceExecutionView execution) {
    return std::visit(
        [&](const auto& form) {
            return ops::hc_mix_workspace_capacity_bytes(form, first, last, execution);
        },
        weights);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcParameters::FinalForm& weights,
                                            std::int32_t first, std::int32_t last,
                                            DeviceExecutionView execution) {
    return std::visit(
        [&](const auto& form) {
            return ops::hc_mix_workspace_capacity_bytes(form, first, last, execution);
        },
        weights);
}

} // namespace ninfer::models::qwen3_5::execution
