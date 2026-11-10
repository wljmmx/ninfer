#include "ninfer/ops/gated_residual.h"
#include "ops/gated_residual/mix/launch.h"
#include "ops/gated_residual/mix/launch_split.h"
#include "ops/gated_residual/mix/operands_split.h"

namespace ninfer::ops {
namespace mix = detail::hc::mix;

namespace {
void execute_mix(mix::Weights weights, const Tensor& normalized, WorkspaceArena& workspace,
                 Tensor& output, Tensor* injection, DeviceExecutionView execution) {
    const auto o = mix::validate(weights, normalized, output, injection, execution);
    const auto p = mix::plan(weights.kind, o.columns);
    auto scope   = workspace.scope();
    const auto scratch =
        mix::allocate_workspace(workspace, mix::layout(weights.kind, p, o.columns));
    mix::validate_scratch(weights, normalized, output, injection, scratch.storage);
    mix::launch(weights.kind, o, p, scratch, execution);
}

void execute_separate(mix::SeparateWeights weights, const Tensor& normalized,
                      WorkspaceArena& workspace, Tensor& output, Tensor* injection,
                      DeviceExecutionView execution) {
    const auto o = mix::validate_separate(weights, normalized, output, injection, execution);
    auto scope   = workspace.scope();
    const auto scratch = mix::allocate_workspace(workspace, mix::layout_separate(o.columns));
    mix::validate_separate_scratch(weights, normalized, output, injection, scratch.storage);
    mix::launch_separate(weights, o, scratch, execution);
}
} // namespace

void hc_mix(const Tensor& normalized, const HcMixWeights& weights, WorkspaceArena& workspace,
            Tensor& block_input, Tensor& injection, DeviceExecutionView execution) {
    execute_mix({mix::Kind::Block, weights.down_injection, weights.up}, normalized, workspace,
                block_input, &injection, execution);
}

void hc_mix(const Tensor& normalized, const HcFinalMixWeights& weights, WorkspaceArena& workspace,
            Tensor& sample_hidden, DeviceExecutionView execution) {
    execute_mix({mix::Kind::Final, weights.down, weights.up}, normalized, workspace, sample_hidden,
                nullptr, execution);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcMixWeights& weights, std::int32_t first,
                                            std::int32_t last, DeviceExecutionView execution) {
    return mix::capacity({mix::Kind::Block, weights.down_injection, weights.up}, first, last,
                         execution);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcFinalMixWeights& weights, std::int32_t first,
                                            std::int32_t last, DeviceExecutionView execution) {
    return mix::capacity({mix::Kind::Final, weights.down, weights.up}, first, last, execution);
}

void hc_mix(const Tensor& normalized, const HcSeparateMixWeights& weights,
            WorkspaceArena& workspace, Tensor& block_input, Tensor& injection,
            DeviceExecutionView execution) {
    execute_separate({mix::Kind::Block, weights.down, &weights.inject, weights.up}, normalized,
                     workspace, block_input, &injection, execution);
}

void hc_mix(const Tensor& normalized, const HcFinalSeparateMixWeights& weights,
            WorkspaceArena& workspace, Tensor& sample_hidden, DeviceExecutionView execution) {
    execute_separate({mix::Kind::Final, weights.down, nullptr, weights.up}, normalized, workspace,
                     sample_hidden, nullptr, execution);
}

namespace {
std::size_t separate_capacity(const mix::SeparateWeights& weights, std::int32_t first,
                              std::int32_t last, DeviceExecutionView execution) {
    mix::describe_separate(weights);
    mix::validate_interval(first, last);
    mix::validate_execution(execution);
    return mix::layout_separate(last).bytes;
}
} // namespace

std::size_t hc_mix_workspace_capacity_bytes(const HcSeparateMixWeights& weights,
                                            std::int32_t first, std::int32_t last,
                                            DeviceExecutionView execution) {
    return separate_capacity({mix::Kind::Block, weights.down, &weights.inject, weights.up}, first,
                             last, execution);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcFinalSeparateMixWeights& weights,
                                            std::int32_t first, std::int32_t last,
                                            DeviceExecutionView execution) {
    return separate_capacity({mix::Kind::Final, weights.down, nullptr, weights.up}, first, last,
                             execution);
}
} // namespace ninfer::ops
