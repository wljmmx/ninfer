#include "ninfer/ops/gdn_control.h"
#include "ops/gdn_control/execute.h"
#include "ops/gdn_control/bf16/launch.h"

#include <array>

namespace ninfer::ops {
namespace control = detail::gdn_control;

void gdn_control(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias,
                 GdnControlOutputs out, cudaStream_t stream) {
    control::validate_control(a, b, a_log, dt_bias, out);
    if (a.ne[1] == 0) return;
    const std::array<const Tensor*, 6> tensors{&a, &b, &a_log, &dt_bias, &out.g, &out.beta};
    control::validate_disjoint(nullptr, tensors);
    control::launch_control(a, b, a_log, dt_bias, out, stream);
}

std::size_t gdn_control_proj_workspace_capacity_bytes(const ProjectionWeights& weights,
                                                      std::int32_t first, std::int32_t last) {
    return control::capacity(control::describe_weights(weights).geometry, false, first, last);
}

std::size_t gdn_norm_control_proj_workspace_capacity_bytes(const ProjectionWeights& weights,
                                                           std::int32_t first, std::int32_t last) {
    return control::capacity(control::describe_weights(weights).geometry, true, first, last);
}

void gdn_control_proj(const Tensor& x, const ProjectionWeights& weights, const Tensor& a_log,
                      const Tensor& dt_bias, GdnControlOutputs out, WorkspaceArena& workspace,
                      DeviceExecutionView execution) {
    const auto operands = control::describe_weights(weights);
    control::validate_weights(operands);
    control::validate_projection(x, operands, a_log, dt_bias, out, execution);
    const auto plan    = control::projection_plan(operands.geometry, x.ne[1]);
    auto scope         = workspace.scope();
    const auto scratch = control::allocate_workspace(workspace, operands.geometry, plan, x.ne[1]);
    const std::array<const Tensor*, 5> tensors{&x, &a_log, &dt_bias, &out.g, &out.beta};
    control::validate_disjoint(&operands, tensors, scratch.partial);
    control::execute_projection(x, operands, a_log, dt_bias, out, plan, scratch, execution);
}

void gdn_norm_control_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                           const ProjectionWeights& weights, const Tensor& a_log,
                           const Tensor& dt_bias, GdnNormControlOutputs out,
                           WorkspaceArena& workspace, DeviceExecutionView execution) {
    const auto operands = control::describe_weights(weights);
    control::validate_weights(operands);
    control::validate_projection(x, operands, a_log, dt_bias, out.control, execution);
    control::validate_norm(x, norm_weight, eps, operands.geometry, out.normalized);
    const auto plan = control::norm_plan(operands.geometry, x.ne[1]);
    auto scope      = workspace.scope();
    const auto scratch =
        control::allocate_workspace(workspace, operands.geometry, plan.projection, x.ne[1],
                                    plan.kernel == control::NormKernel::Mma);
    const std::array<const Tensor*, 7> tensors{
        &x, &norm_weight, &a_log, &dt_bias, &out.normalized, &out.control.g, &out.control.beta};
    control::validate_disjoint(&operands, tensors, scratch.partial);
    control::execute_norm(x, norm_weight, eps, operands, a_log, dt_bias, out, plan, scratch,
                          execution);
}

} // namespace ninfer::ops
