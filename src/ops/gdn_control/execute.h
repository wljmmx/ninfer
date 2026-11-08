#pragma once

#include "ops/gdn_control/workspace.h"

namespace ninfer::ops::detail::gdn_control {

void execute_projection(const Tensor& x, const Weights& weights, const Tensor& a_log,
                        const Tensor& dt_bias, GdnControlOutputs out, Plan plan, Scratch scratch,
                        DeviceExecutionView execution);
void execute_norm(const Tensor& x, const Tensor& norm_weight, float eps, const Weights& weights,
                  const Tensor& a_log, const Tensor& dt_bias, GdnNormControlOutputs out,
                  NormPlan plan, Scratch scratch, DeviceExecutionView execution);

} // namespace ninfer::ops::detail::gdn_control
