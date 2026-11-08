#pragma once

#include "ninfer/ops/gdn_control.h"
#include "ops/common/operand_checks.h"

#include <span>

namespace ninfer::ops::detail::gdn_control {

struct Geometry {
    std::int32_t heads, input_rows;
};

struct Weights {
    Geometry geometry;
    const Weight* first;
    const Weight* second;
};

Weights describe_weights(const ProjectionWeights& weights);
void validate_weights(const Weights& weights);
void validate_interval(std::int32_t first, std::int32_t last);
void validate_projection(const Tensor& x, const Weights& weights, const Tensor& a_log,
                         const Tensor& dt_bias, GdnControlOutputs out,
                         DeviceExecutionView execution);
void validate_norm(const Tensor& x, const Tensor& weight, float eps, Geometry geometry,
                   const Tensor& normalized);
void validate_control(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias,
                      GdnControlOutputs out);
void validate_disjoint(const Weights* weights, std::span<const Tensor* const> tensors,
                       DeviceSpan scratch = {});

} // namespace ninfer::ops::detail::gdn_control
