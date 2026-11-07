#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ops/gated_residual/geometry.h"

namespace ninfer::ops::detail::hc {
enum class Form { ExpandNorm, GroupedNorm, Combine, CombineNorm, AddNorm };

struct Operands {
    const void* input;
    const void* block;
    const float* injection;
    const void* weight;
    void* residual;
    void* normalized;
    std::int32_t columns;
    float epsilon;
};

Operands validate(Form form, const Tensor& input, const Tensor* block, const Tensor* injection,
                  const Tensor* weight, float epsilon, Tensor* residual, Tensor* normalized,
                  DeviceExecutionView execution, const char* op);
} // namespace ninfer::ops::detail::hc
