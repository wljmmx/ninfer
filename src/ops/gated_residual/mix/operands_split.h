#pragma once

#include "ninfer/ops/gated_residual.h"
#include "ops/gated_residual/geometry.h"
#include "ops/gated_residual/mix/operands.h"

namespace ninfer::ops::detail::hc::mix {
// The separate-weight form of Kind::Block / Kind::Final: down and inject are bound as their own
// quantized matrices instead of one contiguous BF16 parent.
struct SeparateWeights {
    Kind kind;
    const Weight& down;
    const Weight* inject;
    const Weight& up;
};

struct SeparateOperands {
    const void* normalized;
    const void* down;
    const void* inject;
    const void* up;
    void* output;
    float* injection;
    std::int32_t columns;
};

// Q5 RowSplit planes of one matrix, with the row strides the artifact's padded K implies.
struct Q5Planes {
    const void* codes;
    const void* high;
    const void* scales;
    std::int32_t code_row_bytes;
    std::int32_t high_row_bytes;
    std::int32_t scale_row_bytes;
    std::int32_t padded_k;
};

// Validates the Q5 RowSplit geometry of one of the three matrices and returns its planes. `role`
// names the matrix in a refusal ("down", "inject", "up").
Q5Planes q5_planes(const Weight& weight, const char* role, int rows, int columns);

void describe_separate(SeparateWeights weights);
SeparateOperands validate_separate(SeparateWeights weights, const Tensor& normalized,
                                   Tensor& output, Tensor* injection,
                                   DeviceExecutionView execution);
void validate_separate_scratch(SeparateWeights weights, const Tensor& normalized,
                               const Tensor& output, const Tensor* injection, DeviceSpan scratch);
} // namespace ninfer::ops::detail::hc::mix
