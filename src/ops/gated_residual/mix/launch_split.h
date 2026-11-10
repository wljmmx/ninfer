#pragma once

#include "ops/gated_residual/mix/operands_split.h"
#include "ops/gated_residual/mix/workspace.h"

namespace ninfer::ops::detail::hc::mix {
// Launches the separate-weight forms on the Q5 RowSplit path: the low-rank rows and the injection
// rows through the row-split SIMT contraction, the fold through the fused Q5 Up kernel.
void launch_separate(SeparateWeights weights, SeparateOperands o, Scratch scratch,
                     DeviceExecutionView execution);
} // namespace ninfer::ops::detail::hc::mix
