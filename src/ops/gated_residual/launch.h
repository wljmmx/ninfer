#pragma once
#include "ops/gated_residual/operands.h"

namespace ninfer::ops::detail::hc {
void launch(Form form, Operands operands, DeviceExecutionView execution);
} // namespace ninfer::ops::detail::hc
