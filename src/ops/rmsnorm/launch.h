#pragma once
#include "ops/rmsnorm/operands.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail::rmsnorm {
void launch(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps, Epilogue kind,
            Tensor& out, Operands operands, std::int32_t multiprocessor_count, cudaStream_t stream);
} // namespace ninfer::ops::detail::rmsnorm
