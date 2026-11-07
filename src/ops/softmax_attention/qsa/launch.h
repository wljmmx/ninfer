#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail::qsa {

void launch_query_gate(const Tensor& x, const Weight& weight, Tensor& query, Tensor& gate,
                       cudaStream_t stream);
void launch_kv_index(const Tensor& x, const Weight& weight, Tensor& key, Tensor& value,
                     Tensor& index_query, Tensor& index_key, cudaStream_t stream);
void launch_joint(const Tensor& x, const Weight& weight, Tensor& query, Tensor& key, Tensor& gate,
                  Tensor& value, Tensor& index_query, Tensor& index_key, cudaStream_t stream);

} // namespace ninfer::ops::detail::qsa
