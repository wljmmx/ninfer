#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::models::qwen3_5::execution {

// Diagnosis (NINFER_NAN_PROBE=1): count the BF16 planes of `value` that hold a NaN, per column, and
// print one line for the pass. A run without the variable set does nothing but return, so a probe
// left in a hot path costs a branch. `layer` is only the label the line carries: it names the
// layer the value belongs to, or a code the caller picked for a value outside the layer loop.
void nan_probe_tensor(const char* tag, const Tensor& value, std::size_t layer, cudaStream_t stream);

// NINFER_QSA_DUMP_LAYER selects one indexed layer by its compact index, or every layer with -1.
[[nodiscard]] bool qsa_dump_wants_layer(int layer);

// Diagnosis: write one plane of a layer as raw bytes plus its stream hash. A plane is addressed by
// the pass's column extent, so a prefill chunk and a decode step write different files and a
// comparison can pick the pass it means instead of racing the last write.
void qsa_dump_tensor(const char* variable, const char* name, const Tensor& value, int layer,
                     std::int32_t columns, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
