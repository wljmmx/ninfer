#pragma once

// Diagnostic-only: int8 activation round-trip (quantize -> dequantize) used to measure the
// model-level accuracy cost of A8 activations independently of any kernel change.
// Enabled at runtime by the environment variable NINFER_ACT_A8 (any non-empty value).
// Layout: tokens x k with k contiguous (the Linear operand convention: row t occupies
// [t*k, (t+1)*k)).

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/** Returns true when the round-trip probe is enabled (NINFER_ACT_A8 set). */
bool act_quant_a8_probe_enabled();

/** In-place per-token int8 round-trip on a bf16 activation plane. No-op when disabled. */
void act_quant_a8_roundtrip_bf16(void* data, std::int32_t k, std::int32_t tokens,
                                 cudaStream_t stream);

} // namespace ninfer::ops
