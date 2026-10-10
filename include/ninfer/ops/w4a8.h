#pragma once

// Prototype W4A8 path: int4/int5 quantized weights + per-token int8 activations with an
// int8 tensor-core GEMM (m16n8k32.s8, int32 accumulate). Weights keep the existing packed
// layout, so no new artifact is required and decode is unaffected.
//
// Enabled at runtime by NINFER_W4A8 (any non-empty value). Only the exact shapes and the
// power-of-two token counts the implementation supports take this route; every other call
// is expected to fall back to the caller's normal path.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/** True when NINFER_W4A8 is set. */
bool w4a8_enabled();

/** True when this weight/shape can use the W4A8 route at the given token count. */
bool w4a8_supported(const Weight& weight, std::int32_t tokens);

/** residual += W @ x, evaluated with the W4A8 route. Requires w4a8_supported(). */
void w4a8_linear_add(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream);

/** True when the folded gate/up weight can use the W4A8 SwiGLU route. */
bool w4a8_swiglu_supported(const Weight& weight, std::int32_t tokens);

/** out = silu(gate(x)) * up(x) with the W4A8 route. Requires w4a8_swiglu_supported(). */
void w4a8_swiglu(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

/** True when the q4/q5 attention input projection pair can use the W4A8 route. */
bool w4a8_attn_input_supported(const Weight& query_key_weight, const Weight& gate_value_weight,
                               std::int32_t tokens);

/** Fused q/k from one weight and gate/v from the other, via the W4A8 route. */
void w4a8_attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                          const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& key,
                          Tensor& value, cudaStream_t stream);

} // namespace ninfer::ops
