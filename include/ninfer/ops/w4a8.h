#pragma once

// int8 tensor-core routes (m16n8k32.s8 int8 MMA, int32 accumulate) over the packed Q4/Q5
// weights: the weight is dequantised to int8 and multiplied against int8 activations with one
// scale per 64-wide K group. Weights keep the existing packed layout, so no new artifact is
// required.
//
// Prefill only: every supported shape requires a token count that is a multiple of the route's
// token tile, so decode keeps the caller's normal path.
//
// Enabled by default. Set NINFER_W4A8 to 0 / off / false / no (any case) to fall back to the
// bf16 and q4/q5 kernels; that is the rollback and A/B comparison switch.
//
// Workspace: caller-owned transient storage reported by the w4a8_*_workspace_capacity_bytes()
// queries and scoped to the call. There is no persistent state side effect.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/** True when the W4A8 routes are enabled (the default). */
bool w4a8_enabled();

/** True when this weight/shape can use the W4A8 route at the given token count. */
bool w4a8_supported(const Weight& weight, std::int32_t tokens);

/** True when the folded gate/up weight can use the W4A8 SwiGLU route. */
bool w4a8_swiglu_supported(const Weight& weight, std::int32_t tokens);

/** True when the q4/q5 attention input projection pair can use the W4A8 route. */
bool w4a8_attn_input_supported(const Weight& query_key_weight, const Weight& gate_value_weight,
                               std::int32_t tokens);

/** True when the q4/q5 GDN input projection pair can use the W4A8 route. */
bool w4a8_gdn_input_supported(const Weight& qk_weight, const Weight& value_z_weight,
                              std::int32_t tokens);

/**
 * Transient capacity required by the W4A8 route for every T in the inclusive
 * [min_tokens,max_tokens] interval. Dimensioned for the profiles w4a8_supported() accepts and
 * zero for any other shape, so the planner can add it unconditionally. Invalid intervals throw.
 */
[[nodiscard]] std::size_t w4a8_linear_add_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

/** Transient capacity for the W4A8 SwiGLU route; zero outside its registered profile. */
[[nodiscard]] std::size_t w4a8_swiglu_workspace_capacity_bytes(std::int32_t gate_up_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/** Transient capacity for the W4A8 attention input projection pair; zero outside its profile. */
[[nodiscard]] std::size_t w4a8_attn_input_workspace_capacity_bytes(std::int32_t query_key_rows,
                                                                  std::int32_t gate_value_rows,
                                                                  std::int32_t input_rows,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

/** Transient capacity for the W4A8 GDN input projection pair; zero outside its profile. */
[[nodiscard]] std::size_t w4a8_gdn_input_workspace_capacity_bytes(std::int32_t qk_rows,
                                                                 std::int32_t value_z_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/** residual += W @ x, evaluated with the W4A8 route. Requires w4a8_supported(). */
void w4a8_linear_add(const Tensor& x, const Weight& weight, Tensor& residual, WorkspaceArena& ws,
                     cudaStream_t stream);

/** out = silu(gate(x)) * up(x) with the W4A8 route. Requires w4a8_swiglu_supported(). */
void w4a8_swiglu(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& ws,
                 cudaStream_t stream);

/** Fused q/k from one weight and gate/v from the other, via the W4A8 route. */
void w4a8_attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                          const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& key,
                          Tensor& value, WorkspaceArena& ws, cudaStream_t stream);

/** Fused qk/value (into qkv) and output gate (into z), via the W4A8 route. */
void w4a8_gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                         Tensor& qkv, Tensor& z, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
