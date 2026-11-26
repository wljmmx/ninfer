#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail::ggml {

// The three expert banks of one Qwen4Exp layer, as the container stores them: `experts` rows of `I`
// per bank for gate and up, `experts` rows of `H` for down, each row a run of GGML blocks. A bank's
// row stride is its own encoded width, so row (e, i) of gate starts at `base + (e * I + i) * row_bytes`.
struct ExpertBanks {
    const Weight* gate;
    const Weight* up;
    const Weight* down;
};

// Scratch the launch needs, so a caller can price it before running.
[[nodiscard]] std::size_t routed_swiglu_workspace_bytes(int top_k, int intermediate);

// One routed-SwiGLU layer for `tokens` columns:
//   y[t] = sum over t's routes k of coef[k] * down[e_k] @ ( silu(gate[e_k] @ x[t]) * (up[e_k] @ x[t]) )
// `ids` is [top_k, T] row-major with ids[k*T + t]; `coefficients` has the same shape. `out` is [H, T]
// BF16 and is written once per (token, row) - the routes of a token are summed inside one CTA.
void launch_routed_swiglu(const Tensor& x, const ExpertBanks& banks, int experts, int top_k,
                          const Tensor& ids, const Tensor& coefficients, Tensor& out,
                          WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail::ggml
