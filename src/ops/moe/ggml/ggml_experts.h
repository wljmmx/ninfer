#pragma once

#include "core/arena.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail::ggml {

// One expert bank as the artifact stores it: `experts` runs of `rows_per_expert` encoded rows.
//
// Addressing follows `ExpertMatrixDeviceView`: an expert's first row is `codes + offsets[e]` when the
// bank is indexed, or `codes + e * expert_stride` when it is affine. `host_flags[e]` non-zero means the
// expert is served from the mapped Host tier instead (`host_codes + host_offsets[e]`) - the case the
// streamed pool relies on, and the reason the row pointer is chosen per route rather than per bank.
struct ExpertBank {
    const std::uint8_t* codes       = nullptr;
    const std::uint8_t* host_codes  = nullptr;
    const std::uint64_t* offsets    = nullptr;
    const std::uint64_t* host_offsets = nullptr;
    const std::int32_t* host_flags  = nullptr;
    std::uint64_t expert_stride     = 0;
    std::int32_t row_bytes          = 0; // encoded bytes of one row of this bank
    std::int32_t groups_per_block   = 1;
    std::int32_t type_bytes         = 0;
    std::int32_t type               = 0; // the decode index shared with the Linear dispatch
    std::int32_t rows_per_expert    = 0;
    // The per-expert stride is a POOL stride on the streamed tier, so it is carried for addressing
    // only; `row_bytes` is what one row of this bank's encoding occupies.
};

struct ExpertBanks {
    ExpertBank gate, up, down;
};

// The dense shared expert, when the layer has one: `gate_up` is the fused [2I, H] parent, `down` is
// [H, I], and `coefficients` is FP32 [T].
struct SharedExpert {
    const ExpertBank* gate_up = nullptr;
    const ExpertBank* down    = nullptr;
    const Tensor* coefficients = nullptr;
};

[[nodiscard]] std::size_t routed_swiglu_workspace_bytes(int top_k, int intermediate);

// A bank description from the device view the caller already holds. `k` is the reduction width of
// this bank (the hidden size for gate/up, the intermediate size for down). The row width comes from
// that width and the encoding - never from `code_stride`, which on the streamed tier is the expert
// POOL's slot stride (512 KiB here), not the encoded row.
[[nodiscard]] ExpertBank bank_from(const ExpertMatrixDeviceView& view, int k, QType qtype);

// One routed-SwiGLU layer for `tokens` columns:
//   y[t] = sum over t's participating routes r of coef[t*top_k+r] *
//          down[e] @ ( silu(gate[e] @ x[t]) * (up[e] @ x[t]) )
// plus, when `shared` is given, coef_shared[t] * down_s @ ( silu(gate_s @ x[t]) * up_s @ x[t] ).
//
// `ids`/`coefficients` are [K,T] *stored token-major*: assignment (r, t) lives at t*K + r, which is
// what topk_softmax_route writes and what every routed kernel reads.
//
// `mode` selects the epilogue: Write assigns, AddResidual adds the routed term to what `out` already
// holds. Each (token, row) is written by exactly one CTA, so both forms are race-free.
void launch_routed_swiglu(const Tensor& x, const ExpertBanks& banks, int hidden, int intermediate,
                          int experts, int top_k, const ExpertSubset& subset, MoeEpilogue mode,
                          const Tensor& ids, const Tensor& coefficients, const SharedExpert* shared,
                          Tensor& out, MemoryArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail::ggml
