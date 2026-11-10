#pragma once

#include "ninfer/ops/moe.h"

#include <cstdint>

namespace ninfer::ops::detail::moe_routed_cpu {

// port (sm_89): the CPU half of a gate/up/down layer.
//
// The fused CPU kernels are compiled for one shape family (h2048_i512_e256_k8) and read a single
// [2I,H] gate_up parent. A layer whose experts are three separate matrices cannot wear that row map:
// row I of the gate object is the next expert's first gate row, so the fused kernel would multiply
// by another expert's weights. This entry point takes the three matrices as they are, at whatever
// hidden/intermediate the artifact carries, and computes the same SwiGLU:
//
//   a_t[i]      = SiLU(gate_i . x_t) * (up_i . x_t)              i in [0, I)
//   out_t[h]   += coef(t,e) * (down_h . a_t)                     h in [0, H)
//
// It is the fallback leg: a route the Device pool and the pinned tier both cannot hold is computed
// here, in FP32, and folded into the same residual the Device half writes.
[[nodiscard]] std::size_t split_capacity(std::int32_t intermediate, std::int32_t tokens);

void split_launch(const Tensor& x, const CpuSwiGLUExperts& experts, const ExpertRoutes& routes,
                  ExpertSubset subset, Tensor& destination, MemoryArena& scratch,
                  CpuExecutionView execution, std::int32_t hidden, std::int32_t intermediate,
                  std::int32_t experts_count, std::int32_t top_k);

} // namespace ninfer::ops::detail::moe_routed_cpu
