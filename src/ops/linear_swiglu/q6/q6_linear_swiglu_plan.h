#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The folded gate/up tiles the Q6 SwiGLU route can launch. Column extents above the widest tile
// are served by token slicing, exactly as the Q4 catalog does.
enum class Q6LinearSwiGluScheduleId {
    MmaC8,
    MmaC48,
    MmaC128,
};

struct Q6LinearSwiGluProblem {
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q6LinearSwiGluPlan {
    Q6LinearSwiGluScheduleId schedule;
    std::size_t workspace_bytes;
};

const char* q6_linear_swiglu_schedule_name(Q6LinearSwiGluScheduleId schedule) noexcept;

bool q6_linear_swiglu_admits(const Q6LinearSwiGluProblem& problem) noexcept;
Q6LinearSwiGluPlan q6_linear_swiglu_resolve_plan(const Q6LinearSwiGluProblem& problem);

std::size_t q6_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols);

void q6_linear_swiglu_execute_plan(const Q6LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream);
void q6_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
