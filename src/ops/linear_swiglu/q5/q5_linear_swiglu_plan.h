#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The folded gate/up tiles the Q5 SwiGLU route can launch. Column extents above the widest tile
// are served by token slicing, exactly as the Q4 catalog does.
enum class Q5LinearSwiGluScheduleId {
    MmaC8,
    MmaC48,
    MmaC128,
};

struct Q5LinearSwiGluProblem {
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q5LinearSwiGluPlan {
    Q5LinearSwiGluScheduleId schedule;
    std::size_t workspace_bytes;
};

const char* q5_linear_swiglu_schedule_name(Q5LinearSwiGluScheduleId schedule) noexcept;

bool q5_linear_swiglu_admits(const Q5LinearSwiGluProblem& problem) noexcept;
Q5LinearSwiGluPlan q5_linear_swiglu_resolve_plan(const Q5LinearSwiGluProblem& problem);

std::size_t q5_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols);

void q5_linear_swiglu_execute_plan(const Q5LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream);
void q5_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
