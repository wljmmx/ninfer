#include "core/weight.h"
#include "ops/linear_swiglu/q6/q6_linear_swiglu_plan.h"

#include "ops/linear_swiglu/q6/q6_linear_swiglu_kernels.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

// The 27B MLP gate/up parent: 2 x 17408 rows over K = 5120, i.e. 544 whole 64-row tiles and 80
// whole 64-wide groups. The same exact problem the Q4 route admits; only the codec differs.
constexpr Q6LinearSwiGluProblem kShape{34816, 17408, 5120, 5120, 1};

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q6LinearSwiGluScheduleId schedule;
};

// One column per decoded token runs the narrowest folded tile; the 48-wide tile covers short
// prompts; anything wider keeps the 128-wide tile and is split by the launcher.
constexpr RouteSpec kRoutes[] = {
    {1, 8, Q6LinearSwiGluScheduleId::MmaC8},
    {9, 48, Q6LinearSwiGluScheduleId::MmaC48},
    {49, kAnyCols, Q6LinearSwiGluScheduleId::MmaC128},
};

bool supported_shape(const Q6LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == kShape.gate_up_rows &&
           problem.output_rows == kShape.output_rows && problem.k == kShape.k &&
           problem.padded_k == kShape.padded_k;
}

} // namespace

const char* q6_linear_swiglu_schedule_name(Q6LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case Q6LinearSwiGluScheduleId::MmaC8:
        return "linear_swiglu.q5.mma.folded.c8";
    case Q6LinearSwiGluScheduleId::MmaC48:
        return "linear_swiglu.q5.mma.folded.c48";
    case Q6LinearSwiGluScheduleId::MmaC128:
        return "linear_swiglu.q5.mma.folded.c128";
    }
    return "linear_swiglu.q5.unknown";
}

bool q6_linear_swiglu_admits(const Q6LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q6LinearSwiGluPlan q6_linear_swiglu_resolve_plan(const Q6LinearSwiGluProblem& problem) {
    if (!q6_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "q5 linear_swiglu: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) {
            return {route.schedule, 0};
        }
    }
    throw std::logic_error("q5 linear_swiglu: admitted problem has no covering route");
}

std::size_t q6_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q5 linear_swiglu: invalid column interval");
    }
    // Every route is fused, so the Op needs no scratch at any column count.
    (void)q6_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, min_cols});
    (void)q6_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols});
    return 0;
}

void q6_linear_swiglu_execute_plan(const Q6LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    const Q6LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q6LinearSwiGluPlan resolved = q6_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("q5 linear_swiglu: plan does not match the exact problem");
    }
    switch (plan.schedule) {
    case Q6LinearSwiGluScheduleId::MmaC8:
        q6_linear_swiglu_mma_c8_launch(x, w, out, stream);
        return;
    case Q6LinearSwiGluScheduleId::MmaC48:
        q6_linear_swiglu_mma_c48_launch(x, w, out, stream);
        return;
    case Q6LinearSwiGluScheduleId::MmaC128:
        q6_linear_swiglu_mma_c128_launch(x, w, out, stream);
        return;
    }
    throw std::logic_error("q5 linear_swiglu: unknown schedule");
}

void q6_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const Q6LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    q6_linear_swiglu_execute_plan(q6_linear_swiglu_resolve_plan(problem), x, w, out, ws, stream);
}

} // namespace ninfer::ops::detail
