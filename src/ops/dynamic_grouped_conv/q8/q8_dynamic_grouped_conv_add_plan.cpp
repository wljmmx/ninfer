#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct Plan {
    Q8DynamicConvAddSchedule schedule;
    std::size_t workspace_bytes;
};

Plan resolve_plan(int rows, int input_rows, int width, int batch) {
    // port (sm_89): C is the layer's channel count, and the schedules tile it by 128 - the old
    // 4096/17408 whitelist was the 35B pair. Qwen3.8-Flash-Next's draft head is C 15360 (and its
    // smaller sibling 7680), which is also a multiple of the tile. The channel count the projection
    // writes back into (rows) is the residual's, and it is templated like C.
    if (input_rows % 128 != 0)
        throw std::invalid_argument(
            "linear dynamic grouped conv add: C must be a multiple of 128");
    if (width < 2 || width > 16 || batch < 1 || batch > 8)
        throw std::invalid_argument("linear dynamic grouped conv add: invalid W/B profile");
    const int columns = width * batch;
    using S           = Q8DynamicConvAddSchedule;
    const S schedule  = columns <= 88 ? S::TiledMma : S::MmaK128;
    return {schedule, static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t)};
}
} // namespace

std::size_t q8_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    int rows, int input_rows, int min_width, int max_width, int min_batch, int max_batch) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument(
            "linear dynamic grouped conv add workspace: invalid W/B interval");
    // Every route shares one projected BF16 matrix; its capacity is monotonic in W and B.
    return resolve_plan(rows, input_rows, max_width, max_batch).workspace_bytes;
}

const char* q8_linear_dynamic_grouped_conv_add_route_name(int rows, int input_rows, int width,
                                                          int batch) {
    switch (resolve_plan(rows, input_rows, width, batch).schedule) {
    case Q8DynamicConvAddSchedule::TiledMma:
        return "dynamic_grouped_conv_add.q8.tiled_mma.materialized_bf16";
    case Q8DynamicConvAddSchedule::MmaK128:
        return "dynamic_grouped_conv_add.q8.r64_c64_k128.materialized_bf16";
    }
    throw std::logic_error("linear dynamic grouped conv add: invalid production schedule");
}

void q8_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& weight,
                                                 const Tensor& base, const Tensor& delta,
                                                 Tensor& residual, WorkspaceArena& workspace,
                                                 cudaStream_t stream) {
    const Plan plan = resolve_plan(residual.ne[0], x.ne[0], x.ne[1], x.ne[2]);
    auto scope      = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(plan.workspace_bytes);
    Tensor projected(storage.data, DType::BF16, {residual.ne[0], x.ne[1], x.ne[2]});
    q8_dynamic_grouped_conv_add_materialized_launch(plan.schedule, x, weight, base, delta, residual,
                                                    projected, stream);
}
} // namespace ninfer::ops::detail
