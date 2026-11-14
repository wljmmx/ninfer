#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include "core/device.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include <cuda_bf16.h>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

// Rows is the residual's channel count (H), so the group count follows it: H = 16 * groups, and the
// finish delta holds one row pair per group. Both used to be spelled 5120/320.
template <int Rows>
__device__ __forceinline__ void finish_value(int row, int col, int width, float current,
                                             float previous, const __nv_bfloat16* base,
                                             const __nv_bfloat16* delta, __nv_bfloat16* residual) {
    constexpr int kGroups = Rows / 16;
    const int index = col * Rows + row, di = col * 2 * kGroups + row / 16;
    float value = fmaf(__bfloat162float(base[2 * Rows + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value =
            fmaf(__bfloat162float(base[3 * Rows + row]) + __bfloat162float(delta[di + kGroups]),
                 previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

using Launch = Q8Launch;

template <int Rows, int InputRows, int TileColumns>
void tiled_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int Warps =
        InputRows == 4096 ? (TileColumns <= 40 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    using Geometry    = Q8LinearGeometry<Rows, InputRows>;
    using Schedule    = Q8A16SlicedKMmaSchedule<TileColumns, Warps, 1, Warps == 8 ? 2 : 3,
                                                Q8ScaleAccess::Shared, Activation>;
    const int columns = x.ne[1];
    LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), Rows};
    const dim3 grid(Rows / 16, (columns + TileColumns - 1) / TileColumns);
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, TileColumns, false>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream);
    CUDA_CHECK(cudaGetLastError());
}

// Live columns stay dynamic; only the eight-column MMA accumulator layout is specialized.
template <int Rows, int InputRows, std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{
        &tiled_projection<Rows, InputRows, 8 * (1 + static_cast<int>(I))>...};
}

// One set per (channel count, projection input width). The 35B pair is 5120 x {4096, 17408}; the
// Qwen3.8-Flash-Next draft is 2560 x {4096, 7680}.
constexpr auto tiled_5120_4096  = make_launchers<5120, 4096>(std::make_index_sequence<11>{});
constexpr auto tiled_5120_17408 = make_launchers<5120, 17408>(std::make_index_sequence<11>{});
constexpr auto tiled_2560_4096  = make_launchers<2560, 4096>(std::make_index_sequence<11>{});
constexpr auto tiled_2560_7680  = make_launchers<2560, 7680>(std::make_index_sequence<11>{});

const std::array<Launch, 11>* tiled_set(std::int32_t rows, std::int32_t input_rows) {
    if (rows == 5120 && input_rows == 4096) return &tiled_5120_4096;
    if (rows == 5120 && input_rows == 17408) return &tiled_5120_17408;
    if (rows == 2560 && input_rows == 4096) return &tiled_2560_4096;
    if (rows == 2560 && input_rows == 7680) return &tiled_2560_7680;
    throw std::invalid_argument("linear dynamic grouped conv add: no tiled instance for rows=" +
                                std::to_string(rows) + " input_rows=" +
                                std::to_string(input_rows));
}

template <int Rows>
__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= Rows) return;
    const int index = col * Rows + row;
    finish_value<Rows>(row, col, width, __bfloat162float(projected[index]),
                       col % width ? __bfloat162float(projected[index - Rows]) : 0.0f, base, delta,
                       residual);
}

void finish_dispatch(std::int32_t rows, const Tensor& projected, const Tensor& base,
                     const Tensor& delta, Tensor& residual, int width, std::int32_t tokens,
                     cudaStream_t stream) {
    const dim3 grid((rows + 255) / 256, tokens);
    const auto* projected_data = static_cast<const __nv_bfloat16*>(projected.data);
    const auto* base_data      = static_cast<const __nv_bfloat16*>(base.data);
    const auto* delta_data     = static_cast<const __nv_bfloat16*>(delta.data);
    auto* residual_data        = static_cast<__nv_bfloat16*>(residual.data);
    if (rows == 5120) {
        finish_kernel<5120><<<grid, 256, 0, stream>>>(projected_data, base_data, delta_data,
                                                      residual_data, width);
    } else if (rows == 2560) {
        finish_kernel<2560><<<grid, 256, 0, stream>>>(projected_data, base_data, delta_data,
                                                      residual_data, width);
    } else {
        throw std::invalid_argument(
            "linear dynamic grouped conv add: no finish instance for the channel count");
    }
    CUDA_CHECK(cudaGetLastError());
}

void materialized(Q8DynamicConvAddSchedule schedule, const Tensor& x, const Weight& weight,
                  const Tensor& base, const Tensor& delta, Tensor& residual, Tensor& projected,
                  cudaStream_t stream) {
    const std::int32_t rows = residual.ne[0];
    const int tokens        = x.ne[1] * x.ne[2];
    const Tensor flat       = x.view({x.ne[0], tokens});
    Tensor result           = projected.view({rows, tokens});
    switch (schedule) {
    case Q8DynamicConvAddSchedule::TiledMma: {
        const auto* set = tiled_set(rows, x.ne[0]);
        (*set)[(tokens - 1) / 8](flat, weight, result, stream);
        break;
    }
    case Q8DynamicConvAddSchedule::MmaK128:
        launch_q8_a16_mma_r64x32_t64_k128_a1(flat, weight, result, stream);
        break;
    }
    finish_dispatch(rows, projected, base, delta, residual, x.ne[1], tokens, stream);
}
} // namespace

void q8_dynamic_grouped_conv_add_materialized_launch(Q8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base, const Tensor& delta,
                                                     Tensor& residual, Tensor& projected,
                                                     cudaStream_t stream) {
    materialized(schedule, x, weight, base, delta, residual, projected, stream);
}
} // namespace ninfer::ops::detail
