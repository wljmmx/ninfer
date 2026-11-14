#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_output.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kFirstSmallT = 1;
constexpr std::int32_t kLastSmallT  = 40;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// The draft head is gate_up = 2 x out at the draft width: 34816 = 2 x 17408 at k 5120 for the 35B
// draft, 15360 = 2 x 7680 at k 2560 for the Qwen3.8-Flash-Next draft. Both the paired-row policy
// and the static-K specialization follow the two numbers, so they are template parameters.
template <int GateUpRows, int InputRows, int Capacity>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry      = Q8LinearGeometry<GateUpRows, InputRows>;
    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    using Schedule =
        Q8A16SlicedKMmaSchedule<Capacity, Capacity == 24 ? 8 : 4, 1, 2, Q8ScaleAccess::Shared>;
    using RowPolicy = Q8SwiGluPairedRows<kIntermediate>;
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((kIntermediate % RowPolicy::kOutputRowsPerCta) == 0);

    const LinearBf16Output ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{};

    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, Capacity, false>, RowPolicy>(
        q8_linear_operands(x, weight), ignored_output, epilogue, stream);
    CUDA_CHECK(cudaGetLastError());
}

template <int GateUpRows, int InputRows, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<GateUpRows, InputRows, 8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto kWideLaunchers  = make_launchers<34816, 5120>(std::make_index_sequence<kLastSmallT / 8>{});
constexpr auto kDraftLaunchers = make_launchers<15360, 2560>(std::make_index_sequence<kLastSmallT / 8>{});

} // namespace

void q8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             cudaStream_t stream) {
    if (x.ne[1] < kFirstSmallT || x.ne[1] > kLastSmallT) {
        throw std::invalid_argument("Q8 DFlash2 LinearSwiGLU small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    if (weight.n == 34816 && x.ne[0] == 5120) {
        kWideLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == 15360 && x.ne[0] == 2560) {
        kDraftLaunchers[index](x, weight, out, stream);
        return;
    }
    throw std::invalid_argument(
        "Q8 DFlash2 LinearSwiGLU small-T: no instance for gate_up=" + std::to_string(weight.n) +
        " k=" + std::to_string(x.ne[0]));
}

} // namespace ninfer::ops::detail
