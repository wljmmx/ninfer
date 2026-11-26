// The GGML-encoded Linear: warp-per-row over the stored blocks, no dequantised copy of the weight.
//
// See ggml_gemv.cuh for the per-group decode and the invariants. This file is the launch: one warp
// per output row, `kWarpsPerBlock` rows per CTA, every column of the activation walked inside the same
// warp so the weight row is read once per launch rather than once per column.

#include "ops/linear/ggml/ggml_linear.h"

#include "core/device.h"
#include "core/ggml_types.h"
#include "ops/linear/ggml/ggml_gemv.cuh"

#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail::ggml {
namespace {

constexpr int kWarpsPerBlock = 4;

// `type` is the dispatch index of the encoding (the same numbering the decode test uses).
__device__ __forceinline__ float row_of(int type, const std::uint8_t* row, const __nv_bfloat16* x,
                                        int k, int groups_per_block, int type_bytes) {
    switch (type) {
    case 0: return gemv_row<0>(row, x, k, groups_per_block, type_bytes);
    case 1: return gemv_row<1>(row, x, k, groups_per_block, type_bytes);
    case 2: return gemv_row<2>(row, x, k, groups_per_block, type_bytes);
    case 3: return gemv_row<3>(row, x, k, groups_per_block, type_bytes);
    case 4: return gemv_row<4>(row, x, k, groups_per_block, type_bytes);
    case 5: return gemv_row<5>(row, x, k, groups_per_block, type_bytes);
    case 6: return gemv_row<6>(row, x, k, groups_per_block, type_bytes);
    case 7: return gemv_row<7>(row, x, k, groups_per_block, type_bytes);
    case 8: return gemv_row<8>(row, x, k, groups_per_block, type_bytes);
    case 9: return gemv_row<9>(row, x, k, groups_per_block, type_bytes);
    case 10: return gemv_row<10>(row, x, k, groups_per_block, type_bytes);
    }
    return 0.0f;
}

template <bool Accumulate>
__global__ void ggml_linear_kernel(const std::uint8_t* weights, const __nv_bfloat16* x,
                                   __nv_bfloat16* y, int k, int n, int columns, int row_bytes,
                                   int groups_per_block, int type_bytes, int type) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp >= n) { return; }
    const std::uint8_t* row = weights + std::size_t(warp) * std::size_t(row_bytes);
    const int lane          = threadIdx.x & 31;
    for (int column = 0; column < columns; ++column) {
        const float acc =
            row_of(type, row, x + std::size_t(column) * k, k, groups_per_block, type_bytes);
        if (lane == 0) {
            __nv_bfloat16* slot = y + std::size_t(warp) + std::size_t(column) * n;
            if constexpr (Accumulate) {
                *slot = __float2bfloat16(__bfloat162float(*slot) + acc);
            } else {
                *slot = __float2bfloat16(acc);
            }
        }
    }
}

// The dispatch index of each encoding. The order is this port's own; the tests name it too, so a
// mismatch shows up as a wrong number rather than as a silent success.
int type_index(QType format) {
    switch (format) {
    case QType::GGML_Q4_K: return 0;
    case QType::GGML_Q5_K: return 1;
    case QType::GGML_Q6_K: return 2;
    case QType::GGML_Q2_0: return 3;
    case QType::GGML_IQ4_NL: return 4;
    case QType::GGML_IQ4_XS: return 5;
    case QType::GGML_IQ2_XXS: return 6;
    case QType::GGML_IQ2_XS: return 7;
    case QType::GGML_IQ2_S: return 8;
    case QType::GGML_IQ3_XXS: return 9;
    case QType::GGML_IQ3_S: return 10;
    default: throw std::invalid_argument("ggml linear: not a GGML block encoding");
    }
}

struct Launch {
    const std::uint8_t* weight;
    const __nv_bfloat16* x;
    __nv_bfloat16* y;
    int k, n, columns, row_bytes, groups_per_block, type_bytes, type, grid;
};

// The geometry checks and launch shape both entries share; only the write differs.
Launch prepare(const Tensor& x, const Weight& w, Tensor& out) {
    const auto geometry = core::ggml_block_geometry(w.qtype);
    if (w.k <= 0 || w.n <= 0 || x.ne[1] <= 0) {
        throw std::invalid_argument("ggml linear: empty geometry");
    }
    if (std::int64_t(w.k) % std::int64_t(geometry.block) != 0) {
        throw std::invalid_argument("ggml linear: K is not a whole number of encoded blocks");
    }
    if (w.qdata == nullptr) { throw std::invalid_argument("ggml linear: weight has no payload"); }
    // A group-wide lane assignment needs whole 32-value groups in every block.
    if (geometry.block % 32 != 0) {
        throw std::invalid_argument("ggml linear: block is not a multiple of 32 values");
    }
    Launch launch{};
    launch.weight           = static_cast<const std::uint8_t*>(w.qdata);
    launch.x                = static_cast<const __nv_bfloat16*>(x.data);
    launch.y                = static_cast<__nv_bfloat16*>(out.data);
    launch.k                = w.k;
    launch.n                = w.n;
    launch.columns          = x.ne[1];
    launch.groups_per_block = int(geometry.block) / 32;
    launch.row_bytes        = w.k / int(geometry.block) * int(geometry.type_bytes);
    launch.type_bytes       = int(geometry.type_bytes);
    launch.type             = type_index(w.qtype);
    launch.grid             = (w.n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    return launch;
}

} // namespace

void launch_linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const auto launch = prepare(x, w, out);
    ggml_linear_kernel<false><<<launch.grid, kWarpsPerBlock * 32, 0, stream>>>(
        launch.weight, launch.x, launch.y, launch.k, launch.n, launch.columns, launch.row_bytes,
        launch.groups_per_block, launch.type_bytes, launch.type);
    CUDA_CHECK(cudaGetLastError());
}

void launch_linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const auto launch = prepare(x, w, residual_out);
    ggml_linear_kernel<true><<<launch.grid, kWarpsPerBlock * 32, 0, stream>>>(
        launch.weight, launch.x, launch.y, launch.k, launch.n, launch.columns, launch.row_bytes,
        launch.groups_per_block, launch.type_bytes, launch.type);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::ggml
