// port (sm_89): the split layer's host hand-off. Two jobs, both about doing one transfer where the
// engine used to do four, and one rounding where the merge used to run on the CPU.
//
// The read-back of a split layer is x, residual, ids and coefficients. They are four independent
// device buffers, so the engine issued four copies and paid the submission each time: measured
// 0.10 ms per layer against 0.02 ms for the single 8 KiB copy the merge needs. Packing them into
// one contiguous block turns that into one pack kernel plus one copy of the whole block.
//
// The merge writes `residual = RNE_bf16(residual + host_term + gpu_term)` with the accumulation
// order and the rounding rule the host loop used, so moving it to the device yields the same value
// bit for bit - which is what lets it replace the D2H join, the host merge and the write-back.
#include "ninfer/ops/moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {
namespace {

inline constexpr std::size_t align16(std::size_t value) { return (value + 15U) & ~std::size_t{15}; }

// One layout function, used by the size helper and by the kernel, so the two cannot disagree.
struct PackLayout {
    std::size_t residual;
    std::size_t ids;
    std::size_t coefficients;
    std::size_t total;
};

PackLayout pack_layout(std::size_t x_bytes, std::size_t residual_bytes, std::size_t ids_bytes,
                       std::size_t coefficient_bytes) {
    PackLayout layout{};
    layout.residual     = align16(x_bytes);
    layout.ids          = align16(layout.residual + residual_bytes);
    layout.coefficients = align16(layout.ids + ids_bytes);
    layout.total        = align16(layout.coefficients + coefficient_bytes);
    return layout;
}

__global__ void pack_kernel(const std::uint16_t* x, const std::uint16_t* residual,
                            const std::int32_t* ids, const float* coefficients, int x_count,
                            int ids_count, std::uint16_t* out_x, std::uint16_t* out_residual,
                            std::int32_t* out_ids, float* out_coefficients) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (i < x_count) {
        out_x[i]        = x[i];
        out_residual[i] = residual[i];
    }
    if (i < ids_count) {
        out_ids[i]          = ids[i];
        out_coefficients[i] = coefficients[i];
    }
}

__global__ void merge_kernel(const float* host_term, const float* gpu_term,
                             std::uint16_t* residual, int count) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (i >= count) { return; }
    const float base = __uint_as_float(static_cast<std::uint32_t>(residual[i]) << 16);
    const float sum  = (base + host_term[i]) + gpu_term[i];
    // Round to nearest even, the rule __float2bfloat16_rn applies: truncating biases every layer
    // downward and compounds over the stack.
    std::uint32_t bits      = __float_as_uint(sum);
    const std::uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7fffu + lsb;
    residual[i] = static_cast<std::uint16_t>(bits >> 16);
}

} // namespace

std::size_t moe_pack_host_fetch_bytes(std::size_t x_bytes, std::size_t residual_bytes,
                                      std::size_t ids_bytes, std::size_t coefficient_bytes) {
    return pack_layout(x_bytes, residual_bytes, ids_bytes, coefficient_bytes).total;
}

void moe_pack_host_fetch(const void* x, const void* residual, const void* ids,
                         const void* coefficients, int x_count, int ids_count, void* packed,
                         cudaStream_t stream) {
    auto* bytes       = static_cast<std::byte*>(packed);
    const auto layout = pack_layout(static_cast<std::size_t>(x_count) * 2U,
                                    static_cast<std::size_t>(x_count) * 2U,
                                    static_cast<std::size_t>(ids_count) * 4U,
                                    static_cast<std::size_t>(ids_count) * 4U);
    auto* out_x   = reinterpret_cast<std::uint16_t*>(bytes);
    auto* out_res = reinterpret_cast<std::uint16_t*>(bytes + layout.residual);
    auto* out_ids = reinterpret_cast<std::int32_t*>(bytes + layout.ids);
    auto* out_coe = reinterpret_cast<float*>(bytes + layout.coefficients);
    const int count = x_count > ids_count ? x_count : ids_count;
    const int block = 256;
    const int grid  = (count + block - 1) / block;
    pack_kernel<<<grid, block, 0, stream>>>(static_cast<const std::uint16_t*>(x),
                                            static_cast<const std::uint16_t*>(residual),
                                            static_cast<const std::int32_t*>(ids),
                                            static_cast<const float*>(coefficients), x_count,
                                            ids_count, out_x, out_res, out_ids, out_coe);
}

void moe_merge_host_term(const void* host_term, const void* gpu_term, void* residual, int count,
                         cudaStream_t stream) {
    const int block = 256;
    const int grid  = (count + block - 1) / block;
    merge_kernel<<<grid, block, 0, stream>>>(static_cast<const float*>(host_term),
                                             static_cast<const float*>(gpu_term),
                                             static_cast<std::uint16_t*>(residual), count);
}

} // namespace ninfer::ops
