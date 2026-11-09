// Implements: include/ninfer/ops/scaled_add.h
// Finite dispatch: a 16-byte pack route when D and the alignment allow it (a pack then never
// crosses a token, so its coefficient is uniform), the scalar route otherwise.
#include "ops/launcher/scaled_add.h"

#include "ops/common/math.h"
#include "ops/kernel/scaled_add.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void scaled_add_launch(const Tensor& source, const Tensor& coefficient, Tensor& destination,
                       cudaStream_t stream) {
    const std::int64_t d = destination.ne[0];
    const std::int64_t n = destination.numel();
    constexpr int kBlock = 256;
    constexpr int kMaxGrid = 4096;
    const auto source_addr = reinterpret_cast<std::uintptr_t>(source.data);
    const auto dest_addr   = reinterpret_cast<std::uintptr_t>(destination.data);
    if ((d % 8) == 0 && ((source_addr | dest_addr) & (alignof(Bf16x8Pack) - 1)) == 0) {
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            kMaxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(kBlock)))));
        scaled_add_bf16x8_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(source.data), static_cast<const float*>(coefficient.data),
            static_cast<Bf16x8Pack*>(destination.data), d, packs);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int grid = static_cast<int>(
        std::min<std::int64_t>(kMaxGrid, std::max<std::int64_t>(1, div_up(n, std::int64_t{kBlock}))));
    scaled_add_scalar_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source.data), static_cast<const float*>(coefficient.data),
        static_cast<__nv_bfloat16*>(destination.data), d, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
