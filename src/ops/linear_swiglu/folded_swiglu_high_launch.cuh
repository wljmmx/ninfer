#pragma once

// Launch plumbing shared by the Q5 and Q6 folded gate/up SwiGLU routes.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/token_slices.h"
#include "ops/linear_swiglu/folded_swiglu_high_mma.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

template <class Atom, int HB, class Cfg, bool Full>
void launch_folded_high(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int PM = Cfg::BM / 2;
    const int t      = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(t, Cfg::BN)),
                    static_cast<unsigned>(div_up(out.ne[0], PM)));
    const auto* high = static_cast<const std::uint8_t*>(weight.qhigh);
    if constexpr (Full) {
        folded_swiglu_high_mma_kernel<Cfg, Atom, HB, true><<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata), high,
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], t, weight.padded_shape[1]);
    } else {
        folded_swiglu_high_mma_kernel<Cfg, Atom, HB, false><<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata), high,
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], t, weight.padded_shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Atom, int HB, class Cfg>
void launch_folded_high_route(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    const bool full = (x.ne[1] % Cfg::BN) == 0;
    for_each_token_slice(x.ne[1], Cfg::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_folded_high<Atom, HB, Cfg, true>(x_slice, weight, out_slice, stream);
        } else {
            launch_folded_high<Atom, HB, Cfg, false>(x_slice, weight, out_slice, stream);
        }
    });
}

} // namespace ninfer::ops::detail
