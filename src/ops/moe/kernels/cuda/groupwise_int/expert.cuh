#pragma once

#include "ops/moe/kernels/cuda/groupwise_int/launch.h"
#include "ops/moe/kernels/cuda/output.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/integer.cuh"
#include "core/pdl.cuh"

namespace ninfer::ops::detail::moe_cuda::groupwise_int {

struct ExpertPlanes {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    int row;
};

struct PackedExpertMatrix {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;

    template <int Rows>
    __device__ __forceinline__ ExpertPlanes resolve(int expert) const {
        return {codes, high, scales, expert * Rows};
    }
};

struct StridedExpertMatrix {
    PackedExpertMatrix planes;
    std::uint64_t code_stride, high_stride, scale_stride;

    template <int Rows>
    __device__ __forceinline__ ExpertPlanes resolve(int expert) const {
        return {planes.codes + expert * code_stride,
                planes.high ? planes.high + expert * high_stride : nullptr,
                planes.scales + expert * scale_stride, 0};
    }
};

struct IndexedExpertMatrix {
    PackedExpertMatrix planes;
    const std::uint64_t* offsets;
    // port (sm_89): Q20's second leg. Null on every path that was not asked for it, which is why
    // the branch below costs one null test per expert rather than a separate kernel variant.
    PackedExpertMatrix host;
    const std::uint64_t* host_offsets;
    const std::int32_t* host_flags;

    template <int Rows>
    __device__ __forceinline__ ExpertPlanes resolve(int expert) const {
        if (host_flags != nullptr && host_flags[expert] != 0) {
            const auto offset = host_offsets[expert];
            return {host.codes + offset, host.high ? host.high + offset : nullptr,
                    host.scales + offset, 0};
        }
        const auto offset = offsets[expert];
        return {planes.codes + offset, planes.high ? planes.high + offset : nullptr,
                planes.scales + offset, 0};
    }
};

inline PackedExpertMatrix matrix(const ExpertMatrixDeviceView& view) {
    return {reinterpret_cast<const std::uint8_t*>(view.codes),
            reinterpret_cast<const std::uint8_t*>(view.high),
            reinterpret_cast<const std::uint8_t*>(view.scales)};
}

inline PackedExpertMatrix matrix(const Weight& weight) {
    return {static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales)};
}

// port (sm_89): the second leg as the kernel wants it - three plane bases, no strides, because a
// pinned slot is one contiguous expert image exactly like a pool slot.
inline PackedExpertMatrix host_matrix(const ExpertMatrixDeviceView& view) {
    return {reinterpret_cast<const std::uint8_t*>(view.host_codes),
            reinterpret_cast<const std::uint8_t*>(view.host_high),
            reinterpret_cast<const std::uint8_t*>(view.host_scales)};
}

template <class Function>
void with_expert_addresses(const CudaSwiGLUExperts& experts, const RoutedSwiGLUProfile& profile,
                           Function&& function) {
    const auto& gu   = experts.bank.gate_up;
    const auto& down = experts.bank.down;
    if (gu.offsets) {
        // Q20: the second leg travels with the indexed form only. An affine or packed bank is a
        // layout whose experts are addressed by stride, and the host tier cannot serve that shape.
        function(IndexedExpertMatrix{matrix(gu), gu.offsets, host_matrix(gu), gu.host_offsets,
                                     gu.host_flags},
                 IndexedExpertMatrix{matrix(down), down.offsets, host_matrix(down),
                                     down.host_offsets, down.host_flags});
        return;
    }
    const auto packed = [](const ExpertMatrixDeviceView& view, QType type, int n, int k) {
        const auto elements = std::uint64_t(n) * k;
        const bool q8       = type == QType::Q8_G32_FP16;
        const auto high     = type == QType::Q5_G64_FP16   ? elements / 8
                              : type == QType::Q6_G64_FP16 ? elements / 4
                                                           : 0;
        return view.code_stride == (q8 ? elements : elements / 2) && view.high_stride == high &&
               view.scale_stride == elements / (q8 ? 16 : 32);
    };
    const auto& p = profile.ffn;
    if (packed(gu, p.gate_up_type, 2 * p.intermediate, p.hidden) &&
        packed(down, p.down_type, p.hidden, p.intermediate)) {
        function(matrix(gu), matrix(down));
        return;
    }
    function(
        StridedExpertMatrix{matrix(gu), gu.code_stride, gu.high_stride, gu.scale_stride},
        StridedExpertMatrix{matrix(down), down.code_stride, down.high_stride, down.scale_stride});
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
