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

// port (sm_89): a split bank keeps the gate object and the up object apart, so an expert's gate
// rows and up rows are one image each in *different* allocations. `gate_rows` is the number of
// rows an image holds (Shape::intermediate); the row below it belongs to the up object, never to
// the gate object's successor. Every access type below states `kSplit` so the kernels can build
// the two-plane form only where it exists and never pay for the test on the fused layout.
struct SplitExpertPlanes {
    ExpertPlanes gate, up;
    int gate_rows;
};

__device__ __forceinline__ ExpertPlanes gate_up_row(const ExpertPlanes& p, int row) {
    return {p.codes, p.high, p.scales, p.row + row};
}

// The fused layout: rows [0,I) are gate and [I,2I) are up inside the same image.
__device__ __forceinline__ ExpertPlanes gate_up_row(const SplitExpertPlanes& p, int row) {
    return row < p.gate_rows
               ? ExpertPlanes{p.gate.codes, p.gate.high, p.gate.scales, p.gate.row + row}
               : ExpertPlanes{p.up.codes, p.up.high, p.up.scales,
                              p.up.row + row - p.gate_rows};
}

struct PackedExpertMatrix {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    static constexpr bool kSplit = false;

    template <int Rows>
    __device__ __forceinline__ ExpertPlanes resolve(int expert) const {
        return {codes, high, scales, expert * Rows};
    }
};

struct StridedExpertMatrix {
    PackedExpertMatrix planes;
    std::uint64_t code_stride, high_stride, scale_stride;
    static constexpr bool kSplit = false;

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
    static constexpr bool kSplit = false;

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

template <class Half>
struct SplitExpertMatrix {
    Half gate, up;
    static constexpr bool kSplit = true;

    template <int Rows>
    __device__ __forceinline__ SplitExpertPlanes resolve(int expert) const {
        return {gate.template resolve<Rows>(expert), up.template resolve<Rows>(expert), Rows / 2};
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
    // Split gate/up: two objects instead of one fused parent. The operand check has already
    // refused a bank that names both forms, so this test only decides which row map the kernels
    // are compiled with.
    if (experts.gate.codes != nullptr) {
        if (experts.gate.offsets) {
            function(SplitExpertMatrix<IndexedExpertMatrix>{
                         IndexedExpertMatrix{matrix(experts.gate), experts.gate.offsets,
                                            host_matrix(experts.gate), experts.gate.host_offsets,
                                            experts.gate.host_flags},
                         IndexedExpertMatrix{matrix(experts.up), experts.up.offsets,
                                            host_matrix(experts.up), experts.up.host_offsets,
                                            experts.up.host_flags}},
                     IndexedExpertMatrix{matrix(down), down.offsets, host_matrix(down),
                                         down.host_offsets, down.host_flags});
            return;
        }
        const auto strided = [](const ExpertMatrixDeviceView& view) {
            StridedExpertMatrix result{matrix(view), view.code_stride, view.high_stride,
                                       view.scale_stride};
            return result;
        };
        function(SplitExpertMatrix<StridedExpertMatrix>{strided(experts.gate), strided(experts.up)},
                 StridedExpertMatrix{matrix(down), down.code_stride, down.high_stride,
                                    down.scale_stride});
        return;
    }
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
