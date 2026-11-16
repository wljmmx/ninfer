#pragma once

#include "core/memory.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// Existing dense Op ABI uses Weight axes directly; callers establish BF16/FP32 and layout.
inline Tensor as_dense(const Weight& weight) {
    const auto dtype = weight.qtype == QType::FP32 ? DType::FP32 : DType::BF16;
    auto* data       = const_cast<void*>(weight.qdata);
    switch (weight.ndim) {
    case 1:
        return Tensor(data, dtype, {weight.shape[0]});
    case 2:
        return Tensor(data, dtype, {weight.shape[0], weight.shape[1]});
    case 3:
        return Tensor(data, dtype, {weight.shape[0], weight.shape[1], weight.shape[2]});
    default:
        return Tensor(data, dtype,
                      {weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]});
    }
}

// A quantized payload may end in one FP32 divisor word per declared domain. The default is the
// single matrix divisor every NVFP4 parent has always stored; a descriptor that keeps one divisor
// per expert declares that array here, so the count, the plane offset and the payload follow.
struct WeightDivisors {
    std::uint64_t count = 1;
};

// Geometry describes one complete encoded parent. Regions address its logical elements;
// padding and scale words do not belong to that element domain.
struct WeightGeometry {
    QType format       = QType::BF16;
    QuantLayout layout = QuantLayout::Contiguous;
    std::vector<std::uint64_t> shape;
    std::uint64_t elements            = 0;
    std::uint64_t bytes               = 0;
    std::uint64_t alignment           = 256;
    std::uint64_t padded_columns      = 0;
    std::uint64_t group_size          = 0;
    std::uint64_t code_bytes_per_row  = 0;
    std::uint64_t high_bytes_per_row  = 0;
    std::uint64_t scale_bytes_per_row = 0;
    std::uint64_t code_bytes          = 0;
    std::uint64_t high_offset         = 0;
    std::uint64_t high_bytes          = 0;
    std::uint64_t scale_offset        = 0;
    std::uint64_t scale_bytes         = 0;
    std::uint64_t divisor_offset      = 0;
    // port (sm_89): bench-compatible geometry the expert bank needs. Appended, and named after
    // the existing scale_bytes_per_row so one plane keeps one name.
    std::uint64_t scale_block_rows = 1;
    std::uint64_t divisor_count    = 0;
    std::uint64_t row_bytes        = 0;
    std::uint64_t rows_per_page    = 0;
    std::uint64_t page_count       = 0;
    // port (sm_89) V10-13: the expert face of a quantized parent. `matrix_rows` is the row count of
    // one matrix the shape names and `scale_bytes_per_matrix` the scale bytes that matrix owns, so
    // for a batched [E, N, K] parent expert e's scale block starts at scale_offset +
    // e * scale_bytes_per_matrix. `divisor_rows` is the number of logical rows one divisor word
    // covers, so a declared array names the expert rows even when the shape is one stacked matrix;
    // it is 0 when the payload has no divisor plane.
    std::uint64_t matrix_rows            = 0;
    std::uint64_t scale_bytes_per_matrix = 0;
    std::uint64_t divisor_rows           = 0;
};

[[nodiscard]] WeightGeometry weight_geometry(QType format, QuantLayout layout,
                                             std::span<const std::uint64_t> shape,
                                             WeightDivisors divisors = {});
[[nodiscard]] std::uint64_t weight_element_count(std::span<const std::uint64_t> shape);

struct WeightParent {
    WeightGeometry geometry;
    const std::byte* data      = nullptr;
    float weight_scale_divisor = 0.0F;
    // port (sm_89): bench-compatible residency and per-expert divisor metadata.
    Device device = Device{};
    std::span<const float> weight_divisors;
    // port (sm_89): artifact object this parent describes, so a streamed tier can map it back to
    // its byte ranges without re-deriving the load plan.
    std::uint64_t source_object = 0;
};

struct WeightRegion {
    const WeightParent* parent = nullptr;
    std::uint64_t begin        = 0;
    std::uint64_t end          = 0;
};

struct WeightView {
    std::vector<std::uint64_t> shape;
    std::vector<WeightRegion> parts;
};

// Coalesce adjacent logical regions without moving their bytes. Fails for a gather or
// multiple parents; those remain explicit inputs for a consumer supporting that form.
[[nodiscard]] WeightRegion contiguous_weight_region(const WeightView& view);
[[nodiscard]] bool is_complete_weight(const WeightView& view);
[[nodiscard]] std::uint64_t weight_scale_offset(const WeightGeometry& geometry, std::uint64_t row,
                                                std::uint64_t group);

struct WeightRowPlanes {
    const std::byte* codes        = nullptr;
    const std::byte* high         = nullptr;
    const std::byte* scales       = nullptr;
    std::uint64_t row_begin       = 0;
    std::uint64_t row_count       = 0;
    std::uint64_t code_row_bytes  = 0;
    std::uint64_t high_row_bytes  = 0;
    std::uint64_t scale_row_bytes = 0;
    // NVFP4 keeps the parent scale base; use its row origin and block geometry.
    bool swizzled_scales = false;
};

[[nodiscard]] WeightRowPlanes weight_row_planes(const WeightRegion& region);
[[nodiscard]] Tensor weight_tensor(const WeightView& view,
                                   std::initializer_list<std::int32_t> internal_shape);
// Existing Weight ABI: complete quantized parents, direct regions, and RowSplit row views.
// Arbitrary FP8/NVFP4 regions use their explicit planes until a native consumer supports them.
[[nodiscard]] Weight native_weight(const WeightView& view, float input_divisor = 0.0F);

} // namespace ninfer

