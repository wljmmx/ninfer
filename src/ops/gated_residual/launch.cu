#include "ops/gated_residual/launch.h"
#include "ops/gated_residual/kernel/norm.cuh"

#include <algorithm>
#include <limits>

namespace ninfer::ops::detail::hc {
namespace {
enum class NormSchedule { Pairs256, Vectors320, StreamingVectors320 };

constexpr int kExpandVectorStart        = 2048;
constexpr int kGroupedStreamingStart    = 2816;
constexpr int kDisjointStreamingStart   = 1536;
constexpr int kDisjointMaterializeStart = 1841;
constexpr int kDisjointMaterializeEnd   = 3072;
constexpr int kInPlaceMaterializeStart  = 1824;
constexpr int kInPlaceMaterializeEnd    = 3680;

template <Form Kind, NormSchedule Schedule = NormSchedule::Pairs256>
void norm_rows(Operands o, DeviceExecutionView e) {
    const std::int64_t rows = std::int64_t(o.columns) * kStreams;
    for (std::int64_t first = 0; first < rows;) {
        const auto count = std::min(rows - first, std::int64_t(std::numeric_limits<int>::max()));
        if constexpr (Schedule != NormSchedule::Pairs256)
            norm_vector_kernel<Kind, Schedule == NormSchedule::StreamingVectors320>
                <<<unsigned(count), 320, 0, e.stream>>>(o, first);
        else
            norm_kernel<Kind, 256><<<unsigned(count), 256, 0, e.stream>>>(o, first);
        CUDA_CHECK(cudaGetLastError());
        first += count;
    }
}

template <Form Kind>
void norm(Operands o, DeviceExecutionView e) {
    if constexpr (Kind == Form::ExpandNorm) {
        if (o.columns >= kExpandVectorStart) return norm_rows<Kind, NormSchedule::Vectors320>(o, e);
    } else if constexpr (Kind == Form::GroupedNorm) {
        if (o.columns >= kGroupedStreamingStart)
            return norm_rows<Kind, NormSchedule::StreamingVectors320>(o, e);
    }
    norm_rows<Kind>(o, e);
}

template <Form Kind>
void combine(Operands o, DeviceExecutionView e) {
    constexpr int block                = 256;
    const std::int64_t vectors         = std::int64_t(o.columns) * kStreams * (kWidth / 8);
    constexpr std::int64_t max_vectors = std::int64_t(std::numeric_limits<int>::max()) * block;
    for (std::int64_t first = 0; first < vectors;) {
        const auto count = std::min(vectors - first, max_vectors);
        combine_vector_kernel<Kind, block>
            <<<unsigned((count + block - 1) / block), block, 0, e.stream>>>(o, first, count);
        CUDA_CHECK(cudaGetLastError());
        first += count;
    }
}

template <Form Kind>
void combine_norm(Operands o, DeviceExecutionView e) {
    const bool alias = o.input == o.residual;
    // Input eviction priority improves the disjoint transition region, with a small
    // cold-cache cost at its upper edge. It is unsuitable for the in-place working set.
    if (!alias && o.columns >= kDisjointStreamingStart && o.columns < kDisjointMaterializeStart)
        return norm_rows<Kind, NormSchedule::StreamingVectors320>(o, e);
    // Materialization wins over both fused schedules in the intermediate working set.
    const bool materialize =
        alias ? o.columns >= kInPlaceMaterializeStart && o.columns <= kInPlaceMaterializeEnd
              : o.columns >= kDisjointMaterializeStart && o.columns <= kDisjointMaterializeEnd;
    if (materialize) {
        combine<Kind == Form::AddNorm ? Form::AddNorm : Form::Combine>(o, e);
        o.input = o.residual;
        // This input was just produced by the preceding kernel. Preserve the qualified
        // materialized schedule rather than applying the standalone input-eviction policy.
        norm_rows<Form::GroupedNorm>(o, e);
    } else {
        norm<Kind>(o, e);
    }
}
} // namespace

void launch(Form form, Operands o, DeviceExecutionView e) {
    switch (form) {
    case Form::ExpandNorm:
        norm<Form::ExpandNorm>(o, e);
        break;
    case Form::GroupedNorm:
        norm<Form::GroupedNorm>(o, e);
        break;
    case Form::Combine:
        combine<Form::Combine>(o, e);
        break;
    case Form::CombineNorm:
        combine_norm<Form::CombineNorm>(o, e);
        break;
    case Form::AddNorm:
        combine_norm<Form::AddNorm>(o, e);
        break;
    }
}
} // namespace ninfer::ops::detail::hc
