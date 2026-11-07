#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/launch.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* label) {
    if (!condition) throw std::invalid_argument(std::string("qsa_tail_extract: invalid ") + label);
}

bool aligned(const void* p, std::size_t bytes) {
    return p && reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
}

void metadata(const Tensor& t, int batch, const char* label) {
    require(t.dtype == DType::I32 && t.ne[0] == batch && t.ne[1] == 1 && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.is_contiguous() && aligned(t.data, 4),
            label);
}

// Contiguous feature columns, with independent column and slot/batch strides. The smaller
// outer extent is at most eight, so validation is bounded even for large pools and records.
void columns(const Tensor& t, DType dtype, int features, int count, int outer, const char* label) {
    const std::uint64_t element = dtype == DType::BF16 ? 2 : 4;
    const std::uint64_t width   = features * element;
    require(count > 0 && outer > 0 && t.dtype == dtype && t.ne[0] == features && t.ne[1] == count &&
                t.ne[2] == outer && t.ne[3] == 1 && t.nb[0] == static_cast<std::int64_t>(element) &&
                aligned(t.data, element),
            label);
    std::uint64_t span = width;
    for (int d = 1; d <= 2; ++d) {
        if (t.ne[d] == 1) continue;
        require(t.nb[d] >= static_cast<std::int64_t>(width) && t.nb[d] % element == 0, label);
        const auto stride = static_cast<std::uint64_t>(t.nb[d]);
        const auto extent = static_cast<std::uint64_t>(t.ne[d] - 1);
        require(stride <= (std::numeric_limits<std::int64_t>::max() - span) / extent, label);
        span += stride * extent;
    }
    require(span <= std::numeric_limits<std::uintptr_t>::max() -
                        reinterpret_cast<std::uintptr_t>(t.data),
            label);
    const int small = count <= outer ? 1 : 2, large = 3 - small;
    const auto small_stride = t.ne[small] == 1 ? width : std::uint64_t(t.nb[small]);
    const auto large_stride = t.ne[large] == 1 ? width : std::uint64_t(t.nb[large]);
    // Only the two nearest columns of the other progression can intersect this column.
    for (int delta = 1; delta < t.ne[small]; ++delta) {
        const auto distance = delta * small_stride, q = distance / large_stride;
        if (q < std::uint64_t(t.ne[large])) require(distance - q * large_stride >= width, label);
        if (q + 1 < std::uint64_t(t.ne[large]))
            require((q + 1) * large_stride - distance >= width, label);
    }
}

void state(const QsaTailStateView& v, int batch) {
    metadata(v.slots, batch, "state slots");
    const int slots = v.raw_keys.ne[2];
    columns(v.raw_keys, DType::BF16, 128, 3, slots, "tail raw keys");
    columns(v.first_rope, DType::I32, 3, slots, 1, "tail coordinates");
}

} // namespace

void qsa_tail_extract(const QsaTailStateView& before, const QsaRawRecordReadView& records,
                      const Tensor& accepted, QsaTailStateView after,
                      DeviceExecutionView execution) {
    const int batch = accepted.ne[0], capacity = records.raw_keys.ne[1];
    require(batch >= 1 && batch <= 8, "batch extent");
    require(execution.multiprocessor_count > 0, "SM count");
    metadata(accepted, batch, "accepted columns");
    metadata(records.base_positions, batch, "record base positions");
    metadata(records.valid_columns, batch, "record valid columns");
    state(before, batch);
    state(after, batch);
    columns(records.raw_keys, DType::BF16, 128, capacity, batch, "record raw keys");
    columns(records.rope_positions, DType::I32, 3, capacity, batch, "record coordinates");
    // Only Device-selected slots are accessed. Overlapping pool bounds do not imply that
    // selected regions overlap; the caller guarantees their cross-row/plane relationships.
    detail::qsa::launch_tail_extract(before, records, accepted, after, execution);
}

} // namespace ninfer::ops
