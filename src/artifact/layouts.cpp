#include "artifact/layouts.h"

#include "artifact/formats.h"
#include "artifact/schema.h"

namespace ninfer::artifact {

// port (sm_89) V10-13: how many FP32 divisor words the payload ends with. A batched block-scale
// parent stores one per expert matrix by construction - that is what makes the bank a bank rather
// than one matrix with a shared divisor - and every other layout keeps the single matrix divisor.
// An object that states a count replaces the derived one, which is the only way a rank-two stack
// can say that each expert row block owns its own word.
std::uint64_t declared_divisor_count(const TensorObject& object, QuantLayout layout) {
    if (object.divisors) { return object.divisors; }
    if (layout == QuantLayout::BatchedBlockScaleK16M128x4 && !object.shape.empty()) {
        return object.shape.front();
    }
    return 1;
}

WeightGeometry describe_tensor(const TensorObject& object) {
    try {
        const auto format = parse_format(object.format);
        const auto layout = parse_layout(object.layout);
        auto geometry     = weight_geometry(format, layout, object.shape,
                                            WeightDivisors{declared_divisor_count(object, layout)});
        if (geometry.bytes != object.bytes || object.offset % geometry.alignment) {
            throw ArtifactError("encoded size or object alignment differs from layout");
        }
        return geometry;
    } catch (const std::exception& error) { throw ArtifactError(object.id + ": " + error.what()); }
}

} // namespace ninfer::artifact
