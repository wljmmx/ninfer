// ninfer::ops - scaled_add wrapper: validates the operands and dispatches to the launcher.
// Host-compiled; never includes the kernel header.
#include "ninfer/ops/scaled_add.h"

#include "ops/launcher/scaled_add.h"

#include <stdexcept>

namespace ninfer::ops {

void scaled_add(const Tensor& source, const Tensor& coefficient, Tensor& destination,
                cudaStream_t stream) {
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16) {
        throw std::invalid_argument("scaled_add: source/destination must be BF16");
    }
    if (coefficient.dtype != DType::FP32) {
        throw std::invalid_argument("scaled_add: coefficient must be FP32");
    }
    if (source.ne[0] <= 0 || source.ne[1] <= 0 || source.ne[0] != destination.ne[0] ||
        source.ne[1] != destination.ne[1] || source.ne[2] != 1 || source.ne[3] != 1 ||
        destination.ne[2] != 1 || destination.ne[3] != 1) {
        throw std::invalid_argument("scaled_add: source/destination must be equal [D,T] matrices");
    }
    if (coefficient.ne[0] != source.ne[1] || coefficient.ne[1] != 1 || coefficient.ne[2] != 1 ||
        coefficient.ne[3] != 1) {
        throw std::invalid_argument("scaled_add: coefficient must be [T]");
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || !coefficient.is_contiguous()) {
        throw std::invalid_argument("scaled_add: operands must be contiguous");
    }
    if (source.data == nullptr || destination.data == nullptr || coefficient.data == nullptr) {
        throw std::invalid_argument("scaled_add: operands must be non-null");
    }
    detail::scaled_add_launch(source, coefficient, destination, stream);
}

} // namespace ninfer::ops
