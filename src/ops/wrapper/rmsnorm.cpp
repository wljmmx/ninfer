// ninfer::ops - rmsnorm wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/gated_rmsnorm.h"

#include "ops/launcher/rmsnorm.h" // detail::rmsnorm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("rmsnorm: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("rmsnorm: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* b_label, const char* op) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(std::string(op) + ": x/" + b_label + " shapes must match");
        }
    }
}

struct ByteRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

std::size_t checked_bf16_bytes(std::int64_t elements, const char* op) {
    if (static_cast<std::uint64_t>(elements) > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::overflow_error(std::string(op) + ": tensor byte count overflows size_t");
    }
    return static_cast<std::size_t>(elements) * 2;
}

ByteRange checked_range(const void* data, std::size_t bytes, const char* op) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error(std::string(op) + ": tensor byte range overflows uintptr_t");
    }
    return {begin, begin + bytes};
}

void require_disjoint(ByteRange output, ByteRange input, const char* op) {
    if (output.begin < input.end && input.begin < output.end) {
        throw std::invalid_argument(std::string(op) + ": output must not overlap an input");
    }
}

void rmsnorm_impl(const Tensor& x, const Tensor& weight, float eps, detail::RmsEpilogue epilogue,
                  const Tensor* z, Tensor& out, std::int32_t multiprocessor_count,
                  cudaStream_t stream, const char* op) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        (z != nullptr && z->dtype != DType::BF16)) {
        throw std::invalid_argument(std::string(op) + ": x/weight/z/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument(std::string(op) + ": eps must be positive and finite");
    }

    const std::int64_t n = z != nullptr ? x.numel() : numel_allow_zero(x, "x");
    require_same_shape(x, out, "out", op);
    if (z != nullptr) { require_same_shape(x, *z, "z", op); }
    if (weight.ne[0] != x.ne[0] || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": weight must be 1-D with ne[0] == x.ne[0]");
    }
    if (n == 0) { return; }
    const std::int64_t rows = n / x.ne[0];
    if (z == nullptr && rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || !weight.is_contiguous() || !out.is_contiguous() ||
        (z != nullptr && !z->is_contiguous())) {
        throw std::invalid_argument(std::string(op) + ": x/weight/z/out must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || out.data == nullptr ||
        (z != nullptr && z->data == nullptr)) {
        throw std::invalid_argument(std::string(op) + ": x/weight/z/out data must be non-null");
    }

    if (z != nullptr) {
        if (multiprocessor_count <= 0) {
            throw std::invalid_argument(std::string(op) +
                                        ": positive multiprocessor count required");
        }
        // Matching BF16 shapes have the same checked byte count; avoid rescanning the views.
        const auto bytes       = checked_bf16_bytes(n, op);
        const ByteRange output = checked_range(out.data, bytes, op);
        require_disjoint(output, checked_range(x.data, bytes, op), op);
        require_disjoint(output, checked_range(weight.data, checked_bf16_bytes(x.ne[0], op), op),
                         op);
        require_disjoint(output, checked_range(z->data, bytes, op), op);
    }

    detail::rmsnorm_launch(x, weight, eps, epilogue, z, out, multiprocessor_count, stream);
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps,
                 unit_offset ? detail::RmsEpilogue::Offset : detail::RmsEpilogue::Plain, nullptr,
                 out, 0, stream, "rmsnorm");
}

void silu_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                        Tensor& out, DeviceExecutionView execution) {
    rmsnorm_impl(x, weight, eps, detail::RmsEpilogue::SiluGated, &z, out,
                 execution.multiprocessor_count, execution.stream, "silu_gated_rmsnorm");
}

void sigmoid_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                           Tensor& out, DeviceExecutionView execution) {
    rmsnorm_impl(x, weight, eps, detail::RmsEpilogue::SigmoidGated, &z, out,
                 execution.multiprocessor_count, execution.stream, "sigmoid_gated_rmsnorm");
}

} // namespace ninfer::ops
