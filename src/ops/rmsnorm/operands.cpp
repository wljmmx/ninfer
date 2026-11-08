#include "ops/rmsnorm/operands.h"
#include "ops/common/operand_checks.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::rmsnorm {
namespace {
void require(bool condition, const char* op, const char* message) {
    if (!condition) throw std::invalid_argument(std::string(op) + ": " + message);
}

std::int64_t elements(const Tensor& tensor, bool positive, const char* op) {
    bool empty = false;
    for (int d = 0; d < 4; ++d) {
        require(tensor.ne[d] >= 0 && (!positive || tensor.ne[d] > 0), op, "invalid tensor extent");
        empty |= tensor.ne[d] == 0;
    }
    if (empty) return 0;
    std::int64_t count = 1;
    for (int d = 0; d < 4; ++d) {
        if (count > std::numeric_limits<std::int64_t>::max() / tensor.ne[d])
            throw std::overflow_error(std::string(op) + ": element count overflows");
        count *= tensor.ne[d];
    }
    return count;
}

void same_shape(const Tensor& x, const Tensor& other, const char* op) {
    for (int d = 0; d < 4; ++d) require(x.ne[d] == other.ne[d], op, "tensor shapes must match");
}
} // namespace

Operands validate(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps,
                  const Tensor& out, Epilogue kind, std::int32_t sms, const char* op) {
    require(x.dtype == DType::BF16 && weight.dtype == DType::BF16 && out.dtype == DType::BF16 &&
                (!gate || gate->dtype == DType::BF16),
            op, "operands must be BF16");
    require(eps > 0 && std::isfinite(eps), op, "epsilon must be positive and finite");
    const auto count = elements(x, gated(kind), op);
    same_shape(x, out, op);
    if (gate) same_shape(x, *gate, op);
    require(weight.ne[0] == x.ne[0] && weight.ne[1] == 1 && weight.ne[2] == 1 && weight.ne[3] == 1,
            op, "weight must be a vector of the reduced dimension");
    if (!count) return {x.ne[0], 0, false};
    require(!gated(kind) || sms > 0, op, "physical SM count must be positive");
    const std::array<const Tensor*, 4> tensors{&x, &weight, &out, gate};
    std::uintptr_t addresses = 0;
    for (const auto* t : tensors) {
        if (!t) continue;
        require(t->is_contiguous() && operand::aligned(t->data, 2), op,
                "operands require contiguous BF16-aligned backing");
        addresses |= reinterpret_cast<std::uintptr_t>(t->data);
    }
    const auto bytes  = operand::bytes(count, 2, op);
    const auto input  = operand::range(x.data, bytes, op);
    const auto gain   = operand::range(weight.data, operand::bytes(x.ne[0], 2, op), op);
    const auto output = operand::range(out.data, bytes, op);
    operand::disjoint(output, input, op);
    operand::disjoint(output, gain, op);
    if (gate)
        operand::disjoint(output, operand::range(gate->data, bytes, op), op);
    else
        operand::disjoint(input, gain, op);
    return {x.ne[0], count / x.ne[0], (addresses & 3u) == 0};
}
} // namespace ninfer::ops::detail::rmsnorm
