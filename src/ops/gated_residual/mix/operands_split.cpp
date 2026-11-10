#include "ops/gated_residual/mix/operands_split.h"
#include "ops/common/operand_checks.h"

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::detail::hc::mix {
namespace {
constexpr const char* kOp = "hc_mix";

void require(bool condition, const std::string& reason) {
    if (!condition) throw std::invalid_argument(std::string(kOp) + ": " + reason);
}

std::string named(const char* role, const char* reason) {
    return std::string(role) + " " + reason;
}

// The three planes of one matrix. They are *not* one flat range: an artifact that stores `down`
// and `inject` as row regions of one object interleaves the regions' plane blocks, so the regions
// are disjoint only plane by plane.
void append_planes(std::vector<operand::Range>& ranges, const Weight& w, const Q5Planes& p) {
    ranges.push_back(operand::range(w.qdata, std::size_t(w.n) * std::size_t(p.code_row_bytes), kOp));
    ranges.push_back(operand::range(w.qhigh, std::size_t(w.n) * std::size_t(p.high_row_bytes), kOp));
    ranges.push_back(
        operand::range(w.scales, std::size_t(w.n) * std::size_t(p.scale_row_bytes), kOp));
}

// Every plane byte the decoder reads, so the caller's overlap checks see exactly the read set.
std::vector<operand::Range> weight_ranges(SeparateWeights weights) {
    std::vector<operand::Range> ranges;
    append_planes(ranges, weights.down, q5_planes(weights.down, "down", kRank, kResidualWidth));
    append_planes(ranges, weights.up,
                  q5_planes(weights.up, "up", kResidualWidth, kRank));
    if (weights.inject) {
        append_planes(ranges, *weights.inject,
                      q5_planes(*weights.inject, "inject", kStreams, kResidualWidth));
    }
    return ranges;
}

operand::Range tensor(const Tensor& t, DType dtype, int a, int b, int c) {
    require(t.dtype == dtype && t.ne[0] == a && t.ne[1] == b && t.ne[2] == c && t.ne[3] == 1,
            "invalid tensor shape or dtype");
    require(t.is_contiguous() && operand::aligned(t.data, 16),
            "tensors require contiguous, 16-byte-aligned backing");
    return operand::range(t.data, t.bytes(), kOp);
}
} // namespace

Q5Planes q5_planes(const Weight& w, const char* role, int rows, int columns) {
    require(w.qtype == QType::Q5_G64_FP16,
            named(role, "requires the Q5_G64_FP16 groupwise format"));
    require(w.layout == QuantLayout::RowSplit, named(role, "requires the RowSplit layout"));
    require(w.ndim == 2 && w.n == rows && w.k == columns && w.shape[0] == rows &&
                w.shape[1] == columns && w.padded_shape[0] == rows,
            named(role, "has the wrong geometry"));
    require(w.group_size == 64 && w.group == 64, named(role, "requires 64-wide groups"));
    require(w.padded_shape[1] >= columns && w.padded_shape[1] % 128 == 0,
            named(role, "requires padded K aligned to 128"));
    const int groups = w.padded_shape[1] / 64;
    Q5Planes out{};
    out.padded_k        = w.padded_shape[1];
    out.code_row_bytes  = groups * 32;
    out.high_row_bytes  = groups * 8;
    out.scale_row_bytes = groups * 2;
    require(w.qdata && w.qhigh && w.scales, named(role, "has no quantized planes"));
    require(w.payload_bytes >= std::size_t(rows) * std::size_t(out.code_row_bytes),
            named(role, "code plane is too small"));
    require(w.high_plane_bytes >= std::size_t(rows) * std::size_t(out.high_row_bytes),
            named(role, "high-bit plane is too small"));
    require(w.scale_ne[0] == groups && w.scale_ne[1] == rows &&
                w.scale_nb[0] == 2 && w.scale_nb[1] == out.scale_row_bytes,
            named(role, "scale plane geometry disagrees with its storage"));
    require(operand::aligned(w.qdata, 16) && operand::aligned(w.qhigh, 16) &&
                operand::aligned(w.scales, 4),
            named(role, "planes require aligned device backing"));
    out.codes  = w.qdata;
    out.high   = w.qhigh;
    out.scales = w.scales;
    return out;
}

void describe_separate(SeparateWeights weights) {
    (void)q5_planes(weights.down, "down", kRank, kResidualWidth);
    (void)q5_planes(weights.up, "up", kResidualWidth, kRank);
    if (weights.kind == Kind::Block) {
        require(weights.inject != nullptr, "the block form requires its injection matrix");
        (void)q5_planes(*weights.inject, "inject", kStreams, kResidualWidth);
    } else {
        require(weights.inject == nullptr, "the final form carries no injection matrix");
    }
}

SeparateOperands validate_separate(SeparateWeights weights, const Tensor& normalized,
                                   Tensor& output, Tensor* injection,
                                   DeviceExecutionView execution) {
    describe_separate(weights);
    require(execution.multiprocessor_count > 0, "physical SM count must be positive");
    const int m = normalized.ne[2];
    require(m > 0, "columns must be positive");
    const auto inputs = weight_ranges(weights);
    const auto z      = tensor(normalized, DType::BF16, kWidth, kStreams, m);
    const auto out    = tensor(output, DType::BF16, kWidth, m, 1);
    operand::disjoint(z, out, kOp);
    for (const auto input : inputs) {
        operand::disjoint(z, input, kOp);
        operand::disjoint(out, input, kOp);
    }
    operand::disjoint(std::span<const operand::Range>{inputs}, kOp);
    float* injection_data = nullptr;
    if (injection) {
        const auto j = tensor(*injection, DType::FP32, kStreams, m, 1);
        operand::disjoint(j, z, kOp);
        operand::disjoint(j, out, kOp);
        for (const auto input : inputs) operand::disjoint(j, input, kOp);
        injection_data = static_cast<float*>(injection->data);
    }
    return {normalized.data,
            weights.down.qdata,
            weights.inject ? weights.inject->qdata : nullptr,
            weights.up.qdata,
            output.data,
            injection_data,
            m};
}

void validate_separate_scratch(SeparateWeights weights, const Tensor& normalized,
                               const Tensor& output, const Tensor* injection, DeviceSpan scratch) {
    const auto work = operand::range(scratch.data, scratch.bytes, kOp);
    operand::disjoint(work, operand::range(normalized.data, normalized.bytes(), kOp), kOp);
    operand::disjoint(work, operand::range(output.data, output.bytes(), kOp), kOp);
    if (injection)
        operand::disjoint(work, operand::range(injection->data, injection->bytes(), kOp), kOp);
    for (const auto input : weight_ranges(weights)) operand::disjoint(work, input, kOp);
}
} // namespace ninfer::ops::detail::hc::mix
