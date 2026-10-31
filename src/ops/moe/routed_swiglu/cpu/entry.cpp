#include "ninfer/ops/moe.h"
#include "ops/common/host/operands.h"
#include "ops/moe/routed_swiglu/cpu/dispatch.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops {
namespace {
namespace impl            = detail::moe_routed_cpu;
namespace kernel          = detail::moe_cpu;
namespace operand         = common::operand;
constexpr int max_experts = 512;
constexpr auto op         = "routed_swiglu CPU";

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

class Operands {
public:
    Operands(const Tensor& x, Tensor& out, MemoryArena& workspace, std::size_t capacity,
             int hidden) {
        operand::tensor(x, DType::BF16, {hidden, x.ne[1]}, 2, op, "x", Device::cpu());
        operand::tensor(out, DType::FP32, {hidden, x.ne[1]}, 4, op, "destination", Device::cpu());
        operand::device(workspace.device(), Device::cpu(), op, "workspace");
        require(capacity <= workspace.capacity() - workspace.used(),
                "routed_swiglu CPU: insufficient workspace");
        output_ = operand::range(out, op);
        if (capacity) {
            const auto* begin =
                static_cast<const std::byte*>(workspace.storage().data) + workspace.used();
            require(operand::aligned(begin, 256),
                    "routed_swiglu CPU: workspace cursor must be 256-byte aligned");
            scratch_ = operand::range(begin, capacity, op);
            operand::disjoint(output_, scratch_, op);
        }
        read(x.data, x.bytes());
    }

    void tensor(const Tensor& value, DType dtype, std::initializer_list<int> shape,
                const char* name) const {
        operand::tensor(value, dtype, shape, dtype_size(dtype), op, name, Device::cpu());
        read(value.data, value.bytes());
    }

    void bank(const WeightBankView& bank, QType type, QuantLayout format_layout, int rows,
              int columns, int count, const std::array<bool, max_experts>& selected) const {
        require(bank.parent && bank.parent->data, "routed_swiglu CPU: missing weight bank");
        operand::device(bank.parent->device, Device::cpu(), op, "weight bank");
        const auto& layout       = bank.layout;
        const auto& geometry     = bank.parent->geometry;
        const auto elements      = std::size_t(rows) * columns;
        const bool nvfp4         = type == QType::NVFP4;
        const auto native_layout = layout.layout == QuantLayout::BatchedBlockScaleK16M128x4
                                       ? QuantLayout::BlockScaleK16M128x4
                                       : layout.layout;
        const auto high_bytes    = type == QType::Q5_G64_FP16   ? elements / 8
                                   : type == QType::Q6_G64_FP16 ? elements / 4
                                                                : 0;
        require(layout.format == type && native_layout == format_layout &&
                    layout.region.count == unsigned(count) &&
                    layout.region.rows == unsigned(rows) && layout.columns == unsigned(columns),
                "routed_swiglu CPU: weight bank differs from the profile");
        require(geometry.group_size == (nvfp4 ? 16 : 64) &&
                    geometry.padded_columns == unsigned(columns) &&
                    geometry.scale_block_rows == 1 && layout.has_divisors == nvfp4,
                "routed_swiglu CPU: invalid weight geometry");
        // port (sm_89): a host-tier bank packs each expert into one segment, so the expected
        // layout has rebased plane offsets instead of the shared-array ones. A fused bank cannot be
        // packed that way - its three plane runs sit hundreds of MB apart - and names each plane's
        // origin through its own per-plane table instead, so the offsets are not comparable and only
        // the plane sizes and the region are checked.
        auto expected = describe_weight_bank(geometry, layout.region);
        if (bank.expert_offsets) {
            expected.codes.offset  = 0;
            expected.high.offset   = layout.codes.expert_bytes;
            expected.scales.offset = layout.codes.expert_bytes + layout.high.expert_bytes;
        }
        const bool per_plane = bank.codes_offsets != nullptr || bank.high_offsets != nullptr ||
                               bank.scales_offsets != nullptr;
        const bool planes_agree =
            per_plane
                ? layout.region == expected.region && layout.format == expected.format &&
                      layout.layout == expected.layout && layout.columns == expected.columns &&
                      layout.codes.expert_stride == expected.codes.expert_stride &&
                      layout.high.expert_stride == expected.high.expert_stride &&
                      layout.scales.expert_stride == expected.scales.expert_stride &&
                      layout.has_divisors == expected.has_divisors
                : layout == expected;
        require(layout.codes.expert_bytes == elements / 2 &&
                    layout.high.expert_bytes == high_bytes &&
                    layout.scales.expert_bytes == elements / (nvfp4 ? 16 : 32) &&
                    planes_agree,
                "routed_swiglu CPU: invalid native weight planes");
        require(bank.parent->weight_divisors.size() == geometry.divisor_count,
                "routed_swiglu CPU: invalid divisor storage");
        for (int expert = 0; expert < count; ++expert) {
            if (!selected[expert]) { continue; }
            for (const auto role : {ExpertPlaneKind::Codes, ExpertPlaneKind::High,
                                   ExpertPlaneKind::Scales}) {
                const auto& plane = role == ExpertPlaneKind::Codes  ? layout.codes
                                    : role == ExpertPlaneKind::High ? layout.high
                                                                    : layout.scales;
                if (!plane.expert_bytes) { continue; }
                const auto memory = bank.plane(role, expert);
                require(operand::aligned(memory.data, 16),
                        "routed_swiglu CPU: weight plane must be 16-byte aligned");
                read(memory.data, memory.bytes);
            }
            if (nvfp4) {
                const auto* divisor = bank.parent->weight_divisors.data() + layout.divisor_begin +
                                      expert * layout.divisor_stride;
                require(std::isfinite(*divisor) && *divisor > 0,
                        "routed_swiglu CPU: invalid expert divisor");
                read(divisor, sizeof(float));
            }
        }
    }

private:
    void read(const void* pointer, std::size_t bytes) const {
        if (!bytes) { return; }
        const auto range = operand::range(pointer, bytes, op);
        operand::disjoint(output_, range, op);
        if (scratch_.begin != scratch_.end) { operand::disjoint(scratch_, range, op); }
    }

    operand::Range output_{}, scratch_{};
};

} // namespace

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                   ExpertSubsetKind subset, std::int32_t first,
                                                   std::int32_t last) {
    const auto& shape = impl::shape_dispatch(profile);
    impl::validate_capacity(subset, first, last);
    return subset == ExpertSubsetKind::None ? 0 : shape.capacity(last);
}

void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CpuSwiGLUExperts& experts, const ExpertRoutes& routes, ExpertSubset subset,
                   Tensor& destination, MemoryArena& workspace, CpuExecutionView execution,
                   std::uint32_t* missed_routes) {
    // The CPU half of a split layer runs exactly the routes its mask kept, so it never skips one.
    (void)missed_routes;
    const auto& shape = impl::shape_dispatch(profile);
    const int tokens  = x.ne[1];
    impl::validate_capacity(subset.kind, tokens, tokens);
    const auto capacity = subset.kind == ExpertSubsetKind::None ? 0 : shape.capacity(tokens);
    // port (sm_89): NINFER_CPU_TIMING separates the per-call setup (operand validation, route
    // bookkeeping, the scratch cursor) from the kernels. At a split layer's granularity the setup is
    // paid once per layer per token, so it competes with the arithmetic.
    const bool timing  = std::getenv("NINFER_CPU_TIMING") != nullptr;
    const auto started = std::chrono::steady_clock::now();
    const Operands operands(x, destination, workspace, capacity, profile.ffn.hidden);
    if (subset.kind == ExpertSubsetKind::None) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }

    operands.tensor(routes.ids, DType::I32, {profile.top_k, tokens}, "route IDs");
    operands.tensor(routes.coefficients, DType::FP32, {profile.top_k, tokens}, "coefficients");
    const std::int32_t* mask = nullptr;
    if (subset.kind == ExpertSubsetKind::MaskOnes || subset.kind == ExpertSubsetKind::MaskZeros) {
        operands.tensor(subset.mask, DType::I32, {profile.experts}, "mask");
        mask = static_cast<const std::int32_t*>(subset.mask.data);
        // port (sm_89): one pass over the words instead of one branch per expert. The mask is
        // validated on every call - 40 times per split token - and a branch per entry was pure
        // per-call overhead: the same check as a branchless or of `value & ~1` costs a few cycles
        // per word and never mispredicts.
        std::uint32_t non_binary = 0;
        for (int e = 0; e < profile.experts; ++e) {
            non_binary |= static_cast<std::uint32_t>(mask[e]) & ~std::uint32_t{1};
        }
        require(non_binary == 0, "routed_swiglu CPU: mask must be binary");
    }
    const auto* ids          = static_cast<const std::int32_t*>(routes.ids.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    std::array<bool, max_experts> selected{};
    bool has_work = false;
    for (int t = 0; t < tokens; ++t) {
        const auto offset = std::size_t(t) * profile.top_k;
        for (int j = 0; j < profile.top_k; ++j) {
            const int expert = ids[offset + j];
            if (!(expert >= 0 && expert < profile.experts &&
                  std::isfinite(coefficients[offset + j]))) {
                // port (sm_89): report the offending values. The two causes need different fixes:
                // an out-of-range id means the router or its input is already wrong, a non-finite
                // coefficient means the layer produced NaN.
                std::fprintf(stderr,
                             "[cpu-route] t=%d j=%d id=%d experts=%d coef=%.9g finite=%d tokens=%d\n",
                             t, j, expert, profile.experts, static_cast<double>(coefficients[offset + j]),
                             int(std::isfinite(coefficients[offset + j])), tokens);
            }
            require(expert >= 0 && expert < profile.experts &&
                        std::isfinite(coefficients[offset + j]),
                    "routed_swiglu CPU: invalid route ID or coefficient");
            for (int previous = 0; previous < j; ++previous) {
                require(expert != ids[offset + previous], "routed_swiglu CPU: duplicate route ID");
            }
            if (kernel::participates(subset.kind, mask, expert)) {
                selected[expert] = true;
                has_work         = true;
            }
        }
    }
    if (!has_work) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }
    operands.bank(experts.gate_up, profile.ffn.gate_up_type, profile.ffn.gate_up_layout,
                  2 * profile.ffn.intermediate, profile.ffn.hidden, profile.experts, selected);
    operands.bank(experts.down, profile.ffn.down_type, profile.ffn.down_layout, profile.ffn.hidden,
                  profile.ffn.intermediate, profile.experts, selected);
    auto scope = workspace.scope();
    const kernel::Invocation args{.input        = static_cast<const std::uint16_t*>(x.data),
                                  .ids          = ids,
                                  .coefficients = coefficients,
                                  .experts      = experts,
                                  .subset       = subset.kind,
                                  .mask         = mask,
                                  .output       = static_cast<float*>(destination.data),
                                  .tokens       = tokens};
    const auto setup_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    const auto launch_begin = std::chrono::steady_clock::now();
    shape.launch(args, workspace, execution);
    if (timing) {
        const auto launch_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - launch_begin).count();
        static std::uint64_t calls = 0;
        static double acc[2]       = {0.0, 0.0};
        acc[0] += setup_ms;
        acc[1] += launch_ms;
        if (++calls % 40 == 0) {
            std::fprintf(stderr, "[cpu] setup=%.3fms kernel=%.3fms per layer\n", acc[0] / 40.0,
                         acc[1] / 40.0);
            for (double& value : acc) { value = 0.0; }
        }
    }
}

} // namespace ninfer::ops
