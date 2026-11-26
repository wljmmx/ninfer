#include "core/nvtx.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/ggml/ggml_experts.h"
#include "ops/moe/operands.h"
#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/workspace.h"
#include "ops/moe/kernels/cuda/groupwise_int/workspace.h"
#include "ops/moe/kernels/cuda/block_scaled/workspace.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {
namespace kernel       = detail::moe_cuda;
namespace integer      = kernel::groupwise_int;
namespace block_scaled = kernel::block_scaled;
namespace impl         = detail::moe_routed;

struct CallStorage {
    integer::ExpertStorage integer;
    block_scaled::Storage block_scaled;
};

template <class Arena>
CallStorage allocate(Arena& arena, const impl::Plan& plan, int tokens, ExpertSubsetKind subset,
                     bool shared, bool split_activation, kernel::ShapeKind kind) {
    if (plan.implementation == kernel::ExpertImplementation::GgmlBlock) {
        // One CTA's staging area: it does not depend on `tokens`. The span is priced here and taken
        // again at launch time, because the two arenas this template serves hand back different span
        // types and the core is not wired to the banks yet.
        (void)arena.alloc_bytes(
            detail::ggml::routed_swiglu_workspace_bytes(plan.ggml.top_k, plan.ggml.intermediate), 256);
        return CallStorage{};
    }
    if (plan.implementation == kernel::ExpertImplementation::BlockScaled) {
        CallStorage out;
        out.block_scaled = block_scaled::allocate(arena, plan.block_scaled, tokens,
                                                  subset != ExpertSubsetKind::None, shared);
        return out;
    }
    if (subset == ExpertSubsetKind::None) {
        CallStorage out;
        if (!shared) { return out; }
        kernel::with_shape(kind, [&]<class Shape>() {
            out.integer = integer::allocate_shared<Shape>(arena, tokens);
        });
        return out;
    }
    CallStorage out;
    kernel::with_shape(kind, [&]<class Shape>() {
        out.integer = plan.integer.grouped
                          ? integer::allocate_grouped<Shape>(arena, tokens, plan.integer.group, shared,
                                                             split_activation)
                          : integer::allocate_token<Shape>(arena, tokens, shared);
    });
    return out;
}

std::size_t capacity(const RoutedSwiGLUProfile& profile, const SwiGLUProfile* shared,
                     ExpertSubsetKind subset, DType type, MoeEpilogue mode, int first, int last) {
    const auto& shape = impl::shape_dispatch(profile);
    detail::validate_interval(first, last);
    detail::validate_subset_kind(subset);
    detail::validate_output(type, mode);
    if (shared) { detail::validate_shared_profile(*shared, profile.ffn.hidden); }
    const auto boundaries =
        subset == ExpertSubsetKind::None ? std::span<const int>{} : shape.allocation_boundaries;
    return detail::interval_capacity(first, last, boundaries, impl::kSliceTokens, [&](int tokens) {
        WorkspaceLayoutBuilder layout;
        const auto plan = shape.select(tokens, shared != nullptr);
        // `shape_kind` answers in terms of the groupwise-int shapes, which a GGML plan does not have;
        // only the branches that need it are allowed to ask.
        const bool ggml = plan.implementation == kernel::ExpertImplementation::GgmlBlock;
        (void)allocate(layout, plan, tokens, subset, shared != nullptr,
                       profile.ffn.gate_up_type == QType::Q8_G32_FP16,
                       ggml ? kernel::ShapeKind::H2560 : kernel::shape_kind(profile));
        return layout.peak_bytes(1);
    });
}

void execute(const Tensor& x, const RoutedSwiGLUProfile& profile, const CudaSwiGLUExperts& experts,
             const ExpertRoutes& routes, ExpertSubset subset, const ScaledSwiGLUTerm* shared,
             MoeEpilogue mode, Tensor& destination, MemoryArena& workspace,
             CudaExecutionView execution, std::uint32_t* missed) {
    const auto& shape = impl::shape_dispatch(profile);
    detail::MoeOperands operands("routed_swiglu", execution);
    const int tokens = operands.input(x, profile.ffn.hidden);
    operands.output(destination, profile.ffn.hidden, tokens, mode);
    operands.subset(subset, profile.experts);
    if (subset.kind != ExpertSubsetKind::None) {
        operands.routes(routes, profile.top_k, tokens, false);
        operands.experts(experts, profile);
    }
    std::optional<SwiGLUProfile> shared_profile;
    if (shared) {
        shared_profile = swiglu_profile(shared->weights);
        operands.shared(shared->weights, profile.ffn.hidden);
        operands.coefficient(shared->coefficients, tokens, false);
    }
    auto workspace_scope = workspace.scope();
    MemoryArena scratch(
        detail::aligned_scratch(workspace, capacity(profile, shared_profile ? &*shared_profile : nullptr,
                                            subset.kind, destination.dtype, mode, tokens, tokens)));
    operands.workspace(scratch.storage());
    operands.finish();
    nvtx::ScopedRange range(nvtx::Name::RoutedSwiGLU, nvtx::Category::Moe, tokens);
    if (subset.kind == ExpertSubsetKind::None && !shared) {
        if (mode == MoeEpilogue::Write) { kernel::launch_zero(destination, execution); }
        return;
    }
    for (std::int64_t start = 0; start < tokens; start += impl::kSliceTokens) {
        const int offset = static_cast<int>(start);
        const int count  = std::min(tokens - offset, impl::kSliceTokens);
        const auto input = x.slice(1, offset, count);
        auto output      = destination.slice(1, offset, count);
        const ExpertRoutes selection =
            subset.kind == ExpertSubsetKind::None
                ? ExpertRoutes{}
                : ExpertRoutes{routes.ids.slice(1, offset, count),
                               routes.coefficients.slice(1, offset, count)};
        std::optional<ScaledSwiGLUTerm> term;
        if (shared) {
            term = ScaledSwiGLUTerm{shared->weights, shared->coefficients.slice(0, offset, count)};
        }
        auto scope         = scratch.scope();
        const auto plan    = shape.select(count, shared != nullptr);
        const bool ggml    = plan.implementation == kernel::ExpertImplementation::GgmlBlock;
        if (ggml) {
            // Handled before `allocate()`: that helper's ggml arm exists to PRICE the scratch during
            // planning, and calling it here would hand the same byte range out twice.
            if (missed != nullptr) {
                throw std::invalid_argument(
                    "routed_swiglu: missed_routes is not implemented for the GGML expert path");
            }
            if (term) {
                throw std::invalid_argument(
                    "routed_swiglu: the GGML expert path does not carry the shared-expert term yet");
            }
            if (experts.gate.codes == nullptr || experts.up.codes == nullptr ||
                experts.bank.down.codes == nullptr) {
                throw std::invalid_argument(
                    "routed_swiglu: the GGML expert path needs the split gate/up views");
            }
            const auto& routed = profile.ffn;
            detail::ggml::ExpertBanks banks{
                detail::ggml::bank_from(experts.gate, routed.hidden, routed.gate_up_type),
                detail::ggml::bank_from(experts.up, routed.hidden, routed.gate_up_type),
                detail::ggml::bank_from(experts.bank.down, routed.intermediate, routed.down_type)};
            try {
                detail::ggml::launch_routed_swiglu(input, banks, routed.hidden, routed.intermediate,
                                                   profile.experts, profile.top_k, subset, mode,
                                                   selection.ids, selection.coefficients, nullptr,
                                                   output, scratch, execution.stream);
            } catch (const std::bad_alloc&) {
                throw std::runtime_error("routed_swiglu: the GGML expert branch ran out of memory");
            }
            continue;
        }
        const auto buffers = allocate(scratch, plan, count, subset.kind, shared != nullptr,
                                      profile.ffn.gate_up_type == QType::Q8_G32_FP16,
                                      kernel::shape_kind(profile));
        if (plan.implementation == kernel::ExpertImplementation::BlockScaled) {
            // The block-scaled kernels skip a masked-out route without counting it. Reporting a zero
            // there would be worse than refusing: a predicted skip would read "no miss" from a path
            // that never looked.
            if (missed != nullptr) {
                throw std::invalid_argument(
                    "routed_swiglu: missed_routes is not implemented for the block-scaled paths");
            }
            block_scaled::launch(input, experts, selection, subset, term ? &*term : nullptr, mode,
                                 output, plan.block_scaled, buffers.block_scaled, execution);
            continue;
        }
        const auto& storage = buffers.integer;
        const auto kind     = kernel::shape_kind(profile);
        if (subset.kind == ExpertSubsetKind::None) {
            integer::launch_shared_swiglu(input, *term, mode, output, storage, plan.integer.shared,
                                          kind, execution);
            continue;
        }
        if (plan.integer.grouped) {
            integer::launch_group_count_scan(selection, subset, plan.integer.group, kind,
                                              storage.groups, execution, missed);
            if (plan.integer.group.adaptive) {
                integer::launch_token_swiglu(input, profile, experts, selection, subset,
                                             term ? &*term : nullptr, mode, output,
                                             plan.integer.token, storage, {}, execution, missed, true);
            }
            integer::launch_group_index(selection, kind, storage.groups, execution);
            integer::launch_grouped_swiglu(input, profile, experts, selection,
                                           term ? &*term : nullptr, mode, output,
                                           plan.integer.group, storage, execution);
        } else {
            integer::launch_token_swiglu(input, profile, experts, selection, subset,
                                         term ? &*term : nullptr, mode, output, plan.integer.token,
                                         storage, {}, execution, missed);
        }
    }
}
} // namespace

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                   ExpertSubsetKind subset, DType type,
                                                   MoeEpilogue mode, int first, int last) {
    return capacity(profile, nullptr, subset, type, mode, first, last);
}

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                   const SwiGLUProfile& shared,
                                                   ExpertSubsetKind subset, DType type,
                                                   MoeEpilogue mode, int first, int last) {
    return capacity(profile, &shared, subset, type, mode, first, last);
}

void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, MoeEpilogue mode, Tensor& destination,
                   MemoryArena& workspace, CudaExecutionView execution,
                   std::uint32_t* missed_routes) {
    execute(x, profile, experts, routes, subset, nullptr, mode, destination, workspace, execution,
            missed_routes);
}

void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, const ScaledSwiGLUTerm& shared, MoeEpilogue mode,
                   Tensor& destination, MemoryArena& workspace, CudaExecutionView execution,
                   std::uint32_t* missed_routes) {
    execute(x, profile, experts, routes, subset, &shared, mode, destination, workspace, execution,
            missed_routes);
}

void route_coverage(const Tensor& ids, const Tensor& mask, std::int32_t member,
                    std::uint32_t* excluded, CudaExecutionView execution) {
    if (excluded == nullptr) {
        throw std::invalid_argument("route_coverage: excluded must receive the count");
    }
    if (ids.dtype != DType::I32 || ids.ne[0] <= 0 || ids.ne[1] <= 0 || !ids.is_contiguous()) {
        throw std::invalid_argument("route_coverage: ids must be contiguous I32 [K,T], K,T > 0");
    }
    if (mask.dtype != DType::I32 || mask.ne[0] <= 0 || !mask.is_contiguous()) {
        throw std::invalid_argument("route_coverage: mask must be contiguous I32 [E], E > 0");
    }
    if (member != 0 && member != 1) {
        throw std::invalid_argument("route_coverage: member must be 0 or 1");
    }
    integer::launch_route_coverage(static_cast<const std::int32_t*>(ids.data), ids.ne[0] * ids.ne[1],
                                   static_cast<const std::int32_t*>(mask.data), mask.ne[0], member,
                                   excluded, execution);
}
} // namespace ninfer::ops
