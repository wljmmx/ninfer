#include "ops/moe/routed_swiglu/cuda/dispatch.h"

#include <array>

namespace ninfer::ops::detail::moe_routed {
namespace {

// The Qwen4Exp row with GGML-encoded experts: H=2560, I=640, E=512, K=10. Its scratch is one CTA's
// staging area (top_k x intermediate products plus the per-route coefficients) and does not grow with
// the token count, so the boundary list only keeps the pricer from quoting slices it never runs.
constexpr int kGgmlFirst = 1;
constexpr std::array kAllocationBoundaries{kGgmlFirst};
constexpr int kTopK         = 10;
constexpr int kIntermediate = 640;

Plan select_plan(int tokens, bool shared) {
    (void)tokens;
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::GgmlBlock;
    plan.ggml           = {.top_k = kTopK, .intermediate = kIntermediate, .shared = shared};
    return plan;
}

} // namespace

const ShapePolicy& ggml_h2560_i640_e512_k10() {
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = kAllocationBoundaries};
    return policy;
}

} // namespace ninfer::ops::detail::moe_routed
