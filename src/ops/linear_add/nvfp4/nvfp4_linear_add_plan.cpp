#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

#if defined(NINFER_ENABLE_NVFP4)
enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
    A4,
};

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 || (input_rows != 6144 && input_rows != 17408)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4LinearAddRoute::A16;
    }
    if (!allows_a4(policy)) { throw std::invalid_argument("nvfp4 linear_add: unsupported policy"); }
    const std::int32_t first_a4 = input_rows == 6144 ? 17 : 8;
    return tokens >= first_a4 ? Nvfp4LinearAddRoute::A4 : Nvfp4LinearAddRoute::A16;
}
#endif

} // namespace

#if defined(NINFER_ENABLE_NVFP4)
std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                       std::int32_t input_rows, LinearPolicy policy,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearAddRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearAddRoute::A16) {
        nvfp4_linear_add_a16_launch(x, weight, residual, stream);
        return;
    }
    auto scope                     = workspace.scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_a4_launch(x, weight, residual, scratch, stream);
}
#else
namespace {
[[noreturn]] void reject_nvfp4_linear_add() {
    throw std::invalid_argument(
        "nvfp4 linear_add: NVFP4 weights require a Blackwell (sm_120a) build with NVFP4 "
        "tensor cores. This engine was built for RTX 4090 (sm_89); load a groupwise-int "
        "artifact instead.");
}
} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                       std::int32_t, std::int32_t) {
    reject_nvfp4_linear_add();
}

void nvfp4_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                                WorkspaceArena&, cudaStream_t) {
    reject_nvfp4_linear_add();
}
#endif

} // namespace ninfer::ops::detail
