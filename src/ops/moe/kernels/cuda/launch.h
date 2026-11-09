#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail::moe_cuda {

template <int H, int I, int E, int K>
struct SwiGLUShape {
    static constexpr int hidden = H, intermediate = I, experts = E, top_k = K;
    static constexpr int control_rows = E + 1;
    static constexpr int score_rows   = (E + 4) / 4 * 4;
};

using SwiGLUH2048I512E256K8  = SwiGLUShape<2048, 512, 256, 8>;
using SwiGLUH2560I640E512K10 = SwiGLUShape<2560, 640, 512, 10>;

// port (sm_89): the groupwise-int kernels are written against one shape at a time, so every path
// that used to write `using S = SwiGLUH2048I512E256K8` now selects the shape from the profile. The
// selector is an enum rather than a template parameter at the API boundary because the plan and the
// workspace live on the host side of the call.
enum class ShapeKind { H2048, H2560 };

inline ShapeKind shape_kind(const RoutedSwiGLUProfile& profile) {
    return profile.ffn.hidden == SwiGLUH2560I640E512K10::hidden ? ShapeKind::H2560
                                                               : ShapeKind::H2048;
}

template <class Function>
inline void with_shape(ShapeKind kind, Function&& function) {
    if (kind == ShapeKind::H2560) {
        function.template operator()<SwiGLUH2560I640E512K10>();
    } else {
        function.template operator()<SwiGLUH2048I512E256K8>();
    }
}

enum class ExpertImplementation { GroupwiseInt, BlockScaled };

inline void launch_zero(Tensor& destination, CudaExecutionView execution) {
    CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(), execution.stream));
}

} // namespace ninfer::ops::detail::moe_cuda
