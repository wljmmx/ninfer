// sm_89 (RTX 4090) v2-tuned split-2 exact residual route for the Q5 linear_add
// decode path (T == 2..13). The v1.2.0 reference routes this window to the
// small-T MMA with the atomic residual epilogue, which is already ported as
// launch_q5_linear_add_small_t_mma (ops/linear/q5/q5_small_t_mma.cu).
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "ops/linear/q5/q5_launch.h"

namespace ninfer::ops::detail {

void q5_linear_add_split2_exact_sm89_launch(const Tensor& x, const Weight& w,
                                            Tensor& residual_out, cudaStream_t stream) {
    launch_q5_linear_add_small_t_mma(x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
