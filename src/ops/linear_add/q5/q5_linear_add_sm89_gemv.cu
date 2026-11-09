// sm_89 (RTX 4090) v2-tuned residual GEMV for the Q5 linear_add decode path
// (T == 1). Ported from the v1.2.0 reference; the kernel itself lives in
// ops/linear/q5/q5_rowsplit_gemv.cuh (already ported).
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void q5_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream) {
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out          = static_cast<__nv_bfloat16*>(residual_out.data);

    if (w.n == 5120 && w.k == 6144 && w.padded_shape[1] == 6144) {
        q5_rowsplit_gemv_residual_launch_kernel<5120, 6144, 8, 3, true>(xp, codes, high, scales,
                                                                         out, stream);
    } else if (w.n == 2560 && w.k == 6144 && w.padded_shape[1] == 6144) {
        // Qwen4Exp GDN output (H = 2560). Same 16-group tile walk (6144 = 96 groups = 6 tiles)
        // and the same warp-per-row/8-rows-per-block/3-stage/stage-x recipe: 2560 = 320 x 8
        // keeps the grid at 320 blocks (>= 2 per SM on 128 SMs), and x (12 KiB) plus the
        // 3-deep weight buffers (15.75 KiB) stay far inside the 100 KiB shared budget.
        q5_rowsplit_gemv_residual_launch_kernel<2560, 6144, 8, 3, true>(xp, codes, high, scales,
                                                                        out, stream);
    } else if (w.n == 5120 && w.k == 17408 && w.padded_shape[1] == 17408) {
        q5_rowsplit_gemv_residual_launch_kernel<5120, 17408, 16, 2, false>(xp, codes, high, scales,
                                                                           out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add GEMV: unsupported exact shape");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
