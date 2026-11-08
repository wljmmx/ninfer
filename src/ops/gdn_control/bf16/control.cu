#include "ops/gdn_control/bf16/launch.h"
#include "ops/gdn_control/bf16/epilogue.cuh"
#include "core/device.h"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::gdn_control {
namespace {
__launch_bounds__(256) __global__ void control_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, const float* a_log,
                               const float* bias, float* g, float* beta, std::int64_t elements) {
    const auto first  = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = std::int64_t(gridDim.x) * blockDim.x;
    for (auto i = first; i < elements; i += stride) {
        const int head = i % 48;
        g[i]           = decay(__bfloat162float(a[i]), a_log[head], bias[head]);
        beta[i]        = sigmoid(__bfloat162float(b[i]));
    }
}
} // namespace

void launch_control(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& bias,
                    GdnControlOutputs out, cudaStream_t stream) {
    const auto elements   = out.g.numel();
    const unsigned blocks = static_cast<unsigned>((elements + 255) / 256);
    control_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a.data), static_cast<const __nv_bfloat16*>(b.data),
        static_cast<const float*>(a_log.data), static_cast<const float*>(bias.data),
        static_cast<float*>(out.g.data), static_cast<float*>(out.beta.data), elements);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::gdn_control
