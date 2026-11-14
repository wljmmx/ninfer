#include "ops/launcher/rmsnorm_pack_tail.h"

#include "core/device.h"
#include "ops/kernel/rmsnorm_pack_tail.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <int Rows>
void launch_rows(const Tensor& input, const Tensor& weight, Tensor& output, cudaStream_t stream) {
    // The CTA covers the row in whole pair strides: 5120 pairs into 512 threads (5 each), 2560 into
    // 256 (also 5 each).
    constexpr int threads = Rows == 2560 ? 256 : 512;
    const dim3 grid(input.ne[1] - 1, input.ne[2]);
    rmsnorm_pack_tail_kernel<Rows, threads>
        <<<grid, threads, 0, stream>>>(static_cast<const __nv_bfloat162*>(input.data),
                                       static_cast<const __nv_bfloat162*>(weight.data),
                                       static_cast<__nv_bfloat162*>(output.data), input.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void rmsnorm_pack_tail_launch(const Tensor& input, const Tensor& weight, Tensor& output,
                              cudaStream_t stream) {
    switch (input.ne[0]) {
    case 2560:
        return launch_rows<2560>(input, weight, output, stream);
    case 5120:
        return launch_rows<5120>(input, weight, output, stream);
    }
    throw std::invalid_argument("rmsnorm_pack_tail: no instance for the channel count");
}

} // namespace ninfer::ops::detail
