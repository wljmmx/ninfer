#pragma once

// bf16 activations -> int8 + one float scale per 64-wide value group, which is the shape the
// int8 m16n8k32 mma consumes together with the per-(row, group) weight scales.
//
// The group width of 64 is not arbitrary: it is the weight quantisation group, so one activation
// scale lines up with one weight scale and the product folds into a single multiply per group
// after the integer dot product. That is the same factoring the CPU int8 kernel uses.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::moe_cuda::groupwise_int {

// x: bf16 [k][tokens] row-major, out: int8 [k][tokens], scales: float [k/64][tokens].
// One block per token, one thread per 64-wide group: k/64 threads, so k <= 64 * 1024.
__global__ void quantize_a8_kernel(const __nv_bfloat16* __restrict__ x,
                                   std::int8_t* __restrict__ out, float* __restrict__ scales,
                                   int k, int tokens) {
    const int tok = blockIdx.x;
    const int g   = threadIdx.x;
    const int k0  = g * 64;
    if (k0 >= k) { return; }

    const int n = min(64, k - k0);
    float maxv  = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float v = __bfloat162float(x[(k0 + i) * tokens + tok]);
        maxv          = fmaxf(maxv, fabsf(v));
    }
    // Symmetric int8: 127 is the largest magnitude a signed byte holds.
    const float scale = maxv > 0.0f ? maxv / 127.0f : 1.0f;
    const float inv   = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        const float v      = __bfloat162float(x[(k0 + i) * tokens + tok]) * inv;
        out[(k0 + i) * tokens + tok] = static_cast<std::int8_t>(__float2int_rn(v));
    }
    scales[g * tokens + tok] = scale;
}

inline void launch_quantize_a8(const void* x, std::int8_t* out, float* scales, int k, int tokens,
                               cudaStream_t stream) {
    const int groups = (k + 63) / 64;
    quantize_a8_kernel<<<tokens, groups, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), out, scales, k, tokens);
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
