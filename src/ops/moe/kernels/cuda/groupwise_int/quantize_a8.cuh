#pragma once

// bf16 activations -> int8 + one float scale per 64-wide value group, which is the shape the int8
// m16n8k32 mma consumes together with the per-(row, group) weight scales.
//
// The group width of 64 is not arbitrary: it is the weight quantisation group, so one activation
// scale lines up with one weight scale and the product folds into a single multiply per group
// after the integer dot product. That is the same factoring the CPU int8 kernel uses.
//
// Layout is one row per token, matching how the model hands activations in and how the gate_up
// kernel reads a token's k run: x[tok][k], out[tok][k], scales[tok][k/64]. The earlier version
// wrote k-major, which no reader in the tree asks for.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::moe_cuda::groupwise_int {

// x: bf16 [tokens][k] row-major, out: int8 [tokens][k], scales: float [tokens][k/64].
// One block per token, one thread per 64-wide group: k/64 threads, so k <= 64 * 1024.
__global__ void quantize_a8_kernel(const __nv_bfloat16* __restrict__ x,
                                   std::int8_t* __restrict__ out, float* __restrict__ scales,
                                   int k, int tokens) {
    const int tok = blockIdx.x;
    const int g   = threadIdx.x;
    const int k0  = g * 64;
    if (tok >= tokens || k0 >= k) { return; }

    const int n         = min(64, k - k0);
    const auto* src     = x + std::int64_t(tok) * k + k0;
    auto* dst           = out + std::int64_t(tok) * k + k0;
    float maxv          = 0.0f;
    for (int i = 0; i < n; ++i) { maxv = fmaxf(maxv, fabsf(__bfloat162float(src[i]))); }
    // Symmetric int8: 127 is the largest magnitude a signed byte holds.
    const float scale = maxv > 0.0f ? maxv / 127.0f : 1.0f;
    const float inv   = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        const float v = __bfloat162float(src[i]) * inv;
        dst[i]        = static_cast<std::int8_t>(__float2int_rn(v));
    }
    scales[std::int64_t(tok) * ((k + 63) / 64) + g] = scale;
}

inline void launch_quantize_a8(const void* x, std::int8_t* out, float* scales, int k, int tokens,
                               cudaStream_t stream) {
    const int groups = (k + 63) / 64;
    if (tokens <= 0) { return; }
    quantize_a8_kernel<<<tokens, groups, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), out, scales, k, tokens);
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
