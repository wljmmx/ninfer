#pragma once

// Implements: include/ninfer/ops/scaled_add.h
// One coefficient per token, applied to a row-major-by-D BF16 plane and folded into a BF16
// destination. The fastest (D) axis is inside one token, so a 16-byte pack never crosses a
// coefficient boundary when D is a multiple of eight.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void scaled_add_scalar_kernel(const __nv_bfloat16* source, const float* coefficient,
                                         __nv_bfloat16* destination, std::int64_t d,
                                         std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int64_t t = i / d;
        const float value    = static_cast<float>(source[i]) * coefficient[t];
        destination[i]       = __float2bfloat16_rn(static_cast<float>(destination[i]) + value);
    }
}

// D is the fastest axis and a pack of eight stays inside one token, so the coefficient is uniform
// across the pack.
__global__ void scaled_add_bf16x8_kernel(const Bf16x8Pack* source, const float* coefficient,
                                         Bf16x8Pack* destination, std::int64_t d,
                                         std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const float scale = coefficient[(i * 8) / d];
        const Bf16x8Pack s = load_vec<Bf16x8Pack>(source + i);
        Bf16x8Pack out     = load_vec<Bf16x8Pack>(destination + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float lo = __low2float(out.pair[pair]) +
                             scale * __low2float(s.pair[pair]);
            const float hi = __high2float(out.pair[pair]) +
                             scale * __high2float(s.pair[pair]);
            out.pair[pair] = __floats2bfloat162_rn(lo, hi);
        }
        store_vec(destination + i, out);
    }
}

} // namespace ninfer::ops
