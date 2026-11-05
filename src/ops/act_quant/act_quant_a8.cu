#include "ninfer/ops/act_quant_a8.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdlib>

namespace ninfer::ops {
namespace {

// Both kernels keep their probe semantics bit-for-bit: the vectorized passes load the same
// bf16 elements as bf16x2 pairs and run the identical per-element fabsf/max/lrintf/clamp
// math, so the quantized output is bit-identical to the scalar form; only the access width
// changes. Rows with an odd element count (never hit by the probe hooks: hidden sizes are
// even) fall back to the scalar loops.
__launch_bounds__(512) __global__ void a8_roundtrip_kernel(__nv_bfloat16* __restrict__ data, int k) {
    const int t                     = blockIdx.x;
    __nv_bfloat16* __restrict__ row = data + static_cast<std::size_t>(t) * k;
    const bool paired = (k & 1) == 0;
    float amax       = 0.0f;
    if (paired) {
        auto* pairs = reinterpret_cast<__nv_bfloat162*>(row);
        for (int p = threadIdx.x; p < k / 2; p += blockDim.x) {
            const __nv_bfloat162 v = pairs[p];
            amax = fmaxf(amax, fabsf(__low2float(v)));
            amax = fmaxf(amax, fabsf(__high2float(v)));
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            amax = fmaxf(amax, fabsf(__bfloat162float(row[i])));
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    __shared__ float warp_max[32];
    if ((threadIdx.x & 31) == 0) { warp_max[threadIdx.x >> 5] = amax; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m         = 0.0f;
        const int warps = (blockDim.x + 31) >> 5;
        for (int i = 0; i < warps; ++i) { m = fmaxf(m, warp_max[i]); }
        warp_max[0] = m > 0.0f ? m / 127.0f : 0.0f;
    }
    __syncthreads();
    const float scale = warp_max[0];
    const float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
    if (paired) {
        auto* pairs = reinterpret_cast<__nv_bfloat162*>(row);
        for (int p = threadIdx.x; p < k / 2; p += blockDim.x) {
            const __nv_bfloat162 v = pairs[p];
            int q0                 = static_cast<int>(lrintf(__low2float(v) * inv));
            int q1                 = static_cast<int>(lrintf(__high2float(v) * inv));
            q0                     = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
            q1                     = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
            pairs[p]               = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                          static_cast<float>(q1) * scale);
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            int q = static_cast<int>(lrintf(__bfloat162float(row[i]) * inv));
            q     = q < -127 ? -127 : (q > 127 ? 127 : q);
            row[i] = __float2bfloat16_rn(static_cast<float>(q) * scale);
        }
    }
}

// One scale per `group` contiguous elements, matching the dense W4A8 quantiser (kG = 64) and the
// weight groups it multiplies. Sixteen lanes cooperate on one group, each taking a contiguous run
// of bf16x2 pairs, so a warp covers two groups as one coalesced pass. Every admitted group width
// is a multiple of 16, so per_lane (group/16 elements) is always an even count and one lane's
// run is whole bf16x2 pairs - the same elements as the scalar form, read wide.
__launch_bounds__(512) __global__ void a8_roundtrip_group_kernel(__nv_bfloat16* __restrict__ data, int k, int group) {
    constexpr int kLanesPerGroup = 16;
    const int t                  = blockIdx.x;
    __nv_bfloat16* __restrict__ row = data + static_cast<std::size_t>(t) * k;
    const int groups = k / group;
    const int per_lane = group / kLanesPerGroup;
    const int pairs = per_lane / 2;
    const int gpr  = blockDim.x / kLanesPerGroup;
    for (int g0 = 0; g0 < groups; g0 += gpr) {
        const int g = g0 + static_cast<int>(threadIdx.x) / kLanesPerGroup;
        if (g >= groups) { continue; }
        const int base = g * group + (static_cast<int>(threadIdx.x) % kLanesPerGroup) * per_lane;
        auto* out_pairs = reinterpret_cast<__nv_bfloat162*>(row + base);
        float amax     = 0.0f;
        for (int p = 0; p < pairs; ++p) {
            const __nv_bfloat162 v = out_pairs[p];
            amax = fmaxf(amax, fabsf(__low2float(v)));
            amax = fmaxf(amax, fabsf(__high2float(v)));
        }
        for (int o = 1; o < kLanesPerGroup; o <<= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int p = 0; p < pairs; ++p) {
            const __nv_bfloat162 v = out_pairs[p];
            int q0                 = static_cast<int>(lrintf(__low2float(v) * inv));
            int q1                 = static_cast<int>(lrintf(__high2float(v) * inv));
            q0                     = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
            q1                     = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
            out_pairs[p]           = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                          static_cast<float>(q1) * scale);
        }
    }
}

} // namespace

bool act_quant_a8_probe_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_ACT_A8");
        return value != nullptr && value[0] != '\0';
    }();
    return enabled;
}

int act_quant_a8_group_size() {
    // NINFER_ACT_A8_TOKEN keeps the old per-token scale; NINFER_ACT_A8_GROUP overrides the group
    // width, which lets the accuracy sweep pick a width without a rebuild.
    static const int group = [] {
        const char* value = std::getenv("NINFER_ACT_A8");
        if (value == nullptr || value[0] == '\0') { return 0; }
        const char* token = std::getenv("NINFER_ACT_A8_TOKEN");
        if (token != nullptr && token[0] != '\0') { return 0; }
        const char* width = std::getenv("NINFER_ACT_A8_GROUP");
        if (width != nullptr && width[0] != '\0') {
            const int parsed = std::atoi(width);
            if (parsed >= 16 && parsed <= 512 && (parsed & 15) == 0) { return parsed; }
        }
        return 64;
    }();
    return group;
}

void act_quant_a8_roundtrip_bf16(void* data, std::int32_t k, std::int32_t tokens,
                                 cudaStream_t stream) {
    if (!act_quant_a8_probe_enabled() || data == nullptr || k <= 0 || tokens <= 0) { return; }
    const int group = act_quant_a8_group_size();
    if (group > 0 && (k % group) == 0) {
        a8_roundtrip_group_kernel<<<tokens, 512, 0, stream>>>(
            static_cast<__nv_bfloat16*>(data), k, group);
    } else {
        a8_roundtrip_kernel<<<tokens, 512, 0, stream>>>(static_cast<__nv_bfloat16*>(data), k);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
