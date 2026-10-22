#include "ninfer/ops/act_quant_a8.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdlib>

namespace ninfer::ops {
namespace {

__global__ void a8_roundtrip_kernel(__nv_bfloat16* __restrict__ data, int k) {
    const int t                     = blockIdx.x;
    __nv_bfloat16* __restrict__ row = data + static_cast<std::size_t>(t) * k;
    float amax                      = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(__bfloat162float(row[i])));
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
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        int q = static_cast<int>(lrintf(__bfloat162float(row[i]) * inv));
        q     = q < -127 ? -127 : (q > 127 ? 127 : q);
        row[i] = __float2bfloat16_rn(static_cast<float>(q) * scale);
    }
}

// One scale per `group` contiguous elements, matching the dense W4A8 quantiser (kG = 64) and the
// weight groups it multiplies. Sixteen lanes cooperate on one group, each taking a contiguous run,
// so a warp covers two groups as one coalesced pass.
__global__ void a8_roundtrip_group_kernel(__nv_bfloat16* __restrict__ data, int k, int group) {
    constexpr int kLanesPerGroup = 16;
    const int t                  = blockIdx.x;
    __nv_bfloat16* __restrict__ row = data + static_cast<std::size_t>(t) * k;
    const int groups = k / group;
    const int per_lane = group / kLanesPerGroup;
    const int gpr      = blockDim.x / kLanesPerGroup;
    for (int g0 = 0; g0 < groups; g0 += gpr) {
        const int g = g0 + static_cast<int>(threadIdx.x) / kLanesPerGroup;
        if (g >= groups) { continue; }
        const int base = g * group + (static_cast<int>(threadIdx.x) % kLanesPerGroup) * per_lane;
        float amax     = 0.0f;
        for (int i = 0; i < per_lane; ++i) {
            amax = fmaxf(amax, fabsf(__bfloat162float(row[base + i])));
        }
        for (int o = 1; o < kLanesPerGroup; o <<= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int i = 0; i < per_lane; ++i) {
            int q = static_cast<int>(lrintf(__bfloat162float(row[base + i]) * inv));
            q     = q < -127 ? -127 : (q > 127 ? 127 : q);
            row[base + i] = __float2bfloat16_rn(static_cast<float>(q) * scale);
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
