#pragma once

// K8V4 stores its value plane as NVFP4 group16; the codec runs through official CUDA 13
// conversion intrinsics on every architecture, so this path is available on Ada (sm_89)
// as well as Blackwell.
#if defined(NINFER_ENABLE_K8V4)
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/k8v4/operands.h"

namespace ninfer::ops::detail {
struct K8V4KvTiledValues {
    using Scale                      = std::uint8_t;
    static constexpr int kCodeBytes  = 128;
    static constexpr int kScaleItems = 16;

    __device__ __forceinline__ static int4 expand(const std::uint8_t* codes, Scale scale) {
        return kv_cache_nvfp4_dequant_f16x8(codes, scale);
    }
};
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_K8V4
