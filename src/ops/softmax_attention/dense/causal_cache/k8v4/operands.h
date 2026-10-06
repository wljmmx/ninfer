#pragma once

// K8V4 stores its value plane as NVFP4 group16; the codec runs through official CUDA 13
// conversion intrinsics on every architecture, so this path is available on Ada (sm_89)
// as well as Blackwell.
#if defined(NINFER_ENABLE_K8V4)
#include "ops/softmax_attention/common/causal_operands.h"

namespace ninfer::ops::detail {
template <bool Writable>
using K8V4KvCacheView = QuantizedCausalCacheView<std::uint8_t, __half, std::uint8_t, Writable>;
using K8V4KvReadView  = K8V4KvCacheView<false>;

} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_K8V4
