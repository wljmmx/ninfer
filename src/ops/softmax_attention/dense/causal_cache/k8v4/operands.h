#pragma once

// K8V4 stores its value plane as NVFP4 group16 and therefore shares the Blackwell-only
// E2M1 hardware codec (quantize pack and scaled decode). This translation unit is
// compiled out on non-Blackwell architectures; dispatch rejects K8V4 KV there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/softmax_attention/common/causal_operands.h"

namespace ninfer::ops::detail {
template <bool Writable>
using K8V4KvCacheView = QuantizedCausalCacheView<std::uint8_t, __half, std::uint8_t, Writable>;
using K8V4KvReadView  = K8V4KvCacheView<false>;

} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
