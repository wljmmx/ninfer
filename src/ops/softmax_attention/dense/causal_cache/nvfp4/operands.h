#pragma once

// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/softmax_attention/common/causal_operands.h"

namespace ninfer::ops::detail {
template <bool Writable>
using Nvfp4KvCacheView =
    QuantizedCausalCacheView<std::uint8_t, std::uint8_t, std::uint8_t, Writable>;
using Nvfp4KvReadView = Nvfp4KvCacheView<false>;

} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
