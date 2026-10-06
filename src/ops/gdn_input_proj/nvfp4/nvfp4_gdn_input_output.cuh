#pragma once

// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
using Nvfp4GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
