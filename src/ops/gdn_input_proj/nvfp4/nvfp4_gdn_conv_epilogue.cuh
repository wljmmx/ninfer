#pragma once

// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/gdn_input_proj/gdn_conv_output.cuh"

namespace ninfer::ops::detail {
template <int Tokens>
struct Nvfp4GdnConvEpilogue {
    [[maybe_unused]] static constexpr int kRowTokens = Tokens;

    template <class Output>
    __device__ __forceinline__ void apply_row(Output output, int row, int,
                                              const float (&values)[Tokens], int) const {
        output.store_row(row, values);
    }
};
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
