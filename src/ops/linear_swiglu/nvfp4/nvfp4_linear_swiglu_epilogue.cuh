#pragma once

// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/common/math.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
template <int RowsPerBranch>
struct Nvfp4SwiGluRows {
    static constexpr bool kPaired = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int rows) const {
        return begin + row % RowsPerBranch + (row >= RowsPerBranch ? rows / 2 : 0);
    }
};

struct Nvfp4SwiGluEpilogue {
    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, silu(gate) * up);
    }
};
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
