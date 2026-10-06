#pragma once

// Exact E4M3-to-BF16 operand widening shared by the small-T and GEMM A16 routes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops::detail {

__device__ __forceinline__ unsigned fp8_e4m3x2_to_bf16x2_bits(unsigned packed) {
    // Official CUDA FP8 conversion intrinsic (cuda_fp8.h, CUDA 13.4). On sm_100+ family
    // targets it lowers to the native cvt.rn.bf16x2.e4m3x2 instruction; on every other
    // architecture, including the Ada (sm_89) target, CUDA 13.4's own architecture
    // dispatch (__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__ in cuda_fp8.h)
    // selects the toolkit's emulation path. Every E4M3 value is exactly representable
    // in BF16, so both paths produce identical bits.
    const __nv_bfloat162_raw pair =
        __nv_cvt_fp8x2_to_bf162raw(static_cast<__nv_fp8x2_storage_t>(packed), __NV_E4M3);
    return static_cast<unsigned>(pair.x) | (static_cast<unsigned>(pair.y) << 16U);
}

__device__ __forceinline__ int fp8_a16_shared_col_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

} // namespace ninfer::ops::detail
