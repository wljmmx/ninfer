#pragma once

#include "ops/op_check.h"

#include <limits>

namespace ninfer::test::gdn_control {

inline constexpr double kFp32UnitRoundoff = std::numeric_limits<float>::epsilon() / 2.0;

// Against the complete FP64 formula, expf contributes up to two ULP from the correctly
// rounded result and log1pf one ULP. Allow 16u for decay's addition, two exponentials,
// logarithm and multiplication; sigmoid's exponential, addition and reciprocal need 8u.
// Absolute floors cover small outputs, where relative error is poorly conditioned.
inline constexpr PointwiseCriterion kDecay{1.0e-7, 16.0 * kFp32UnitRoundoff};
inline constexpr PointwiseCriterion kSigmoid{1.0e-8, 8.0 * kFp32UnitRoundoff};

// The longest registered BF16 dot has K=5120, with about 320 FP32 MMA accumulation
// steps. gamma(320) supplies a 19.1 ppm accumulation-depth estimate; add decay's 16u
// epilogue budget. This finite gross cap is a numerical quality limit, not a worst-case
// bound for arbitrary dot-product conditioning. Keep the stricter 1.4 ppm relative L2
// criterion to reject distributed error. The whole projection family uses this budget.
inline constexpr double kProjectionAccumulationDepth = 5120.0 / 16.0;
inline constexpr double kProjectionAccumulationError =
    kProjectionAccumulationDepth * kFp32UnitRoundoff /
    (1.0 - kProjectionAccumulationDepth * kFp32UnitRoundoff);
inline constexpr ReductionCriterion kProjection{1.4e-6, 5.0e-7,
                                                kProjectionAccumulationError + kDecay.relative};

inline constexpr ReductionCriterion kNormalized{1.75e-3, 1.0e-4, 4.0e-3};

// The control budget belongs to the normalization route, not to a particular kernel.
//
// Fused route (norm scaled inside the projection): the dot accumulates from the represented
// activation in FP32, so the remaining error is the reduction budget above plus the epilogue.
// Measured relative L2 on the registered geometries is ~1e-7 for g and beta.
inline constexpr ReductionCriterion kNormControlFused{8.0e-4, 1.5e-4, 2.0e-3};

// Materialized route (RMSNorm writes the BF16 normalized parent, then the projection reads it):
// every term of every dot carries one extra BF16 storage rounding. For independent roundings the
// relative L2 of the dot stays within the storage unit roundoff, and the softplus/sigmoid
// epilogue passes it through. Measured worst cases on RTX 4090: K=5120 M=64/8192 zero-norm-gain
// g 5.7e-4, K=2560 M=16 zero-norm-gain g 8.35e-4, K=2560 M=1..8192 g 3.3e-4..5.1e-4, beta
// 8.2e-5..3.2e-4. The allowance below is 0.6 of the BF16 unit roundoff (2^-9 = 1.95e-3 for an
// 8-bit significand), which still rejects any mechanism error: dropping the (1+w) gain, a swapped
// head or a mis-scaled row moves relative L2 by two to three orders of magnitude more.
inline constexpr ReductionCriterion kNormControlMaterialized{1.2e-3, 1.5e-4, 3.0e-3};

// Mirrors gdn_control's own norm plan: only these two geometries fuse the normalization, every
// other (H,K,columns) materializes it. Keep the two in step.
inline constexpr bool fused_norm(int input_rows, int columns) {
    return (input_rows == 5120 && columns <= 42) || (input_rows == 2048 && columns <= 16);
}

inline constexpr ReductionCriterion norm_control_criterion(int input_rows, int columns) {
    return fused_norm(input_rows, columns) ? kNormControlFused : kNormControlMaterialized;
}

} // namespace ninfer::test::gdn_control
