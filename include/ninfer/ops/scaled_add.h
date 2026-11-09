#pragma once

// ninfer::ops - per-token scaled accumulation.
//
// Qwen4Exp keeps the shared expert outside the routed banks and gates it with one per-token
// coefficient (the sigmoid of its own logit). The routed epilogue applies that coefficient inside
// its FP32 reduction; a shared expert run as its own Linear pair produces a BF16 matrix instead, so
// the layer needs the same "coefficient[t] * term[:,t]" applied before it is folded into the
// residual. This is that operation and nothing else.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: scaled_add
 *
 * Math / indexing:
 *   destination[d,t] = destination[d,t] + coefficient[t] * source[d,t]
 *
 * Logical shapes:
 *   source is contiguous BF16 [D,T]; destination is contiguous BF16 [D,T] with the same extents;
 *   coefficient is contiguous FP32 [T]. D and T are positive. The coefficient is applied as FP32;
 *   the sum is rounded once into the BF16 destination.
 *
 * Effects:
 *   Updates every element of destination in place. source and destination must not alias, and
 *   coefficient must not overlap either. No workspace and no persistent state.
 *
 * Capture/replay reads the current contents of the three operands.
 */
void scaled_add(const Tensor& source, const Tensor& coefficient, Tensor& destination,
                cudaStream_t stream);

} // namespace ninfer::ops
