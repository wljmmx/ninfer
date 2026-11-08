#pragma once

#include "core/tensor.h"
#include "core/device.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Applies RMSNorm over the fastest dimension D=ne[0]. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   gain[d]  = unit_offset ? 1 + weight[d] : weight[d]
 *   ideal[d,r] = x[d,r] * inv_r * gain[d].
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors, weight is contiguous BF16 [D], and eps
 * is positive and finite. Nonnegative empty shapes perform no device work. Nonempty backing is
 * naturally BF16-aligned. Input, weight, and output must not overlap. The oracle evaluates `ideal`
 * naively in FP64 from the represented inputs. The BF16 output is promoted and compared directly
 * with that result; output storage rounding belongs to the Op's numerical criterion, not the
 * oracle. Kernel reduction, staging, and accumulator precision are implementation choices. There
 * is no workspace or persistent state side effect.
 */
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream);

/**
 * Applies RMS normalization over ne[0] and an elementwise SiLU gate. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   ideal[d,r] = x[d,r] * inv_r * weight[d] * SiLU(z[d,r]).
 *
 * `x`, `z`, and `out` are same-shaped contiguous BF16 tensors with positive extents,
 * `weight` is contiguous BF16 [D], and eps is positive and finite. This form does not apply a
 * unit offset to weight. The output must not overlap any input; read-only inputs may share
 * backing. Inputs remain unchanged and the output is fully written. The oracle evaluates
 * `ideal` naively in FP64 from the represented inputs. The BF16 output is promoted and compared
 * directly with that result; output storage rounding belongs to the Op's numerical criterion,
 * not the oracle. Kernel reduction, staging,
 * and accumulator precision are implementation choices. There is no workspace or persistent state
 * side effect. `execution` supplies the stream and the selected device's positive physical SM
 * count for launch selection.
 */
void silu_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                        Tensor& out, DeviceExecutionView execution);

/**
 * Applies RMS normalization over ne[0] and an elementwise sigmoid gate. For each logical row r:
 *
 *   inv_r        = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   ideal[d,r]   = x[d,r] * inv_r * weight[d] * sigmoid(z[d,r]).
 *
 * Shapes, direct weight multiplier, effects, non-overlap, zero-workspace and execution rules
 * are the same as silu_gated_rmsnorm. Every positive D and remaining extent is supported within
 * the view's represented range and available storage. In particular, BF16 [128,48,T,B] uses
 * independent 128-element rows and a shared BF16 weight [128]. Norm and gate form one complete
 * mathematical operation: there is no public BF16 norm intermediate. The independent FP64 oracle
 * evaluates the full formula from represented inputs; the BF16 output is compared directly with
 * that result under the suite's numerical criterion, without reproducing private casts.
 */
void sigmoid_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                           Tensor& out, DeviceExecutionView execution);

} // namespace ninfer::ops
