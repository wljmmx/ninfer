#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
// port (sm_89): this tree keeps the projection weight types in weight_input.h; the upstream
// bench tree splits them into projection_weights.h.
#include "ninfer/ops/weight_input.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct GdnControlOutputs {
    Tensor g, beta;
};

struct GdnNormControlOutputs {
    Tensor normalized;
    GdnControlOutputs control;
};

/**
 * Op: GDN control preparation.
 *
 * Math / indexing:
 *   g[h,m] = -exp(a_log[h]) * softplus(a[h,m] + dt_bias[h])
 *   beta[h,m] = sigmoid(b[h,m]).
 *
 * Supported domain:
 *   Contiguous BF16 a/b [48,M], FP32 a_log/dt_bias [48], FP32 g/beta [48,M],
 *   with backing naturally aligned to its element type.
 *   M is nonnegative; an empty matrix performs no device work.
 *
 * Numeric / effects:
 *   The complete formula is evaluated over the represented inputs. FP32 outputs are compared
 *   directly with an independent FP64 oracle. All inputs and outputs must be disjoint;
 *   inputs remain unchanged and both outputs are fully written. No persistent state or workspace.
 *
 * Execution:
 *   Work is submitted to stream; capture/replay observes the current operand contents.
 */
void gdn_control(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias,
                 GdnControlOutputs out, cudaStream_t stream);

/**
 * Op: GDN control projection.
 *
 * Math / indexing:
 *   a[h,m] = sum_k Wa[h,k] * x[k,m]
 *   b[h,m] = sum_k Wb[h,k] * x[k,m]
 *   g[h,m] = -exp(a_log[h]) * softplus(a[h,m] + dt_bias[h])
 *   beta[h,m] = sigmoid(b[h,m]).
 *
 * Supported domain:
 *   x is contiguous BF16 [K,M]; a_log/dt_bias are FP32 [H]; outputs are FP32 [H,M].
 *   Combined contiguous BF16 parents [2H,K] admit (H,K)=(48,5120),(32,2048),(48,2560).
 *   Rows [0,H) are A and [H,2H) are B. Paired BF16 parents [H,K] admit (48,5120).
 *   BF16 x and weight backing must be 16-byte aligned; FP32 backing must be 4-byte aligned.
 *   Every valid activation permission admits this BF16 route. Every positive M is supported
 *   within represented views and available storage. A [K,T,B] block is presented as [K,M],
 *   M=T*B, column=t+T*b.
 *
 * Numeric / effects:
 *   The complete formula consumes represented x directly; there is no input normalization or
 *   public BF16 projection intermediate. FP32 outputs are compared directly with an independent
 *   FP64 oracle. Private reduction, staging and accumulator precision are implementation choices.
 *   All inputs, outputs and allocated scratch are disjoint. Inputs remain unchanged; both outputs
 *   are fully written. Combined weights produce zero-copy row views, without runtime repacking.
 *
 * Workspace / execution:
 *   Caller-owned transient workspace is scoped to the call. The capacity query covers every M in
 *   an inclusive interval and rejects invalid profiles/intervals. It inspects metadata without
 *   requiring uploaded backing. execution supplies stream and the physical positive SM count;
 *   device facts may change launch decomposition, never mathematical behavior. Capture/replay
 *   reads current contents at the bound addresses. There is no persistent state effect.
 */
void gdn_control_proj(const Tensor& x, const ProjectionWeights& weights, const Tensor& a_log,
                      const Tensor& dt_bias, GdnControlOutputs out, WorkspaceArena& workspace,
                      DeviceExecutionView execution);

[[nodiscard]] std::size_t
gdn_control_proj_workspace_capacity_bytes(const ProjectionWeights& weights, std::int32_t first,
                                          std::int32_t last);

/**
 * Op: input RMSNorm and GDN control projection.
 *
 * Math / indexing:
 *   n[k,m] = x[k,m] / sqrt(mean_j(x[j,m]^2) + eps) * (1 + norm_weight[k])
 *   normalized_ideal[k,m] = n[k,m]
 *   a[h,m] = sum_k Wa[h,k] * n[k,m]; b[h,m] = sum_k Wb[h,k] * n[k,m]
 *   g[h,m] = -exp(a_log[h]) * softplus(a[h,m] + dt_bias[h])
 *   beta[h,m] = sigmoid(b[h,m]).
 *
 * Supported domain:
 *   Projection operands/effects follow gdn_control_proj for (H,K)=(48,5120),(32,2048),(48,2560).
 *   norm_weight is contiguous BF16 [K], normalized is BF16 [K,M]; both are 16-byte aligned.
 *   eps is positive and finite.
 *   (H,K)=(48,2560) has no fused form: its forty K tiles admit no whole-K split-32 normalizing
 *   MMA, and the fused SIMT kernel is compiled for K=5120, so that geometry materializes the
 *   normalization (one RMSNorm writing normalized) and then runs its projection route. Fusing and
 *   materializing remain implementation choices for the other geometries, as before.
 *
 * Numeric:
 *   normalized is the explicit BF16 output for other projections. Promoted normalized and FP32
 *   controls are compared directly with the complete independent FP64 formula. The oracle does
 *   not reproduce output storage rounding. Controls derive from n, without a required rounding
 *   through normalized. Private operand staging and fusion/materialization are implementation
 *   choices. The additional norm input and output obey the same non-overlap and preservation
 *   rules. Workspace and capture/replay follow the projection contract above.
 */
void gdn_norm_control_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                           const ProjectionWeights& weights, const Tensor& a_log,
                           const Tensor& dt_bias, GdnNormControlOutputs out,
                           WorkspaceArena& workspace, DeviceExecutionView execution);

[[nodiscard]] std::size_t
gdn_norm_control_proj_workspace_capacity_bytes(const ProjectionWeights& weights, std::int32_t first,
                                               std::int32_t last);

} // namespace ninfer::ops
