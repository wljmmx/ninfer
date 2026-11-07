#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: qsa_input_proj
 *
 * Math / indexing:
 * Each output s is an independent contiguous row segment of one matrix projection:
 *   ideal_s[r,m] = sum_k logical_weight[begin_s+r,k] * BF16_value(x[k,m]).
 * There is no bias, normalization, RoPE, or gate activation. The parent is already in the
 * following row order; the Op does not apply an additional head permutation:
 *
 *   Q8 [12288,2560]:  query [0,6144), gate [6144,12288).
 *   BF16 [1664,2560]: key [0,512), value [512,1024), index_query [1024,1536),
 *                     index_key [1536,1664).
 *   BF16 [13952,2560]: query [0,6144), key [6144,6656), gate [6656,12800),
 *                      value [12800,13312), index_query [13312,13824),
 *                      index_key [13824,13952).
 *
 * Logical shapes / supported domain:
 * x is BF16 [2560,M]. Each output is BF16 [segment_rows,M]. All activation tensors are
 * contiguous, non-null, and 16-byte aligned. M is any positive column extent representable
 * by the tensors and available storage. For a layer batch, M=T*B; the caller may view the
 * continuous segments as [256,24,T,B], [256,2,T,B], [128,4,T,B], and [128,T,B].
 * The two-output parent is Q8_G32_FP16 RowSplit with G32 FP16 scales. The four- and six-output
 * parents are BF16 Contiguous. Logical and padded weight shapes are equal to the registered
 * [N,2560] shapes above. Weight code/data planes and Q8 scales are non-null and 16-byte aligned.
 * Weight inputs are native Weight views; each call admits only its corresponding parent form.
 *
 * Numeric:
 * All routes use A16 compute and write BF16 outputs. The independent oracle decodes the logical
 * weight and evaluates every complete dot product naively in FP64 from represented inputs.
 * BF16 outputs are promoted and compared to the unrounded ideal result under the Op's A16
 * criterion. Private dequantization, accumulation, and reduction order are implementation
 * choices. Every physical input column is computed; masked layer inputs supply finite safe
 * values before invoking this Op.
 *
 * Effects:
 * All outputs are fully written and mutually disjoint. They overlap neither x nor any weight
 * plane. Inputs and weights are preserved. Outputs may occupy disjoint slices of one allocation.
 * There is no state, cache, metadata, frontier, or allocation side effect.
 *
 * Workspace / execution:
 * No caller-owned global scratch or hidden Device allocation. Results are written directly
 * to the supplied segments. Work is submitted to stream and supports CUDA Graph capture/replay;
 * operand readiness and lifetime through the last consumer belong to the caller.
 */
void qsa_input_proj(const Tensor& x, const Weight& query_gate_weight, Tensor& query, Tensor& gate,
                    cudaStream_t stream);

void qsa_input_proj(const Tensor& x, const Weight& kv_index_weight, Tensor& key, Tensor& value,
                    Tensor& index_query, Tensor& index_key, cudaStream_t stream);

void qsa_input_proj(const Tensor& x, const Weight& joint_weight, Tensor& query, Tensor& key,
                    Tensor& gate, Tensor& value, Tensor& index_query, Tensor& index_key,
                    cudaStream_t stream);

} // namespace ninfer::ops
