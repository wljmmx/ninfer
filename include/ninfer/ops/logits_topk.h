#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// How many candidates one column may ask for. Selection costs one vocabulary scan per candidate, so
// the cap bounds the output size and the pass count together.
inline constexpr std::int32_t kLogitsTopkCapacity = 64;

/**
 * Op: Per-column top-K logits, their log-probabilities, and the column log-normalizer
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c], and
 *
 *     m[c] = max_{r<valid_rows} l[r,c],
 *     Z[c] = log(sum_{r=0..valid_rows-1} exp(l[r,c])).
 *
 *   For every column c the output holds the k largest candidates ordered by l descending with the
 *   lower token id breaking ties. For rank k' in [0,k):
 *
 *     ids[c,k']         = r  of the k'-th candidate,   stored at index k' + c*k
 *     logprobs[c,k']    = l[r,c] - m[c] - log(sum_r exp(l[r,c]-m[c])), stored at index k' + c*k
 *     log_normalizer[c] = Z[c].
 *
 *   log_normalizer is the same normalizer target_logprobs subtracts, so for a column whose argmax
 * is r, logprobs[c,0] equals that op's target logprob of r.
 *
 * Logical shapes:
 *   logits is [physical_rows,C] BF16, ids is [k,C] I32, logprobs is [k,C] FP32, and log_normalizer
 *   is [C] FP32, with C>0, 1<=valid_rows<=physical_rows and
 *   1<=k<=min(valid_rows,kLogitsTopkCapacity). Columns are the last dimension throughout, so the
 *   candidates of column c occupy the contiguous block [c*k,(c+1)*k) of ids and logprobs. Physical
 *   rows [valid_rows,physical_rows) participate in neither the ordering nor the normalizer.
 *
 * Supported domain:
 *   logits is contiguous finite BF16, ids is contiguous I32, logprobs and log_normalizer are
 *   contiguous FP32. Storage has its dtype's natural alignment.
 *
 * Numeric:
 *   The ordering is exact: candidates are compared as packed (value,id) keys, never approximated.
 *   log_normalizer and every logprob use the two-pass max-then-shifted-sum evaluation with the same
 *   reduction association target_logprobs uses (one CTA of 256 threads per column, one thread per
 *   strided row), so the two ops agree bit for bit on the same logits column.
 *
 * Effects:
 *   Writes every output element and preserves logits. No output may overlap logits or another
 *   output.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   One CTA per column; no persistent state. The kernel scans the column k+1 times.
 */
void logits_topk(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& ids,
                 Tensor& logprobs, Tensor& log_normalizer, cudaStream_t stream);

} // namespace ninfer::ops
