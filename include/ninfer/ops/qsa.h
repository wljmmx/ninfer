#pragma once

#include "core/device.h"
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

// Non-owning views of one layer. Dimension zero is contiguous; outer strides may retain
// pool/layer gaps or interleaving. BF16/I32 bases and used strides are naturally aligned,
// positive, and non-overlapping. All tensors reside on Device; metadata is contiguous I32 [B].
struct QsaTailStateView {
    Tensor raw_keys;   // BF16 [128,3,S].
    Tensor first_rope; // I32 [3,S].
    Tensor slots;      // I32 [B], independently selects this pool's state slots.
};

struct QsaRawRecordReadView {
    Tensor raw_keys;       // BF16 [128,R,B], actual raw index-key projection words.
    Tensor rope_positions; // I32 [3,R,B], actual three-axis coordinates.
    Tensor base_positions; // I32 [B], F0: record column zero's input position.
    Tensor valid_columns;  // I32 [B], u: available record prefix.
};

/**
 * Op: qsa_tail_extract
 *
 * Math / indexing:
 * For row b, let F0=records.base_positions[b], a=accepted_columns[b], F1=F0+a,
 * r=F1%4, s=F1-r, src=before.slots[b], dst=after.slots[b]. For every feature c and j in [0,3):
 *   after.raw_keys[c,j,dst] = BF16 +0                              if j>=r,
 *                            before.raw_keys[c,s+j-(F0-F0%4),src] if s+j<F0,
 *                            records.raw_keys[c,s+j-F0,b]         otherwise.
 *   after.first_rope[:,dst] = [0,0,0]                         if r==0,
 *                            before.first_rope[:,src]        if s<F0,
 *                            records.rope_positions[:,s-F0,b] otherwise.
 * before is the canonical state at F0: its first F0%4 raw columns contain the unfinished
 * four-token block in chronological order, the other columns are BF16 +0; first_rope holds
 * that block's first token coordinates, or three I32 zeros when F0%4==0.
 *
 * Logical shapes / supported domain:
 * View rules above apply. B is in [1,8]; R and both pools' S are positive. Each state's two
 * planes agree on S, both record planes agree on R/B, and before/after may have different S.
 * accepted_columns is contiguous Device I32 [B]. Each row has 0<=a<=u<=R, F0>=0, and F0+u
 * representable as nonnegative I32. The caller supplies the input-prefix length, not an emitted
 * token count, and ensures that before corresponds to F0. Source/destination slots are valid
 * even for a=0. Host-visible layouts are checked without reading Device metadata to Host;
 * count/index values and selected-region alias relationships are caller guarantees.
 *
 * Numeric / effects:
 * Exact BF16 word/I32 copying and zero filling: copied bits, including zero signs and NaN
 * payloads, are preserved; no finite-value requirement or floating-point evaluation applies.
 * Only the selected destination's three raw columns and three coordinates are written.
 * Unselected state, records and metadata are preserved. Only record columns needed from [0,a)
 * are read; no rejected suffix or earlier completed block is scanned. a=0 copies the canonical
 * source to a distinct destination and preserves every state bit when in-place.
 * Destination regions are mutually disjoint and cannot overlap another row's source or any
 * records/metadata. Read-only sources may be shared. A destination plane may exactly alias its
 * corresponding source plane with identical addressed layout; other write/read overlap is
 * forbidden, including between raw and coordinate planes. Aliasing is defined by selected
 * addresses, not pool descriptor identity or slot numbers. In-place results use original source
 * words. No cache, count, frontier or state-publication effect occurs.
 *
 * Workspace / execution:
 * No caller-owned global scratch, sizing query, or hidden Device allocation. Work is submitted
 * to execution.stream, with a positive execution.multiprocessor_count. CUDA Graph replay reads
 * the current Device metadata. The caller owns operand readiness and storage lifetime.
 */
void qsa_tail_extract(const QsaTailStateView& before, const QsaRawRecordReadView& records,
                      const Tensor& accepted_columns, QsaTailStateView after,
                      DeviceExecutionView execution);

} // namespace ninfer::ops
