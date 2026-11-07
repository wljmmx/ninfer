#pragma once

#include "core/device.h"
#include "core/paged_kv_cache.h"
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

struct QsaBatchView {
    Tensor positions;     // I32 [T,B], original-token ordinals.
    Tensor valid_columns; // I32 [B].
    Tensor table_rows;    // I32 [B].
};

struct QsaProjectionView {
    Tensor query;       // BF16 [256,24,T,B].
    Tensor key;         // BF16 [256,2,T,B].
    Tensor value;       // BF16 [256,2,T,B].
    Tensor index_query; // BF16 [128,4,T,B].
    Tensor index_key;   // BF16 [128,T,B], raw projection.
};

struct QsaPrepareParameters {
    Tensor query_norm;       // BF16 [256], offset weights: gain = 1 + FP32(weight).
    Tensor key_norm;         // BF16 [256].
    Tensor index_query_norm; // BF16 [128].
    Tensor index_key_norm;   // BF16 [128].
    float rms_epsilon = 1e-6F;
    float rope_theta  = 1e7F;
};

struct QsaIndexCacheView {
    Tensor keys;         // BF16 [128,16,1,G], page-major.
    Tensor block_tables; // I32 [logical_pages,table_rows], identical to main.block_tables.
};

struct QsaCacheView {
    PagedKVBatchLayerView main;
    QsaIndexCacheView index;
};

struct QsaRawRecordWriteView {
    Tensor raw_keys;       // BF16 [128,R,B].
    Tensor rope_positions; // I32 [3,R,B].
    Tensor offsets;        // I32 [B], first record column written by this call.
};

struct QsaBlockSelection {
    Tensor block_ids;    // Device I32 [512,T,B], logical four-token block ids.
    Tensor block_counts; // Device I32 [T,B].
};

struct QsaIndexerExecutionEnvelope {
    std::uint32_t max_visible_tokens = 0; // Fixed capture capacity, in original tokens.
};

/** Capacity over every positive T in [min_tokens,max_tokens], with fixed B and envelope.
 * Includes only Op-local scratch, measured from a 256-byte-aligned workspace cursor.
 */
[[nodiscard]] std::size_t qsa_indexer_workspace_capacity_bytes(std::int32_t min_tokens,
                                                               std::int32_t max_tokens,
                                                               std::int32_t batch_size,
                                                               QsaIndexerExecutionEnvelope envelope,
                                                               DeviceExecutionView execution);

/**
 * Op: qsa_indexer
 *
 * Math / indexing:
 * For each live (t,b), p=positions[t,b], C=floor((p+1)/4), and g in [0,C):
 *   score[g] = sum_{h=0..3} max(0, sum_{d=0..127} index_query[d,h,t,b]*index.K[d,g]).
 * The common positive 1/sqrt(128) factor is omitted because only ranking is observable.
 * Select min(512,C) highest computed scores; exact numeric ties prefer smaller logical g.
 * Index g addresses block_tables[g/16,table_rows[b]], row g%16 of the index plane.
 * Four heads produce one shared selection. The incomplete token tail is not included here.
 * C<=512 selects every g without reading Q, keys or page-table contents.
 *
 * Domain / numeric:
 * T>0, B=1..8. index_query is contiguous, 16-byte-aligned BF16 [128,4,T,B]. Cache keys
 * are naturally aligned BF16 [128,16,1,G], feature-contiguous with positive, non-overlapping
 * page-major outer strides. Tables have contiguous I32 entries and non-overlapping row strides.
 * Positions are contiguous I32 [T,B]; valid_columns and table_rows are Device I32 [B].
 * Each row has 0<=n<=T; live positions are consecutive nonnegative original-token ordinals.
 * Every live p+1 is <= envelope.max_visible_tokens <=262144, within the table capacity.
 * Envelope zero permits only inert queries. Live scoring operands and their FP32 scores are
 * finite. BF16 values enter FP32 accumulation directly; scores have no BF16 rounding boundary.
 * Mathematical scores and selection boundaries are independently qualified against FP64.
 * Device counts, positions, selectors and readable live pages are caller guarantees.
 *
 * Effects / output:
 * Outputs are naturally aligned, contiguous I32 with the shapes above. Each live count is
 * min(512,C); its first count ids are distinct and in [0,C), in unspecified order. Inert counts
 * are zero and do not read positions, Q, keys or table selectors. Ids beyond count are undefined.
 * Inputs and caches are preserved. Outputs, inputs and used workspace must not write/read
 * overlap; read-only views may share storage. There is no persistent state or publication.
 *
 * Workspace / execution:
 * Scratch comes from the caller's arena and is reused across bounded query windows, without
 * hidden allocation. Fixed T/B, views, envelope and device facts determine the launch topology.
 * Actual per-query lengths, mixed batch contexts and all-select/inert cases are handled inside
 * the same kernels, including Graph replay. Metadata is not read back by the Host. Downstream
 * consumers read only count ids; same-stream ordering supplies the dependency.
 */
void qsa_indexer(const Tensor& index_query, const QsaIndexCacheView& cache,
                 const QsaBatchView& batch, QsaIndexerExecutionEnvelope envelope,
                 WorkspaceArena& workspace, QsaBlockSelection& out, DeviceExecutionView execution);

/**
 * Op: qsa_prepare
 *
 * Math / indexing:
 * N_D(x,w) = x * (1+FP32(w)) / sqrt(sum(x*x)/D + rms_epsilon), independently per head.
 * R(z,rho) rotates pairs (i,i+32), i in [0,32), by rho[axis(i)]*theta^(-2*i/64);
 * axis(i)=i%3 on canonical three-axis rho. Current A=1 coordinates broadcast to three axes.
 * R[i]=z[i]*cos-z[i+32]*sin,
 * R[i+32]=z[i+32]*cos+z[i]*sin; dimensions [64,D) retain z.
 * For each row b and t<valid_columns[b], p=positions[t,b]:
 *   query[:,h,t,b]       = BF16(R(N_256(projected.query[:,h,t,b],query_norm),rho[t,:,b]));
 *   index_query[:,h,t,b] = BF16(R(N_128(projected.index_query[:,h,t,b],index_query_norm),rho));
 *   k                   = R(N_256(projected.key[:,h,t,b],key_norm),rho);
 *   v                   = projected.value[:,h,t,b].
 * BFloat16 main caches store BF16(k) and FP16_RNE(v). For quantized main caches, k enters
 * the fixed FP32 H256/16 Hadamard and storage encoder directly from norm/RoPE, with no
 * intermediate BF16 rounding. INT8-G64 and FP8 transform K only; NVFP4-G16 and K8V4 transform
 * both K and V. V enters from the represented BF16 projection. Codes, represented scales,
 * zero and saturation rules are those of kv_cache_append; its BF16 K input boundary does not
 * apply to this fused computation. Q/index-Q and the compressed index cache remain BF16 and
 * do not receive this main-cache Hadamard. The attention consumer owns the corresponding Q
 * transform and any value-basis inversion.
 * Raw index-key words and actual coordinates are copied to records[:,offsets[b]+t,b];
 * A=1 replicates the coordinate to all three record axes.
 * With F=positions[0,b] and n=valid_columns[b], each newly completed block
 * g in [floor(F/4),floor((F+n)/4)) writes:
 *   u = BF16(mean(raw_index_key[4*g:4*g+4])); index.K[g] =
 * BF16(R(N_128(u,index_key_norm),rho[4*g])). Raw keys preceding F come from before's canonical tail
 * at F; other keys come from projected. A block starting before F uses before.first_rope; otherwise
 * it uses this call's coordinates. Main p addresses table[p/64,table_rows[b]], row p%64; index g
 * uses the same table at g/16, row g%16. Only complete blocks are written. No causal
 * visibility/frontier is published.
 *
 * Logical shapes / supported domain:
 * T>0, B in [1,8], A=1 or 3. Projections and both outputs have the shapes above, are contiguous
 * and 16-byte aligned. Norm weights are contiguous, naturally aligned BF16 vectors. Positions
 * are contiguous I32 [T,B]; rho is contiguous I32 [T,A,B]. Metadata is contiguous Device I32 [B].
 * Tail/records follow QsaTailStateView's feature-contiguous, naturally aligned stride rules.
 * Main-cache data and scale planes:
 *   BFloat16: BF16 K / FP16 V [256,64,2,G]; no scales.
 *   Int8Group64: I8 K/V [256,64,2,G]; FP16 scales [4,64,2,G].
 *   Fp8E4M3Row256: FP8_E4M3FN K/V [256,64,2,G]; FP16 scales [1,64,2,G].
 *   Nvfp4Group16: packed U8 K/V [128,64,2,G]; U8 E4M3 scales [16,64,2,G].
 *   Fp8KeyNvfp4Value: the FP8 key planes and NVFP4 value planes above.
 * Cache planes are page-major with contiguous features and non-overlapping positive outer
 * strides; tables have contiguous I32 page entries and a non-overlapping row stride.
 * Both caches share G and the identical block_tables view. R and the state pool size are positive.
 * Every row has 0<=n<=T and 0<=offset<=offset+n<=R; R may be smaller than T.
 * Live positions are consecutive, nonnegative ordinals below 262144. before corresponds to F.
 * RoPE coordinates are independent nonnegative I32 values. Epsilon/theta are positive finite
 * floats; live projection/norm values are finite. The caller provides valid live slots/page
 * mappings and disjoint writes across rows. Host validation does not read Device metadata.
 *
 * Numeric / effects:
 * Norm and rotation have no intermediate represented tensor. The pooled mean's BF16 rounding
 * is semantic; raw records preserve input words exactly. Q/IQ/index keys and BFloat16-cache K
 * are BF16 approximations to the complete formula, independently qualified against a
 * high-precision oracle. Quantized K includes the storage encoder's representation error;
 * V uses its profile's exact persistent encoding. Q/IQ for t>=n are exactly BF16 +0; inert columns
 * do not read projections/positions/state or write cache/records. An empty row need not have valid
 * slot/table selectors. before, metadata, page tables and untouched cache/record regions remain
 * unchanged. query may exactly alias projected.query, and index_query may independently exactly
 * alias projected.index_query (same address/shape/strides). Partial overlap and all other
 * write/read overlap are forbidden. Device-selected state/cache/record alias relationships are
 * caller guarantees, including across different views of an interleaved allocation.
 *
 * Workspace / execution:
 * Zero global scratch, no sizing query or hidden allocation. execution supplies the stream and
 * positive SM count. CUDA Graph replay reads current Device metadata; callers own operand
 * readiness/lifetime, allocation, tail extraction, acceptance, and state publication.
 */
void qsa_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const Tensor& rope_positions,
                 const QsaTailStateView& before, QsaCacheView cache, QsaRawRecordWriteView records,
                 Tensor& query, Tensor& index_query, DeviceExecutionView execution);

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
