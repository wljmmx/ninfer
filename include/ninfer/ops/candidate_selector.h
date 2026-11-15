#pragma once

#include "core/tensor.h"
#include "core/arena.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Capacity for every K/B pair in the inclusive intervals, K=1..15 and B=1..8.
[[nodiscard]] std::size_t candidate_selector_path_workspace_capacity_bytes(std::int32_t min_steps,
                                                                           std::int32_t max_steps,
                                                                           std::int32_t min_batch,
                                                                           std::int32_t max_batch);

/**
 * Op: candidate_selector_path
 *
 * For K in [1,15] and B in [1,8], the inputs are contiguous candidate_ids I32 [16,K,B],
 * unary_scores FP32 [16,K,B], projected_hidden BF16 [256,K,B], anchors I32 [B],
 * predecessor_codebook and successor_codebook BF16 [256,248320], base_positions I32 [B], and a
 * device-resident SamplingConfig[B]. Candidate rank is the fastest axis. The 16 candidate ids in
 * each row are distinct, and all candidate and anchor token ids lie in [0,248077); the registered
 * vocabulary, artifact binding, and linear_topk producer establish that trusted value contract.
 *
 * Starting with predecessor=anchors[b], each position i in [0,K) computes:
 *
 *   edge[c] = unary_scores[c,i,b]
 *           + sum_r predecessor_codebook[r,predecessor]
 *                   * projected_hidden[r,i,b]
 *                   * successor_codebook[r,candidate_ids[c,i,b]].
 *
 * A row with configs[b].temperature<=0 selects the lowest candidate rank attaining max(edge) and
 * writes its exact one-hot distribution. A positive-temperature row writes the FP32 softmax of
 * edge/temperature, then draws a candidate with counter key
 * (configs[b].seed,base_positions[b]+i,kSamplePurposeDFlash2Proposal). The selected global id is
 * written to drafts[i,b] and becomes the next predecessor. The Op ignores all other
 * SamplingConfig fields and never updates token_counts.
 *
 * drafts is contiguous I32 [K,B] and proposal_q is contiguous FP32 [16,K,B]. Both outputs are
 * completely overwritten. Inputs, outputs, codebooks, and the config array must be pairwise
 * non-overlapping. The Op has no persistent state or internal allocation. Caller workspace is
 * transient and must not overlap any input or output.
 *
 * Probe, off unless NINFER_SELECTOR_EDGE_PROBE is set: a lattice-route launch then also keeps its
 * score lattice in a 120 KiB device buffer owned by the Op, which candidate_selector_edge_probe
 * reads back. A direct-route launch allocates nothing and clears what a previous lattice launch left.
 */
void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                             cudaStream_t stream);

/**
 * The probe behind NINFER_SELECTOR_EDGE_PROBE. Returns the last lattice-route launch's scores, one
 * [16 candidates, 16 predecessor ranks] block per column, flat as
 * ((column * 16 + predecessor_rank) * 16 + candidate_rank), which is the kernel's own indexing.
 * `columns` is steps * batch and `batch` the launch's batch. Returns nullptr, with `columns` 0, when
 * the probe is off or the last launch took the direct route (K<=4), so a caller never reads a stale
 * lattice as if it belonged to the round it just ran.
 */
const float* candidate_selector_edge_probe(std::int32_t& columns, std::int32_t& batch);

} // namespace ninfer::ops
