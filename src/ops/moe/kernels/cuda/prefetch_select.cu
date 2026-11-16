// port (sm_89), P2.2: the prediction region's placement, decided on the Device.
//
// Until this kernel existed, the region's fill was a host loop: `stage_predicted` walked a vector of
// ids the host had, asked the pool "is this one resident", and called set_prefetch_experts() with the
// survivors. On the Device form of the prediction hand-off (`set_predicted_device`) the ids never
// reach the host at all, so that loop could not run - the Device form resolved the ids against the
// region mirror, counted them, and placed nothing. Measured: `[pf-region] hits=0 calls=640 (0.0%)`.
//
// This is the selection half of that missing consumer, and it is the half that decides:
//
//   * which predicted ids are worth placing - the pool's own placement table says "already has a
//     slot", and a prediction that is already resident is not a prediction;
//   * which region slot each one lands in - the ring's cursor is a Device word, so the sequence of
//     placements is a Device fact and the host cannot re-derive it (and does not have to: it receives
//     the pairs this kernel writes).
//
// The bytes are NOT copied here. The only Device-addressable source of an expert's image is the pinned
// host tier, and an expert the tier holds is already addressed by the demand path without any copy
// (`kExpertPlacementHost`); the artifact - the source that actually matters for a cold expert - is a
// host mapping that only the host-side uploader can DMA out of. So the copy stays where the pool's
// upload path is (prepare/progress), and what moved to the Device is the decision it implements.
//
// The output is deliberately in the region's own space, not in pool slots: `pairs[i] = {expert,
// region}` and the host turns a region index into a slot through the pool that reserved it
// (ExpertCache::adopt_prefetch_placements refuses an index that names anything else).
//
// Nothing here synchronises. The kernel runs on the caller's stream, behind the router that produced
// the ids; the host reads its result after the synchronise the layer already performs for its route
// read-back.
#include "ninfer/ops/moe.h"

#include "core/expert_cache.h" // kAbsentSlot - the pool's marker, stated once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {
namespace {

// One block per predicted id, one thread doing the decision.
//
// The decision is serial per id on purpose: the cursor is a single ring position and the pair list is
// a compacted list, so both are read-modify-write words. With a guess of tens of ids the contention
// is nothing (three atomics per placed id) and the alternative - a parallel scan that recomputes the
// same prefix sums on the host - would move the decision back where it came from.
__global__ void prefetch_select_kernel(const std::int32_t* ids, int count,
                                       const std::uint32_t* placement_table, std::uint32_t experts,
                                       std::uint32_t* cursor, std::uint32_t regions,
                                       std::uint32_t* pairs, std::uint32_t capacity,
                                       std::uint32_t* counts) {
    const int index = static_cast<int>(blockIdx.x);
    if (index >= count || threadIdx.x != 0) { return; }
    const std::int32_t id = ids[index];
    if (id < 0 || static_cast<std::uint32_t>(id) >= experts) {
        atomicAdd(counts + 4, 1u); // out of this bank's range: a router/pool width disagreement
        return;
    }
    // A guess is top_k x tokens, so the same expert appears once per token it was routed to. The
    // first occurrence is the placement; the later ones would take a second region slot for the same
    // expert and evict themselves.
    for (int earlier = 0; earlier < index; ++earlier) {
        if (ids[earlier] == id) {
            atomicAdd(counts + 5, 1u); // duplicate id inside one guess
            return;
        }
    }
    if (placement_table != nullptr && placement_table[id] != kAbsentSlot) {
        // Already addressable - from the pool, or from this bank's region. Either way a placement
        // now would move bytes for a route that has an address, and the region's slots are the
        // scarce thing here.
        atomicAdd(counts + 1, 1u);
        return;
    }
    const std::uint32_t slot = atomicAdd(counts + 0, 1u);
    if (slot >= capacity || regions == 0) {
        atomicAdd(counts + 2, 1u); // the ring is full for this guess (counted, not truncated away)
        return;
    }
    const std::uint32_t region = atomicAdd(cursor, 1u) % regions;
    pairs[2 * slot + 0]        = static_cast<std::uint32_t>(id);
    pairs[2 * slot + 1]        = region;
}

} // namespace

void moe_prefetch_select(const std::int32_t* ids, int count, const std::uint32_t* placement_table,
                         std::uint32_t experts, std::uint32_t* cursor, std::uint32_t regions,
                         std::uint32_t* pairs, std::uint32_t capacity, std::uint32_t* counts,
                         cudaStream_t stream) {
    if (count <= 0 || ids == nullptr || cursor == nullptr || pairs == nullptr ||
        counts == nullptr || regions == 0 || capacity == 0) {
        return;
    }
    // One block per id: the block index IS the id's position, so the duplicate scan above is a
    // position comparison rather than a second buffer.
    prefetch_select_kernel<<<count, 32, 0, stream>>>(ids, count, placement_table, experts, cursor,
                                                     regions, pairs, capacity, counts);
}

} // namespace ninfer::ops
