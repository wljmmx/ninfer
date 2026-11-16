// port (sm_89): S2-3 - expert id -> slot, resolved on the Device.
//
// The staging half (S1) keeps a per-bank table on the host and uploads it to a Device mirror. This
// kernel is the consumer that makes the mirror worth having: given the ids a layer routed to, it
// produces the slot each id was placed in, so a Device-side placement can name its destination
// without a host round trip and without a second copy of the table.
//
// C1/C2: there are TWO tables and two spaces, and the kernel now says which one it is reading:
//
//   * the PLACEMENT table is expert -> pool slot (`export_slots`, indexed by expert id). The answer
//     is the address of an expert's bytes. Its generation is step_generation(bank).
//   * the PREDICTION REGION's table is region slot -> expert (`export_prefetch_owners`, indexed by
//     region). The answer is the region INDEX that holds the id, which is a position inside one bank's
//     ring, NOT an address. Its generation is prefetch_generation(bank).
//
// The two spaces have different lengths on every bank that has a region, and a slot number from one is
// a plausible-looking number in the other - which is why the caller names the kind instead of handing
// over "a table" and hoping. The host side compares the two (prefetch_region_slot) before an index is
// ever used as an address.
//
// The "no slot" word is the pool's own (kAbsentSlot in core/expert_cache.h), not a value invented
// here: a table written by export_slots() and read by this kernel must agree bit for bit, or a route
// with no slot would read as slot 0 and address another expert's bytes.
#include "ninfer/ops/moe.h"

#include "core/expert_cache.h" // kAbsentSlot - the pool's marker, stated once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {
namespace {

// One thread per id. The count is a layer's top_k * tokens (tens of values), so a small block is
// enough; the point is that the table stays where it is read, not throughput.
__global__ void route_slots_kernel(const std::int32_t* ids, int count,
                                   const std::uint32_t* slot_table, std::uint32_t experts,
                                   std::uint32_t* slots, std::uint32_t* misses) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    if (i >= count) { return; }
    const std::int32_t id = ids[i];
    // Out of range first: a negative id would index before the table, and `id >= experts` reads
    // past it, so neither may reach the lookup.
    if (id < 0 || static_cast<std::uint32_t>(id) >= experts) {
        slots[i] = kAbsentSlot;
        if (misses != nullptr) { atomicAdd(misses, 1u); }
        return;
    }
    const std::uint32_t slot = slot_table[id];
    slots[i] = slot;
    if (slot == kAbsentSlot && misses != nullptr) { atomicAdd(misses, 1u); }
}

// C2: the region's own query. The table is region -> expert, so "which slot holds this id" is a search
// over the region (its length is the region's slot count: a handful of words on this pool). The answer
// is the region INDEX - the same space the table is written in - and the host maps it to a pool slot
// through the pool's own region list, so an index can never be mistaken for an address here.
__global__ void route_region_slots_kernel(const std::int32_t* ids, int count,
                                          const std::uint32_t* region_owners, std::uint32_t regions,
                                          std::uint32_t* region_indices, std::uint32_t* misses) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    if (i >= count) { return; }
    const std::int32_t id = ids[i];
    if (id < 0) {
        region_indices[i] = kAbsentSlot;
        if (misses != nullptr) { atomicAdd(misses, 1u); }
        return;
    }
    const auto expert = static_cast<std::uint32_t>(id);
    for (std::uint32_t region = 0; region < regions; ++region) {
        if (region_owners[region] == expert) {
            region_indices[i] = region;
            return;
        }
    }
    region_indices[i] = kAbsentSlot;
    if (misses != nullptr) { atomicAdd(misses, 1u); }
}

} // namespace

void moe_route_slots(const std::int32_t* ids, int count, const std::uint32_t* slot_table,
                     std::uint32_t experts, std::uint32_t* slots, std::uint32_t* misses,
                     cudaStream_t stream) {
    if (count <= 0 || ids == nullptr || slot_table == nullptr || slots == nullptr) { return; }
    constexpr int block = 256;
    const int grid      = (count + block - 1) / block;
    route_slots_kernel<<<grid, block, 0, stream>>>(ids, count, slot_table, experts, slots, misses);
}

void moe_route_region_slots(const std::int32_t* ids, int count,
                            const std::uint32_t* region_owners, std::uint32_t regions,
                            std::uint32_t* region_indices, std::uint32_t* misses,
                            cudaStream_t stream) {
    if (count <= 0 || ids == nullptr || region_owners == nullptr || region_indices == nullptr ||
        regions == 0) {
        return;
    }
    constexpr int block = 256;
    const int grid      = (count + block - 1) / block;
    route_region_slots_kernel<<<grid, block, 0, stream>>>(ids, count, region_owners, regions,
                                                          region_indices, misses);
}

void moe_route_slots(const std::int32_t* ids, int count, const std::uint32_t* table,
                     std::uint32_t entries, RouteTableKind kind, std::uint32_t* slots,
                     std::uint32_t* misses, cudaStream_t stream) {
    if (kind == RouteTableKind::PrefetchRegion) {
        moe_route_region_slots(ids, count, table, entries, slots, misses, stream);
        return;
    }
    moe_route_slots(ids, count, table, entries, slots, misses, stream);
}

} // namespace ninfer::ops
