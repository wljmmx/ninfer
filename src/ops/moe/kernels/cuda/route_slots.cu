// port (sm_89): S2-3 - expert id -> pool slot, resolved on the Device.
//
// The staging half (S1) keeps a per-bank expert->slot table on the host and uploads it to a Device
// mirror. This kernel is the consumer that makes the mirror worth having: given the ids a layer
// routed to, it produces the slot each id was placed in, so a Device-side placement can name its
// destination without a host round trip and without a second copy of the table.
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

} // namespace

void moe_route_slots(const std::int32_t* ids, int count, const std::uint32_t* slot_table,
                     std::uint32_t experts, std::uint32_t* slots, std::uint32_t* misses,
                     cudaStream_t stream) {
    if (count <= 0 || ids == nullptr || slot_table == nullptr || slots == nullptr) { return; }
    constexpr int block = 256;
    const int grid      = (count + block - 1) / block;
    route_slots_kernel<<<grid, block, 0, stream>>>(ids, count, slot_table, experts, slots, misses);
}

} // namespace ninfer::ops
