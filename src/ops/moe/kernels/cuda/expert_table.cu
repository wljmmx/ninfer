// port (sm_89): GH3 - the placement word, expanded on the Device.
//
// The publisher used to build the expert tables on the host: a pool offset table, a pinned-tier offset
// table, and a mask that said "one of the two has an address". Every structure added to the placement
// (the prediction region was the third) added a host loop and a row to push, and the mask - the one
// array the GEMM consults on every route - was the host's opinion rather than a consequence of where
// the experts actually were.
//
// This kernel makes it a consequence. One word per expert says where it lives, and the four arrays
// the kernels read are derived from it: the two offset tables, the host flag, and the mask. The host
// now publishes 4 bytes per moved expert instead of 12, and never writes the mask at all.
#include "ninfer/ops/moe.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {
namespace {

// One thread per logical expert. The table is 256 words on this model, so this is a single small
// block - the point is not throughput, it is that the work happens where the tables are read and the
// host is out of the loop.
__global__ void expand_placements_kernel(const std::uint32_t* placements, int count,
                                         std::uint64_t pool_slot_bytes,
                                         std::uint64_t pool_bias_bytes,
                                         std::uint64_t host_slot_bytes,
                                         std::uint64_t* pool_offsets, std::int32_t* pool_mask,
                                         std::uint64_t* host_offsets, std::int32_t* host_flags) {
    const int expert = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (expert >= count) { return; }
    const std::uint32_t placement = placements[expert];
    // Absent first: its marker has every bit set, so it would otherwise read as a host slot.
    const bool absent = placement == kExpertPlacementAbsent;
    const bool host   = !absent && (placement & kExpertPlacementHost) != 0;
    const std::uint32_t slot = placement & ~kExpertPlacementHost;
    // The mask means exactly "this route resolves to an address", which is the same rule the host's
    // complement uses for the CPU half. Both legs collapse into it, so a route served by the pinned
    // tier is not a miss.
    if (pool_mask != nullptr) { pool_mask[expert] = absent ? 0 : 1; }
    if (host_flags != nullptr) { host_flags[expert] = host ? 1 : 0; }
    if (pool_offsets != nullptr) {
        pool_offsets[expert] =
            (!absent && !host)
                ? static_cast<std::uint64_t>(slot) * pool_slot_bytes + pool_bias_bytes
                : 0;
    }
    if (host_offsets != nullptr) {
        host_offsets[expert] = host ? static_cast<std::uint64_t>(slot) * host_slot_bytes : 0;
    }
}

} // namespace

void moe_expand_expert_placements(const std::uint32_t* placements, int count,
                                  std::uint64_t pool_slot_bytes, std::uint64_t pool_bias_bytes,
                                  std::uint64_t host_slot_bytes,
                                  std::uint64_t* pool_offsets, std::int32_t* pool_mask,
                                  std::uint64_t* host_offsets, std::int32_t* host_flags,
                                  cudaStream_t stream) {
    if (count <= 0 || placements == nullptr) { return; }
    constexpr int block = 256;
    const int grid      = (count + block - 1) / block;
    expand_placements_kernel<<<grid, block, 0, stream>>>(placements, count, pool_slot_bytes,
                                                         pool_bias_bytes, host_slot_bytes,
                                                         pool_offsets, pool_mask, host_offsets,
                                                         host_flags);
}

} // namespace ninfer::ops
