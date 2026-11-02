#include "core/ring_admission.h"

#include <cstdint>

namespace ninfer::ops {
namespace {

__global__ void ring_admit_kernel(const int* routes, int values, const std::uint32_t* candidates,
                                  const int* resident, std::uint32_t expert_count,
                                  std::uint32_t ring_base, std::uint32_t ring_slots,
                                  RingAdmitDescriptor* descriptors,
                                  std::uint32_t* descriptor_count, std::uint32_t capacity) {
    // 256 experts fit in four 64-bit words. Routes repeat (8 lanes x top-k), so the bitmap is what
    // turns "this step's routes" into "this step's distinct experts" without a sort.
    __shared__ unsigned long long seen[4];
    if (threadIdx.x < 4) { seen[threadIdx.x] = 0; }
    __syncthreads();

    for (int index = static_cast<int>(threadIdx.x); index < values;
         index += static_cast<int>(blockDim.x)) {
        const int expert = routes[index];
        if (expert < 0 || static_cast<std::uint32_t>(expert) >= expert_count) { continue; }
        const auto word = static_cast<unsigned>(expert) >> 6U;
        const auto bit  = 1ULL << (static_cast<unsigned>(expert) & 63U);
        if (word >= 4U) { continue; }
        if ((atomicOr(&seen[word], bit) & bit) != 0ULL) { continue; }
        // An expert already in the pool is not a miss. Admitting it would move it out of the hot
        // segment and copy bytes over themselves.
        if (resident != nullptr && resident[expert] != 0) { continue; }
        const std::uint32_t host_slot = candidates[expert];
        if (host_slot == kRingAdmitAbsent) { continue; }
        const std::uint32_t slot = atomicAdd(descriptor_count, 1U);
        // A descriptor past the ring's capacity must not be written and must not consume a slot.
        // The slot comes from the descriptor's own index, so the first `capacity` admissions name
        // `capacity` distinct ring slots no matter how the threads interleave.
        if (slot >= capacity || ring_slots == 0) { continue; }
        const std::uint32_t ring_slot = ring_base + (slot % ring_slots);
        descriptors[slot] = RingAdmitDescriptor{.expert    = static_cast<std::uint32_t>(expert),
                                                .host_slot = host_slot,
                                                .ring_slot = ring_slot};
    }
}

} // namespace

void ring_admit_launch(const void* routes, std::int32_t values, const std::uint32_t* candidates,
                       const std::int32_t* resident, std::uint32_t expert_count,
                       std::uint32_t ring_base, std::uint32_t ring_slots,
                       RingAdmitDescriptor* descriptors, std::uint32_t* descriptor_count,
                       std::uint32_t capacity, cudaStream_t stream) {
    if (routes == nullptr || candidates == nullptr || descriptors == nullptr ||
        descriptor_count == nullptr || values <= 0 || ring_slots == 0 || capacity == 0) {
        return;
    }
    ring_admit_kernel<<<1, 64, 0, stream>>>(static_cast<const int*>(routes), values, candidates,
                                            resident, expert_count, ring_base, ring_slots,
                                            descriptors, descriptor_count, capacity);
}

} // namespace ninfer::ops
