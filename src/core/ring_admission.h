#pragma once

// Q20's missing half: device-side admission.
//
// The fixed-S ring is currently filled by the host from this step's route list. That makes the host
// the owner of the ring's contents, which is exactly what a captured layer cannot tolerate: the
// decision has to come from the Device, and the host may only move bytes it was told to move.
//
// The Device is not allowed to program a PCIe transfer. What it can do is choose and describe:
// this kernel reads the route ids and the pinned candidate table, picks the distinct experts that
// are outside the pool, assigns each a ring slot through a device-side cursor, and writes
// descriptors into a queue. The host drains that queue and performs the copies; it never decides
// which expert enters the ring.

#include <cstdint>
#include <cuda_runtime_api.h>

namespace ninfer::ops {

// One request the Device made: copy the candidate at `host_slot` into pool slot `ring_slot`, which
// from then on holds `expert`.
struct RingAdmitDescriptor {
    std::uint32_t expert    = 0;
    std::uint32_t host_slot = 0;
    std::uint32_t ring_slot = 0;
    std::uint32_t reserved  = 0;
};

inline constexpr std::uint32_t kRingAdmitAbsent = 0xffffffffU;

// `routes` is the Device route-id buffer (the same one the layer already wrote). `candidates` is
// indexed by expert and holds the pinned slot, or kRingAdmitAbsent. `resident` is the pool's
// membership mask (1 = already resident), or null; an expert the pool already holds is not a miss
// and is never admitted. The ring slot is derived from the descriptor's own index, so the first
// `capacity` admissions name distinct slots.
void ring_admit_launch(const void* routes, std::int32_t values, const std::uint32_t* candidates,
                       const std::int32_t* resident, std::uint32_t expert_count,
                       std::uint32_t ring_base, std::uint32_t ring_slots,
                       RingAdmitDescriptor* descriptors, std::uint32_t* descriptor_count,
                       std::uint32_t capacity, cudaStream_t stream);

} // namespace ninfer::ops
