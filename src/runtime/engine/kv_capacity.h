#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>

namespace ninfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

// Split of the per-page device cost into the part that only exists while a page is resident and
// the part that must exist for every logical page (appendix BP).
//
// Both fields default to 0, which keeps the residency model inactive and leaves every existing
// caller byte-identical. Measured on the RTX 4090 / qwen3.8-27b / rk4v4-e8 configuration, the
// 983,044 B stride of a fully resident pool decomposes exactly as
//     payload = 983,040 B  (K/V planes of the 16 full-attention layers == 15,360 B/token x 64)
//     ledger  =       4 B  (one int32 block-table row per logical page at max_concurrency = 1)
// The page pool is sized by *physical* page groups; the block table is sized by *logical* page
// groups, so charging payload per resident page and ledger per logical page is what lets logical
// capacity exceed what the device can hold resident.
struct KvResidencyStrides {
    std::size_t payload_bytes_per_resident_page_group = 0;
    std::size_t ledger_bytes_per_logical_page_group   = 0;
};

struct KvResidencyResolution {
    std::uint32_t logical_page_groups  = 0;
    std::uint32_t resident_page_groups = 0;
    // Anchor reservation (at the curve minimum, fully resident) + payload for the resident pages
    // beyond it + ledger for every logical page beyond it.
    std::size_t device_reservation_bytes         = 0;
    // Payload of the logical pages that are not resident; these live on the host side.
    std::size_t host_payload_bytes               = 0;
    // What the fully resident model would demand for the same logical capacity.
    std::size_t fully_resident_reservation_bytes = 0;
    bool        residency_active                 = false;
};

// Resolves an explicit logical capacity together with the residency the device budget can hold.
// This is a planning model, not a runnable mode: it reports what tiering would need and never
// relaxes the "physical pages >= logical pages" invariant of the decoder state.
[[nodiscard]] KvResidencyResolution resolve_kv_capacity_with_residency(
    const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
    const KvResidencyStrides& strides, std::size_t available_runtime_bytes);

} // namespace ninfer::runtime
