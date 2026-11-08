#pragma once

#include "ninfer/types.h"
#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::runtime {

// One logical context allocation. The Main KV arena is sized in page groups; every complete
// StateImage beyond the execution lanes is a separate device allocation with its own stride.
// Upstream 049d5fb4 replaced the independent `kv_capacity` / `extra_state_slots` knobs with one
// pair, because both grow against the same device budget and sizing them separately can overcommit
// it. This struct is that pair.
struct ContextCapacity {
    std::uint32_t primary_page_groups = 0;
    std::uint32_t device_state_slots  = 0;

    [[nodiscard]] friend constexpr bool operator==(ContextCapacity,
                                                   ContextCapacity) noexcept = default;
};

// The three accounts a context pool owns, priced together:
//
//   KV      `SequenceCapacityCurve::reservation_bytes(pages)` - the finalized plan's reservation
//           at the curve minimum, plus a byte stride per additional page group;
//   State   one more complete StateImage per slot beyond the execution lanes;
//   Work    the transient workspace and the metadata ledger that must exist for either.
//
// The base curve's minimum already contains the execution lanes' StateImages and the plan's own
// workspace, so `minimum_state_slots` is the lane count, not zero, and `workspace_bytes` is the
// *unplanned* remainder (capture staging, graph allowance, transport scratch).
struct ContextCapacityCurve {
    SequenceCapacityCurve kv;
    std::uint32_t minimum_state_slots           = 0;
    std::size_t bytes_per_additional_state_slot = 0;
    std::size_t workspace_bytes                 = 0;
    // Metadata that grows with either capacity, not with payload (appendix: the directory ledger).
    std::size_t control_bytes_per_page_group = 0;
    std::size_t control_bytes_per_state_slot = 0;

    // KV reservation plus the extra StateImage strides. Throws `std::invalid_argument` when the
    // capacity is outside the storage domain and `std::overflow_error` on size_t overflow.
    [[nodiscard]] std::size_t reservation_bytes(ContextCapacity) const;
    [[nodiscard]] std::size_t control_bytes(ContextCapacity) const;
    // reservation + control + workspace: what the plan takes from the device in total.
    [[nodiscard]] std::size_t total_bytes(ContextCapacity) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t pages) const;
};

// The quoted plan. `total_reservation_bytes` is what must fit in `available_bytes`; the three
// components stay separate so an operator can see which book set the limit.
struct ContextCapacityPlan {
    ContextCapacity capacity;
    std::uint32_t maximum_page_groups = 0;
    std::uint32_t resolved_tokens     = 0;

    std::size_t kv_reservation_bytes        = 0;
    std::size_t state_reservation_bytes     = 0;
    std::size_t workspace_reservation_bytes = 0;
    std::size_t control_reservation_bytes   = 0;
    std::size_t total_reservation_bytes     = 0;
    std::size_t available_bytes             = 0;
    std::size_t planned_slack_bytes         = 0;

    // True when the checkpoint target (2 x lanes + one slot per interval) shaped the StateImage
    // count instead of the caller's explicit `extra` figure.
    bool checkpoints_targeted    = false;
    std::uint32_t target_slots   = 0;
};

// Jointly resolves page groups and StateImage slots against one device budget.
//
// `cache.device_state_slots` keeps its local meaning: extra complete StateImages *beyond* the
// execution lanes. An explicit KV policy pins the page count and never grows it; an automatic
// policy spends the residual bytes on checkpoints first, then on pages, then gives any bytes the
// discrete slot count could not use back to pages.
//
// `checkpoint_interval == 0` disables the checkpoint target. The function is a planning model: it
// reserves nothing and mutates nothing. `resolve_kv_capacity` remains the page-only path for
// callers that do not opt in (NINFER_CONTEXT_JOINT_PLAN).
[[nodiscard]] ContextCapacityPlan plan_context_capacity(const KvCapacityPolicy& policy,
                                                        const ContextCacheOptions& cache,
                                                        const ContextCapacityCurve& curve,
                                                        std::size_t available_runtime_bytes,
                                                        std::uint32_t checkpoint_interval);

} // namespace ninfer::runtime
