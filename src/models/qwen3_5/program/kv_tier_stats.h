#pragma once

#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

// appendix FH (M3 3.2d): the counters behind the `kv tiers` line, in the Strata "expert tiers"
// style - what each tier holds now, plus, since the start, what has moved between them. They live
// here rather than on ProgramImpl for two reasons: the only two paths that update them (the
// demotion in storage/sequence.cpp and the restore in transactions/context_transaction.cpp) share
// no other object, and a diagnostic must not change the layout of the engine's main state. This is
// the same process-wide pattern the page-score keep map already uses (g_scored_keep_by_row).
// Diagnostics only: nothing reads these counters back.
struct KvTierStats {
    std::uint64_t spilled_pages         = 0;  // Device -> Host, pages that reached Host
    std::uint64_t spilled_bytes         = 0;  // Device -> Host, the batch's payload
    std::uint64_t fetched_pages         = 0;  // Host -> Device, the restore's pages
    std::uint64_t fetched_bytes         = 0;  // Host -> Device, the restore's payload
    std::uint64_t preempted_checkpoints = 0;  // checkpoints that gave up their Host room
    // appendix GC (FZ layer 0): the eviction window's retired frontier - how many rounds used it,
    // and where the last one left it. Diagnostics only, like the rest of this struct.
    std::uint64_t frontier_rounds       = 0;
    std::uint32_t frontier_advanced_to  = 0;
};

// One instance per program: the inline function's local static is shared across translation
// units, which is exactly the scope "since the start" needs.
inline KvTierStats& kv_tier_stats() noexcept {
    static KvTierStats stats;
    return stats;
}

} // namespace ninfer::models::qwen3_5::detail
