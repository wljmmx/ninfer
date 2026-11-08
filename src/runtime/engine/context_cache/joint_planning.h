#pragma once

// T156 (upstream 049d5fb4): joint capacity planning and prefix-checkpoint retention.
//
// The context pool holds three kinds of bytes that used to be sized independently:
//   * KV pages (main + backend),
//   * recurrent state images (the GDN/ReplaySSM state that must travel with the KV),
//   * the transient execution workspace.
// Sizing each of them against the whole pool independently lets the three together
// exceed the device budget; the union is what has to fit. These functions are the
// planning-only half of that: pure arithmetic over a total budget, with no reservations
// and no side effects, so the caller can quote a plan before touching an allocator.
//
// The retention half answers a different question: given a byte budget for retained
// prefix checkpoints, which ones are worth keeping. Pinned candidates are always kept
// (they are in flight); the rest are ranked by frontier length, evidence of reuse, and
// recency, and dropped from the bottom until the budget is met. This is deliberately a
// selection policy, not a replacement for ResourceManager's owner/points bookkeeping:
// the manager still owns the handles; this only picks the subset the budget can carry.

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

namespace ninfer::runtime::context_cache {

// One request's (or one working point's) demand for the three pools.
struct JointDemand {
    std::uint64_t kv_bytes        = 0;  // main + backend KV pages
    std::uint64_t state_bytes     = 0;  // recurrent state images
    std::uint64_t workspace_bytes = 0;  // transient execution workspace
};

struct JointBudget {
    std::uint64_t kv_bytes        = 0;
    std::uint64_t state_bytes     = 0;
    std::uint64_t workspace_bytes = 0;
    std::uint64_t headroom_bytes  = 0;  // kept free for the pool; never handed out
    bool          feasible        = false;
};

// Assigns a total device budget across the three pools.
//
// Priority is fixed: workspace first (an execution without it cannot start), then
// state (a restored prefix without its state is not reusable), then KV (the remainder).
// The headroom is subtracted before anything is assigned and is reported back so the
// caller quotes the same number it planned with.
[[nodiscard]] inline JointBudget plan_joint_capacity(std::uint64_t total_bytes,
                                                     const JointDemand& demand,
                                                     std::uint64_t headroom_bytes = 0) {
    JointBudget out;
    out.headroom_bytes = std::min(headroom_bytes, total_bytes);
    const std::uint64_t usable = total_bytes - out.headroom_bytes;
    if (demand.workspace_bytes > usable) { return out; }
    if (demand.state_bytes > usable - demand.workspace_bytes) { return out; }
    const std::uint64_t after_workspace_and_state =
        usable - demand.workspace_bytes - demand.state_bytes;
    // KV is the residual pool: it gets whatever the two mandatory pools did not take.
    out.workspace_bytes = demand.workspace_bytes;
    out.state_bytes     = demand.state_bytes;
    out.kv_bytes        = after_workspace_and_state;
    out.feasible        = true;
    return out;
}

// A retained prefix candidate, in the units the selection policy ranks.
struct PrefixCandidate {
    std::uint64_t handle      = 0;
    std::uint32_t frontier    = 0;      // tokens covered; longer prefixes are worth more
    std::uint64_t bytes       = 0;      // what keeping it costs
    bool          reused      = false;  // evidence that a later request asked for it
    std::uint64_t last_demand = 0;      // logical clock of the last reuse
    bool          pinned      = false;  // in flight; must be kept regardless of budget
};

// Selects the retained set under a byte budget.
//
// Pinned candidates are always kept. Among the rest, ranking is: reused before never
// reused, then longer frontier, then more recent demand, then lower handle (stable).
// The least valuable candidate is dropped until the total fits. If even the pinned set
// exceeds the budget the selection returns only the pinned set and reports it as
// over budget, so the caller can decide to fail the admission rather than silently
// corrupt the plan.
struct PrefixSelection {
    std::vector<std::uint64_t> handles;       // kept, in input-relative rank order
    std::uint64_t              bytes_kept = 0;
    bool                       over_budget = false;  // pinned alone exceeded the budget
};

[[nodiscard]] inline PrefixSelection
select_prefix_checkpoints(std::vector<PrefixCandidate> candidates, std::uint64_t budget_bytes) {
    PrefixSelection out;
    if (candidates.empty()) { return out; }

    std::vector<std::size_t> pinned;
    std::vector<std::size_t> optional;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        (candidates[i].pinned ? pinned : optional).push_back(i);
    }

    std::uint64_t pinned_bytes = 0;
    for (const auto i : pinned) {
        pinned_bytes += candidates[i].bytes;
        out.handles.push_back(candidates[i].handle);
        out.bytes_kept += candidates[i].bytes;
    }
    if (pinned_bytes > budget_bytes) {
        out.over_budget = true;
        return out;
    }

    std::stable_sort(optional.begin(), optional.end(), [&](std::size_t a, std::size_t b) {
        const auto& x = candidates[a];
        const auto& y = candidates[b];
        if (x.reused != y.reused) { return x.reused; }
        if (x.frontier != y.frontier) { return x.frontier > y.frontier; }
        if (x.last_demand != y.last_demand) { return x.last_demand > y.last_demand; }
        return x.handle < y.handle;
    });

    for (const auto i : optional) {
        const auto& candidate = candidates[i];
        if (out.bytes_kept + candidate.bytes > budget_bytes) { continue; }
        out.handles.push_back(candidate.handle);
        out.bytes_kept += candidate.bytes;
    }
    return out;
}

}  // namespace ninfer::runtime::context_cache
