#pragma once

// P4.2: the split geometry an attention dispatch actually used, exposed READ-ONLY to models.
//
// WHY THIS EXISTS. `CausalKvPartition` (capacity / target / key_shift) is composed inside the ops
// dispatch (`int8/plan.cpp` `make_int8_kv_causal_plan`) and `Schedule::kKeyRows` is a template
// constant of the instance that runs (`int8/instances.h`), so neither ever left the ops layer. A
// models-side policy that wants to price a split - `decide_cpu_kv_split`
// (models/qwen3_5/execution/cpu_path_policy.h) - therefore had no way to describe the split it was
// asked about. This header is that description and nothing else: no decision reads it to change
// what a dispatch does, and nothing here is a second source of any geometry.
//
// WHO WRITES, AND WHY THAT IS THE ONLY WRITE. `note_int8_kv_plan` is called from exactly one place:
// `rk_kv_cpu_split_gate` (int8/host_partial.h), from inside the same per-instance lambda that
// launches the grouped MMA. The caller passes the *same* `partition` object and
// `Instance::Schedule::kKeyRows` as `key_tile` that the launch itself received, and the family is
// the instance's own template dtype through `rk_host_kv_codecs(DType).name`. So the record is not
// a copy of a copy: it is composed from the arguments of the call it describes. There is no second
// place that recomputes capacity/target/key_shift, hence nothing that can drift from the launch.
//
// WHEN IT IS WRITTEN. On every gated attention call, before the knob (`NINFER_KV_CPU_PARTITION`) is
// even consulted - the channel has to describe the call that ran, not only the calls a diagnostic
// switch armed. Cost is a handful of stores immediately before a kernel launch.
//
// WHEN IT IS VALID. One thread-local slot, overwritten by every gated call on that thread, never
// cleared on its own. A reader must read it inside the same call sequence that produced it (models
// reads it immediately after `ops::causal_softmax_attention` returns, on the same worker thread);
// `sequence` is a per-thread counter bumped on every write so a reader can tell "the record of the
// call I just made" from "an older one" without assuming ordering. A stale record can change a log
// line and nothing else - the dispatch never reads this slot.
//
// WHAT IT DOES NOT COVER, STATED RATHER THAN FILLED IN. The plain-int8 entry
// (`int8_kv_append_attention`'s `grouped<>` instances) and the width > 16 batch-wide kernels do not
// call the gate, so they leave no record; a reader then sees `sequence == 0` (or a previous call's
// record) and must not invent geometry from it.

#include "ops/softmax_attention/common/causal_partition.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct Int8KvPlanRecord {
    // The dispatcher's own partition object. Held as the struct, not as three copied ints, so the
    // only `active(visible)` formula in the tree stays CausalKvPartition::active.
    CausalKvPartition partition{};
    int key_rows    = 0;     // Instance::Schedule::kKeyRows of the launched instance
    int query_heads = 0;     // the dispatched G's QHeads
    int kv_heads    = 0;     // the dispatched G's KVHeads
    int width       = 0;     // the call's token tile
    int batch       = 0;     // the call's batch
    const char* family = "none";  // RkHostKvDType spelling of the launched instance
    std::uint64_t sequence = 0;   // 0 = this thread has not gated a call yet
};

namespace int8_kv_plan_detail {

[[nodiscard]] inline Int8KvPlanRecord& slot() {
    static thread_local Int8KvPlanRecord record;
    return record;
}

[[nodiscard]] inline std::uint64_t& sequence() {
    static thread_local std::uint64_t count = 0;
    return count;
}

} // namespace int8_kv_plan_detail

// The one writer (see the header comment for why the gate is the only caller).
inline void note_int8_kv_plan(const Int8KvPlanRecord& record) {
    auto& slot      = int8_kv_plan_detail::slot();
    const auto next = ++int8_kv_plan_detail::sequence();
    slot            = record;
    slot.sequence   = next;
}

// The last gated call on this thread.
[[nodiscard]] inline const Int8KvPlanRecord& int8_kv_plan_last() {
    return int8_kv_plan_detail::slot();
}

} // namespace ninfer::ops::detail
