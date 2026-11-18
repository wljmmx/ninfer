#pragma once

// V10-75: how big the pinned, mapped host-tier block asks the allocator for.
//
// `allocate_host_arena` (runtime/engine/model_instance.cpp) answers three different questions with
// one number today, and two of them are not the same question:
//
//   unknown   the box cannot be measured at all (no Windows commit accounting, or the call failed).
//             The ask then stands as named: an unreadable budget must not be read as zero, or a
//             platform without the query would silently run a smaller tier than it asked for.
//   measured  the page file has bytes left. Clamp to them - a mapped block the commit budget
//             cannot back is the failure this clamp exists to prevent.
//   measured  the page file has NOTHING left (headroom 0). This is not "do not clamp", it is the
//             one answer that says the block cannot exist: a refused mapped allocation is terminal
//             in this process (the measurement behind that note lives next to allocate_host_arena),
//             so asking for the full candidate here buys nothing and costs the process's commit.
//
// The third case used to fall into the first: the guard was `headroom != 0`, so a measured zero
// took the "do not clamp" arm and asked for everything - and the refusal line the same function
// carries for "no commit headroom left" could never print. Observed on this box (35B-e2, RTX 4090)
// with the V10-73 arena leak holding commit:
// `[host] arena allocated 8846 MiB (asked 8846, commit headroom 0)` - 8.6 GiB granted against a
// budget that read zero, because Windows happened to grow the page file underneath it. The same ask
// against a page file that does not grow is the terminal refusal.
//
// `exact` is the explicit-override form (NINFER_EXPERT_HOST_ARENA_MB): a size the caller named is
// asked for as named, with no clamp at all, so an impossible experiment cannot silently run at a
// smaller arena and then be compared against a real one. A refusal there is the caller's error.

#include <algorithm>
#include <cstdint>
#include <optional>

namespace ninfer::runtime {

// Bytes to ask for, given the candidate, the commit budget (nullopt = unknown) and whether the
// caller named the size. Never returns more than `asked` and never invents a budget.
inline std::uint64_t host_arena_ask(std::uint64_t                        asked,
                                    const std::optional<std::uint64_t>& budget, bool exact) {
    if (exact || !budget.has_value()) { return asked; }
    return std::min(asked, *budget);
}

} // namespace ninfer::runtime
