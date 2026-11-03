#pragma once

// Q67/S1: one lane's place in a merged prefill pass.
//
// Each row carries its own KV address row, its own absolute base, how many prompt tokens it
// contributes, and the state slots it reads and writes. That is the whole reason a merged pass can
// stay block-diagonal without a mask: row i's attention only ever reads KV row i, so two prompts in
// one pass cannot see each other even though they share the same kernels.
//
// Lives in its own header because both the engine-facing Program handle and the execution context
// name it, and neither should have to pull the other's dependencies in to say so.

#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

struct PrefillRow {
    std::uint32_t lane             = 0;
    std::uint32_t kv_row           = 0;
    std::uint32_t base             = 0;
    std::uint32_t count            = 0;
    std::int32_t state_source      = 0;
    std::int32_t state_destination = 0;
};

} // namespace ninfer::models::qwen3_5::execution
