#pragma once

// One switch, read in one place.
//
// A byte record holds the main KV run and the state image, never the draft/backend KV a speculative
// backend keeps beside it. That made the restore path refuse every spec engine outright, which is
// also why the cache looked inert under the console's default `--spec mtp`.
//
// Restoring it anyway is sound HERE because the draft path never decides what is emitted: every
// proposal is verified by the target model, so a draft cache that missed the restored prefix can
// only lose acceptance, not correctness. Measured on this port (RTX 4090, 2026-11-22, same prompt
// twice, greedy):
//
//   MTP      cold 43/122 (35.2%)  | restored 43/121 (35.5%)  | output identical
//   DFlash2  cold 38/155 (24.5%)  | restored 38/155 (24.5%)  | output identical
//   163k     restore 11.7 s, cache 163,680 of 163,687 (100.0%), TTFT 14.3 s
//
// So the default is on. `NINFER_LMCACHE_ALLOW_SPEC=0` puts the refusal back, which is what a future
// record format that carries the draft KV would replace rather than complement.
#include <cstdlib>

namespace ninfer::core {

[[nodiscard]] inline bool lmcache_accepts_spec_records() noexcept {
    static const bool allow = [] {
        const char* value = std::getenv("NINFER_LMCACHE_ALLOW_SPEC");
        return !(value != nullptr && std::atoi(value) == 0);
    }();
    return allow;
}

} // namespace ninfer::core
