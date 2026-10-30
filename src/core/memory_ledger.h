#pragma once

// port (sm_89): one ledger for the three tiers. Nothing allocates without asking it first.
//
// Why this exists: today each module sizes itself from whatever it can see - the expert arena takes
// a quarter of free RAM, the host KV arena is a flag, the device pool takes what is left after KV -
// and no module can see the others. Two of them each taking "a quarter" is half, and the first
// symptom is not "too much expert mirror", it is "KV started missing". At 125B the expert payload
// alone exceeds RAM, so the partition has to be decided in one place.
//
// The tiers are a partition, not a copy of everything:
//   Device (hot)  - experts the router is likely to take next, plus the KV pages every step reads
//   Host   (warm) - experts evicted from hot that are still likely to come back; NEVER the whole payload
//   Disk   (cold)  - everything else, which is just the artifact; it has a bandwidth quota, not a cap
//
// This file is the bookkeeping half - step (1) of 1y.7: register and print, do not change any
// allocation yet. Step (2) makes the two existing sizers ask it instead of guessing; step (3) adds
// eviction and migration. Keeping (1) behaviour-neutral is what makes it safe to land first.

#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace ninfer::core {

enum class Tier { Device, Host, Disk };

struct LedgerEntry {
    Tier tier        = Tier::Device;
    std::string_view owner;
    std::uint64_t bytes = 0;
};

class MemoryLedger {
public:
    // A tier's ceiling. Device and Host come from the scheduler's budget; Disk has no ceiling, only
    // a bandwidth quota, so it is registered with kNoCap.
    static constexpr std::uint64_t kNoCap = ~0ull;

    void set_cap(Tier tier, std::uint64_t bytes) { cap(slot(tier)) = bytes; }

    // Returns false when the reservation does not fit. Callers are expected to lower their own
    // ask rather than to ignore it - that is the whole point of a ledger.
    bool reserve(Tier tier, std::string_view owner, std::uint64_t bytes) {
        const auto index = slot(tier);
        if (bytes > cap(index) - used(index)) { return false; }
        entries_.push_back({tier, owner, bytes});
        used(index) += bytes;
        return true;
    }

    [[nodiscard]] std::uint64_t used(Tier tier) const { return used(slot(tier)); }
    [[nodiscard]] std::uint64_t cap(Tier tier) const { return cap(slot(tier)); }

    // The rule that separates a partition from a bulk load: a warm tier that can hold the whole
    // payload is not a warm tier, it is the whole model in RAM wearing a different name.
    [[nodiscard]] bool warm_is_partition(std::uint64_t payload_bytes) const {
        return used(Tier::Host) < payload_bytes;
    }

    void report(std::uint64_t payload_bytes) const {
        const auto line = [](Tier tier) { return tier == Tier::Device ? "vram"
                                        : tier == Tier::Host   ? "ram"
                                                               : "disk"; };
        for (const auto tier : {Tier::Device, Tier::Host, Tier::Disk}) {
            char cap_text[32];
            if (cap(tier) == kNoCap) {
                std::snprintf(cap_text, sizeof(cap_text), "none");
            } else {
                std::snprintf(cap_text, sizeof(cap_text), "%llu MiB",
                              static_cast<unsigned long long>(cap(tier) >> 20));
            }
            std::fprintf(stderr, "[ledger] %-4s used=%llu MiB cap=%s", line(tier),
                         static_cast<unsigned long long>(used(tier) >> 20), cap_text);
            for (const auto& entry : entries_) {
                if (entry.tier != tier) { continue; }
                std::fprintf(stderr, " | %.*s=%llu MiB", static_cast<int>(entry.owner.size()),
                             entry.owner.data(),
                             static_cast<unsigned long long>(entry.bytes >> 20));
            }
            std::fputc('\n', stderr);
        }
        std::fprintf(stderr, "[ledger] warm is %s partition (used %llu MiB of %llu MiB payload)\n",
                     warm_is_partition(payload_bytes) ? "a" : "NOT a",
                     static_cast<unsigned long long>(used(Tier::Host) >> 20),
                     static_cast<unsigned long long>(payload_bytes >> 20));
        std::fflush(stderr);
    }

private:
    static constexpr std::size_t slot(Tier tier) {
        return tier == Tier::Device ? 0U : tier == Tier::Host ? 1U : 2U;
    }
    std::uint64_t& cap(std::size_t index) { return caps_[index]; }
    std::uint64_t cap(std::size_t index) const { return caps_[index]; }
    std::uint64_t& used(std::size_t index) { return used_[index]; }
    std::uint64_t used(std::size_t index) const { return used_[index]; }

    std::uint64_t caps_[3] = {0, 0, kNoCap};
    std::uint64_t used_[3] = {0, 0, 0};
    std::vector<LedgerEntry> entries_;
};

} // namespace ninfer::core
