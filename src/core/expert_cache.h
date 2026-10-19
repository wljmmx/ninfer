#pragma once

#include "core/expert_bank.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/weight_view.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

// port (sm_89): the expert cache planned on the bench line, expressed against master's
// "one 2D parent per expert" bank shape.
//
// A bank keeps `fixed` experts resident in the pool and serves the rest through a streaming
// window; uploads are batched per slot and observed without blocking the caller.
struct ExpertCacheConfig {
    std::uint64_t capacity_bytes = 0; // pool bytes, 256-aligned
    std::size_t max_batch_bytes   = 32U * 1024U * 1024U;
    std::uint32_t upload_slots    = 2;

    friend bool operator==(const ExpertCacheConfig&, const ExpertCacheConfig&) = default;
};

struct ExpertMatrixSpec {
    WeightGeometry geometry;
};

struct ExpertBankSpec {
    ExpertMatrixSpec gate_up;
    ExpertMatrixSpec down;
    std::uint32_t count = 0; // experts in this bank (master stores one 2D parent each)
    bool resident       = true; // every expert already lives in the pool
};

struct ExpertBankLayout {
    std::uint32_t count          = 0;
    bool resident                = true;
    std::uint64_t payload_bytes  = 0; // codes + high + scales, no padding
    std::uint64_t entry_bytes    = 0; // 256-aligned image of one expert
    LayoutRegion codes, high, scales;
    LayoutRegion down_codes, down_high, down_scales;

    friend bool operator==(const ExpertBankLayout&, const ExpertBankLayout&) = default;
};

struct ExpertCacheRequirements {
    ExpertCacheConfig config;
    std::vector<ExpertBankLayout> banks;
    std::size_t entry_count     = 0;
    std::size_t offset_bytes    = 0;
    std::size_t max_entry_bytes = 0;

    friend bool operator==(const ExpertCacheRequirements& a, const ExpertCacheRequirements& b) {
        return a.config == b.config && a.entry_count == b.entry_count &&
               a.offset_bytes == b.offset_bytes && a.banks.size() == b.banks.size() &&
               a.max_entry_bytes == b.max_entry_bytes;
    }
};

[[nodiscard]] ExpertCacheRequirements describe_expert_cache(std::span<const ExpertBankSpec>,
                                                           const ExpertCacheConfig&);

struct ExpertKey {
    std::uint32_t bank   = 0;
    std::uint32_t expert = 0;

    friend bool operator==(const ExpertKey&, const ExpertKey&) = default;
};

// Where one expert currently lives in the pool.
struct ExpertSlot {
    ExpertKey key;
    std::uint64_t offset = 0;
    bool ready           = false;
};

struct ExpertCacheProgress {
    std::size_t submitted_batches = 0;
    bool required_published       = true;
};

class ExpertCache {
public:
    ExpertCache(std::span<const ExpertBankSpec>, const ExpertCacheRequirements&,
                const DeviceContext&, std::span<const ExpertBankPair> views);
    ~ExpertCache();
    ExpertCache(const ExpertCache&)            = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    // Declares the experts the next layer needs; they are uploaded by progress().
    void prepare(std::span<const ExpertKey> required);

    // Uploads one batch if a slot is free. Never blocks.
    [[nodiscard]] ExpertCacheProgress progress();

    // Waits for every in-flight batch. Used at startup and before a required miss.
    void wait_for_progress();

    // Pool offset of a published expert, or nullopt when it is not resident.
    [[nodiscard]] std::optional<std::uint64_t> offset(ExpertKey key) const;

    [[nodiscard]] std::uint64_t used_bytes() const noexcept;
    [[nodiscard]] std::uint64_t capacity_bytes() const noexcept;
    [[nodiscard]] std::uint64_t uploads() const noexcept;
    [[nodiscard]] std::uint64_t evictions() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer




