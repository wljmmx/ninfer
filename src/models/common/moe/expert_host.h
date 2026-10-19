#pragma once

#include "artifact/reader.h"
#include "core/host_reserve.h"
#include "core/weight_bank.h"
#include "models/common/moe/weights.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::io {
class ReadOnlyFile;
}

namespace ninfer::models::moe {

// port (sm_89): Host expert tier.
//
// Each expert owns a 64 KiB aligned segment per stored matrix, carved out of one HostReserve.
// Segments are committed and read from the artifact on first use, then released under an LRU
// quota. WeightBankView's per-expert offset tables make the addresses independent of residency,
// so a released expert keeps its view and is reloaded in place.
class ExpertHostTier final : public HostResidency {
public:
    ExpertHostTier(const artifact::Reader& reader, std::span<const BankDescription> banks,
                   std::size_t quota_bytes);
    ~ExpertHostTier() override;
    ExpertHostTier(const ExpertHostTier&)            = delete;
    ExpertHostTier& operator=(const ExpertHostTier&) = delete;

    [[nodiscard]] const WeightParent* parent(std::uint32_t bank, bool gate_up) const;
    [[nodiscard]] const std::uint64_t* offsets(std::uint32_t bank, bool gate_up) const;

    void ensure(ExpertBankId bank, std::span<const std::int32_t> experts,
                const std::int32_t* resident_mask) override;

    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserve_.size(); }

    [[nodiscard]] std::size_t committed_bytes() const noexcept {
        return reserve_.committed_bytes();
    }

    [[nodiscard]] std::size_t peak_committed_bytes() const noexcept {
        return reserve_.peak_committed_bytes();
    }

    [[nodiscard]] std::size_t quota_bytes() const noexcept { return quota_; }

    [[nodiscard]] std::uint64_t loads() const noexcept { return loads_; }

    [[nodiscard]] std::uint64_t hits() const noexcept { return hits_; }

    [[nodiscard]] std::uint64_t evictions() const noexcept { return evictions_; }

private:
    struct Matrix {
        WeightParent parent;
        std::uint64_t base      = 0; // Offset of expert 0 inside the reserve.
        std::uint64_t segment   = 0; // Bytes per expert, 64 KiB aligned.
        std::uint32_t count     = 0;
        std::vector<float> divisors; // Owned Host metadata; parent borrows it.
        // Per expert: segment start relative to the parent. Plane offsets are rebased onto the
        // packed segment layout by bind_banks.
        std::vector<std::uint64_t> offsets;
        // Per expert: chunks of the segment image, image_offset relative to the segment.
        std::vector<std::vector<ExpertFileChunk>> chunks;
    };

    [[nodiscard]] bool resident(const Matrix& matrix, std::uint32_t expert) const;
    void load(Matrix& matrix, std::uint32_t expert);
    void release(Matrix& matrix, std::uint32_t expert);
    void enforce_quota(ExpertBankId bank, std::uint32_t keep_expert);

    HostReserve reserve_;
    std::size_t quota_ = 0;
    std::vector<std::array<Matrix, 2>> banks_; // [gate_up, down]
    std::vector<std::vector<std::uint64_t>> last_use_;
    std::vector<std::vector<std::uint8_t>> live_;
    std::vector<std::unique_ptr<io::ReadOnlyFile>> files_;
    std::uint64_t epoch_     = 0;
    std::uint64_t loads_     = 0;
    std::uint64_t hits_      = 0;
    std::uint64_t evictions_ = 0;
};

} // namespace ninfer::models::moe
