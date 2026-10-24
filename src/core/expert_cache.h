#pragma once

#include "core/expert_bank.h"
#include "core/expert_file.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/weight_view.h"

#include <cuda_runtime.h>

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

struct ExpertCacheKey {
    std::uint32_t bank   = 0;
    std::uint32_t expert = 0;

    friend bool operator==(const ExpertCacheKey&, const ExpertCacheKey&) = default;
};

// Where one expert currently lives in the pool.
struct ExpertSlot {
    ExpertCacheKey key;
    std::uint64_t offset = 0;
    bool ready           = false;
};

struct ExpertCacheProgress {
    std::size_t submitted_batches = 0;
    bool required_published       = true;
};

class ExpertCache {
public:
    // port (sm_89): when a file source is given, streaming experts are fetched from the
    // artifact into pinned staging instead of being read from a resident Host replica.
    ExpertCache(std::span<const ExpertBankSpec>, const ExpertCacheRequirements&,
                const DeviceContext&, std::span<const ExpertBankPair> views,
                std::span<const ExpertFileBank> files = {},
                const ExpertBlobReader* reader        = nullptr);
    ~ExpertCache();
    ExpertCache(const ExpertCache&)            = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

// Declares the experts the next layer needs; they are uploaded by progress(). `stream` is the
// stream the consumer runs on: uploads queued there are ordered before the kernels that read them,
// so no device-wide synchronisation is needed.
void prepare(std::span<const ExpertCacheKey> required, cudaStream_t stream = nullptr);

    // Names the experts of one bank the pool should hold. Experts not listed are treated as
    // absent: prepare() skips them and offset() reports nullopt, which lets the caller mask them
    // out of the GPU job and hand them to the host tier. At most resident_capacity() experts are
    // honoured; the rest are dropped from the end of the list.
    // port (sm_89): restricts a bank to `experts` and returns whether that changed the set. Experts
    // that stay resident keep their slot, so a caller only republishes its offset table when this
    // returns true. The rest of the bank streams in again on their next use.
    bool set_resident_experts(std::uint32_t bank, std::span<const std::uint32_t> experts);

    // Uploads one batch if a slot is free. Never blocks.
    [[nodiscard]] ExpertCacheProgress progress(cudaStream_t stream = nullptr);

    // Waits for every in-flight batch. Used at startup and before a required miss.
    // port (sm_89): when a stream is given the uploads are ordered on it, so the caller does
    // not need a device-wide sync between a layer's copies and its kernel.
    void wait_for_progress(cudaStream_t stream = nullptr);

    // Pool offset of a published expert, or nullopt when it is not resident.
    [[nodiscard]] std::optional<std::uint64_t> offset(ExpertCacheKey key) const;

    [[nodiscard]] std::uint64_t used_bytes() const noexcept;
    [[nodiscard]] std::uint64_t capacity_bytes() const noexcept;
    // port (sm_89): hands device memory back to (or takes it from) the driver without moving a
    // published offset. The reservation stays where it is, so bank views and offset tables stay
    // valid; only the backing segments of the highest slots come and go. Experts of a dropped slot
    // run report nullopt again and stream in on their next use, which is why the caller must be at
    // a layer boundary. Returns the capacity in effect afterwards.
    [[nodiscard]] std::uint64_t resize(std::uint64_t target_bytes);
    // port (sm_89): the Device pool every published entry lives in. An indexed expert bank points
    // its plane bases into this span and adds `offset(key)` per expert.
    [[nodiscard]] ConstMemorySpan pool() const;
    // port (sm_89): bytes between two experts of the same bank inside the pool. Banks address
    // through a per-expert offset table, so this is only the slot alignment unit.
    [[nodiscard]] std::uint64_t slot_bytes() const;
    // port (sm_89): byte origin of a bank window, so a strided bank view can rebase onto it. The
    // pool is a flat slot space, so this is always 0; it stays so such a view keeps compiling.
    [[nodiscard]] std::uint64_t window_origin(std::uint32_t bank) const;
    [[nodiscard]] std::uint64_t uploads() const noexcept;
    [[nodiscard]] std::uint64_t evictions() const noexcept;
    [[nodiscard]] std::uint64_t hits() const noexcept;
    // port (sm_89): how many banks' worth of slots the pool currently backs. Kept as the reported
    // pool size unit; the pool itself is one flat slot space.
    [[nodiscard]] std::uint64_t windows() const noexcept;

    // Experts of one bank the pool can hold at once. Equals the widest bank when a whole layer
    // fits; smaller when the pool only holds part of one, which is the case a host tier covers.
    [[nodiscard]] std::uint32_t resident_capacity() const noexcept;

    // port (sm_89): global slot of an expert, or nullopt when the pool does not hold it. The slot
    // is stable across set_resident_experts() calls for experts that stay, which is what lets a
    // caller rebuild its offset table without reuploading anything.
    [[nodiscard]] std::optional<std::uint32_t> expert_slot(std::uint32_t bank,
                                                           std::uint32_t expert) const noexcept;

    // port (sm_89): true when the pool can hold every bank's whole layer at the same time, so no
    // layer ever evicts another and the resident set is fixed for the pool's lifetime. That is the
    // only configuration a captured decode graph can use: replay does not run the host code that
    // would otherwise pick and upload experts.
    [[nodiscard]] bool can_hold_every_layer() const noexcept;

    // port (sm_89): true when every slot of this bank's run is backed by device memory. A resize
    // can drop the highest runs, so a pinned stream checks this before trusting a cached offset.
    [[nodiscard]] bool window_resident(std::uint32_t bank) const noexcept;

    // port (sm_89): uploads every expert of every bank so a capture never records an upload.
    // Requires can_hold_every_layer(); does nothing otherwise.
    void preload_all();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer




