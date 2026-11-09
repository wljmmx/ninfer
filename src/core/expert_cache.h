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
    // Address-space reservation ceiling for runtime growth. 0 means "same as capacity_bytes"
    // (the historical behaviour: the pool could only shrink). When set higher, the VMM range is
    // reserved to this size at construction and only the capacity_bytes slots are mapped, so
    // resize() can grow up to the ceiling by mapping segments from the driver on demand.
    std::uint64_t capacity_limit_bytes = 0;
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
    // port (sm_89): A1. `hot_count` is how many of the leading entries are the hot set; the rest are
    // the spill ring and are reported separately, because the two segments have different lifetimes:
    // the hot segment may only move at an epoch boundary, the ring moves every step. With the default
    // (all entries hot) the two counters move together, i.e. exactly the previous behaviour.
    bool set_resident_experts(std::uint32_t bank, std::span<const std::uint32_t> experts,
                              std::uint32_t hot_count = UINT32_MAX);

    // Q20: the Device chose an expert and a ring slot; the host has already copied the bytes. This
    // adopts that pair as residency without assigning a slot of its own and without queueing an
    // upload. A slot previously owned by another expert is released to the shared free list, so a
    // ring that rotates every layer does not leak. Returns false when the pair cannot be represented
    // (out of range, or the slot belongs to the prediction region).
    bool adopt_resident_slot(std::uint32_t bank, std::uint32_t expert, std::uint32_t slot);

    // Q20: a fixed-S ring per bank, taken out of the shared free list once. The ring slots have
    // stable ids, which is what lets the Device name a destination without the host choosing it.
    // Without this reservation, "ring slot 3" is just a position in a list and its address belongs
    // to whichever expert the allocator put there.
    void configure_ring(std::uint32_t bank, std::uint32_t slots);
    [[nodiscard]] std::uint32_t ring_slot_count(std::uint32_t bank) const noexcept;
    [[nodiscard]] std::optional<std::uint32_t> ring_slot(std::uint32_t bank,
                                                         std::uint32_t index) const noexcept;

    // A1: the two generations, split. `hot_generation` moves only when the hot segment's membership
    // does; `step_generation` moves whenever the whole set does. A publisher that keeps its Device
    // tables in two segments compares each against its own counter, which is what lets an
    // epoch-boundary hot re-placement coexist with a per-step ring delta.
    [[nodiscard]] std::uint64_t hot_generation(std::uint32_t bank) const noexcept;
    [[nodiscard]] std::uint64_t step_generation(std::uint32_t bank) const noexcept;
    // A1: the third segment - the prediction region, if one was carved. It moves on a placement, not
    // on a residency decision, and it must not invalidate either of the other two.
    [[nodiscard]] std::uint64_t prefetch_generation(std::uint32_t bank) const noexcept;
    // The region's experts in ring order, with holes (kAbsentSlot) where a slot has been reused but
    // the publisher has not rewritten it. The publisher walks this and skips the holes.
    [[nodiscard]] const std::vector<std::uint32_t>& prefetch_owners(std::uint32_t bank) const;

    // port (sm_89): Q30/P6. Places experts the pool does NOT already hold into a per-bank ring of
    // reserved slots and returns the keys that need an upload. Experts already resident are ignored:
    // a prediction that is already there is not a prediction, and the delta is what sizes the region.
    std::uint32_t set_prefetch_experts(std::uint32_t bank, std::span<const std::uint32_t> experts,
                                       std::vector<ExpertCacheKey>& placed);
    [[nodiscard]] bool prefetch_slot(std::uint32_t slot) const noexcept;
    [[nodiscard]] std::uint32_t prefetch_slots_per_bank() const noexcept;

    // port (sm_89): the pinned RAM tier the host half keeps its working set in. When set, an upload
    // of an expert that tier already holds copies from locked memory instead of the pageable artifact
    // mapping, which is worth ~0.34 ms a layer: the driver can DMA straight out of a pinned buffer
    // but has to bounce through its own staging for a mapped one. Attach after both exist; before
    // that, uploads fall back to the artifact reader.
    void attach_arena(const ExpertArenaSource* arena) noexcept;

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
    // port (sm_89): how many experts this stream actually moved into the pool on the last turn.
    // It is the only honest miss count - the residency mask is published from the same decision that
    // places the routes, so asking it "was this a miss?" always answers no.
    [[nodiscard]] std::uint32_t last_uploaded() const noexcept;
    // Diagnostic: how many of these experts are mapped into the pool but their bytes have not landed.
    // A route the mask calls resident whose slot is not ready reads whatever the slot held before, and
    // that is silent - the arithmetic is right, the weights are not. Counted by name rather than by
    // hashing the payload, because the payload's address is the thing under test.
    [[nodiscard]] std::uint32_t unready_count(std::uint32_t bank,
                                              std::span<const std::uint32_t> experts) const;
    // Bumped whenever a bank's resident set actually changes. The publisher uses it to prove a skipped
    // publication was skipped for the right reason: "the table did not move" is a claim about the
    // cache, so it is the cache that has to say so (A3).
    [[nodiscard]] std::uint64_t layout_generation() const noexcept;
    // port (sm_89): the same fact per bank. `layout_generation()` is one counter for the whole pool, so
    // a bank that published at generation 1 looks stale the moment any of the other 39 banks places an
    // expert - which is 39 times out of 40. A validity check has to compare like with like: what this
    // bank's Device tables hold against what this bank's set now is. Measured the hard way: a per-bank
    // table generation checked against the global counter fired on the default path with `1 vs 40`.
    [[nodiscard]] std::uint64_t bank_generation(std::uint32_t bank) const noexcept;
    // Where the images came from, and what the cold path cost to read. The warm-tier decision (1y.7
    // step 3f) needs both sides: an arena hit skips the read entirely, so "is a managed copy worth
    // its admission cost" is a ratio between read_us and the copy that filling it costs.
    struct UploadSourceStats {
        std::uint64_t arena_experts = 0; // served from the pinned warm copy
        std::uint64_t arena_probes  = 0; // arena hits deliberately served cold, to keep that side measured
        std::uint64_t file_experts  = 0; // read out of the artifact mapping
        double read_us              = 0.0; // host time spent in read_entry, cumulative
    };
    [[nodiscard]] UploadSourceStats upload_sources() const;
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





