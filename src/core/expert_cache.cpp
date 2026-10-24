#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include "core/vmm_pool.h"
#include "core/device_scope.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::uint64_t kAbsent = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t kAbsentSlot = std::numeric_limits<std::uint32_t>::max();

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

LayoutRegion add_plane(LayoutBuilder& builder, const WeightGeometry& geometry, bool codes,
                       bool high, bool scales) {
    const auto bytes = codes    ? geometry.code_bytes
                       : high   ? geometry.high_bytes
                                : geometry.scale_bytes;
    if (!bytes) { return {}; }
    return builder.add(static_cast<std::size_t>(bytes), 256, "expert plane");
}

} // namespace

ExpertCacheRequirements describe_expert_cache(std::span<const ExpertBankSpec> banks,
                                              const ExpertCacheConfig& config) {
    if (!config.capacity_bytes || config.capacity_bytes % 256 || !config.upload_slots) {
        throw std::invalid_argument("expert cache requires aligned capacity and upload slots");
    }
    ExpertCacheRequirements out;
    out.config = config;
    LayoutBuilder offsets;
    for (const auto& bank : banks) {
        if (bank.gate_up.geometry.shape.size() != 2 || bank.down.geometry.shape.size() != 2) {
            throw std::invalid_argument("expert bank requires one 2D parent per expert");
        }
        ExpertBankLayout layout;
        layout.count    = bank.count;
        layout.resident = bank.resident;
        if (!layout.count) { throw std::invalid_argument("expert bank has no experts"); }
        LayoutBuilder entry;
        layout.codes  = add_plane(entry, bank.gate_up.geometry, true, false, false);
        layout.high   = add_plane(entry, bank.gate_up.geometry, false, true, false);
        layout.scales = add_plane(entry, bank.gate_up.geometry, false, false, true);
        // The down matrix follows the same plane order inside the same entry image. Its offsets
        // must be kept: an uploader and an indexed bank both address down through them.
        layout.down_codes  = add_plane(entry, bank.down.geometry, true, false, false);
        layout.down_high   = add_plane(entry, bank.down.geometry, false, true, false);
        layout.down_scales = add_plane(entry, bank.down.geometry, false, false, true);
        layout.payload_bytes = bank.gate_up.geometry.code_bytes + bank.gate_up.geometry.high_bytes +
                               bank.gate_up.geometry.scale_bytes + bank.down.geometry.code_bytes +
                               bank.down.geometry.high_bytes + bank.down.geometry.scale_bytes;
        layout.entry_bytes = entry.finish(256, "expert entry");
        out.max_entry_bytes = std::max(out.max_entry_bytes,
                                       static_cast<std::size_t>(layout.entry_bytes));
        out.banks.push_back(layout);
        if (!layout.resident) { out.entry_count += layout.count; }
    }
    out.offset_bytes = offsets.finish();
    return out;
}

struct ExpertCache::Impl {
    struct Entry {
        std::uint64_t offset  = kAbsent;
        std::uint64_t last_use = 0;
        bool ready             = false;
    };

    ExpertCacheConfig config;
    std::vector<ExpertBankLayout> banks;
    std::vector<ExpertBankPair> views;
    std::vector<ExpertFileBank> files;
    const ExpertBlobReader* file_reader = nullptr;
    // One upload turn carries several experts: the file read is serialised per image anyway, but
    // each image would otherwise pay its own cudaStreamSynchronize (~2.5 ms of WDDM submission on
    // sm_89). The staging buffer therefore holds one slot image per batch position.
    static constexpr std::size_t kUploadBatch = 4;
    // port (sm_89): smallest hot set a bank keeps when the pool cannot hold whole layers. One
    // decode step routes top_k=8 experts, so 8 lets every layer coexist even in a tiny pool; a
    // narrower window than that cannot hold one step's route and would thrash.
    static constexpr std::uint64_t kMinimumHotSet = 8;
    std::size_t max_entry_bytes               = 0;
    std::unique_ptr<PinnedHostBuffer> staging;
    // port (sm_89): the pool is a VMM reservation rather than one cudaMalloc, so whole segments can
    // be handed back to the driver while the pool is alive. Every expert offset indexes into this
    // reservation, so shrinking the pool never moves a published entry.
    static constexpr std::size_t kSegmentBytes = 8u << 20;
    std::unique_ptr<VmmPool> pool;
    // Bytes between two bank windows. Expert offsets inside a window stay `slot * slot_bytes`.
    std::uint64_t window_stride = 0;
    // Windows the reservation can hold, and windows currently backed by device memory.
    std::uint64_t windows_max  = 0;
    std::uint64_t windows_live = 0;
    std::vector<std::size_t> first; // flat entry index of each bank's expert 0
    std::vector<Entry> entries;
    std::vector<ExpertCacheKey> required;
    std::uint64_t used = 0, uploads = 0, evictions = 0, epoch = 0, hits = 0;
    std::uint64_t slot_bytes = 0;
    std::vector<std::uint64_t> window_of;
    std::vector<std::uint64_t> last_window_use;
    std::vector<std::uint64_t> window_hits;
    // Slot index of each expert inside its bank window, or kAbsentSlot when the pool does not hold
    // it. The default is the identity, so a fully resident bank keeps addressing expert i at slot i.
    std::vector<std::vector<std::uint32_t>> slot_of;
    // Upper bound on how many experts one bank window may hold. Zero means "no bound", which makes
    // window_slots() take the whole pool and leaves room for exactly one bank: every layer then
    // reuploads the entire window. A cap trades resident experts per bank for more coexisting
    // windows, which is what keeps a split pool from reuploading hundreds of megabytes per layer.
    std::uint64_t window_cap = 0;
    int ordinal = 0;

    Impl(std::span<const ExpertBankSpec> specs, const ExpertCacheRequirements& req,
         const DeviceContext& device, std::span<const ExpertBankPair> bank_views,
         std::span<const ExpertFileBank> file_banks, const ExpertBlobReader* reader)
        : config(req.config), banks(req.banks), views(bank_views.begin(), bank_views.end()),
          files(file_banks.begin(), file_banks.end()), file_reader(reader),
          max_entry_bytes(req.max_entry_bytes),
          staging(req.max_entry_bytes
                      ? std::make_unique<PinnedHostBuffer>(
                            static_cast<std::size_t>(req.max_entry_bytes) * kUploadBatch)
                      : nullptr) {
        if (req != describe_expert_cache(specs, req.config)) {
            throw std::invalid_argument("expert cache requirements differ from their sources");
        }
        if (!bank_views.empty() && views.size() != banks.size()) {
            throw std::invalid_argument("expert cache views differ from their banks");
        }
        if (bank_views.empty()) { views.resize(banks.size()); }
        std::size_t flat = 0;
        for (const auto& bank : banks) {
            first.push_back(flat);
            if (!bank.resident) { flat += bank.count; }
        }
        entries.resize(flat);
        // port (sm_89): a strided bank must advance by a whole layout block. This artifact stores
        // row_split_k128 codes at 1024 bytes/row, so one block is 128 KiB.
        constexpr std::uint64_t block = 128 * 1024;
        slot_bytes = (req.max_entry_bytes + block - 1) / block * block;
        // One window per bank means every layer evicts the previous bank and reuploads a whole
        // window. Splitting the pool into two windows halves that traffic, but it also halves the
        // experts a bank can keep, so it only pays when the pool is nearly large enough for a whole
        // layer: there the reupload is the dominant cost (measured 82 -> 92 tok/s at 251 slots),
        // while at half a layer the lost Device work outweighs it (92 -> 88 at 135 slots).
        const auto pool_slots = req.config.capacity_bytes / slot_bytes;
        std::uint64_t widest_bank = 0;
        std::uint64_t layer_count = 0;
        for (const auto& bank : req.banks) {
            if (bank.resident) { continue; }
            widest_bank = std::max(widest_bank, std::uint64_t{bank.count});
            ++layer_count;
        }
        // port (sm_89): a window holds one bank's resident set, and eviction happens per window. As
        // long as a window can hold a whole layer, a pool that misses even one layer pays a full
        // layer reupload every step: 39 windows over 40 layers measured 20.1 tok/s against 203.9
        // when every layer fit. Divide the slots rather than the windows, so each bank keeps a
        // smaller permanent set and no layer is ever evicted. A bank that cannot hold its whole
        // layer then reports capacity < count, which sends the rest of the layer down the existing
        // split path: resident experts run on the Device, the others on the host tier.
        //
        // That is the shape a model larger than the card needs: keep a per-layer hot set, leave
        // every other expert in host memory, and never spill VRAM to a window swap.
        const auto whole_layers = widest_bank * layer_count;
        if (widest_bank != 0 && layer_count != 0) {
            if (pool_slots >= whole_layers) {
                window_cap = widest_bank;
            } else {
                window_cap = std::max<std::uint64_t>(kMinimumHotSet, pool_slots / layer_count);
            }
        }
        if (const char* cap = std::getenv("NINFER_EXPERT_WINDOW_SLOTS")) {
            const auto requested = std::strtoull(cap, nullptr, 10);
            if (requested > 0) { window_cap = requested; }
        }
        window_of.assign(banks.size(), kAbsent);
        last_window_use.assign(banks.size(), 0);
        window_hits.assign(banks.size(), 0);
        slot_of.resize(banks.size());
        // port (sm_89): only the experts that fit the window start resident. The identity mapping
        // is right when the window holds the whole layer; a smaller window must start with the rest
        // absent, or a stale slot index would point past the window.
        const auto initial_slots = window_slots();
        for (std::size_t b = 0; b < banks.size(); ++b) {
            slot_of[b].resize(banks[b].count);
            for (std::uint32_t e = 0; e < banks[b].count; ++e) {
                slot_of[b][e] = e < initial_slots ? e : kAbsentSlot;
            }
        }
        ordinal    = device.device;
        // port (sm_89): reserve one contiguous address range and back every window with its own
        // segments. A window keeps its offset for the pool's lifetime; only its backing pages can
        // be returned to the driver by resize().
        window_stride = window_slots() * slot_bytes;
        if (window_stride == 0) { throw std::invalid_argument("expert cache holds no window"); }
        const auto reserved = static_cast<std::size_t>(
            (req.config.capacity_bytes + kSegmentBytes - 1) / kSegmentBytes * kSegmentBytes);
        pool = std::make_unique<VmmPool>(Device::cuda(ordinal), reserved, kSegmentBytes);
        windows_max  = reserved / window_stride;
        windows_live = windows_max;
    }

    // Keeps the first `experts` entries that fit in one window and marks every other expert of the
    // bank absent. The caller recomputes and republishes its offset table afterwards. Returns true
    // when the resident set changed. Experts that stay resident keep their slot, so a step that
    // routes to experts already present uploads nothing and the window is never discarded.
    bool set_resident(std::uint32_t bank, std::span<const std::uint32_t> experts) {
        if (bank >= banks.size()) { throw std::out_of_range("expert cache bank"); }
        auto& slots = slot_of[bank];
        const auto limit = window_slots();
        std::vector<std::uint32_t> wanted;
        wanted.reserve(std::min<std::size_t>(experts.size(), limit));
        std::vector<bool> keep(slots.size(), false);
        for (const std::uint32_t expert : experts) {
            if (expert >= slots.size()) { throw std::out_of_range("expert cache expert"); }
            if (wanted.size() >= limit) { break; }
            if (keep[expert]) { continue; }
            keep[expert] = true;
            wanted.push_back(expert);
        }
        bool changed = false;
        for (std::size_t expert = 0; expert < slots.size(); ++expert) {
            if ((slots[expert] != kAbsentSlot) != keep[expert]) {
                changed = true;
                break;
            }
        }
        // Consecutive steps usually route to the same experts. Keeping the window when the set is
        // unchanged turns a per-token reupload into a hit.
        if (!changed) { return false; }
        // An expert that stays keeps its slot: only those leaving release their slot and only the
        // arrivals lose their cached image.
        std::vector<std::uint32_t> next(slots.size(), kAbsentSlot);
        std::vector<bool> taken(limit, false);
        for (std::size_t expert = 0; expert < slots.size(); ++expert) {
            if (slots[expert] == kAbsentSlot || !keep[expert]) { continue; }
            next[expert]         = slots[expert];
            taken[slots[expert]] = true;
        }
        std::uint32_t free_slot = 0;
        for (const std::uint32_t expert : wanted) {
            if (next[expert] != kAbsentSlot) { continue; }
            while (free_slot < limit && taken[free_slot]) { ++free_slot; }
            if (free_slot >= limit) { break; }
            next[expert]              = free_slot;
            taken[free_slot]          = true;
            entries[first[bank] + expert].ready = false;
        }
        for (std::size_t expert = 0; expert < slots.size(); ++expert) {
            if (slots[expert] == kAbsentSlot || keep[expert]) { continue; }
            entries[first[bank] + expert].ready = false;
        }
        slots = std::move(next);
        required.erase(std::remove_if(required.begin(), required.end(),
                                      [&](const ExpertCacheKey& key) {
                                          return key.bank == bank &&
                                                 slot_of[bank][key.expert] == kAbsentSlot;
                                      }),
                       required.end());
        return true;
    }

    std::size_t index(ExpertCacheKey key) const {
        if (key.bank >= banks.size() || banks[key.bank].resident ||
            key.expert >= banks[key.bank].count) {
            throw std::out_of_range("expert cache key");
        }
        return first[key.bank] + key.expert;
    }

    // port (sm_89): slots are uniform. Banks may differ in entry size (a layer whose down matrix
    // carries a high plane is 128 KiB larger), and an evicted slot is handed to another bank, so
    // only a pool of max_entry_bytes slots can be reused safely.
    std::uint64_t window_slots() const {
        std::uint64_t widest = 0;
        for (const auto& bank : banks) {
            if (!bank.resident) { widest = std::max(widest, std::uint64_t{bank.count}); }
        }
        // A pool smaller than one layer still serves it, one expert short of a whole window: the
        // window shrinks to the pool and the host tier is expected to cover the remainder.
        const std::uint64_t slots = config.capacity_bytes / slot_bytes;
        auto limit = std::min(widest, slots);
        if (window_cap != 0) { limit = std::min(limit, window_cap); }
        return limit;
    }

    std::uint64_t window_count() const {
        return windows_live;
    }

    // Bytes the pool is allowed to use right now.
    std::uint64_t live_bytes() const { return windows_live * window_stride; }

    // Every bank's whole layer fits at once: one window per bank, never evicted.
    bool holds_every_layer() const noexcept {
        std::uint64_t needed = 0;
        std::uint64_t widest = 0;
        for (const auto& bank : banks) {
            if (bank.resident) { continue; }
            ++needed;
            widest = std::max<std::uint64_t>(widest, bank.count);
        }
        return needed != 0 && widest <= window_slots() && needed <= windows_live;
    }

    // Uploads the whole resident set once. A captured graph freezes the expert addresses, so every
    // expert the decode may route to has to be present before the capture runs; afterwards the
    // steady state touches no upload path at all.
    void preload_every_layer() {
        if (!pool || !holds_every_layer()) { return; }
        DeviceScope scope(Device::cuda(ordinal));
        std::size_t staged = 0;
        for (std::uint32_t b = 0; b < banks.size(); ++b) {
            if (banks[b].resident) { continue; }
            for (std::uint32_t e = 0; e < banks[b].count; ++e) {
                if (entries[first[b] + e].ready) { continue; }
                upload_one({b, e}, nullptr, false, staged);
                if (++staged == kUploadBatch) {
                    const auto settled = cudaStreamSynchronize(nullptr);
                    if (settled != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(settled)); }
                    staged = 0;
                }
            }
        }
        if (staged != 0) {
            const auto settled = cudaStreamSynchronize(nullptr);
            if (settled != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(settled)); }
        }
    }

    // Hands device memory back to (or takes it from) the driver. Windows above the new count are
    // dropped, so their experts stream again; the caller must be at a layer boundary because an
    // unmapped window that a kernel still reads is a use-after-free.
    std::uint64_t resize(std::uint64_t target_bytes) {
        if (!pool) { return 0; }
        auto wanted = target_bytes / window_stride;
        wanted      = std::max<std::uint64_t>(wanted, 1);
        wanted      = std::min<std::uint64_t>(wanted, windows_max);
        if (wanted == windows_live) { return live_bytes(); }
        DeviceScope scope(Device::cuda(ordinal));
        const auto settled = cudaDeviceSynchronize();
        if (settled != cudaSuccess) {
            throw std::runtime_error(std::string("expert pool resize: ") +
                                     cudaGetErrorString(settled));
        }
        // Segments are fixed size, so the backing range is the window bytes rounded up.
        const auto keep = static_cast<std::size_t>(
            (wanted * window_stride + kSegmentBytes - 1) / kSegmentBytes * kSegmentBytes);
        if (wanted < windows_live) {
            for (std::uint32_t b = 0; b < banks.size(); ++b) {
                if (window_of[b] == kAbsent || window_of[b] < wanted) { continue; }
                window_of[b] = kAbsent;
                for (std::uint32_t e = 0; e < banks[b].count; ++e) {
                    entries[first[b] + e].ready = false;
                }
            }
            pool->release(pool->mapped() - keep, keep);
        } else {
            pool->acquire(keep - pool->mapped());
        }
        windows_live = wanted;
        return live_bytes();
    }

    // port (sm_89): a bank owns one contiguous run of slots, so its experts form a strided
    // matrix. Windows are replaced whole: addressing stays affine and no per-expert offset
    // table can go stale while a kernel reads it.
    void ensure_window(std::uint32_t bank) {
        if (window_of[bank] != kAbsent) { last_window_use[bank] = ++epoch; ++window_hits[bank]; ++hits; return; }
        const auto windows = window_count();
        if (windows == 0) { throw std::runtime_error("expert cache holds no complete window"); }
        std::vector<bool> used(windows, false);
        for (std::uint32_t b = 0; b < banks.size(); ++b) {
            if (window_of[b] != kAbsent) { used[window_of[b]] = true; }
        }
        std::uint64_t chosen = kAbsent;
        for (std::uint64_t w = 0; w < windows; ++w) {
            if (!used[w]) { chosen = w; break; }
        }
        if (chosen == kAbsent) {
            std::uint32_t victim = static_cast<std::uint32_t>(banks.size());
            // port (sm_89): layers are visited in a fixed cycle, so plain recency evicts exactly
            // the layer needed next. Keep the most used windows and stream the rest.
            std::uint64_t coldest = std::numeric_limits<std::uint64_t>::max();
            for (std::uint32_t b = 0; b < banks.size(); ++b) {
                if (window_of[b] != kAbsent && window_hits[b] < coldest) {
                    coldest = window_hits[b];
                    victim = b;
                }
            }
            if (victim == banks.size()) {
                throw std::runtime_error("expert cache has no evictable window");
            }
            chosen = window_of[victim];
            window_of[victim] = kAbsent;
            for (std::uint32_t expert = 0; expert < banks[victim].count; ++expert) {
                entries[first[victim] + expert].ready = false;
            }
            ++evictions;
        }
        window_of[bank]       = chosen;
        last_window_use[bank] = ++epoch;
    }


    void upload_one(ExpertCacheKey key, cudaStream_t stream, bool settle = true,
                    std::size_t staging_slot = 0) {
        const auto& bank = banks[key.bank];
        ensure_window(key.bank);
        auto& entry = entries[index(key)];
        const auto slot = slot_of[key.bank][key.expert];
        if (slot == kAbsentSlot) { return; }
        entry.offset = window_of[key.bank] * window_stride +
                       std::uint64_t(slot) * slot_bytes;
        if (!pool || pool->mapped() == 0) {
            throw std::runtime_error("expert pool has no Device storage");
        }
        const auto limit = pool->mapped();
        if (entry.offset == kAbsent || entry.offset > limit ||
            bank.entry_bytes > limit - entry.offset) {
            throw std::runtime_error("expert slot is inconsistent: offset=" +
                                     std::to_string(entry.offset) + " entry=" +
                                     std::to_string(bank.entry_bytes) + " pool=" +
                                     std::to_string(limit) + " bank=" +
                                     std::to_string(key.bank) + " expert=" +
                                     std::to_string(key.expert));
        }
        auto* base = static_cast<std::byte*>(pool->data()) + entry.offset;

        const ExpertFileEntry* file = nullptr;
        if (file_reader && key.bank < files.size() &&
            key.expert < files[key.bank].experts.size()) {
            const auto& candidate = files[key.bank].experts[key.expert];
            if (candidate.image_bytes) { file = &candidate; }
        }
        if (file) {
            // port (sm_89): a source that already holds the bytes in place (the Host replica)
            // names the slices, so the transfer goes straight from host memory to the pool.
            std::vector<ExpertBlobSpan> direct;
            if (file_reader->spans(*file, direct) && !direct.empty()) {
                for (const auto& span : direct) {
                    if (span.image_offset > file->image_bytes ||
                        span.bytes > file->image_bytes - span.image_offset) {
                        throw std::runtime_error("expert span lies outside its entry image");
                    }
                    const auto copied =
                        cudaMemcpyAsync(base + span.image_offset, span.source,
                                        static_cast<std::size_t>(span.bytes),
                                        cudaMemcpyHostToDevice, stream);
                    if (copied != cudaSuccess) {
                        throw std::runtime_error(cudaGetErrorString(copied));
                    }
                }
                if (settle) {
                    const auto settled = cudaStreamSynchronize(stream);
                    if (settled != cudaSuccess) {
                        throw std::runtime_error(cudaGetErrorString(settled));
                    }
                }
            } else {
                // The artifact is the source: one fetch into staging, one copy to the pool.
                if (!staging || file->image_bytes > max_entry_bytes ||
                    staging->size() < (staging_slot + 1) * max_entry_bytes) {
                    throw std::runtime_error("expert file image exceeds the staging buffer");
                }
                auto* host_base = static_cast<std::byte*>(staging->data()) +
                                  staging_slot * max_entry_bytes;
                file_reader->read_entry(
                    *file, {host_base, static_cast<std::size_t>(file->image_bytes)});
                const auto status =
                    cudaMemcpyAsync(base, host_base, static_cast<std::size_t>(file->image_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
                if (settle) { (void)cudaStreamSynchronize(stream); }
            }
        } else {
            const auto& view = views[key.bank];
            if (!view.gate_up.valid() || !view.down.valid()) {
                throw std::runtime_error("expert has neither an artifact entry nor a Host replica");
            }
            const auto copy  = [&](ExpertPlaneKind kind, LayoutRegion region, bool down) {
                if (!region.bytes) { return; }
                const auto span = down ? view.down.plane(kind, key.expert)
                                       : view.gate_up.plane(kind, key.expert);
                if (span.device != Device::cpu() || span.bytes != region.bytes) {
                    throw std::runtime_error("expert cache source is not a matching Host plane");
                }
                const auto status = cudaMemcpyAsync(base + region.offset, span.data, region.bytes,
                                                    cudaMemcpyHostToDevice, stream);
                if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
            };
            copy(ExpertPlaneKind::Codes, bank.codes, false);
            copy(ExpertPlaneKind::High, bank.high, false);
            copy(ExpertPlaneKind::Scales, bank.scales, false);
            copy(ExpertPlaneKind::Codes, bank.down_codes, true);
            copy(ExpertPlaneKind::High, bank.down_high, true);
            copy(ExpertPlaneKind::Scales, bank.down_scales, true);
            if (settle) { (void)cudaStreamSynchronize(stream); }
        }
        entry.ready    = true;
        entry.last_use = ++epoch;
        ++uploads;
    }
};

ExpertCache::ExpertCache(std::span<const ExpertBankSpec> specs,
                         const ExpertCacheRequirements& requirements, const DeviceContext& device,
                         std::span<const ExpertBankPair> views,
                         std::span<const ExpertFileBank> files, const ExpertBlobReader* reader)
    : impl_(std::make_unique<Impl>(specs, requirements, device, views, files, reader)) {}

ExpertCache::~ExpertCache() = default;

void ExpertCache::prepare(std::span<const ExpertCacheKey> required) {
    // Experts the pool does not hold are dropped here: the caller masks them out of the GPU job and
    // serves them from the host tier, so uploading them would only evict a resident expert.
    impl_->required.clear();
    for (const auto& key : required) {
        if (key.bank < impl_->slot_of.size() && key.expert < impl_->slot_of[key.bank].size() &&
            impl_->slot_of[key.bank][key.expert] != kAbsentSlot) {
            impl_->required.push_back(key);
        }
    }
}

bool ExpertCache::set_resident_experts(std::uint32_t bank,
                                       std::span<const std::uint32_t> experts) {
    return impl_->set_resident(bank, experts);
}

std::optional<std::uint32_t> ExpertCache::expert_slot(std::uint32_t bank,
                                                      std::uint32_t expert) const noexcept {
    if (bank >= impl_->slot_of.size() || expert >= impl_->slot_of[bank].size()) {
        return std::nullopt;
    }
    const auto slot = impl_->slot_of[bank][expert];
    if (slot == kAbsentSlot) { return std::nullopt; }
    return slot;
}

ExpertCacheProgress ExpertCache::progress() {
    std::size_t staged = 0;
    for (const auto& key : impl_->required) {
        const auto& entry = impl_->entries[impl_->index(key)];
        if (entry.ready) { continue; }
        if (staged == Impl::kUploadBatch) { break; }
        impl_->upload_one(key, nullptr, false, staged);
        ++staged;
    }
    if (staged == 0) { return {0, true}; }
    // One settle for the whole turn: the copies were queued back to back on the same stream.
    const auto settled = cudaStreamSynchronize(nullptr);
    if (settled != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(settled)); }
    return {staged, false};
}

void ExpertCache::wait_for_progress(cudaStream_t stream) {
    (void)stream;
    while (!progress().required_published) {}
}

std::optional<std::uint64_t> ExpertCache::offset(ExpertCacheKey key) const {
    if (key.bank >= impl_->banks.size() || impl_->banks[key.bank].resident ||
        key.expert >= impl_->banks[key.bank].count) {
        return std::nullopt;
    }
    if (impl_->slot_of[key.bank][key.expert] == kAbsentSlot) { return std::nullopt; }
    const auto& entry = impl_->entries[impl_->index(key)];
    if (!entry.ready) { return std::nullopt; }
    return entry.offset;
}

std::uint64_t ExpertCache::used_bytes() const noexcept {
    std::uint64_t windows = 0;
    for (const auto window : impl_->window_of) {
        if (window != std::numeric_limits<std::uint64_t>::max()) { ++windows; }
    }
    return windows * impl_->window_slots() * impl_->slot_bytes;
}
std::uint64_t ExpertCache::capacity_bytes() const noexcept { return impl_->live_bytes(); }
std::uint64_t ExpertCache::slot_bytes() const { return impl_->slot_bytes; }
std::uint64_t ExpertCache::resize(std::uint64_t target_bytes) { return impl_->resize(target_bytes); }
std::uint64_t ExpertCache::window_origin(std::uint32_t bank) const {
    if (bank >= impl_->window_of.size() || impl_->window_of[bank] == kAbsent) {
        throw std::out_of_range("expert bank has no resident window");
    }
    return impl_->window_of[bank] * impl_->window_stride;
}
ConstMemorySpan ExpertCache::pool() const {
    // The whole reserved range: a caller may address any window the pool can still provide, and an
    // offset is only ever valid while resize() keeps that window mapped.
    return {impl_->pool->data(), impl_->pool->size(), Device::cuda(impl_->ordinal)};
}
std::uint64_t ExpertCache::uploads() const noexcept { return impl_->uploads; }
std::uint64_t ExpertCache::evictions() const noexcept { return impl_->evictions; }
std::uint64_t ExpertCache::hits() const noexcept { return impl_->hits; }
std::uint64_t ExpertCache::windows() const noexcept { return impl_->window_count(); }
std::uint32_t ExpertCache::resident_capacity() const noexcept {
    const auto slots = impl_->live_bytes() / impl_->slot_bytes;
    const auto widest = impl_->window_slots();
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(slots, widest));
}

bool ExpertCache::can_hold_every_layer() const noexcept { return impl_->holds_every_layer(); }

bool ExpertCache::window_resident(std::uint32_t bank) const noexcept {
    return bank < impl_->window_of.size() && impl_->window_of[bank] != kAbsent;
}

void ExpertCache::preload_all() { impl_->preload_every_layer(); }

} // namespace ninfer





