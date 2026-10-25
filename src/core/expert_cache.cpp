#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include "core/vmm_pool.h"
#include "core/device_scope.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
    // port (sm_89): one layer's uploads are independent copies - distinct slots, distinct sources,
    // one direction - so the batch API can submit them as one. A submission costs ~10 us on this
    // driver whatever its size, and a split layer makes up to a dozen of them; the batch form also
    // lets the driver order them against each other instead of the host ordering them in turn.
    struct UploadOp {
        void* destination  = nullptr;
        const void* source = nullptr;
        std::size_t bytes  = 0;
    };
    // port (sm_89): smallest hot set a bank keeps when the pool cannot hold whole layers. One
    // decode step routes top_k=8 experts, so 8 lets every layer coexist even in a tiny pool; a
    // narrower window than that cannot hold one step's route and would thrash.
    static constexpr std::uint64_t kMinimumHotSet = 8;
    std::size_t max_entry_bytes               = 0;
    std::unique_ptr<PinnedHostBuffer> staging;
    // port (sm_89): the stream the consumer runs on. Uploads queued there are ordered before the
    // kernels that read them, so a Host-replica upload needs no barrier at all.
    cudaStream_t upload_stream = nullptr;
    // True when experts come from the artifact and therefore pass through the shared staging
    // buffer, whose slots cannot be reused until their copy has landed.
    bool staging_needed = false;
    // Which path each upload actually took, so the timing can say what the collect phase was
    // doing instead of inferring it from a constructor flag.
    std::uint64_t stat_direct = 0;
    std::uint64_t stat_staged = 0;
    std::uint64_t stat_views  = 0;
    // Counted at the one place every upload passes through, so the figure cannot disagree with the
    // batch it describes the way the per-branch counters did.
    std::uint64_t stat_push = 0;
    // Reused across layers. Allocating these per call looked free and was not: a split layer pays
    // the upload turn forty times a token, and on this allocator seventy-two entries of scratch is
    // most of the collect budget on a layer that only has one expert to move.
    std::vector<UploadOp> upload_ops;
    std::vector<void*> upload_destinations;
    std::vector<const void*> upload_sources;
    std::vector<std::size_t> upload_sizes;
    // port (sm_89): the pool is a VMM reservation rather than one cudaMalloc, so whole segments can
    // be handed back to the driver while the pool is alive. Every expert offset indexes into this
    // reservation, so shrinking the pool never moves a published entry.
    static constexpr std::size_t kSegmentBytes = 8u << 20;
    std::unique_ptr<VmmPool> pool;
    // port (sm_89): the pool is one flat slot space. Every expert lives at `slot * slot_bytes`
    // measured from the reservation's base, so no address is derived from a window number. Slots
    // are handed out from one shared free list rather than to fixed per-bank ranges: a bank's
    // resident set can then grow or shrink on demand instead of being pinned to a share of the
    // pool that another layer may be short of.
    std::uint64_t slot_cap   = 0; // slots one bank is expected to hold (the split hint)
    std::uint64_t slots_max  = 0; // slots the reservation can back
    std::uint64_t slots_live = 0; // slots backed by device memory right now
    std::vector<std::size_t> first; // flat entry index of each bank's expert 0
    std::vector<Entry> entries;
    std::vector<ExpertCacheKey> required;
    std::uint64_t used = 0, uploads = 0, evictions = 0, epoch = 0, hits = 0;
    std::uint64_t slot_bytes = 0;
    // Global slot of each expert, or kAbsentSlot when the pool does not hold it. A fully resident
    // bank keeps the identity, which is what lets a captured graph address expert i directly.
    std::vector<std::vector<std::uint32_t>> slot_of;
    // Every slot not owned by an expert. Free slots are shared: a bank that drops experts releases
    // them here and another bank's next resident set may take them.
    std::vector<std::uint32_t> free_slots;
    // How many experts of each bank own a slot right now.
    std::vector<std::uint32_t> resident_count;
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
        // port (sm_89): a bank owns a fixed run of slots. As long as a bank can hold a whole layer,
        // a pool that misses even one layer pays a full layer reupload every step: 39 windows over
        // 40 layers measured 20.1 tok/s against 203.9 when every layer fit. Divide the slots rather
        // than the banks, so each bank keeps a smaller permanent set and no layer is ever evicted.
        // A bank that cannot hold its whole layer then reports capacity < count, which sends the
        // rest of the layer down the existing split path: resident experts run on the Device, the
        // others on the host tier.
        //
        // That is the shape a model larger than the card needs: keep a per-layer hot set, leave
        // every other expert in host memory, and never spill VRAM to a slot swap.
        const auto whole_layers = widest_bank * layer_count;
        std::uint64_t cap = 1;
        if (widest_bank != 0 && layer_count != 0) {
            if (pool_slots >= whole_layers) {
                cap = widest_bank;
            } else {
                cap = std::max<std::uint64_t>(kMinimumHotSet, pool_slots / layer_count);
            }
        }
        if (const char* cap_env = std::getenv("NINFER_EXPERT_WINDOW_SLOTS")) {
            const auto requested = std::strtoull(cap_env, nullptr, 10);
            if (requested > 0) { cap = requested; }
        }
        ordinal    = device.device;
        for (const auto& bank : files) {
            for (const auto& expert : bank.experts) {
                if (expert.image_bytes) {
                    staging_needed = true;
                    break;
                }
            }
            if (staging_needed) { break; }
        }
        // port (sm_89): reserve one contiguous address range and back it with fixed-size segments.
        // Slot i always addresses the same place inside the reservation; only the backing pages can
        // be returned to the driver by resize().
        if (pool_slots == 0) {
            throw std::invalid_argument("expert cache holds no complete expert slot");
        }
        const auto reserved = static_cast<std::size_t>(
            (req.config.capacity_bytes + kSegmentBytes - 1) / kSegmentBytes * kSegmentBytes);
        pool = std::make_unique<VmmPool>(Device::cuda(ordinal), reserved, kSegmentBytes);
        slots_max  = reserved / slot_bytes;
        slots_live = pool_slots;
        // A pool narrower than one hot set per bank cannot give every bank its own run: narrow the
        // hot set instead of addressing past the reservation.
        if (layer_count != 0) {
            const auto per_bank = slots_max / layer_count;
            if (per_bank == 0) {
                throw std::invalid_argument("expert pool cannot hold one slot per bank");
            }
            cap = std::min(cap, per_bank);
        }
        slot_cap = cap;
        free_slots.clear();
        free_slots.reserve(static_cast<std::size_t>(slots_live));
        // Pushed high to low so the first pop hands out slot 0: a fresh bank then keeps the identity
        // mapping it needs for a graph, exactly as the per-bank runs used to lay it out.
        for (std::uint64_t slot = slots_live; slot > 0; --slot) {
            free_slots.push_back(static_cast<std::uint32_t>(slot - 1));
        }
        slot_of.resize(banks.size());
        resident_count.assign(banks.size(), 0);
        // port (sm_89): only the experts that fit the per-bank hint start resident. The identity
        // mapping is right when the hint holds the whole layer; a smaller hint must start with the
        // rest absent, or a stale slot index would point at another bank's expert.
        for (std::size_t b = 0; b < banks.size(); ++b) {
            slot_of[b].assign(banks[b].count, kAbsentSlot);
            if (banks[b].resident) { continue; }
            for (std::uint32_t e = 0; e < banks[b].count && e < slot_cap; ++e) {
                if (free_slots.empty()) {
                    throw std::invalid_argument("expert pool cannot seed every bank's hot set");
                }
                slot_of[b][e] = free_slots.back();
                free_slots.pop_back();
                ++resident_count[b];
            }
        }
    }

    // Makes `experts` the bank's resident set, in list order. The caller recomputes and republishes
    // its offset table afterwards. Returns true when the resident set changed. Experts that stay
    // resident keep their slot, so a step that routes to experts already present uploads nothing
    // and never discards them; slots of the ones leaving go back to the shared free list, and
    // arrivals take whatever the list has left.
    bool set_resident(std::uint32_t bank, std::span<const std::uint32_t> experts) {
        if (bank >= banks.size()) { throw std::out_of_range("expert cache bank"); }
        auto& slots = slot_of[bank];
        std::vector<std::uint32_t> wanted;
        wanted.reserve(experts.size());
        std::vector<bool> keep(slots.size(), false);
        for (const std::uint32_t expert : experts) {
            if (expert >= slots.size()) { throw std::out_of_range("expert cache expert"); }
            if (keep[expert]) { continue; }
            keep[expert] = true;
            wanted.push_back(expert);
        }
        // A bank may ask for more slots than the pool has left; the shared free list decides how
        // many it actually gets, and the caller reads the result back off the offset table.
        bool changed = false;
        for (std::size_t expert = 0; expert < slots.size(); ++expert) {
            if ((slots[expert] != kAbsentSlot) != keep[expert]) {
                changed = true;
                break;
            }
        }
        // Consecutive steps usually route to the same experts. Keeping the set when it is unchanged
        // turns a per-token reupload into a hit.
        if (!changed) { return false; }
        std::uint32_t held = 0;
        for (std::size_t expert = 0; expert < slots.size(); ++expert) {
            if (slots[expert] == kAbsentSlot) { continue; }
            if (keep[expert]) {
                ++held;
                continue;
            }
            // Leaving: hand the slot to whoever needs it next. This happens before the arrivals are
            // placed, so a swap of equal size always has the slots it is about to need.
            free_slots.push_back(slots[expert]);
            slots[expert]                       = kAbsentSlot;
            entries[first[bank] + expert].ready = false;
        }
        std::uint32_t taken = 0;
        for (const std::uint32_t expert : wanted) {
            if (held + taken >= wanted.size()) { break; }
            if (slots[expert] != kAbsentSlot) { continue; }
            if (free_slots.empty()) { break; }
            slots[expert]                       = free_slots.back();
            free_slots.pop_back();
            entries[first[bank] + expert].ready = false;
            ++taken;
        }
        resident_count[bank] = held + taken;
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
    // carries a high plane is 128 KiB larger), so only a slot of max_entry_bytes can be handed to
    // any bank.
    std::uint64_t slot_capacity() const { return slot_cap; }

    // Bytes the pool is allowed to use right now.
    std::uint64_t live_bytes() const { return slots_live * slot_bytes; }

    // Every bank's whole layer fits at once. Slots are shared, so what matters is the total: with
    // room for the sum of every bank, no bank is ever shortened.
    bool holds_every_layer() const noexcept {
        std::uint64_t needed = 0;
        for (const auto& bank : banks) {
            if (bank.resident) { continue; }
            needed += bank.count;
        }
        return needed != 0 && needed <= slots_live;
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

    // Hands device memory back to (or takes it from) the driver. Slots above the new count are
    // dropped, so their experts stream again; the caller must be at a layer boundary because an
    // unmapped slot that a kernel still reads is a use-after-free.
    std::uint64_t resize(std::uint64_t target_bytes) {
        if (!pool) { return 0; }
        auto wanted = target_bytes / slot_bytes;
        wanted      = std::max<std::uint64_t>(wanted, 1);
        wanted      = std::min<std::uint64_t>(wanted, slots_max);
        if (wanted == slots_live) { return live_bytes(); }
        DeviceScope scope(Device::cuda(ordinal));
        const auto settled = cudaDeviceSynchronize();
        if (settled != cudaSuccess) {
            throw std::runtime_error(std::string("expert pool resize: ") +
                                     cudaGetErrorString(settled));
        }
        // Segments are fixed size, so the backing range is the slot bytes rounded up.
        const auto keep = static_cast<std::size_t>(
            (wanted * slot_bytes + kSegmentBytes - 1) / kSegmentBytes * kSegmentBytes);
        if (wanted < slots_live) {
            // Experts above the new limit lose their slot; the reservation still owns those
            // addresses, so a grow later hands them out again from the free list.
            for (std::uint32_t b = 0; b < banks.size(); ++b) {
                if (banks[b].resident) { continue; }
                for (std::uint32_t e = 0; e < banks[b].count; ++e) {
                    const auto slot = slot_of[b][e];
                    if (slot == kAbsentSlot || slot < wanted) { continue; }
                    slot_of[b][e]                        = kAbsentSlot;
                    entries[first[b] + e].ready          = false;
                    --resident_count[b];
                }
            }
            free_slots.erase(std::remove_if(free_slots.begin(), free_slots.end(),
                                            [&](std::uint32_t slot) { return slot >= wanted; }),
                             free_slots.end());
            pool->release(pool->mapped() - keep, keep);
        } else {
            for (std::uint64_t slot = slots_live; slot < wanted; ++slot) {
                free_slots.push_back(static_cast<std::uint32_t>(slot));
            }
            pool->acquire(keep - pool->mapped());
        }
        slots_live = wanted;
        return live_bytes();
    }

    void upload_one(ExpertCacheKey key, cudaStream_t stream, bool settle = true,
                    std::size_t staging_slot = 0, std::vector<UploadOp>* batch = nullptr) {
        const auto& bank = banks[key.bank];
        auto& entry = entries[index(key)];
        const auto slot = slot_of[key.bank][key.expert];
        if (slot == kAbsentSlot) { return; }
        if (slot >= slots_live) {
            throw std::runtime_error("expert slot is not backed: slot=" + std::to_string(slot) +
                                     " live=" + std::to_string(slots_live));
        }
        entry.offset = std::uint64_t(slot) * slot_bytes;
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
                ++stat_direct;
                for (const auto& span : direct) {
                    if (span.image_offset > file->image_bytes ||
                        span.bytes > file->image_bytes - span.image_offset) {
                        throw std::runtime_error("expert span lies outside its entry image");
                    }
                    if (batch != nullptr) {
                        ++stat_push;
                        batch->push_back({base + span.image_offset, span.source, span.bytes});
                        continue;
                    }
                    const auto copied =
                        cudaMemcpyAsync(base + span.image_offset, span.source,
                                        static_cast<std::size_t>(span.bytes),
                                        cudaMemcpyHostToDevice, stream);
                    if (copied != cudaSuccess) {
                        throw std::runtime_error(cudaGetErrorString(copied));
                    }
                }
                if (batch == nullptr && settle) {
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
                ++stat_staged;
                if (batch != nullptr) {
                    ++stat_push;
                    batch->push_back({base, host_base, static_cast<std::size_t>(file->image_bytes)});
                    entry.ready    = true;
                    entry.last_use = ++epoch;
                    ++uploads;
                    return;
                }
                const auto status =
                    cudaMemcpyAsync(base, host_base, static_cast<std::size_t>(file->image_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
                if (settle) { (void)cudaStreamSynchronize(stream); }
            }
        } else {
            const auto& view = views[key.bank];
            ++stat_views;
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
                if (batch != nullptr) {
                    ++stat_push;
                    batch->push_back({base + region.offset, span.data, region.bytes});
                    return;
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
            if (batch == nullptr && settle) { (void)cudaStreamSynchronize(stream); }
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

void ExpertCache::prepare(std::span<const ExpertCacheKey> required, cudaStream_t stream) {
    // Experts the pool does not hold are dropped here: the caller masks them out of the GPU job and
    // serves them from the host tier, so uploading them would only evict a resident expert.
    impl_->upload_stream = stream;
    impl_->required.clear();
    for (const auto& key : required) {
        if (key.bank < impl_->slot_of.size() && key.expert < impl_->slot_of[key.bank].size() &&
            impl_->slot_of[key.bank][key.expert] != kAbsentSlot &&
            impl_->slot_of[key.bank][key.expert] < impl_->slots_live) {
            impl_->required.push_back(key);
            // A hit is an expert the pool already holds: the step costs no upload for it.
            if (impl_->entries[impl_->index(key)].ready) { ++impl_->hits; }
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

ExpertCacheProgress ExpertCache::progress(cudaStream_t stream) {
    if (stream == nullptr) { stream = impl_->upload_stream; }
    const auto timing_begin = std::chrono::steady_clock::now();
    std::size_t staged = 0;
    auto& batch = impl_->upload_ops;
    batch.clear();
    for (const auto& key : impl_->required) {
        const auto& entry = impl_->entries[impl_->index(key)];
        if (entry.ready) { continue; }
        if (staged == Impl::kUploadBatch) { break; }
        impl_->upload_one(key, stream, false, staged, &batch);
        ++staged;
    }
    const auto collect_begin = std::chrono::steady_clock::now();
    if (staged == 0) { return {0, true}; }
    // port (sm_89): the batch these were collected into is one submission instead of one per
    // slice. The copies are independent by construction - a slot, a source and a direction each -
    // so the batch's unspecified internal order changes nothing, and the kernel that reads them is
    // queued behind the whole batch on this stream.
    if (!batch.empty()) {
        bool submitted = false;
        if (stream != nullptr && batch.size() > 1) {
            auto& destinations = impl_->upload_destinations;
            auto& sources      = impl_->upload_sources;
            auto& sizes        = impl_->upload_sizes;
            destinations.clear();
            sources.clear();
            sizes.clear();
            for (const auto& op : batch) {
                destinations.push_back(op.destination);
                sources.push_back(op.source);
                sizes.push_back(op.bytes);
            }
            cudaMemcpyAttributes attributes{};
            attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
            attributes.flags          = cudaMemcpyFlagPreferOverlapWithCompute;
            std::size_t attribute_index = 0;
            const auto status           = cudaMemcpyBatchAsync(
                destinations.data(), sources.data(), sizes.data(), batch.size(), &attributes,
                &attribute_index, 1, stream);
            submitted = status == cudaSuccess;
            if (!submitted && status != cudaErrorNotSupported && status != cudaErrorInvalidValue) {
                throw std::runtime_error(cudaGetErrorString(status));
            }
        }
        if (!submitted) {
            for (const auto& op : batch) {
                const auto status =
                    cudaMemcpyAsync(op.destination, op.source, op.bytes, cudaMemcpyHostToDevice,
                                    stream);
                if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
            }
        }
    }
    // port (sm_89): the artifact path stages through one pinned buffer per batch position, so the
    // next batch would overwrite an image that is still being copied. The Host-replica path copies
    // straight out of the arena and needs no barrier; its copies are ordered by the consumer stream
    // and the kernel that reads them, so nothing has to be waited for at all.
    if (impl_->staging_needed) {
        const auto settled = cudaStreamSynchronize(stream);
        if (settled != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(settled)); }
    }
    // The upload budget was one number and it did not say what it was spent on. Split it: the
    // collect phase is host work (slot reads, and the artifact read when a staged image is needed),
    // the submit phase is the API call plus, on the staged path, the barrier that pins it down.
    // Operations per layer is what decides whether the batch form can help at all.
    if (std::getenv("NINFER_UPLOAD_TIMING") != nullptr) {
        const auto timing_end = std::chrono::steady_clock::now();
        static std::uint64_t calls = 0;
        static double acc[4]       = {0.0, 0.0, 0.0, 0.0};
        const auto elapsed = [](auto from, auto to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        acc[0] += elapsed(timing_begin, collect_begin);
        acc[1] += elapsed(collect_begin, timing_end);
        acc[2] += static_cast<double>(batch.size());
        double layer_bytes = 0.0;
        for (const auto& op : batch) { layer_bytes += static_cast<double>(op.bytes); }
        acc[3] += layer_bytes;
        static std::uint64_t seen_direct = 0, seen_staged = 0, seen_views = 0;
        const auto direct_now = impl_->stat_direct;
        const auto staged_now = impl_->stat_staged;
        const auto views_now  = impl_->stat_views;
        const auto direct_delta = direct_now - seen_direct;
        const auto staged_delta = staged_now - seen_staged;
        const auto views_delta  = views_now - seen_views;
        seen_direct = direct_now;
        seen_staged = staged_now;
        seen_views  = views_now;
        static std::uint64_t seen_push = 0;
        const auto push_delta = impl_->stat_push - seen_push;
        seen_push             = impl_->stat_push;
        if (++calls % 40 == 0) {
            std::fprintf(stderr,
                         "[upload] collect=%.3fms submit=%.3fms ops=%.1f mb=%.2f per layer "
                         "direct=%.1f staged=%.1f views=%.1f\n",
                         acc[0] / 40.0, acc[1] / 40.0, acc[2] / 40.0, acc[3] / 40.0 / 1048576.0,
                         static_cast<double>(direct_delta) / 40.0,
                         static_cast<double>(staged_delta) / 40.0,
                         static_cast<double>(views_delta) / 40.0);
            std::fprintf(stderr, "[upload] pushed=%.1f ops per layer (cross-check)\n",
                         static_cast<double>(push_delta) / 40.0);
            acc[0] = acc[1] = acc[2] = acc[3] = 0.0;
        }
        (void)direct_delta;
        (void)staged_delta;
    }
    return {staged, false};
}

void ExpertCache::wait_for_progress(cudaStream_t stream) {
    while (!progress(stream).required_published) {}
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
    std::uint64_t ready = 0;
    for (const auto& entry : impl_->entries) {
        if (entry.ready) { ++ready; }
    }
    return ready * impl_->slot_bytes;
}
std::uint64_t ExpertCache::capacity_bytes() const noexcept { return impl_->live_bytes(); }
std::uint64_t ExpertCache::slot_bytes() const { return impl_->slot_bytes; }
std::uint64_t ExpertCache::resize(std::uint64_t target_bytes) { return impl_->resize(target_bytes); }
std::uint64_t ExpertCache::window_origin(std::uint32_t bank) const {
    // port (sm_89): the pool is a flat slot space, so a bank's plane bases are already absolute and
    // nothing is offset per bank. Kept so a caller that rebases a strided view keeps compiling.
    if (bank >= impl_->banks.size()) { throw std::out_of_range("expert cache bank"); }
    return 0;
}
ConstMemorySpan ExpertCache::pool() const {
    // The whole reserved range: a caller may address any slot the pool can still provide, and an
    // offset is only ever valid while resize() keeps that slot mapped.
    return {impl_->pool->data(), impl_->pool->size(), Device::cuda(impl_->ordinal)};
}
std::uint64_t ExpertCache::uploads() const noexcept { return impl_->uploads; }
std::uint64_t ExpertCache::evictions() const noexcept { return impl_->evictions; }
std::uint64_t ExpertCache::hits() const noexcept { return impl_->hits; }
std::uint64_t ExpertCache::windows() const noexcept { return impl_->slot_capacity() != 0
                                                                    ? impl_->slots_live / impl_->slot_capacity()
                                                                    : 0; }
std::uint32_t ExpertCache::resident_capacity() const noexcept {
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(impl_->slots_live, impl_->slot_capacity()));
}

bool ExpertCache::can_hold_every_layer() const noexcept { return impl_->holds_every_layer(); }

bool ExpertCache::window_resident(std::uint32_t bank) const noexcept {
    return bank < impl_->resident_count.size() &&
           impl_->resident_count[bank] == impl_->banks[bank].count;
}

void ExpertCache::preload_all() { impl_->preload_every_layer(); }

} // namespace ninfer





