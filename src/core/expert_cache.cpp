#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include "core/vmm_pool.h"
#include "core/device_scope.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::uint64_t kAbsent = std::numeric_limits<std::uint64_t>::max();
// Q30/P6: slots per bank in the prediction region. Off unless asked for. The caller sizes it from the
// measured delta between "predicted for the next layer" and "already in the pool", which is much
// smaller than top-k - about one expert a layer on this artifact, not eight.
std::uint32_t prefetch_slots_from_env() {
    const char* value = std::getenv("NINFER_PREFETCH_SLOTS");
    if (value == nullptr || *value == '\0') { return 0; }
    return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
}
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
    // Pinned RAM tier to prefer over the artifact mapping, when one is attached and holds the expert.
    const ExpertArenaSource* arena_source = nullptr;
    // One upload turn carries several experts: the file read is serialised per image anyway, but
    // each image would otherwise pay its own cudaStreamSynchronize (~2.5 ms of WDDM submission on
    // sm_89). The staging buffer therefore holds one slot image per batch position.
    static constexpr std::size_t kUploadBatch = 4;
    // port (sm_89): the staging slots are a ring, not a per-turn scratch pad. A staged image may not
    // be overwritten until its copy has landed, and the old shape enforced that with one
    // cudaStreamSynchronize per turn - forty of them a decode token, ~0.2-0.3 ms each on this driver.
    // With a ring each slot only waits for the copy that used it kStagingRing slots ago, which is a
    // dozen layers back and therefore certainly finished, so the wait costs nothing and the copies
    // stay asynchronous.
    static constexpr std::size_t kStagingRing = 32;
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
    // port (sm_89): Q60. A coalesced turn reads the range that covers its keys once, into pinned
    // memory, and then hands each expert's slices to the batch straight from there - no staging ring
    // position, no second copy, one open/seek instead of one per expert. Sized on first use and kept:
    // a layer pays for this forty times a token, so reallocating per turn is the one thing that would
    // make the coalesced form slower than the per-expert one.
    std::unique_ptr<PinnedHostBuffer> cover;
    std::uint64_t cover_bytes    = 0;
    std::uint64_t stat_coalesced = 0;
    std::uint64_t stat_cover_reads = 0;
    std::unique_ptr<PinnedHostBuffer> staging;
    // Ring cursor for the slots above, and one event per slot recording when that slot's copy was
    // queued. Before a slot is reused its event is waited on, which is the only ordering the ring
    // needs: the data path itself is already ordered by the consumer stream.
    std::size_t staging_cursor = 0;
    // Experts moved on the last progress() turn, for the path controller's miss estimate.
    std::uint32_t last_staged = 0;
// Incremented on every set_resident() that changed something (see layout_generation()).
std::uint64_t layout_generation = 0;
// ... and once per changed bank, which is the only form a per-bank publisher can compare against.
std::vector<std::uint64_t> bank_generation;
    std::array<cudaEvent_t, kStagingRing> staging_event{};
    std::vector<std::size_t> staging_used;
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
    std::uint64_t stat_arena  = 0;
    std::uint64_t stat_arena_experts = 0;
    std::uint64_t stat_arena_probes  = 0;
    std::uint64_t stat_file_experts  = 0;
    double stat_read_us              = 0.0;
    // Counted at the one place every upload passes through, so the figure cannot disagree with the
    // batch it describes the way the per-branch counters did.
    std::uint64_t stat_push = 0;
    // Q52: API submissions, which is the number "one submit per layer" is about. `stat_push` counts
    // slices and stays near 11.6 by construction (six planes an expert); a batch of those slices is
    // one submission, and that is what the driver is asked to do.
    std::uint64_t stat_submits = 0;
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
    // port (sm_89): heterogeneous banks. A bank's expert image has its own padded width, and a
    // single stride charges the widest bank to every slot. On the 35B-A3B MTP artifact that is 40
    // banks at 1.875 MiB plus the MTP layer at 3.25 MiB; the uniform stride turned an 18.8 GiB pool
    // into 5,774 of 10,496 resident experts (prefill 127.8 tok/s). When the banks disagree the slot
    // space is priced per bank: bank b owns a contiguous run at its own stride, laid out as
    // [every bank's live run][every bank's spare run]. Live ids stay below slots_live and spares
    // above it, so the mapped range is still a prefix and a resize only moves the ids at the edge.
    bool per_bank_layout = false;
    std::vector<std::uint64_t> bank_stride;    // padded expert width per bank, 0 when resident
    std::vector<std::uint32_t> bank_slot_base; // first slot id of the bank's live run
    std::vector<std::uint32_t> bank_slot_max;  // ids the bank's ceiling can back
    std::vector<std::uint32_t> bank_slot_live; // ids the bank has backed right now
    std::vector<std::vector<std::uint32_t>> bank_free; // per-bank free ids (live run first)
    std::vector<std::uint64_t> slot_offset;    // slot id -> byte offset in the reservation
    std::vector<std::uint32_t> slot_owner;     // slot id -> bank
    // 0 free, 1 prediction region, 2 fixed ring. One array for both roles: a slot is at most one.
    static constexpr std::uint8_t kRoleNone = 0, kRolePrefetch = 1, kRoleRing = 2;
    std::vector<std::uint8_t> slot_role;
    // Global slot of each expert, or kAbsentSlot when the pool does not hold it. A fully resident
    // bank keeps the identity, which is what lets a captured graph address expert i directly.
    std::vector<std::vector<std::uint32_t>> slot_of;
    // Every slot not owned by an expert. Free slots are shared: a bank that drops experts releases
    // them here and another bank's next resident set may take them.
    std::vector<std::uint32_t> free_slots;
    // How many experts of each bank own a slot right now.
    std::vector<std::uint32_t> resident_count;
    // A1: the two table segments' generations, per bank. `step_gen` moves on any set change and is
    // what the per-step segment is checked against; `hot_gen` moves only when the hot membership
    // does, so a boundary re-placement cannot make the per-step check fail.
    std::vector<std::uint64_t> hot_gen, step_gen, prefetch_gen;
    std::vector<std::vector<std::uint32_t>> prev_hot; // sorted hot set at the last decision
    // Q30/P6: the prediction region - slots reserved at the top of the pool, never handed to a
    // resident set. Separate from the spill ring on purpose: the ring holds exactly this step's
    // misses (C1) and is sized from the widest step (B4), while predictions belong to the NEXT layer;
    // sharing would make them the oldest arrivals, i.e. the first thing FIFO drops.
    std::uint32_t prefetch_per_bank = 0;
    std::uint32_t prefetch_base     = 0; // region is [base, slots_live)
    std::vector<std::vector<std::uint32_t>> prefetch_owner;   // bank -> region index -> expert
    std::vector<std::uint32_t> prefetch_cursor;               // bank -> next region index to reuse
    // Q20: fixed ring slots per bank, taken from the bank's own free list by configure_ring().
    std::vector<std::vector<std::uint32_t>> ring_slots;
    int ordinal = 0;

    Impl(std::span<const ExpertBankSpec> specs, const ExpertCacheRequirements& req,
         const DeviceContext& device, std::span<const ExpertBankPair> bank_views,
         std::span<const ExpertFileBank> file_banks, const ExpertBlobReader* reader)
        : config(req.config), banks(req.banks), views(bank_views.begin(), bank_views.end()),
          files(file_banks.begin(), file_banks.end()), file_reader(reader),
          max_entry_bytes(req.max_entry_bytes),
          staging(req.max_entry_bytes
                      ? std::make_unique<PinnedHostBuffer>(
                            static_cast<std::size_t>(req.max_entry_bytes) * kStagingRing)
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
        std::uint64_t widest_bank = 0;
        std::uint64_t layer_count = 0;
        for (const auto& bank : req.banks) {
            if (bank.resident) { continue; }
            widest_bank = std::max(widest_bank, std::uint64_t{bank.count});
            ++layer_count;
        }
        // Per-bank pricing: decide before anything is laid out. Uniform banks (the common case, and
        // every artifact shipped before the 35B MTP one) keep the flat slot space untouched; only a
        // genuine width disagreement switches the pool to per-bank strides. NINFER_EXPERT_UNIFORM_SLOTS
        // pins the old behaviour for an A/B.
        const bool uniform_forced = std::getenv("NINFER_EXPERT_UNIFORM_SLOTS") != nullptr;
        bank_stride.assign(banks.size(), 0);
        for (std::size_t b = 0; b < banks.size(); ++b) {
            if (!banks[b].resident) { bank_stride[b] = align_up(banks[b].entry_bytes, block); }
        }
        // Heterogeneous means "one bank is wide enough to starve the others", not "no two banks are
        // byte-identical": rounding to the block already absorbs a layer whose down matrix carries a
        // high plane (a 128 KiB difference on 1.875 MiB). The payout of pricing per bank is a slot
        // count, and that only changes where experts land once the widest bank is a quarter wider
        // than the narrowest - the MTP case is 3.25 against 1.875. Below the line the flat space is
        // kept, because it is the layout every recorded perplexity byte belongs to.
        std::uint64_t narrowest_stride = 0;
        for (const auto stride : bank_stride) {
            if (stride == 0) { continue; }
            narrowest_stride =
                narrowest_stride == 0 ? stride : std::min(narrowest_stride, stride);
        }
        const bool heterogeneous =
            narrowest_stride != 0 && slot_bytes >= narrowest_stride + narrowest_stride / 4;
        // On by default when the banks disagree: a single stride charges the widest bank to every
        // slot, and on the 35B-A3B MTP artifact (40 banks at 1.875 MiB, the MTP layer at 3.25 MiB)
        // that held 5,740 of 10,496 experts and measured 143 prefill / 84 decode against 10,344 and
        // 2,094 / 404 with per-bank pricing (pp2048+tg128, INT8, -r 5, same machine). Artifacts whose
        // banks agree never enter this branch. NINFER_EXPERT_UNIFORM_SLOTS restores the old layout
        // for an A/B.
        per_bank_layout = heterogeneous && !uniform_forced;
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
        // be returned to the driver by resize(). The reservation is the growth ceiling, not the live
        // pool: VmmPool reserves virtual address space and maps segments on demand, so reserving room
        // for the auto-sized maximum while only capacity_bytes start mapped costs no device memory
        // and makes the L1 dial's upward steps reachable (the 2026-11-05 acceptance measured the old
        // behaviour: slots_max == slots_live, so every grow was clamped back).
        const auto limit_bytes = std::max(req.config.capacity_bytes,
                                          req.config.capacity_limit_bytes);
        const auto reserved = static_cast<std::size_t>(
            (limit_bytes + kSegmentBytes - 1) / kSegmentBytes * kSegmentBytes);
        pool = std::make_unique<VmmPool>(Device::cuda(ordinal), reserved, kSegmentBytes);
        slot_of.resize(banks.size());
        resident_count.assign(banks.size(), 0);
        ring_slots.resize(banks.size());
        if (per_bank_layout) {
            // Equal byte shares: every non-resident bank gets capacity/layers of live bytes and
            // limit/layers of ceiling bytes, charged at its own stride. Live runs come first, so
            // the mapped range is a prefix; the spare runs that follow are what a resize grows into.
            const auto live_share    = req.config.capacity_bytes / layer_count;
            const auto ceiling_share = limit_bytes / layer_count;
            bank_slot_base.assign(banks.size(), 0);
            bank_slot_max.assign(banks.size(), 0);
            bank_slot_live.assign(banks.size(), 0);
            bank_free.assign(banks.size(), {});
            std::uint64_t live_total = 0, max_total = 0;
            for (std::size_t b = 0; b < banks.size(); ++b) {
                if (banks[b].resident) { continue; }
                const auto stride = bank_stride[b];
                bank_slot_live[b] = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    banks[b].count, stride != 0 ? live_share / stride : 0));
                bank_slot_max[b] = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    banks[b].count, stride != 0 ? ceiling_share / stride : 0));
                live_total += bank_slot_live[b];
                max_total  += bank_slot_max[b];
            }
            if (live_total == 0) {
                throw std::invalid_argument("expert pool cannot hold one slot per bank");
            }
            slot_offset.assign(static_cast<std::size_t>(max_total), 0);
            slot_owner.assign(static_cast<std::size_t>(max_total), 0);
            std::uint64_t base = 0, byte = 0;
            for (std::size_t b = 0; b < banks.size(); ++b) { // live runs
                if (banks[b].resident) { continue; }
                bank_slot_base[b] = static_cast<std::uint32_t>(base);
                for (std::uint32_t i = 0; i < bank_slot_live[b]; ++i, ++base) {
                    slot_offset[base] = byte;
                    slot_owner[base]  = static_cast<std::uint32_t>(b);
                    byte += bank_stride[b];
                }
            }
            for (std::size_t b = 0; b < banks.size(); ++b) { // growth tail
                if (banks[b].resident) { continue; }
                for (std::uint32_t i = bank_slot_live[b]; i < bank_slot_max[b]; ++i, ++base) {
                    slot_offset[base] = byte;
                    slot_owner[base]  = static_cast<std::uint32_t>(b);
                    byte += bank_stride[b];
                }
            }
            slots_live = live_total;
            slots_max  = max_total;
            slot_cap   = 0;
            for (std::size_t b = 0; b < banks.size(); ++b) {
                if (banks[b].resident) { continue; }
                auto& list = bank_free[b];
                list.reserve(static_cast<std::size_t>(bank_slot_live[b]));
                // Only the live run is handed out: the spare run sits above the mapped prefix and is
                // not backed until resize() grows the pool, so putting it on the free list would let
                // a bank take a slot the driver has not given us (measured: cudaErrorIllegalAddress
                // inside the first layer that placed beyond its run).
                // Live run pushed high to low, so the first pop hands out the bank's base id: a
                // fresh bank keeps the identity mapping expert e -> base + e.
                for (std::uint32_t i = bank_slot_live[b]; i > 0; --i) {
                    list.push_back(static_cast<std::uint32_t>(bank_slot_base[b] + i - 1));
                }
                slot_cap = std::max<std::uint64_t>(slot_cap, bank_slot_live[b]);
            }
        } else {
            const auto pool_slots = req.config.capacity_bytes / slot_bytes;
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
            if (pool_slots == 0) {
                throw std::invalid_argument("expert cache holds no complete expert slot");
            }
            slots_max  = reserved / slot_bytes;
            slots_live = pool_slots;
            // A pool narrower than one hot set per bank cannot give every bank its own run: narrow
            // the hot set instead of addressing past the live pool (growth adds windows, not a
            // wider one).
            if (layer_count != 0) {
                const auto per_bank = slots_live / layer_count;
                if (per_bank == 0) {
                    throw std::invalid_argument("expert pool cannot hold one slot per bank");
                }
                cap = std::min(cap, per_bank);
            }
            slot_cap = cap;
            free_slots.clear();
            free_slots.reserve(static_cast<std::size_t>(slots_max));
            // Pushed high to low so the first pop hands out slot 0: a fresh bank then keeps the
            // identity mapping it needs for a graph, exactly as the per-bank runs used to lay it out.
            for (std::uint64_t slot = slots_live; slot > 0; --slot) {
                free_slots.push_back(static_cast<std::uint32_t>(slot - 1));
            }
        }
        slot_role.assign(static_cast<std::size_t>(slots_max), kRoleNone);
        // Q20: the fixed ring is a slice of each bank's own window, not extra capacity. Taking it
        // after seeding failed by construction on an 8 GiB pool (106 slots/bank, 39 free in total,
        // 8 x 40 needed), so it is taken here and the hot seed is shortened by the same amount.
        std::uint32_t ring_per_bank = 0;
        if (const char* enabled = std::getenv("NINFER_RING_DEVICE_ADMIT");
            enabled != nullptr && *enabled != '\0' && std::strtol(enabled, nullptr, 10) != 0) {
            ring_per_bank = 8;
            if (const char* value = std::getenv("NINFER_RING_SLOTS");
                value != nullptr && *value != '\0') {
                ring_per_bank = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
            }
            ring_per_bank = std::min<std::uint32_t>(ring_per_bank, slot_cap / 2);
        }
        // Q30/P6: size the prediction region BEFORE seeding and take it out of each bank's seed
        // (hot = capacity - S_p, the same shape B4 uses for the ring). Taking it from the leftovers
        // was tried first and never produced a region: the pool seeds every bank to its cap, so
        // nothing is left to carve.
        prefetch_per_bank = prefetch_slots_from_env();
        if (prefetch_per_bank != 0 && !banks.empty()) {
            const auto room = static_cast<std::uint32_t>(slot_cap / 4);
            prefetch_per_bank = std::min(prefetch_per_bank, room);
        }
        if (per_bank_layout && (ring_per_bank != 0 || prefetch_per_bank != 0)) {
            // Both experiments assume one global slot stride: the ring hands a slot id to a Device
            // kernel that indexes the pool by it, and the prediction region is carved out of the top
            // of a flat slot space. Neither has been re-derived for per-bank strides, so they are
            // refused rather than silently mis-addressing.
            std::fprintf(stderr,
                         "[pool] per-bank slot pricing disables the ring/prefetch experiments\n");
            std::fflush(stderr);
            ring_per_bank     = 0;
            prefetch_per_bank = 0;
        }
        for (std::size_t b = 0; b < banks.size() && ring_per_bank != 0; ++b) {
            auto& list = free_list(static_cast<std::uint32_t>(b));
            if (list.size() < ring_per_bank) {
                throw std::invalid_argument("expert pool cannot reserve a ring for every bank");
            }
            ring_slots[b].reserve(ring_per_bank);
            for (std::uint32_t index = 0; index < ring_per_bank; ++index) {
                const std::uint32_t slot = list.back();
                list.pop_back();
                slot_role[slot] = kRoleRing;
                ring_slots[b].push_back(slot);
            }
        }
        if (ring_per_bank != 0) {
            std::fprintf(stderr, "[ring] fixed slots=%u/bank (%u total), hot seed reduced\n",
                         ring_per_bank, ring_per_bank * static_cast<std::uint32_t>(banks.size()));
            std::fflush(stderr);
        }
        const std::uint32_t reserved_per_bank = prefetch_per_bank + ring_per_bank;
        // port (sm_89): only the experts that fit the per-bank hint start resident. The identity
        // mapping is right when the hint holds the whole layer; a smaller hint must start with the
        // rest absent, or a stale slot index would point at another bank's expert.
        std::uint32_t seed = 0;
        for (std::size_t b = 0; b < banks.size(); ++b) {
            slot_of[b].assign(banks[b].count, kAbsentSlot);
            if (banks[b].resident) { continue; }
            // Per-bank pricing gives every bank its own live run; the uniform space shares one hint.
            const auto cap_b = per_bank_layout ? bank_slot_live[b]
                                               : static_cast<std::uint32_t>(slot_cap);
            const std::uint32_t bank_seed =
                reserved_per_bank < cap_b ? cap_b - reserved_per_bank : cap_b;
            seed = std::max(seed, bank_seed);
            auto& list = free_list(static_cast<std::uint32_t>(b));
            for (std::uint32_t e = 0; e < banks[b].count && e < bank_seed; ++e) {
                if (list.empty()) {
                    throw std::invalid_argument("expert pool cannot seed every bank's hot set");
                }
                slot_of[b][e] = list.back();
                list.pop_back();
                ++resident_count[b];
            }
        }
        if (prefetch_per_bank != 0) {
            const auto per_bank = static_cast<std::uint64_t>(prefetch_per_bank);
            prefetch_base       = static_cast<std::uint32_t>(
                slots_live - per_bank * static_cast<std::uint64_t>(banks.size()));
            std::erase_if(free_slots, [&](std::uint32_t slot) { return slot >= prefetch_base; });
            for (std::uint32_t slot = prefetch_base; slot < slots_live; ++slot) {
                slot_role[slot] = kRolePrefetch;
            }
            prefetch_owner.assign(banks.size(),
                                  std::vector<std::uint32_t>(prefetch_per_bank, kAbsentSlot));
            prefetch_cursor.assign(banks.size(), 0);
            prefetch_gen.assign(banks.size(), 0);
            std::fprintf(stderr, "[pool] prefetch region: %u slot(s)/bank, %u total, hot seed %u\n",
                         prefetch_per_bank,
                         prefetch_per_bank * static_cast<std::uint32_t>(banks.size()), seed);
            std::fflush(stderr);
        }
        if (heterogeneous) {
            // The one line that says which pricing is in force and what it bought: a width
            // disagreement is the only reason the layout can differ, and the residency count is what
            // the prefill rate follows.
            std::uint64_t resident = 0, total = 0;
            for (std::size_t b = 0; b < banks.size(); ++b) {
                total += banks[b].count;
                resident += banks[b].resident ? banks[b].count : resident_count[b];
            }
            std::fprintf(stderr,
                         "[pool] %s layout over %zu bank(s), widest stride %.1f KiB: %llu of %llu "
                         "experts resident\n",
                         per_bank_layout ? "per-bank" : "uniform", banks.size(),
                         static_cast<double>(slot_bytes) / 1024.0,
                         static_cast<unsigned long long>(resident),
                         static_cast<unsigned long long>(total));
            std::fflush(stderr);
        }
    }

    // Q30/P6: the prediction's entry point. Only experts the pool does NOT already hold are placed;
    // the caller filters by the same rule, and this repeats it because a stale prediction list must
    // not displace a slot that is currently serving routes.
    //
    // The region is a ring per bank: the next placement reuses the oldest region slot. With a delta
    // of one or two experts a layer (prediction minus what is already resident), the region needs to
    // be that small, not top-k - an over-sized region only steals slots from the hot set.
    //
    // No upload happens here: the placed keys come back to the caller, which hands them to
    // prepare()/progress() on the compute stream (the tested upload path, and the copy is ordered
    // before the layer that reads it).
    std::uint32_t set_prefetch(std::uint32_t bank, std::span<const std::uint32_t> experts,
                               std::vector<ExpertCacheKey>& placed) {
        if (prefetch_per_bank == 0 || bank >= banks.size()) { return 0; }
        auto& slots  = slot_of[bank];
        auto& owner  = prefetch_owner[bank];
        auto& cursor = prefetch_cursor[bank];
        std::uint32_t count = 0;
        for (const std::uint32_t expert : experts) {
            if (expert >= slots.size()) { continue; }
            if (slots[expert] != kAbsentSlot) { continue; } // already in the pool: nothing to do
            const std::uint32_t region = cursor++ % prefetch_per_bank;
            const std::uint32_t slot   = prefetch_base + bank * prefetch_per_bank + region;
            const std::uint32_t old    = owner[region];
            if (old != kAbsentSlot && old < slots.size() && slots[old] == slot) {
                slots[old]                       = kAbsentSlot;
                entries[first[bank] + old].ready = false;
                ++evictions;
            }
            owner[region]                       = expert;
            slots[expert]                       = slot;
            entries[first[bank] + expert].ready = false;
            placed.push_back(ExpertCacheKey{bank, expert});
            ++count;
        }
        if (count != 0) {
            if (prefetch_gen.size() <= bank) { prefetch_gen.resize(bank + 1, 0); }
            ++prefetch_gen[bank];
        }
        return count;
    }

    // True when this global slot belongs to a bank's prediction region.
    [[nodiscard]] bool prefetch_slot(std::uint32_t slot) const noexcept {
        return slot < slot_role.size() && slot_role[slot] == kRolePrefetch;
    }

    [[nodiscard]] bool ring_slot(std::uint32_t slot) const noexcept {
        return slot < slot_role.size() && slot_role[slot] == kRoleRing;
    }

    // Does `expert` count as a member of the bank's resident set (as opposed to a prediction)? The
    // distinction matters in set_resident: a prediction is not the hot set's to free, and it must not
    // make a step look changed when the hot set is identical.
    [[nodiscard]] bool resident_slot(std::uint32_t bank, std::uint32_t expert) const noexcept {
        if (bank >= slot_of.size() || expert >= slot_of[bank].size()) { return false; }
        const auto slot = slot_of[bank][expert];
        if (slot == kAbsentSlot) { return false; }
        return !prefetch_slot(slot) && !ring_slot(slot);
    }

    // Makes `experts` the bank's resident set, in list order, with the first `hot_count` entries
    // being the hot segment. The caller recomputes and republishes its offset table afterwards.
    // Returns true when the resident set changed. Experts that stay resident keep their slot, so a
    // step that routes to experts already present uploads nothing and never discards them; slots of
    // the ones leaving go back to the shared free list, and arrivals take whatever the list has left.
    bool set_resident(std::uint32_t bank, std::span<const std::uint32_t> experts,
                      std::uint32_t hot_count) {
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
            // A prediction is not a member of the resident set: it must not make the set look
            // changed, or every step that ever prefetched would republish the whole table.
            if (resident_slot(bank, static_cast<std::uint32_t>(expert)) != keep[expert]) {
                changed = true;
                break;
            }
        }
        // A1: the hot segment's own membership. Compared as a sorted set because the caller's order is
        // a policy detail, while a generation has to mean "the same experts are hot".
        const std::uint32_t hot_n =
            hot_count == UINT32_MAX ? static_cast<std::uint32_t>(wanted.size())
                                    : std::min(hot_count, static_cast<std::uint32_t>(wanted.size()));
        std::vector<std::uint32_t> hot_now(wanted.begin(), wanted.begin() + hot_n);
        std::sort(hot_now.begin(), hot_now.end());
        if (prev_hot.size() <= bank) { prev_hot.resize(bank + 1); }
        const bool hot_changed = prev_hot[bank] != hot_now;
        if (hot_changed) { prev_hot[bank] = hot_now; }
        // Consecutive steps usually route to the same experts. Keeping the set when it is unchanged
        // turns a per-token reupload into a hit.
        if (!changed && !hot_changed) { return false; }
        ++layout_generation;
        // The same bump, per bank: this is the one a publisher of this bank's tables can compare
        // against without being invalidated by the other banks (see bank_generation()).
        if (bank_generation.size() <= bank) { bank_generation.resize(bank + 1, 0); }
        ++bank_generation[bank];
        // A1: the split. `step_generation` is what the per-step table segment is checked against and
        // moves whenever anything in the set does; `hot_generation` is the epoch-segment's, and only
        // a hot re-placement moves it - which is what lets a boundary publish leave the per-step
        // check untouched.
        if (step_gen.size() <= bank) { step_gen.resize(bank + 1, 0); }
        if (hot_gen.size() <= bank) { hot_gen.resize(bank + 1, 0); }
        if (changed) { ++step_gen[bank]; }
        if (hot_changed) { ++hot_gen[bank]; }
        if (!changed) { return true; } // hot-only change: nothing in the slot tables moves
        std::uint32_t held = 0;
        for (std::size_t expert = 0; expert < slots.size(); ++expert) {
            if (slots[expert] == kAbsentSlot) { continue; }
            if (keep[expert]) {
                ++held;
                continue;
            }
            if (prefetch_slot(slots[expert])) {
                // The region recycles its own slots. Handing one to the free list would give a
                // prediction slot to another bank's hot set and leave the ring addressing bytes the
                // ring no longer owns.
                slots[expert]                       = kAbsentSlot;
                entries[first[bank] + expert].ready = false;
                continue;
            }
            if (ring_slot(slots[expert])) {
                // Q20: the fixed ring owns these slots and the Device admits into them. The hot set
                // leaving one is a mapping change, not a free.
                slots[expert]                       = kAbsentSlot;
                entries[first[bank] + expert].ready = false;
                continue;
            }
            // Leaving: hand the slot back to this bank's own list. Per-bank pricing means the bytes
            // behind an id are only addressable at this bank's stride, so a slot cannot migrate.
            free_list(bank).push_back(slots[expert]);
            slots[expert]                       = kAbsentSlot;
            entries[first[bank] + expert].ready = false;
        }
        std::uint32_t taken = 0;
        for (const std::uint32_t expert : wanted) {
            if (held + taken >= wanted.size()) { break; }
            if (slots[expert] != kAbsentSlot) { continue; }
            if (free_list(bank).empty()) { break; }
            slots[expert]                       = free_list(bank).back();
            free_list(bank).pop_back();
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


    // Q20 device-side admission: the Device named the expert and the ring slot; the host copied the
    // bytes. This makes that pair the bank's residency without a second slot assignment, and hands
    // the slot's previous owner back to the free list.
    bool adopt_slot(std::uint32_t bank, std::uint32_t expert, std::uint32_t slot) {
        if (bank >= banks.size() || expert >= slot_of[bank].size() || slot >= slots_live ||
            prefetch_slot(slot) || !ring_slot(slot)) {
            return false;
        }
        // The ring is per-bank: a slot configured for another bank names bytes this bank's offset
        // table must never address.
        const auto& owned = ring_slots[bank];
        if (std::find(owned.begin(), owned.end(), slot) == owned.end()) { return false; }
        auto& slots = slot_of[bank];
        if (slots[expert] != slot) {
            // One slot names one expert. Clearing the old owner is what keeps the offset table from
            // pointing two rows at the same bytes.
            for (std::uint32_t other = 0; other < slots.size(); ++other) {
                if (other != expert && slots[other] == slot) {
                    slots[other]                        = kAbsentSlot;
                    entries[first[bank] + other].ready = false;
                }
            }
            if (slots[expert] != kAbsentSlot) {
                if (!prefetch_slot(slots[expert]) && !ring_slot(slots[expert])) {
                    free_list(bank).push_back(slots[expert]);
                }
                entries[first[bank] + expert].ready = false;
            }
            slots[expert]                       = slot;
            entries[first[bank] + expert].ready = true;
            required.erase(std::remove_if(required.begin(), required.end(),
                                          [&](const ExpertCacheKey& key) {
                                              return key.bank == bank && key.expert == expert;
                                          }),
                           required.end());
            if (step_gen.size() <= bank) { step_gen.resize(bank + 1, 0); }
            if (bank_generation.size() <= bank) { bank_generation.resize(bank + 1, 0); }
            ++step_gen[bank];
            ++bank_generation[bank];
        } else {
            entries[first[bank] + expert].ready = true;
        }
        resident_count[bank] = static_cast<std::uint32_t>(
            std::count_if(slots.begin(), slots.end(), [](auto value) { return value != kAbsentSlot; }));
        return true;
    }

        // Q20: take a fixed run of slots out of the bank's own free list. The ids are stable for the life
    // of the pool, so the Device can name one and the host can address it without a second lookup.
    void configure_ring(std::uint32_t bank, std::uint32_t slots) {
        if (bank >= banks.size() || slots == 0) { return; }
        auto& owned = ring_slots[bank];
        if (!owned.empty()) { return; }
        auto& list = free_list(bank);
        if (list.size() < slots) {
            throw std::invalid_argument("expert pool cannot reserve a ring for every bank");
        }
        if (slot_role.size() < slots_max) { slot_role.resize(slots_max, kRoleNone); }
        owned.reserve(slots);
        for (std::uint32_t index = 0; index < slots; ++index) {
            const std::uint32_t slot = list.back();
            list.pop_back();
            slot_role[slot] = kRoleRing;
            owned.push_back(slot);
        }
        std::fprintf(stderr, "[ring] bank=%u fixed slots=%u\n", bank, slots);
        std::fflush(stderr);
    }

    std::size_t index(ExpertCacheKey key) const {        if (key.bank >= banks.size() || banks[key.bank].resident ||
            key.expert >= banks[key.bank].count) {
            throw std::out_of_range("expert cache key");
        }
        return first[key.bank] + key.expert;
    }

    // port (sm_89): slots are uniform. Banks may differ in entry size (a layer whose down matrix
    // carries a high plane is 128 KiB larger), so only a slot of max_entry_bytes can be handed to
    // any bank.
    //
    // Per-bank pricing keeps that rule for banks that share a width and only splits the space when
    // they genuinely disagree; `free_list` and `slot_address` are the two places the split shows,
    // so nothing downstream multiplies a slot id by a stride any more.
    std::vector<std::uint32_t>& free_list(std::uint32_t bank) {
        return per_bank_layout ? bank_free[bank] : free_slots;
    }
    [[nodiscard]] std::uint64_t slot_address(std::uint32_t slot) const {
        if (!per_bank_layout) { return std::uint64_t{slot} * slot_bytes; }
        return slot < slot_offset.size() ? slot_offset[slot] : kAbsent;
    }
    std::uint64_t slot_capacity() const { return slot_cap; }

    // Bytes the pool is allowed to use right now.
    std::uint64_t live_bytes() const {
        if (!per_bank_layout) { return slots_live * slot_bytes; }
        std::uint64_t bytes = 0;
        for (std::size_t b = 0; b < banks.size(); ++b) {
            if (banks[b].resident) { continue; }
            bytes += std::uint64_t{bank_slot_live[b]} * bank_stride[b];
        }
        return bytes;
    }

    // Every bank's whole layer fits at once. With per-bank pricing each bank has its own run, so the
    // question is per bank; the uniform space shares one list, so the total is what matters.
    bool holds_every_layer() const noexcept {
        if (per_bank_layout) {
            for (std::size_t b = 0; b < banks.size(); ++b) {
                if (banks[b].resident) { continue; }
                if (bank_slot_live[b] < banks[b].count) { return false; }
            }
            return true;
        }
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
        std::uint64_t wanted = 0;
        std::size_t keep = 0;
        if (per_bank_layout) {
            // Live ids are ordered [every bank's live run][every bank's spare run], so the new edge
            // is the number of ids whose address starts below the backing we are about to keep.
            keep = static_cast<std::size_t>(
                (align_up(std::max<std::uint64_t>(target_bytes, 1), kSegmentBytes)));
            if (keep > pool->size()) { keep = pool->size(); }
            while (wanted < slot_offset.size() &&
                   slot_offset[wanted] + bank_stride[slot_owner[wanted]] <= keep) {
                ++wanted;
            }
        } else {
            wanted = target_bytes / slot_bytes;
            wanted = std::max<std::uint64_t>(wanted, 1);
            wanted = std::min<std::uint64_t>(wanted, slots_max);
            keep   = static_cast<std::size_t>(
                (wanted * slot_bytes + kSegmentBytes - 1) / kSegmentBytes * kSegmentBytes);
        }
        if (wanted == slots_live) { return live_bytes(); }
        DeviceScope scope(Device::cuda(ordinal));
        const auto settled = cudaDeviceSynchronize();
        if (settled != cudaSuccess) {
            throw std::runtime_error(std::string("expert pool resize: ") +
                                     cudaGetErrorString(settled));
        }
        if (wanted < slots_live) {
            // Q20: a fixed ring slot that the shrink would unmap has no replacement address. The
            // ring is not "expert residency" the resize is allowed to drop.
            for (const auto& owned : ring_slots) {
                for (const auto slot : owned) {
                    if (slot >= wanted) {
                        throw std::invalid_argument(
                            "expert pool cannot shrink below its fixed ring slots");
                    }
                }
            }
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
            if (per_bank_layout) {
                for (auto& list : bank_free) {
                    std::erase_if(list, [&](std::uint32_t slot) { return slot >= wanted; });
                }
            } else {
                std::erase_if(free_slots,
                              [&](std::uint32_t slot) { return slot >= wanted; });
            }
            pool->release(pool->mapped() - keep, keep);
        } else {
            if (per_bank_layout) {
                for (std::uint64_t slot = slots_live; slot < wanted; ++slot) {
                    bank_free[slot_owner[slot]].push_back(static_cast<std::uint32_t>(slot));
                }
            } else {
                for (std::uint64_t slot = slots_live; slot < wanted; ++slot) {
                    free_slots.push_back(static_cast<std::uint32_t>(slot));
                }
            }
            pool->acquire(keep - pool->mapped());
        }
        slots_live = wanted;
        if (per_bank_layout) {
            std::fill(bank_slot_live.begin(), bank_slot_live.end(), 0);
            for (std::uint64_t slot = 0; slot < wanted; ++slot) {
                ++bank_slot_live[slot_owner[slot]];
            }
            slot_cap = 0;
            for (std::size_t b = 0; b < banks.size(); ++b) {
                if (banks[b].resident) { continue; }
                slot_cap = std::max<std::uint64_t>(slot_cap, bank_slot_live[b]);
            }
        }
        if (slot_role.size() < wanted) { slot_role.resize(wanted, kRoleNone); }
        // 13.5 (step 5): the layout just moved. Experts above the new edge lost their slots, and
        // the backing below it changed hands with the driver. The published offset tables name
        // slots, and a stream republishes them exactly when the generations below move - so bump
        // every one of them, for every bank. The per-bank vectors are filled lazily on first
        // placement, so they are sized up first: a resize that arrives before any placement (the
        // pool is elastic from construction) must still invalidate the table a placement
        // publisher would otherwise compare as already published. A table left stale by a shrink
        // still names a segment the driver has taken back, which the GPU reports as an illegal
        // address the first time a grouped kernel reads it (found by the resize-under-load
        // acceptance on the e2 artifact).
        ++layout_generation;
        {
            const auto size = banks.size();
            const auto grow = [size](std::vector<std::uint64_t>& generations) {
                if (generations.size() < size) { generations.resize(size, 0); }
                for (auto& generation : generations) { ++generation; }
            };
            grow(bank_generation);
            grow(hot_gen);
            grow(step_gen);
            grow(prefetch_gen);
        }
        return live_bytes();
    }

    // port (sm_89): Q60. Read the range that covers this turn's keys once, into pinned memory.
    //
    // The shape of the turn decides whether that is possible: the keys must be one bank and one
    // object (a fused artifact keeps a layer's experts in a few contiguous runs of one object, an e2
    // artifact gives each expert its own), the covering range must be bounded, and the reader must be
    // able to answer a byte range at all. Anything else returns null and the caller takes the
    // per-expert path it has always taken - so a run without the switch, or on a source that cannot
    // coalesce, behaves exactly as before.
    const std::byte* cover_for(const std::uint32_t* banks, const std::uint32_t* experts,
                               std::size_t count, std::uint64_t& lo_out, std::uint64_t& end_out) {
        static const bool enabled = std::getenv("NINFER_UPLOAD_COALESCE") != nullptr;
        if (!enabled || count == 0 || file_reader == nullptr) { return nullptr; }
        const auto bank = banks[0];
        if (bank >= files.size()) { return nullptr; }
        const bool trace = std::getenv("NINFER_UPLOAD_TIMING") != nullptr;
        std::uint32_t file_index = 0;
        std::uint64_t lo         = ~std::uint64_t{0};
        std::uint64_t hi         = 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (banks[i] != bank || experts[i] >= files[bank].experts.size()) {
                if (trace) { std::fprintf(stderr, "[cover] reject: bank/expert i=%zu\n", i); }
                return nullptr;
            }
            for (const auto& chunk : files[bank].experts[experts[i]].chunks) {
                if (i == 0) {
                    file_index = chunk.file_index;
                } else if (chunk.file_index != file_index) {
                    if (trace) { std::fprintf(stderr, "[cover] reject: file_index mix\n"); }
                    return nullptr;
                }
                lo = std::min(lo, chunk.file_offset);
                hi = std::max(hi, chunk.file_offset + chunk.bytes);
            }
        }
        constexpr std::uint64_t kMaxCover = 64ULL << 20;
        // The point of a cover is that its bytes are the members' bytes. A range that also spans
        // everything between two experts reads - and copies - the whole gap, so the test is
        // contiguity, not just size: the covering range may not exceed what the members need by more
        // than a small slack. This is also the fuse-vs-e2 dispatch, measured rather than declared -
        // a fused bank's experts are adjacent runs and pass, an e2 bank's are separate objects spread
        // across the file and fail it in one comparison.
        constexpr std::uint64_t kCoverSlack = 1ULL << 20;
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i < count; ++i) {
            for (const auto& chunk : files[bank].experts[experts[i]].chunks) { sum += chunk.bytes; }
        }
        if (hi <= lo || hi - lo > kMaxCover || hi - lo > sum + kCoverSlack) {
            if (trace) {
                std::fprintf(stderr, "[cover] reject: range lo=%llu hi=%llu need=%llu\n",
                             static_cast<unsigned long long>(lo),
                             static_cast<unsigned long long>(hi),
                             static_cast<unsigned long long>(sum));
            }
            return nullptr;
        }
        const auto bytes = static_cast<std::size_t>(hi - lo);
        if (!cover || cover->size() < bytes) {
            cover      = std::make_unique<PinnedHostBuffer>(bytes);
            cover_bytes = hi - lo;
        }
        if (!file_reader->read_range(
                file_index, lo,
                {static_cast<std::byte*>(cover->data()), bytes})) {
            if (trace) { std::fprintf(stderr, "[cover] reject: read_range false\n"); }
            return nullptr;
        }
        ++stat_cover_reads;
        lo_out  = lo;
        end_out = hi;
        return static_cast<const std::byte*>(cover->data());
    }

    void upload_one(ExpertCacheKey key, cudaStream_t stream, bool settle = true,
                    std::size_t staging_slot = 0, std::vector<UploadOp>* batch = nullptr,
                    const std::byte* cover_base = nullptr, std::uint64_t cover_lo = 0,
                    std::uint64_t cover_end = 0) {
        const auto& bank = banks[key.bank];
        auto& entry = entries[index(key)];
        const auto slot = slot_of[key.bank][key.expert];
        if (slot == kAbsentSlot) { return; }
        if (slot >= slots_live) {
            throw std::runtime_error("expert slot is not backed: slot=" + std::to_string(slot) +
                                     " live=" + std::to_string(slots_live));
        }
        entry.offset = slot_address(slot);
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
            // port (sm_89): the pinned host tier is the best source when it holds this expert. It is
            // the same working set the CPU half reads, already locked, so the copy is a straight DMA
            // out of pinned memory - and the whole image goes as one operation instead of one per
            // plane. Falls through to the mapped artifact when the arena does not have it.
            if (arena_source != nullptr) {
                std::uint64_t arena_bytes = 0;
                const auto* arena_image =
                    arena_source->arena_image(key.bank, key.expert, arena_bytes);
                // Every Nth arena hit is deliberately served from the artifact instead, so the cold
                // side of the warm-tier decision keeps producing a number while the warm side is
                // winning. A controller that only measures the path it chose cannot tell when the
                // chosen path stops being the right one (1y.7 step 3f).
                static const std::uint64_t probe_every = [] {
                    const char* value = std::getenv("NINFER_ARENA_PROBE");
                    return value != nullptr && *value != '\0'
                               ? std::strtoull(value, nullptr, 10)
                               : std::uint64_t{64};
                }();
                static std::uint64_t arena_hits_seen = 0;
                bool probe_cold = false;
                if (probe_every != 0 && arena_image != nullptr && arena_bytes != 0) {
                    probe_cold = (++arena_hits_seen % probe_every) == 0;
                }
                if (probe_cold) {
                    ++stat_arena_probes;
                    arena_image = nullptr;
                    arena_bytes = 0;
                }
                if (arena_image != nullptr && arena_bytes != 0) {
                    ++stat_arena_experts;
                    // Verification: does the arena actually hold this expert's image? Cheap to ask,
                    // and it separates "the pinned tier hands over the wrong bytes" from everything
                    // else that could make a run produce empty text.
                    static const bool verify = std::getenv("NINFER_ARENA_VERIFY") != nullptr;
                    bool usable = true;
                    if (verify && arena_bytes == file->image_bytes) {
                        static std::size_t checks = 0;
                        if (checks < 40) {
                            ++checks;
                            std::vector<std::byte> reference(
                                static_cast<std::size_t>(file->image_bytes));
                            file_reader->read_entry_cached(
                                *file, {reference.data(), reference.size()});
                            const auto same = std::memcmp(reference.data(), arena_image,
                                                          reference.size()) == 0;
                            if (!same) {
                                std::fprintf(stderr,
                                             "[arena] MISMATCH bank=%u expert=%u bytes=%llu\n",
                                             key.bank, key.expert,
                                             static_cast<unsigned long long>(arena_bytes));
                                usable = false;
                            } else if (checks <= 3) {
                                std::fprintf(stderr, "[arena] match bank=%u expert=%u bytes=%llu\n",
                                             key.bank, key.expert,
                                             static_cast<unsigned long long>(arena_bytes));
                            }
                        }
                    } else if (verify && arena_bytes != file->image_bytes) {
                        static std::size_t mismatched = 0;
                        if (mismatched++ < 5) {
                            std::fprintf(stderr,
                                         "[arena] SIZE bank=%u expert=%u arena=%llu file=%llu\n",
                                         key.bank, key.expert,
                                         static_cast<unsigned long long>(arena_bytes),
                                         static_cast<unsigned long long>(file->image_bytes));
                        }
                        usable = false;
                    }
                    if (usable && batch != nullptr) {
                        ++stat_push;
                        batch->push_back({base, arena_image, static_cast<std::size_t>(arena_bytes)});
                        entry.ready    = true;
                        entry.last_use = ++epoch;
                        ++uploads;
                        ++stat_arena;
                        return;
                    }
                    const auto status =
                        cudaMemcpyAsync(base, arena_image, static_cast<std::size_t>(arena_bytes),
                                        cudaMemcpyHostToDevice, stream);
                    if (status != cudaSuccess) {
                        throw std::runtime_error(cudaGetErrorString(status));
                    }
                    if (settle) { (void)cudaStreamSynchronize(stream); }
                }
            }
            // port (sm_89): the reader can name the slices, so each expert is read once from the
            // page cache into the pinned ring instead of through read_entry's chunk loop.
            std::vector<ExpertBlobSpan> direct;
            if (file_reader->spans(*file, direct) && !direct.empty()) {
                ++stat_direct;
                // port (sm_89): a span names page-cache bytes in the artifact's file mapping, which
                // is not pinned memory. DMA-ing straight out of a file section is what raised
                // cudaErrorAlreadyMapped asynchronously on WDDM - the status only surfaced at the
                // next kernel launch, and only for experts whose file pages happened to collide with
                // an existing mapping. The spans are still the cheap source; the copy now lands in
                // the pinned staging ring first, which is the source the driver can map without
                // ambiguity. Same ring, same event wait, same one-DMA-per-expert shape as read_entry.
                if (!staging || file->image_bytes > max_entry_bytes ||
                    staging->size() < (staging_slot + 1) * max_entry_bytes) {
                    throw std::runtime_error("expert file image exceeds the staging buffer");
                }
                auto* host_base = static_cast<std::byte*>(staging->data()) +
                                  staging_slot * max_entry_bytes;
                if (staging_event[staging_slot] != nullptr) {
                    (void)cudaEventSynchronize(staging_event[staging_slot]);
                }
                for (const auto& span : direct) {
                    if (span.image_offset > file->image_bytes ||
                        span.bytes > file->image_bytes - span.image_offset) {
                        throw std::runtime_error("expert span lies outside its entry image");
                    }
                    std::memcpy(host_base + span.image_offset, span.source,
                                static_cast<std::size_t>(span.bytes));
                }
                if (batch != nullptr) {
                    ++stat_push;
                    staging_used.push_back(staging_slot);
                    batch->push_back(
                        {base, host_base, static_cast<std::size_t>(file->image_bytes)});
                    entry.ready    = true;
                    entry.last_use = ++epoch;
                    ++uploads;
                    return;
                }
                const auto copied =
                    cudaMemcpyAsync(base, host_base, static_cast<std::size_t>(file->image_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (copied != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(copied)); }
                if (settle) {
                    const auto settled = cudaStreamSynchronize(stream);
                    if (settled != cudaSuccess) {
                        throw std::runtime_error(cudaGetErrorString(settled));
                    }
                }
            } else {
                // port (sm_89): Q60. If this turn's cover was read in one go, the expert's slices are
                // already in pinned memory: name them and let the batch copy from there. Every chunk
                // has to lie inside the cover, or the bytes would be someone else's - so the test is
                // per chunk, not per expert.
                if (cover_base != nullptr && batch != nullptr) {
                    bool covered = !file->chunks.empty();
                    for (const auto& chunk : file->chunks) {
                        if (chunk.file_offset < cover_lo ||
                            chunk.bytes > cover_end - chunk.file_offset) {
                            covered = false;
                            break;
                        }
                    }
                    if (covered) {
                        for (const auto& chunk : file->chunks) {
                            ++stat_push;
                            batch->push_back({base + chunk.image_offset,
                                              cover_base + (chunk.file_offset - cover_lo),
                                              chunk.bytes});
                        }
                        entry.ready    = true;
                        entry.last_use = ++epoch;
                        ++uploads;
                        ++stat_coalesced;
                        return;
                    }
                }
                // The artifact is the source: one fetch into staging, one copy to the pool.
                if (!staging || file->image_bytes > max_entry_bytes ||
                    staging->size() < (staging_slot + 1) * max_entry_bytes) {
                    throw std::runtime_error("expert file image exceeds the staging buffer");
                }
                auto* host_base = static_cast<std::byte*>(staging->data()) +
                                  staging_slot * max_entry_bytes;
                // This slot is about to be overwritten. Its previous copy was queued kStagingRing
                // turns ago; waiting on that event is a bookkeeping check, not a stall - but if the
                // ring were ever made small enough to matter, the wait would still be correct.
                if (staging_event[staging_slot] != nullptr) {
                    (void)cudaEventSynchronize(staging_event[staging_slot]);
                }
                {
                    const auto read_begin = std::chrono::steady_clock::now();
                    file_reader->read_entry(
                        *file, {host_base, static_cast<std::size_t>(file->image_bytes)});
                    stat_read_us += std::chrono::duration<double, std::micro>(
                                        std::chrono::steady_clock::now() - read_begin)
                                        .count();
                    ++stat_file_experts;
                }
                ++stat_staged;
                if (batch != nullptr) {
                    ++stat_push;
                    staging_used.push_back(staging_slot);
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
    const auto begin = std::chrono::steady_clock::now();
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
    if (std::getenv("NINFER_UPLOAD_TIMING") != nullptr) {
        static std::uint64_t calls = 0;
        static double ms            = 0.0;
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                  .count();
        if (++calls % 40 == 0) {
            std::fprintf(stderr, "[prepare] %.3fms/layer\n", ms / 40.0);
            std::fflush(stderr);
            ms = 0.0;
        }
    }
}

bool ExpertCache::set_resident_experts(std::uint32_t bank, std::span<const std::uint32_t> experts,
                                       std::uint32_t hot_count) {
    return impl_->set_resident(bank, experts, hot_count);
}

std::uint32_t ExpertCache::unready_count(std::uint32_t bank,
                                         std::span<const std::uint32_t> experts) const {
    auto& state = *impl_;
    if (bank >= state.first.size() || bank >= state.entries.size()) { return 0; }
    std::uint32_t unready = 0;
    for (const std::uint32_t expert : experts) {
        if (expert >= state.slot_of[bank].size()) { continue; }
        if (state.slot_of[bank][expert] == kAbsentSlot) { continue; } // not mapped: a host route
        const auto& entry = state.entries[state.first[bank] + expert];
        if (!entry.ready) { ++unready; }
    }
    return unready;
}

bool ExpertCache::adopt_resident_slot(std::uint32_t bank, std::uint32_t expert,
                                      std::uint32_t slot) {
    return impl_->adopt_slot(bank, expert, slot);
}

void ExpertCache::configure_ring(std::uint32_t bank, std::uint32_t slots) {
    impl_->configure_ring(bank, slots);
}

std::uint32_t ExpertCache::ring_slot_count(std::uint32_t bank) const noexcept {
    return bank < impl_->ring_slots.size()
               ? static_cast<std::uint32_t>(impl_->ring_slots[bank].size())
               : 0;
}

std::optional<std::uint32_t> ExpertCache::ring_slot(std::uint32_t bank,
                                                    std::uint32_t index) const noexcept {
    if (bank >= impl_->ring_slots.size() || index >= impl_->ring_slots[bank].size()) {
        return std::nullopt;
    }
    return impl_->ring_slots[bank][index];
}

std::uint32_t ExpertCache::set_prefetch_experts(std::uint32_t bank,
                                                std::span<const std::uint32_t> experts,
                                                std::vector<ExpertCacheKey>& placed) {
    return impl_->set_prefetch(bank, experts, placed);
}

bool ExpertCache::prefetch_slot(std::uint32_t slot) const noexcept {
    return impl_->prefetch_slot(slot);
}

std::uint32_t ExpertCache::prefetch_slots_per_bank() const noexcept {
    return impl_->prefetch_per_bank;
}

std::uint64_t ExpertCache::hot_generation(std::uint32_t bank) const noexcept {
    return bank < impl_->hot_gen.size() ? impl_->hot_gen[bank] : 0;
}

std::uint64_t ExpertCache::step_generation(std::uint32_t bank) const noexcept {
    return bank < impl_->step_gen.size() ? impl_->step_gen[bank] : 0;
}

std::uint64_t ExpertCache::prefetch_generation(std::uint32_t bank) const noexcept {
    return bank < impl_->prefetch_gen.size() ? impl_->prefetch_gen[bank] : 0;
}

const std::vector<std::uint32_t>& ExpertCache::prefetch_owners(std::uint32_t bank) const {
    static const std::vector<std::uint32_t> none;
    if (bank >= impl_->prefetch_owner.size()) { return none; }
    return impl_->prefetch_owner[bank];
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
    {
        // port (sm_89): Q60. Which keys this turn takes is decided before anything is read, because
        // the coalesced form needs all of them: a range that covers four experts is one read, a range
        // that covers one is the read it always was.
        std::uint32_t pick_bank[Impl::kUploadBatch];
        std::uint32_t pick_expert[Impl::kUploadBatch];
        std::size_t pick_staging[Impl::kUploadBatch];
        std::size_t picked = 0;
        static const bool group_by_bank = std::getenv("NINFER_UPLOAD_COALESCE") != nullptr;
        for (auto it = impl_->required.begin(); it != impl_->required.end(); ++it) {
            const auto& candidate = impl_->entries[impl_->index(*it)];
            if (candidate.ready) { continue; }
            if (picked == Impl::kUploadBatch) { break; }
            // A coalesced turn has to be one bank: the cover is a byte range of one object, so a
            // mixed turn cannot be covered. Grouping costs nothing when the switch is off, and when
            // it is on the turn simply takes what the first key's bank has pending.
            if (picked != 0 && group_by_bank && it->bank != pick_bank[0]) { continue; }
            pick_bank[picked]    = it->bank;
            pick_expert[picked]  = it->expert;
            pick_staging[picked] = impl_->staging_cursor % Impl::kStagingRing;
            ++impl_->staging_cursor;
            ++picked;
        }
        std::uint64_t cover_lo = 0, cover_end = 0;
        const std::byte* cover_base =
            picked == 0 ? nullptr
                        : impl_->cover_for(pick_bank, pick_expert, picked, cover_lo, cover_end);
        for (std::size_t i = 0; i < picked; ++i) {
            impl_->upload_one({pick_bank[i], pick_expert[i]}, stream, false, pick_staging[i], &batch,
                              cover_base, cover_lo, cover_end);
            ++staged;
        }
    }
    const auto collect_begin = std::chrono::steady_clock::now();
    if (staged == 0) { return {0, true}; }
    // port (sm_89): the batch these were collected into is one submission instead of one per
    // slice. The copies are independent by construction - a slot, a source and a direction each -
    // so the batch's unspecified internal order changes nothing, and the kernel that reads them is
    // queued behind the whole batch on this stream.
    if (!batch.empty()) {
        bool submitted = false;
        // Q52: the batch form is one submission, but a submission is not free. At the shape the
        // split path actually has (2.8 slices a layer after the span merge), three interleaved
        // rounds at 8 GiB measured batch 0.158/0.161/0.159 ms ens against per-slice
        // 0.154/0.151/0.151 - the batch API costs 4-5% for copies this small. The criterion is
        // "ops per layer <= 3", and 2.8 already meets it; the batch is therefore opt-in for the
        // case where the slice count grows, not the default.
        static const bool batch_submit = [] {
            const char* value = std::getenv("NINFER_UPLOAD_SINGLE_SUBMIT");
            return value != nullptr && value[0] != '0';
        }();
        if (stream != nullptr && batch.size() > 1 && batch_submit) {
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
            if (submitted) {
                ++impl_->stat_submits;
            } else if (status != cudaErrorNotSupported && status != cudaErrorInvalidValue) {
                throw std::runtime_error(cudaGetErrorString(status));
            }
        }
        if (!submitted) {
            for (const auto& op : batch) {
                const auto status =
                    cudaMemcpyAsync(op.destination, op.source, op.bytes, cudaMemcpyHostToDevice,
                                    stream);
                if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
                ++impl_->stat_submits;
            }
        }
    }
    // port (sm_89): the staging ring replaces the per-turn barrier. Each slot used by this turn gets
    // an event on the stream, and that event is the only thing waited on before the slot is reused -
    // by which time the copy is a dozen layers old. The data path needs no barrier at all: the
    // kernel that reads the destination is queued behind the copy on this same stream.
    for (const auto slot : impl_->staging_used) {
        if (impl_->staging_event[slot] == nullptr) {
            (void)cudaEventCreateWithFlags(&impl_->staging_event[slot], cudaEventDisableTiming);
        }
        (void)cudaEventRecord(impl_->staging_event[slot], stream);
    }
    impl_->staging_used.clear();
    // A/B switch: NINFER_STAGING_SYNC=1 restores the per-turn barrier the ring replaces, so the two
    // can be measured in one session against the same binary. Across sessions this box's decode rate
    // moves by more than the difference being measured.
    if (std::getenv("NINFER_STAGING_SYNC") != nullptr) {
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
        static double acc[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        static double acc_arena = 0.0;
        static double acc_submits = 0.0;
        // acc: collect, submit, bytes, direct, staged, views, pushes. Every per-call delta has to
        // be accumulated here first; dividing a single call's delta by 40 was what made the
        // operation count read 0.1 while the byte total said five megabytes.
        const auto elapsed = [](auto from, auto to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        acc[0] += elapsed(timing_begin, collect_begin);
        acc[1] += elapsed(collect_begin, timing_end);
        double layer_bytes = 0.0;
        for (const auto& op : batch) { layer_bytes += static_cast<double>(op.bytes); }
        acc[2] += layer_bytes;        static std::uint64_t seen_direct = 0, seen_staged = 0, seen_views = 0, seen_arena = 0;
        const auto direct_now = impl_->stat_direct;
        const auto staged_now = impl_->stat_staged;
        const auto views_now  = impl_->stat_views;
        const auto arena_hits = impl_->stat_arena;
        const auto direct_delta = direct_now - seen_direct;
        const auto staged_delta = staged_now - seen_staged;
        const auto views_delta  = views_now - seen_views;
        const auto arena_delta  = arena_hits - seen_arena;
        seen_direct = direct_now;
        seen_staged = staged_now;
        seen_views  = views_now;
        seen_arena  = arena_hits;
        static std::uint64_t seen_push = 0;
        const auto push_delta = impl_->stat_push - seen_push;
        seen_push             = impl_->stat_push;
        static std::uint64_t seen_submits = 0;
        const auto submit_delta           = impl_->stat_submits - seen_submits;
        seen_submits                      = impl_->stat_submits;
        acc_submits += static_cast<double>(submit_delta);
        // Q60: how much of the turn the coalesced form actually covered - one read for how many
        // experts. Without this the switch "working" and the switch "never firing" look the same.
        static std::uint64_t seen_coalesced = 0;
        const auto coalesced_delta          = impl_->stat_coalesced - seen_coalesced;
        seen_coalesced                      = impl_->stat_coalesced;
        acc[3] += static_cast<double>(direct_delta);
        acc[4] += static_cast<double>(staged_delta);
        acc[5] += static_cast<double>(views_delta);
        acc[6] += static_cast<double>(push_delta);
        acc_arena += static_cast<double>(arena_delta);
        if (++calls % 40 == 0) {
            // "per call", not "per layer": `calls` only advances when a layer actually uploads
            // something, so dividing by 40 normalises across the 40 *uploading* layers in the window.
            // Reading it as 40 layers = one token is how 1.187 ms/layer turned into an impossible
            // 47 ms/token next to the expert panel's 1.5-2.5 ms/token.
            std::fprintf(stderr,
                         "[upload] collect=%.3fms submit=%.3fms ops=%.1f submits=%.1f mb=%.2f per "
                         "uploading call (x40 window) direct=%.1f staged=%.1f views=%.1f\n",
                         acc[0] / 40.0, acc[1] / 40.0, acc[6] / 40.0, acc_submits / 40.0,
                         acc[2] / 40.0 / 1048576.0, acc[3] / 40.0, acc[4] / 40.0, acc[5] / 40.0);
            std::fprintf(stderr, "[upload] arena=%.1f experts per uploading call\n", acc_arena / 40.0);
            // Q60: experts answered from a cover read, per uploading call. A number that stays at
            // zero while the switch is on says the turn's keys were not one bank - which is a fact
            // about the picker, not about the reader.
            std::fprintf(stderr, "[upload] coalesced=%.1f experts/call cover_reads=%llu\n",
                         static_cast<double>(coalesced_delta) / 40.0,
                         static_cast<unsigned long long>(impl_->stat_cover_reads));
            // Raw values for the call being reported: if these two disagree while the averages
            // above do, the disagreement is in the accumulation and not in the upload itself.
            std::fprintf(stderr,
                         "[upload] raw this-call: batch=%zu push_total=%llu mb=%.2f staged=%zu\n",
                         batch.size(), static_cast<unsigned long long>(impl_->stat_push),
                         layer_bytes / 1048576.0, staged);
            acc[0] = acc[1] = acc[2] = acc[3] = acc[4] = acc[5] = acc[6] = 0.0;
            acc_arena = 0.0;
            acc_submits = 0.0;
        }
        (void)direct_delta;
        (void)staged_delta;
    }
    return {staged, false};
}

void ExpertCache::wait_for_progress(cudaStream_t stream) {
    // port (sm_89): the caller's report says this is where a fully covered split layer spends its
    // time, and the body below says it should be a no-op when nothing is missing. Those two cannot
    // both be true, so the round count and the staged count are published: a layer that uploads
    // nothing must report rounds=1 staged=0, and a layer that does not is the answer.
    const bool timing    = std::getenv("NINFER_UPLOAD_TIMING") != nullptr;
    const auto begin     = std::chrono::steady_clock::now();
    std::uint64_t rounds = 0;
    std::uint64_t staged_total = 0;
    while (true) {
        const auto state = progress(stream);
        ++rounds;
        staged_total += state.submitted_batches;
        if (state.required_published) { break; }
    }
    impl_->last_staged = static_cast<std::uint32_t>(staged_total);
    // port (sm_89): the staging events this loop waits on are the copies' completion, so this is the
    // moment the arena may reuse the slots it handed out (ExpertArenaSource::uploads_settled).
    if (impl_->arena_source != nullptr) {
        const_cast<ExpertArenaSource*>(impl_->arena_source)->uploads_settled();
    }
    if (timing) {
        static std::uint64_t calls       = 0;
        static double ms                 = 0.0;
        static std::uint64_t rounds_seen = 0;
        static std::uint64_t staged_seen = 0;
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                  .count();
        rounds_seen += rounds;
        staged_seen += staged_total;
        if (++calls % 40 == 0) {
            std::fprintf(stderr, "[wait] %.3fms/layer rounds=%.1f staged=%.1f\n", ms / 40.0,
                         static_cast<double>(rounds_seen) / 40.0,
                         static_cast<double>(staged_seen) / 40.0);
            std::fflush(stderr);
            ms          = 0.0;
            rounds_seen = 0;
            staged_seen = 0;
        }
    }
}

std::uint32_t ExpertCache::last_uploaded() const noexcept { return impl_->last_staged; }

std::uint64_t ExpertCache::layout_generation() const noexcept { return impl_->layout_generation; }

std::uint64_t ExpertCache::bank_generation(std::uint32_t bank) const noexcept {
    return bank < impl_->bank_generation.size() ? impl_->bank_generation[bank] : 0;
}

ExpertCache::UploadSourceStats ExpertCache::upload_sources() const {
    UploadSourceStats stats;
    stats.arena_experts = impl_->stat_arena_experts;
    stats.arena_probes  = impl_->stat_arena_probes;
    stats.file_experts  = impl_->stat_file_experts;
    stats.read_us       = impl_->stat_read_us;
    return stats;
}

void ExpertCache::attach_arena(const ExpertArenaSource* arena) noexcept {
    // port (sm_89): on by default. A bank addressed in place has no arena slot, and the arena is
    // empty until the host tier fills it, so a run that never enables the host arena is unaffected.
    // Verified byte-for-byte against the reader (NINFER_ARENA_VERIFY) and through the split gate:
    // with an 8 GiB arena the generated body is identical to the artifact path
    // (64239db9317155bc) and split float PPL stays 7.765921.
    const char* value = std::getenv("NINFER_ARENA_UPLOAD");
    if (value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) == 0) { return; }
    impl_->arena_source = arena;
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
    if (!impl_->per_bank_layout) {
        std::uint64_t ready = 0;
        for (const auto& entry : impl_->entries) {
            if (entry.ready) { ++ready; }
        }
        return ready * impl_->slot_bytes;
    }
    std::uint64_t bytes = 0;
    for (std::size_t b = 0; b < impl_->banks.size(); ++b) {
        if (impl_->banks[b].resident) { continue; }
        for (std::uint32_t e = 0; e < impl_->banks[b].count; ++e) {
            if (impl_->entries[impl_->first[b] + e].ready) {
                bytes += impl_->banks[b].entry_bytes;
            }
        }
    }
    return bytes;
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
std::uint64_t ExpertCache::windows() const noexcept {
    // How many whole layers the pool holds. Per-bank pricing means each bank has its own run, so
    // the answer is the number of banks that are fully backed, not a ratio of two totals.
    if (impl_->per_bank_layout) {
        std::uint64_t whole = 0;
        for (std::size_t b = 0; b < impl_->banks.size(); ++b) {
            if (impl_->banks[b].resident) { continue; }
            if (impl_->bank_slot_live[b] == impl_->banks[b].count) { ++whole; }
        }
        return whole;
    }
    return impl_->slot_capacity() != 0 ? impl_->slots_live / impl_->slot_capacity() : 0;
}
std::uint32_t ExpertCache::resident_capacity() const noexcept {
    // One conservative answer for the whole pool, so a caller that budgets every bank alike cannot
    // over-commit the narrowest one.
    if (impl_->per_bank_layout) {
        std::uint32_t narrowest = std::numeric_limits<std::uint32_t>::max();
        for (std::size_t b = 0; b < impl_->banks.size(); ++b) {
            if (impl_->banks[b].resident) { continue; }
            narrowest = std::min(narrowest, impl_->bank_slot_live[b]);
        }
        return narrowest == std::numeric_limits<std::uint32_t>::max() ? 0 : narrowest;
    }
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(impl_->slots_live, impl_->slot_capacity()));
}

std::uint32_t ExpertCache::resident_capacity(std::uint32_t bank) const noexcept {
    // The bank's own run, not the narrowest run: per-bank pricing means one bank can be much wider
    // than the rest, and a cap meant for this bank must be read off this bank.
    if (!impl_->per_bank_layout) {
        // One shared slot space: every bank draws from the same list, and the hint was already
        // clamped to the per-bank share when the pool was built, so the live floor is a guard only.
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(impl_->slot_capacity(), impl_->slots_live));
    }
    if (bank >= impl_->bank_slot_live.size()) { return 0; }
    return impl_->bank_slot_live[bank];
}

std::uint64_t ExpertCache::slot_offset(std::uint32_t slot) const noexcept {
    // Address of one slot id. The ring admission path copies into `pool().data + this`, and with
    // per-bank pricing that is no longer `slot * slot_bytes`.
    return impl_->slot_address(slot);
}

std::uint64_t ExpertCache::slot_stride(std::uint32_t bank) const noexcept {
    if (!impl_->per_bank_layout || bank >= impl_->banks.size()) { return impl_->slot_bytes; }
    return impl_->bank_stride[bank];
}

std::uint64_t ExpertCache::slot_bias(std::uint32_t bank) const noexcept {
    if (!impl_->per_bank_layout || bank >= impl_->banks.size()) { return 0; }
    const auto base = impl_->bank_slot_base[bank];
    return impl_->slot_offset[base] - std::uint64_t{base} * impl_->bank_stride[bank];
}

std::uint64_t ExpertCache::resident_experts() const noexcept {
    std::uint64_t count = 0;
    for (std::size_t b = 0; b < impl_->banks.size(); ++b) {
        count += impl_->banks[b].resident ? impl_->banks[b].count : impl_->resident_count[b];
    }
    return count;
}

bool ExpertCache::can_hold_every_layer() const noexcept { return impl_->holds_every_layer(); }

bool ExpertCache::window_resident(std::uint32_t bank) const noexcept {
    return bank < impl_->resident_count.size() &&
           impl_->resident_count[bank] == impl_->banks[bank].count;
}

void ExpertCache::preload_all() { impl_->preload_every_layer(); }

} // namespace ninfer








