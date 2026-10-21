#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::uint64_t kAbsent = std::numeric_limits<std::uint64_t>::max();

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
    std::unique_ptr<PinnedHostBuffer> staging;
    DeviceBuffer pool;
    std::vector<std::size_t> first; // flat entry index of each bank's expert 0
    std::vector<Entry> entries;
    std::vector<ExpertCacheKey> required;
    std::uint64_t used = 0, uploads = 0, evictions = 0, epoch = 0;
    std::uint64_t slot_bytes = 0;
    std::vector<std::uint64_t> window_of;
    std::vector<std::uint64_t> last_window_use;
    std::vector<std::uint64_t> window_hits;
    int ordinal = 0;

    Impl(std::span<const ExpertBankSpec> specs, const ExpertCacheRequirements& req,
         const DeviceContext& device, std::span<const ExpertBankPair> bank_views,
         std::span<const ExpertFileBank> file_banks, const ExpertBlobReader* reader)
        : config(req.config), banks(req.banks), views(bank_views.begin(), bank_views.end()),
          files(file_banks.begin(), file_banks.end()), file_reader(reader),
          staging(req.max_entry_bytes
                      ? std::make_unique<PinnedHostBuffer>(
                            static_cast<std::size_t>(req.max_entry_bytes))
                      : nullptr),
          pool(static_cast<std::size_t>(req.config.capacity_bytes)) {
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
        window_of.assign(banks.size(), kAbsent);
        last_window_use.assign(banks.size(), 0);
        window_hits.assign(banks.size(), 0);
        ordinal    = device.device;
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
        return widest;
    }

    std::uint64_t window_count() const {
        const auto slots = config.capacity_bytes / slot_bytes;
        return window_slots() == 0 ? 0 : slots / window_slots();
    }

    // port (sm_89): a bank owns one contiguous run of slots, so its experts form a strided
    // matrix. Windows are replaced whole: addressing stays affine and no per-expert offset
    // table can go stale while a kernel reads it.
    void ensure_window(std::uint32_t bank) {
        if (window_of[bank] != kAbsent) { last_window_use[bank] = ++epoch; ++window_hits[bank]; return; }
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


    void upload_one(ExpertCacheKey key, cudaStream_t stream) {
        const auto& bank = banks[key.bank];
        ensure_window(key.bank);
        auto& entry = entries[index(key)];
        entry.offset = window_of[key.bank] * window_slots() * slot_bytes + std::uint64_t(key.expert) * slot_bytes;
        if (pool.p == nullptr) { throw std::runtime_error("expert pool has no Device storage"); }
        if (entry.offset == kAbsent || entry.offset > pool.bytes ||
            bank.entry_bytes > pool.bytes - entry.offset) {
            throw std::runtime_error("expert slot is inconsistent: offset=" +
                                     std::to_string(entry.offset) + " entry=" +
                                     std::to_string(bank.entry_bytes) + " pool=" +
                                     std::to_string(pool.bytes) + " bank=" +
                                     std::to_string(key.bank) + " expert=" +
                                     std::to_string(key.expert));
        }
        auto* base = static_cast<std::byte*>(pool.p) + entry.offset;

        const ExpertFileEntry* file = nullptr;
        if (file_reader && key.bank < files.size() &&
            key.expert < files[key.bank].experts.size()) {
            const auto& candidate = files[key.bank].experts[key.expert];
            if (candidate.image_bytes) { file = &candidate; }
        }
        if (file) {
            // The artifact is the source: one fetch into pinned staging, one copy to the pool.
            if (!staging || file->image_bytes > staging->size()) {
                throw std::runtime_error("expert file image exceeds the staging buffer");
            }
            file_reader->read_entry(
                *file, {static_cast<std::byte*>(staging->data()),
                        static_cast<std::size_t>(file->image_bytes)});
            const auto status = cudaMemcpyAsync(base, staging->data(),
                                           static_cast<std::size_t>(file->image_bytes),
                                           cudaMemcpyHostToDevice, stream);
            if (status == cudaSuccess) { (void)cudaStreamSynchronize(stream); }
            if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
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
                const auto status = cudaMemcpy(base + region.offset, span.data, region.bytes,
                                               cudaMemcpyHostToDevice);
                if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
            };
            copy(ExpertPlaneKind::Codes, bank.codes, false);
            copy(ExpertPlaneKind::High, bank.high, false);
            copy(ExpertPlaneKind::Scales, bank.scales, false);
            copy(ExpertPlaneKind::Codes, bank.down_codes, true);
            copy(ExpertPlaneKind::High, bank.down_high, true);
            copy(ExpertPlaneKind::Scales, bank.down_scales, true);
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
    impl_->required.assign(required.begin(), required.end());
}

ExpertCacheProgress ExpertCache::progress() {
    for (const auto& key : impl_->required) {
        const auto& entry = impl_->entries[impl_->index(key)];
        if (entry.ready) { continue; }
        impl_->upload_one(key, nullptr);
        return {1, false};
    }
    return {0, true};
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
std::uint64_t ExpertCache::capacity_bytes() const noexcept { return impl_->config.capacity_bytes; }
std::uint64_t ExpertCache::slot_bytes() const { return impl_->slot_bytes; }
std::uint64_t ExpertCache::window_origin(std::uint32_t bank) const {
    if (bank >= impl_->window_of.size() || impl_->window_of[bank] == kAbsent) {
        throw std::out_of_range("expert bank has no resident window");
    }
    return impl_->window_of[bank] * impl_->window_slots() * impl_->slot_bytes;
}
ConstMemorySpan ExpertCache::pool() const {
    return {impl_->pool.p, impl_->pool.bytes, Device::cuda(impl_->ordinal)};
}
std::uint64_t ExpertCache::uploads() const noexcept { return impl_->uploads; }
std::uint64_t ExpertCache::evictions() const noexcept { return impl_->evictions; }

} // namespace ninfer





