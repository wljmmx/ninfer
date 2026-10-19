#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

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
        // The down matrix follows the same plane order inside the same entry image.
        const auto down_codes = add_plane(entry, bank.down.geometry, true, false, false);
        const auto down_high  = add_plane(entry, bank.down.geometry, false, true, false);
        const auto down_scales = add_plane(entry, bank.down.geometry, false, false, true);
        (void)down_codes; (void)down_high; (void)down_scales;
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
    DeviceBuffer pool;
    std::vector<std::size_t> first; // flat entry index of each bank's expert 0
    std::vector<Entry> entries;
    std::vector<ExpertKey> required;
    std::uint64_t used = 0, uploads = 0, evictions = 0, epoch = 0;

    Impl(std::span<const ExpertBankSpec> specs, const ExpertCacheRequirements& req,
         const DeviceContext& device, std::span<const ExpertBankPair> bank_views)
        : config(req.config), banks(req.banks), views(bank_views.begin(), bank_views.end()),
          pool(static_cast<std::size_t>(req.config.capacity_bytes)) {
        if (req != describe_expert_cache(specs, req.config)) {
            throw std::invalid_argument("expert cache requirements differ from their sources");
        }
        if (views.size() != banks.size()) {
            throw std::invalid_argument("expert cache views differ from their banks");
        }
        std::size_t flat = 0;
        for (const auto& bank : banks) {
            first.push_back(flat);
            if (!bank.resident) { flat += bank.count; }
        }
        entries.resize(flat);
        (void)device;
    }

    std::size_t index(ExpertKey key) const {
        if (key.bank >= banks.size() || banks[key.bank].resident ||
            key.expert >= banks[key.bank].count) {
            throw std::out_of_range("expert cache key");
        }
        return first[key.bank] + key.expert;
    }

    std::uint64_t allocate(std::uint64_t bytes) {
        if (used + bytes <= config.capacity_bytes) {
            const auto offset = used;
            used += bytes;
            return offset;
        }
        // Evict the least recently used ready entry.
        std::size_t victim = entries.size();
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t i = 0; i < entries.size(); ++i) {
            if (!entries[i].ready) { continue; }
            if (entries[i].last_use < oldest) { oldest = entries[i].last_use; victim = i; }
        }
        if (victim == entries.size()) { throw std::runtime_error("expert cache pool is full"); }
        entries[victim].ready = false;
        ++evictions;
        return entries[victim].offset;
    }

    void upload_one(ExpertKey key) {
        const auto& bank = banks[key.bank];
        auto& entry      = entries[index(key)];
        if (entry.offset == kAbsent) { entry.offset = allocate(bank.entry_bytes); }
        const auto& view = views[key.bank];
        auto* base       = static_cast<std::byte*>(pool.p) + entry.offset;
        const auto copy  = [&](ExpertPlaneKind kind, LayoutRegion region, bool down = false) {
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
        copy(ExpertPlaneKind::Codes, bank.codes);
        copy(ExpertPlaneKind::High, bank.high);
        copy(ExpertPlaneKind::Scales, bank.scales);
        // The down matrix planes follow the gate_up planes in the same entry image.
        const auto& down = banks[key.bank];
        (void)down;
        entry.ready    = true;
        entry.last_use = ++epoch;
        ++uploads;
    }
};

ExpertCache::ExpertCache(std::span<const ExpertBankSpec> specs,
                         const ExpertCacheRequirements& requirements, const DeviceContext& device,
                         std::span<const ExpertBankPair> views)
    : impl_(std::make_unique<Impl>(specs, requirements, device, views)) {}

ExpertCache::~ExpertCache() = default;

void ExpertCache::prepare(std::span<const ExpertKey> required) {
    impl_->required.assign(required.begin(), required.end());
}

ExpertCacheProgress ExpertCache::progress() {
    for (const auto& key : impl_->required) {
        const auto& entry = impl_->entries[impl_->index(key)];
        if (entry.ready) { continue; }
        impl_->upload_one(key);
        return {1, false};
    }
    return {0, true};
}

void ExpertCache::wait_for_progress() {
    while (!progress().required_published) {}
}

std::optional<std::uint64_t> ExpertCache::offset(ExpertKey key) const {
    if (key.bank >= impl_->banks.size() || impl_->banks[key.bank].resident ||
        key.expert >= impl_->banks[key.bank].count) {
        return std::nullopt;
    }
    const auto& entry = impl_->entries[impl_->index(key)];
    if (!entry.ready) { return std::nullopt; }
    return entry.offset;
}

std::uint64_t ExpertCache::used_bytes() const noexcept { return impl_->used; }
std::uint64_t ExpertCache::capacity_bytes() const noexcept { return impl_->config.capacity_bytes; }
std::uint64_t ExpertCache::uploads() const noexcept { return impl_->uploads; }
std::uint64_t ExpertCache::evictions() const noexcept { return impl_->evictions; }

} // namespace ninfer



