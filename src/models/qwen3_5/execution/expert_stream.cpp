#include "models/qwen3_5/execution/expert_stream.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::uint64_t rows_of(const WeightGeometry& parent, std::uint64_t elements) {
    if (parent.shape.empty() || parent.shape.back() == 0) {
        throw std::invalid_argument("streamed expert parent has no column axis");
    }
    return elements / parent.shape.back();
}

// port (sm_89): how the hot set of a bank follows the router. Only small batches train it, so a
// prefill chunk (which touches most experts anyway) cannot wash the ranking out; the score decays
// so the set follows the conversation instead of remembering the first prompt forever.
constexpr std::int32_t kHotSetUpdateTokens  = 4;
constexpr std::uint32_t kHotSetDecayPeriod  = 32;
constexpr std::uint32_t kHotSetReorderPeriod = 32;

} // namespace

ExpertStream::ExpertStream(std::uint32_t count, WeightBankLayout gate_up, WeightBankLayout down,
                           WeightGeometry gate_up_parent, WeightGeometry down_parent,
                           std::uint64_t gate_up_object, std::uint64_t down_object)
    : count_(count), gate_up_(gate_up), down_(down), gate_up_parent_(std::move(gate_up_parent)),
      down_parent_(std::move(down_parent)),
      gate_up_object_(gate_up_object), down_object_(down_object) {
    const std::array<std::uint64_t, 2> gate_up_shape{gate_up_.region.rows, gate_up_.columns};
    const std::array<std::uint64_t, 2> down_shape{down_.region.rows, down_.columns};
    gate_up_expert_ = weight_geometry(gate_up_.format, gate_up_.layout, gate_up_shape);
    down_expert_    = weight_geometry(down_.format, down_.layout, down_shape);
    keys_.reserve(count_);
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        keys_.push_back(ExpertCacheKey{0, expert});
    }
    hot_score_.assign(count_, 0.0F);
}

void ExpertStream::attach(ExpertCache& cache, std::uint32_t bank, const ExpertBankLayout& entry) {
    if (bank > 0xffffU || cache.capacity_bytes() < entry.entry_bytes) {
        throw std::invalid_argument("expert stream cannot attach to this cache");
    }
    cache_ = &cache;
    bank_  = bank;
    entry_ = entry;
    for (auto& key : keys_) { key.bank = bank; }
    

    const auto pool = cache.pool();
    const auto base = static_cast<const std::byte*>(pool.data);
    auto& gate_up_view = experts_.bank.gate_up;
    gate_up_view.codes = base + entry.codes.offset;
    gate_up_view.high  = entry.high.bytes ? base + entry.high.offset : nullptr;
    gate_up_view.scales = base + entry.scales.offset;
    gate_up_view.code_stride  = cache.slot_bytes();
    gate_up_view.high_stride  = entry.high.bytes ? cache.slot_bytes() : 0;
    gate_up_view.scale_stride = cache.slot_bytes();
    auto& down_view   = experts_.bank.down;
    down_view         = gate_up_view;
    down_view.codes   = base + entry.down_codes.offset;
    down_view.high    = entry.down_high.bytes ? base + entry.down_high.offset : nullptr;
    down_view.scales  = base + entry.down_scales.offset;
    down_view.code_stride  = cache.slot_bytes();
    down_view.high_stride  = entry.down_high.bytes ? cache.slot_bytes() : 0;
    down_view.scale_stride = cache.slot_bytes();
    experts_.bank.count  = count_;
    experts_.bank.device = pool.device;

    // Publish the per-expert offset table. Every entry is expert * slot for now; the kernel reads
    // its plane base through this table, so a partial resident set only rewrites the entries.
    offsets_.resize(count_);
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        offsets_[expert] = std::uint64_t{expert} * cache.slot_bytes();
    }
    const auto table_bytes = offsets_.size() * sizeof(std::uint64_t);
    offsets_device_        = std::make_unique<DeviceBuffer>(table_bytes);
    const auto copied      = cudaMemcpy(offsets_device_->p, offsets_.data(), table_bytes,
                                        cudaMemcpyHostToDevice);
    if (copied != cudaSuccess) {
        throw std::runtime_error(std::string("expert offset table: ") + cudaGetErrorString(copied));
    }
    const auto* table = static_cast<const std::uint64_t*>(offsets_device_->p);
    gate_up_view.offsets = table;
    down_view.offsets    = table;
    // Indexed addressing names its backing store, so the pool has to travel with the table.
    gate_up_view.indexed_storage = pool;
    down_view.indexed_storage    = pool;

    // Publish which experts the pool will hold. With room for the whole layer every entry is set;
    // otherwise only the leading resident_capacity() experts are, and the caller masks the rest
    // out of the GPU job list.
    const auto resident = cache.resident_capacity();
    std::vector<std::int32_t> mask(count_, 0);
    for (std::uint32_t expert = 0; expert < std::min(count_, resident); ++expert) {
        mask[expert] = 1;
    }
    mask_device_ = std::make_unique<DeviceBuffer>(mask.size() * sizeof(std::int32_t));
    mask_host_   = mask;
    const auto mask_bytes = mask.size() * sizeof(std::int32_t);
    const auto mask_copy  = cudaMemcpy(mask_device_->p, mask.data(), mask_bytes,
                                       cudaMemcpyHostToDevice);
    if (mask_copy != cudaSuccess) {
        throw std::runtime_error(std::string("expert residency mask: ") + cudaGetErrorString(mask_copy));
    }
    // port (sm_89): with one window per bank the resident set never changes, so this stream can
    // skip the routing round trip entirely and a captured graph stays valid.
    pinned_ = cache.can_hold_every_layer();
}

void ExpertStream::rebase_window() {
    const auto origin = cache_->window_origin(bank_);
    const auto base   = static_cast<const std::byte*>(cache_->pool().data) + origin;
    experts_.bank.gate_up.codes  = base + entry_.codes.offset;
    experts_.bank.gate_up.high   = entry_.high.bytes ? base + entry_.high.offset : nullptr;
    experts_.bank.gate_up.scales = base + entry_.scales.offset;
    experts_.bank.down.codes     = base + entry_.down_codes.offset;
    experts_.bank.down.high      = entry_.down_high.bytes ? base + entry_.down_high.offset : nullptr;
    experts_.bank.down.scales    = base + entry_.down_scales.offset;
}

bool ExpertStream::fully_resident() const noexcept {
    return cache_ != nullptr && cache_->resident_capacity() >= count_;
}

bool ExpertStream::ensure(const void* device_ids, std::int32_t top_k, std::int32_t tokens,
                          cudaStream_t stream) {
    if (cache_ == nullptr) { throw std::logic_error("expert stream is not attached"); }
    // port (sm_89): every layer is resident, so the window cannot move and the routers do not need
    // to be read back. Rebasing is the whole per-layer work, and it happens during capture so the
    // frozen kernel arguments already point at the right window.
    if (pinned_) {
        if (cache_->window_resident(bank_)) {
            rebase_window();
            return true;
        }
        // A resize dropped this window; fall through so the normal path allocates a new one.
    }
    // One window holds a whole layer when the pool is large enough; otherwise the window shrinks to
    // the pool and the caller splits the layer between the Device pool and the host tier.
    const auto capacity = cache_->resident_capacity();
    if (capacity == 0) { return false; }
    const bool fully = capacity >= count_;
    const auto values = static_cast<std::size_t>(top_k) * static_cast<std::size_t>(tokens);
    const auto bytes  = values * sizeof(std::int32_t);
    // port (sm_89): the host must know the routed experts before it can publish them, so one
    // pinned staging buffer carries the route IDs back and the caller then waits on the stream.
    if (!route_staging_ || route_staging_->size() < bytes) {
        route_staging_ = std::make_unique<PinnedHostBuffer>(bytes);
    }
    const auto status =
        cudaMemcpyAsync(route_staging_->data(), device_ids, bytes, cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) { throw std::runtime_error(std::string("route copy: ") + cudaGetErrorString(status)); }
    const auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) { throw std::runtime_error(std::string("route sync: ") + cudaGetErrorString(sync)); }
    const auto* route_ids = static_cast<const std::int32_t*>(route_staging_->data());

    // Port (sm_89): every routed expert must be resident before the consumer runs, so the
    // required set is deduplicated and published in one batch.
    for (std::size_t i = 0; i < values; ++i) {
        const auto expert = route_ids[i];
        if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) {
            throw std::out_of_range("expert stream received a route outside its bank");
        }
    }
    // port (sm_89): a pool that cannot hold a whole layer keeps a hot set per bank. This step's
    // routes are placed first - they must run on the Device - and the remaining slots go to the
    // experts with the highest decayed score, so the next step has a chance to hit. The cache only
    // uploads the experts that actually changed, and experts that stay keep their slot.
    if (!fully) {
        if (tokens <= kHotSetUpdateTokens) {
            for (std::size_t i = 0; i < values; ++i) {
                hot_score_[static_cast<std::size_t>(route_ids[i])] += 1.0F;
            }
            if (++since_decay_ >= kHotSetDecayPeriod) {
                for (auto& score : hot_score_) { score *= 0.5F; }
                since_decay_ = 0;
            }
        }
        std::vector<std::uint32_t> wanted;
        std::vector<bool> chosen(count_, false);
        wanted.reserve(capacity);
        for (std::size_t i = 0; i < values && wanted.size() < capacity; ++i) {
            const auto expert = static_cast<std::uint32_t>(route_ids[i]);
            if (chosen[expert]) { continue; }
            chosen[expert] = true;
            wanted.push_back(expert);
        }
        if (wanted.size() < capacity) {
            std::vector<std::uint32_t> ranked(count_);
            std::iota(ranked.begin(), ranked.end(), 0U);
            std::stable_sort(ranked.begin(), ranked.end(),
                             [&](std::uint32_t a, std::uint32_t b) {
                                 return hot_score_[a] > hot_score_[b];
                             });
            for (const auto expert : ranked) {
                if (wanted.size() >= capacity) { break; }
                if (chosen[expert]) { continue; }
                chosen[expert] = true;
                wanted.push_back(expert);
            }
        }
        if (cache_->set_resident_experts(bank_, wanted)) {
            std::fill(offsets_.begin(), offsets_.end(), 0);
            std::vector<std::int32_t> mask(count_, 0);
            for (std::uint32_t expert = 0; expert < count_; ++expert) {
                const auto slot = cache_->expert_slot(bank_, expert);
                if (!slot) { continue; }
                offsets_[expert] = static_cast<std::uint64_t>(*slot) * cache_->slot_bytes();
                mask[expert]     = 1;
            }
            mask_host_ = mask;
            const auto table_bytes = offsets_.size() * sizeof(std::uint64_t);
            // Async on the consumer stream: the kernels that read these tables are queued behind
            // them there, while a blocking copy would stall the host twice per layer.
            (void)cudaMemcpyAsync(offsets_device_->p, offsets_.data(), table_bytes,
                                  cudaMemcpyHostToDevice, stream);
            (void)cudaMemcpyAsync(mask_device_->p, mask.data(), mask.size() * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice, stream);
        }
    }
    required_.clear();
    for (std::size_t i = 0; i < values; ++i) {
        const auto expert = route_ids[i];
        required_.push_back(keys_[static_cast<std::size_t>(expert)]);
    }
    std::sort(required_.begin(), required_.end(), [](const ExpertCacheKey& a, const ExpertCacheKey& b) {
        return a.expert < b.expert;
    });
    required_.erase(std::unique(required_.begin(), required_.end(),
                                [](const ExpertCacheKey& a, const ExpertCacheKey& b) {
                                    return a.expert == b.expert;
                                }),
                    required_.end());
    const auto upload_begin = std::chrono::steady_clock::now();
    cache_->prepare(required_, stream);
    try {
        cache_->wait_for_progress(stream);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("pool: ") + error.what());
    }
    if (std::getenv("NINFER_EXPERT_TIMING")) {
        static std::uint64_t calls = 0;
        static double seconds = 0.0;
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - upload_begin).count();
        if (++calls % 40 == 0) {
            std::fprintf(stderr, "[experts] calls=%llu upload=%.3fs avg=%.2fms hits=%llu\n",
                         (unsigned long long)calls, seconds, seconds * 1000.0 / 40.0,
                         (unsigned long long)cache_->hits());
            seconds = 0.0;
        }
    }
    // port (sm_89): the bank's window can move between layers, so the plane bases are
    // rebased onto the window that is resident now. A fixed base would address window 0.
    rebase_window();
    // The uploads were queued on the consumer stream, so the kernels that read them are already
    // ordered after the copies; only the artifact path had to wait, and it did so inside
    // progress(). A device-wide synchronise here used to cost most of the layer's time.
    return true;
}

} // namespace ninfer::models::qwen3_5::execution

