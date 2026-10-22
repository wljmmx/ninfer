#include "models/qwen3_5/execution/expert_stream.h"

#include <cuda_runtime.h>

#include <algorithm>
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
}

bool ExpertStream::ensure(const void* device_ids, std::int32_t top_k, std::int32_t tokens,
                          cudaStream_t stream) {
    if (cache_ == nullptr) { throw std::logic_error("expert stream is not attached"); }
    // One window holds a whole layer, so a pool that cannot hold even one window cannot serve
    // this bank. Report that instead of letting ensure_window throw from deep inside the upload.
    if (cache_->windows() == 0) { return false; }
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
    required_.clear();
    for (std::size_t i = 0; i < values; ++i) {
        const auto expert = route_ids[i];
        if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) {
            throw std::out_of_range("expert stream received a route outside its bank");
        }
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
    cache_->prepare(required_);
    try {
        cache_->wait_for_progress();
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
    {
        const auto origin = cache_->window_origin(bank_);
        const auto base   = static_cast<const std::byte*>(cache_->pool().data) + origin;
        experts_.bank.gate_up.codes  = base + entry_.codes.offset;
        experts_.bank.gate_up.high   = entry_.high.bytes ? base + entry_.high.offset : nullptr;
        experts_.bank.gate_up.scales = base + entry_.scales.offset;
        experts_.bank.down.codes  = base + entry_.down_codes.offset;
        experts_.bank.down.high   = entry_.down_high.bytes ? base + entry_.down_high.offset : nullptr;
        experts_.bank.down.scales = base + entry_.down_scales.offset;
    }
    // port (sm_89): uploads go through the legacy stream while the compute stream is
    // non-blocking; without settling here nothing orders the next kernel after the copies.
    {
        const auto settled = cudaDeviceSynchronize();
        if (settled != cudaSuccess) {
            throw std::runtime_error(std::string("upload settle: ") + cudaGetErrorString(settled));
        }
    }
    return true;
}

} // namespace ninfer::models::qwen3_5::execution
