#pragma once

#include "core/arena.h"
#include "core/expert_cache.h"
#include "core/expert_file.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

class HostExpertResidency;

// port (sm_89): one routed-expert bank whose weights stay in the artifact and stream into a
// shared Device pool. The bank description is fixed at startup; only the per-expert offset table
// is republished as the resident set changes.
class ExpertStream {
public:
    ExpertStream(std::uint32_t count, WeightBankLayout gate_up, WeightBankLayout down,
                 WeightGeometry gate_up_parent, WeightGeometry down_parent,
                 std::uint64_t gate_up_object, std::uint64_t down_object);

    // Startup-only. `entry` is the cache's per-entry plane layout for this bank.
    void attach(ExpertCache& cache, std::uint32_t bank, const ExpertBankLayout& entry);

    // Publishes the experts the next call routes to, then republishes the offset table. Returns
    // false when the pool cannot hold one whole layer at all; the caller then owns the decision
    // (today it raises, later it can fall back to the host tier).
    bool ensure(const void* device_ids, std::int32_t top_k, std::int32_t tokens,
                cudaStream_t stream);

    [[nodiscard]] const ops::CudaSwiGLUExperts& experts() const noexcept { return experts_; }
    [[nodiscard]] std::uint32_t bank_index() const noexcept { return bank_; }
    // Device-side int32 array, one entry per expert: 1 when the pool holds it.
    [[nodiscard]] const void* resident_mask() const noexcept {
        return mask_device_ ? mask_device_->p : nullptr;
    }
    // Host copy of the same mask, for the CPU half of a split layer.
    [[nodiscard]] std::span<const std::int32_t> resident_mask_host() const noexcept {
        return mask_host_;
    }
    [[nodiscard]] bool fully_resident() const noexcept;
    [[nodiscard]] bool attached() const noexcept { return cache_ != nullptr; }
    // Host half of a split tier; null when the Device pool holds every expert.
    void attach_host(HostExpertResidency& host) noexcept { host_ = &host; }
    [[nodiscard]] HostExpertResidency* host() const noexcept { return host_; }

    [[nodiscard]] std::uint32_t count() const noexcept { return count_; }
    [[nodiscard]] std::uint64_t gate_up_object() const noexcept { return gate_up_object_; }
    [[nodiscard]] std::uint64_t down_object() const noexcept { return down_object_; }
    [[nodiscard]] const WeightGeometry& gate_up_parent() const noexcept { return gate_up_parent_; }
    [[nodiscard]] const WeightGeometry& down_parent() const noexcept { return down_parent_; }
    [[nodiscard]] const WeightBankLayout& gate_up_layout() const noexcept { return gate_up_; }
    [[nodiscard]] const WeightBankLayout& down_layout() const noexcept { return down_; }
    [[nodiscard]] const WeightGeometry& gate_up_expert_geometry() const noexcept {
        return gate_up_expert_;
    }
    [[nodiscard]] const WeightGeometry& down_expert_geometry() const noexcept {
        return down_expert_;
    }

private:
    std::uint32_t count_          = 0;
    WeightBankLayout gate_up_;
    WeightBankLayout down_;
    WeightGeometry gate_up_parent_;
    WeightGeometry down_parent_;
    WeightGeometry gate_up_expert_;
    WeightGeometry down_expert_;
    std::uint64_t gate_up_object_ = 0;
    std::uint64_t down_object_    = 0;

    ExpertCache* cache_ = nullptr;
    HostExpertResidency* host_ = nullptr;
    std::uint32_t bank_ = 0;
    ExpertBankLayout entry_{};
    // port (sm_89): the pool holds every layer at once, so this bank's window is fixed and no
    // per-step routing decision - and therefore no host round trip - is needed.
    bool pinned_ = false;
    void rebase_window();
    // port (sm_89): decayed per-expert route counts. A pool that cannot hold a whole layer keeps
    // the top entries of this instead of whatever the current step happened to route to, which
    // used to replace all eight experts on every token and make the hot set useless.
    std::vector<float> hot_score_;
    std::vector<std::uint32_t> resident_order_;
    std::uint32_t since_decay_   = 0;
    std::uint32_t since_reorder_ = 0;
    std::vector<ExpertCacheKey> keys_;
    std::vector<ExpertCacheKey> required_;
    // Per-expert byte offsets the kernel adds to each plane base. Uniform today, but the kernel
    // already addresses through this table, so a partial resident set only has to rewrite entries.
    std::vector<std::uint64_t> offsets_;
    std::unique_ptr<DeviceBuffer> offsets_device_;
    // One entry per expert: 1 when the pool holds it. A partial resident set drives the GPU
    // through ExpertSubsetKind::MaskOnes and leaves the rest to the host tier.
    std::unique_ptr<DeviceBuffer> mask_device_;
    std::vector<std::int32_t> mask_host_;
    std::unique_ptr<PinnedHostBuffer> route_staging_;
    ops::CudaSwiGLUExperts experts_{};
};

} // namespace ninfer::models::qwen3_5::execution
