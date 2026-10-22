#pragma once

#include "core/arena.h"
#include "core/expert_cache.h"
#include "core/expert_file.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

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
    [[nodiscard]] bool attached() const noexcept { return cache_ != nullptr; }

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
    std::uint32_t bank_ = 0;
    ExpertBankLayout entry_{};
    std::vector<ExpertCacheKey> keys_;
    std::vector<ExpertCacheKey> required_;
    // Per-expert byte offsets the kernel adds to each plane base. Uniform today, but the kernel
    // already addresses through this table, so a partial resident set only has to rewrite entries.
    std::vector<std::uint64_t> offsets_;
    std::unique_ptr<DeviceBuffer> offsets_device_;
    std::unique_ptr<PinnedHostBuffer> route_staging_;
    ops::CudaSwiGLUExperts experts_{};
};

} // namespace ninfer::models::qwen3_5::execution
