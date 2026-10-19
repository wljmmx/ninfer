#pragma once

#include "artifact/schema.h"
#include "core/buffer.h"
#include "core/layout.h"

#include <optional>
#include <span>
#include <vector>

namespace ninfer::artifact {

class Reader;

struct PreparedResource {
    ObjectHandle object;
    std::vector<std::byte> data;
};

// Moving this owner preserves the addresses borrowed by parsed frontend resources.
struct PreparedResources {
    PreparedResources()                                        = default;
    PreparedResources(PreparedResources&&) noexcept            = default;
    PreparedResources& operator=(PreparedResources&&) noexcept = default;
    PreparedResources(const PreparedResources&)                = delete;
    PreparedResources& operator=(const PreparedResources&)     = delete;

    const Reader* source = nullptr;
    std::vector<PreparedResource> objects;
    std::uint64_t read_bytes        = 0;
    std::uint64_t owned_value_bytes = 0;
};

enum class HostAccess { CpuOnly, CudaTransfer };

struct PlacementRequest {
    ObjectHandle object;
    Device device;
    HostAccess host_access = HostAccess::CpuOnly;
    // port (sm_89): leave these objects out of Host storage; the host tier reads them on demand.
    bool file_backed = false;
};

struct TensorPlacement {
    ObjectHandle object;
    LayoutRegion region;
};

struct MaterializationPlan {
    const Reader* source     = nullptr;
    std::size_t object_count = 0;
    Device device;
    std::size_t device_capacity_bytes = 0;
    std::size_t host_capacity_bytes   = 0;
    std::size_t host_alignment        = 256;
    std::size_t registered_host_bytes = 0; // Aligned prefix of the Host backing.
    std::vector<TensorPlacement> device_objects;
    // Host placements are relative to their respective registered/ordinary regions.
    std::vector<TensorPlacement> ordinary_host_objects;
    std::vector<TensorPlacement> registered_host_objects;
    PreparedResources resources;
    // port (sm_89): a Host backing already filled from a device replica. When set, materialize
    // keeps it instead of allocating and reading from the artifact.
    std::optional<HostBuffer> prefilled_host;
    // port (sm_89): device payload parked in Host during a demotion; uploaded instead of read.
    std::optional<HostBuffer> prefilled_device;
};

// Preserve the source page offset so aligned interiors can be read directly into Host backing.
[[nodiscard]] LayoutRegion plan_host_tensor_region(LayoutBuilder&, const Reader&, ObjectHandle);

// Deduplicates complete parents independently at each requested location. Tensor placement
// never reads payloads; only already-prepared resource bytes move through this plan.
[[nodiscard]] MaterializationPlan plan_materialization(const Reader& reader,
                                                       std::span<const PlacementRequest> requests,
                                                       PreparedResources resources = {});

} // namespace ninfer::artifact
