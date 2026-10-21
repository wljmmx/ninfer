#pragma once

#include "artifact/reader.h"
#include "core/expert_cache.h"
#include "core/expert_file.h"
#include "core/weight_view.h"

#include <span>
#include <vector>

namespace ninfer::artifact {

// port (sm_89): maps one expert bank's artifact objects onto cache entry images.
//
// `gate_up` and `down` are the per-expert 2D geometries (master stores one object per
// expert); `objects` are their object handles in the same expert order. The produced chunks
// are laid out exactly like `layout`'s entry image, so the cache uploads each expert with a
// single copy after one fetch.
[[nodiscard]] ExpertFileBank describe_expert_files(
    const Reader& reader, std::span<const ObjectHandle> gate_up_objects,
    std::span<const ObjectHandle> down_objects, const WeightGeometry& gate_up,
    const WeightGeometry& down, const ExpertBankLayout& layout);

// port (sm_89): the artifact this model ships stores one fused matrix per (layer, matrix) with the
// experts in consecutive row blocks, rather than one object per expert. Map expert e's six planes
// onto the cache's entry image from the row block it occupies.
[[nodiscard]] ExpertFileBank describe_fused_expert_files(
    const Reader& reader, ObjectHandle gate_up_object, ObjectHandle down_object,
    const WeightGeometry& gate_up_parent, const WeightGeometry& down_parent, std::uint32_t count,
    std::uint32_t gate_up_rows, std::uint32_t down_rows, const ExpertBankLayout& layout);

} // namespace ninfer::artifact
