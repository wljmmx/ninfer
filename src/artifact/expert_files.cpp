#include "artifact/expert_files.h"

#include "core/weight_bank.h"

#include <stdexcept>

namespace ninfer::artifact {
namespace {

void append_plane(ExpertFileEntry& entry, const Reader& reader, ObjectHandle object,
                  std::uint64_t plane_offset, std::uint64_t plane_bytes,
                  std::uint64_t image_offset) {
    if (!plane_bytes) { return; }
    const auto& descriptor = reader.directory().tensor(object);
    const auto logical     = descriptor.offset + plane_offset;
    for (const auto& segment : reader.segments(logical, plane_bytes)) {
        ExpertFileChunk chunk;
        chunk.file_index   = static_cast<std::uint32_t>(segment.file_index);
        chunk.file_offset  = segment.file_offset;
        chunk.bytes        = segment.bytes;
        chunk.image_offset = image_offset + segment.destination_offset;
        entry.chunks.push_back(chunk);
    }
}

} // namespace

ExpertFileBank describe_expert_files(const Reader& reader,
                                     std::span<const ObjectHandle> gate_up_objects,
                                     std::span<const ObjectHandle> down_objects,
                                     const WeightGeometry& gate_up, const WeightGeometry& down,
                                     const ExpertBankLayout& layout) {
    if (gate_up_objects.size() != down_objects.size() || gate_up_objects.size() != layout.count) {
        throw std::invalid_argument("expert file plan disagrees on the expert count");
    }
    ExpertFileBank bank;
    bank.experts.reserve(gate_up_objects.size());
    for (std::size_t expert = 0; expert < gate_up_objects.size(); ++expert) {
        ExpertFileEntry entry;
        entry.image_bytes = layout.entry_bytes;
        append_plane(entry, reader, gate_up_objects[expert], 0, gate_up.code_bytes,
                     layout.codes.offset);
        append_plane(entry, reader, gate_up_objects[expert], gate_up.high_offset,
                     gate_up.high_bytes, layout.high.offset);
        append_plane(entry, reader, gate_up_objects[expert], gate_up.scale_offset,
                     gate_up.scale_bytes, layout.scales.offset);
        append_plane(entry, reader, down_objects[expert], 0, down.code_bytes,
                     layout.down_codes.offset);
        append_plane(entry, reader, down_objects[expert], down.high_offset, down.high_bytes,
                     layout.down_high.offset);
        append_plane(entry, reader, down_objects[expert], down.scale_offset, down.scale_bytes,
                     layout.down_scales.offset);
        bank.experts.push_back(std::move(entry));
    }
    return bank;
}

ExpertFileBank describe_fused_expert_files(const Reader& reader, ObjectHandle gate_up_object,
                                           ObjectHandle down_object,
                                           const WeightGeometry& gate_up_parent,
                                           const WeightGeometry& down_parent, std::uint32_t count,
                                           std::uint32_t gate_up_rows, std::uint32_t down_rows,
                                           const ExpertBankLayout& layout) {
    if (!count || count != layout.count || !gate_up_rows || !down_rows) {
        throw std::invalid_argument("fused expert plan disagrees on the expert count");
    }
    if (gate_up_parent.shape.empty() || down_parent.shape.empty()) {
        throw std::invalid_argument("fused expert parent has no column axis");
    }
    ExpertFileBank bank;
    bank.experts.reserve(count);
    for (std::uint32_t expert = 0; expert < count; ++expert) {
        // Per-expert plane origins come from the same affine description the resident bank uses,
        // so the cache image and the CUDA bank cannot disagree about where a plane begins.
        const auto gu = describe_weight_bank(
            gate_up_parent, {static_cast<std::uint64_t>(expert) * gate_up_rows, 1, gate_up_rows,
                             gate_up_rows});
        const auto dn = describe_weight_bank(
            down_parent, {static_cast<std::uint64_t>(expert) * down_rows, 1, down_rows, down_rows});
        ExpertFileEntry entry;
        entry.image_bytes = layout.entry_bytes;
        append_plane(entry, reader, gate_up_object, gu.codes.offset, gu.codes.expert_bytes,
                     layout.codes.offset);
        append_plane(entry, reader, gate_up_object, gu.high.offset, gu.high.expert_bytes,
                     layout.high.offset);
        append_plane(entry, reader, gate_up_object, gu.scales.offset, gu.scales.expert_bytes,
                     layout.scales.offset);
        append_plane(entry, reader, down_object, dn.codes.offset, dn.codes.expert_bytes,
                     layout.down_codes.offset);
        append_plane(entry, reader, down_object, dn.high.offset, dn.high.expert_bytes,
                     layout.down_high.offset);
        append_plane(entry, reader, down_object, dn.scales.offset, dn.scales.expert_bytes,
                     layout.down_scales.offset);
        bank.experts.push_back(std::move(entry));
    }
    return bank;
}

ExpertFileBank describe_fused_expert_host(ObjectHandle gate_up_object,
                                          ObjectHandle down_object,
                                          const WeightGeometry& gate_up_parent,
                                          const WeightGeometry& down_parent, std::uint32_t count,
                                          std::uint32_t gate_up_rows, std::uint32_t down_rows,
                                          const ExpertBankLayout& layout) {
    if (!count || count != layout.count || !gate_up_rows || !down_rows) {
        throw std::invalid_argument("fused expert plan disagrees on the expert count");
    }
    ExpertFileBank bank;
    bank.experts.reserve(count);
    for (std::uint32_t expert = 0; expert < count; ++expert) {
        const auto gu = describe_weight_bank(
            gate_up_parent, {static_cast<std::uint64_t>(expert) * gate_up_rows, 1, gate_up_rows,
                             gate_up_rows});
        const auto dn = describe_weight_bank(
            down_parent, {static_cast<std::uint64_t>(expert) * down_rows, 1, down_rows, down_rows});
        ExpertFileEntry entry;
        entry.image_bytes = layout.entry_bytes;
        const auto chunk = [](ExpertFileEntry& target, const WeightBankPlane& plane,
                              std::uint32_t object, std::uint64_t image_offset) {
            if (!plane.expert_bytes) { return; }
            target.chunks.push_back({object, plane.offset, plane.expert_bytes, image_offset});
        };
        chunk(entry, gu.codes, gate_up_object.index, layout.codes.offset);
        chunk(entry, gu.high, gate_up_object.index, layout.high.offset);
        chunk(entry, gu.scales, gate_up_object.index, layout.scales.offset);
        chunk(entry, dn.codes, down_object.index, layout.down_codes.offset);
        chunk(entry, dn.high, down_object.index, layout.down_high.offset);
        chunk(entry, dn.scales, down_object.index, layout.down_scales.offset);
        bank.experts.push_back(std::move(entry));
    }
    return bank;
}

} // namespace ninfer::artifact
