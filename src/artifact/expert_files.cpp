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
    // A chunk names the object and the offset inside it - the same convention the retained Host
    // replica reads (host_expert_reader.h) and the one the cache's cover read and in-place
    // addressing expect. The reader resolves the object to the file that holds those bytes, so a
    // chunk that lands in a continuation part is read from that part and never from the entry. The
    // segment walk still splits a plane at the part boundary, so each chunk stays inside one file
    // and can be named where it already is.
    for (const auto& segment : reader.segments(logical, plane_bytes)) {
        ExpertFileChunk chunk;
        chunk.file_index   = static_cast<std::uint32_t>(object.index);
        chunk.file_offset  = plane_offset + segment.destination_offset;
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

ExpertFileBank describe_expert_objects(std::span<const ObjectHandle> gate_up_objects,
                                      std::span<const ObjectHandle> down_objects,
                                      const WeightGeometry& gate_up, const WeightGeometry& down,
                                      const ExpertBankLayout& layout) {
    if (gate_up_objects.size() != down_objects.size() ||
        gate_up_objects.size() != layout.count) {
        throw std::invalid_argument("expert object plan disagrees on the expert count");
    }
    const auto rows_of = [](const WeightGeometry& geometry) {
        return geometry.shape.empty() ? std::uint64_t{0} : geometry.shape.front();
    };
    const auto gate_up_rows = rows_of(gate_up);
    const auto down_rows    = rows_of(down);
    if (!gate_up_rows || !down_rows) {
        throw std::invalid_argument("expert object plan has an empty expert");
    }
    // One geometry describes the expert itself, so the plane origins inside its object come from a
    // single-expert region rather than a row range of the bank.
    const auto gu = describe_weight_bank(gate_up, {0, 1, gate_up_rows, gate_up_rows});
    const auto dn = describe_weight_bank(down, {0, 1, down_rows, down_rows});
    ExpertFileBank bank;
    bank.experts.reserve(gate_up_objects.size());
    for (std::size_t expert = 0; expert < gate_up_objects.size(); ++expert) {
        ExpertFileEntry entry;
        entry.image_bytes = layout.entry_bytes;
        const auto chunk = [](ExpertFileEntry& target, const WeightBankPlane& plane,
                              std::uint32_t object, std::uint64_t image_offset) {
            if (!plane.expert_bytes) { return; }
            target.chunks.push_back({object, plane.offset, plane.expert_bytes, image_offset});
        };
        chunk(entry, gu.codes, gate_up_objects[expert].index, layout.codes.offset);
        chunk(entry, gu.high, gate_up_objects[expert].index, layout.high.offset);
        chunk(entry, gu.scales, gate_up_objects[expert].index, layout.scales.offset);
        chunk(entry, dn.codes, down_objects[expert].index, layout.down_codes.offset);
        chunk(entry, dn.high, down_objects[expert].index, layout.down_high.offset);
        chunk(entry, dn.scales, down_objects[expert].index, layout.down_scales.offset);
        bank.experts.push_back(std::move(entry));
    }
    return bank;
}

ExpertFileBank describe_matrix_expert_files(const Reader& reader,
                                           std::span<const ObjectHandle> objects,
                                           const WeightGeometry& matrix, std::uint32_t count,
                                           std::uint32_t rows, const ExpertBankLayout& layout) {
    if (!count || count != layout.count || !rows || matrix.shape.empty() ||
        objects.size() != count) {
        throw std::invalid_argument("single-matrix expert plan disagrees on the expert count");
    }
    // One object per expert: each expert's matrix is the whole object, so its planes start at the
    // matrix's own offsets. A bank-wide object keeps the experts in consecutive row blocks.
    const bool per_expert = objects.front().index != objects.back().index;
    const auto bank_geometry = [&] {
        const std::uint64_t shape[2] = {static_cast<std::uint64_t>(count) * rows,
                                        matrix.shape.back()};
        return weight_geometry(matrix.format, matrix.layout, shape);
    }();
    ExpertFileBank bank;
    bank.experts.reserve(count);
    for (std::uint32_t expert = 0; expert < count; ++expert) {
        // The same affine description the operator's bank view uses, so the entry image and the
        // kernel cannot disagree about where a plane begins.
        const auto slice = per_expert
                               ? describe_weight_bank(matrix, {0, 1, rows, rows})
                               : describe_weight_bank(
                                     bank_geometry,
                                     {static_cast<std::uint64_t>(expert) * rows, 1, rows, rows});
        ExpertFileEntry entry;
        entry.image_bytes = layout.entry_bytes;
        const auto object_index = objects[per_expert ? expert : 0].index;
        append_plane(entry, reader, ObjectHandle{object_index}, slice.codes.offset,
                     slice.codes.expert_bytes, layout.codes.offset);
        append_plane(entry, reader, ObjectHandle{object_index}, slice.high.offset,
                     slice.high.expert_bytes, layout.high.offset);
        append_plane(entry, reader, ObjectHandle{object_index}, slice.scales.offset,
                     slice.scales.expert_bytes, layout.scales.offset);
        bank.experts.push_back(std::move(entry));
    }
    return bank;
}

ExpertFileBank describe_matrix_expert_objects(std::span<const ObjectHandle> objects,
                                             const WeightGeometry& matrix, std::uint32_t count,
                                             std::uint32_t rows, const ExpertBankLayout& layout) {
    if (!count || count != layout.count || !rows || matrix.shape.empty() ||
        objects.size() != count) {
        throw std::invalid_argument("single-matrix expert plan disagrees on the expert count");
    }
    // Same split as the reader-backed form: one object per expert carries the whole matrix, a
    // bank-wide object keeps the experts in consecutive row blocks.
    const bool per_expert = objects.front().index != objects.back().index;
    const auto bank_geometry = [&] {
        const std::uint64_t shape[2] = {static_cast<std::uint64_t>(count) * rows,
                                        matrix.shape.back()};
        return weight_geometry(matrix.format, matrix.layout, shape);
    }();
    ExpertFileBank bank;
    bank.experts.reserve(count);
    const auto chunk = [](ExpertFileEntry& target, const WeightBankPlane& plane,
                          std::uint32_t object, std::uint64_t image_offset) {
        if (!plane.expert_bytes) { return; }
        target.chunks.push_back({object, plane.offset, plane.expert_bytes, image_offset});
    };
    for (std::uint32_t expert = 0; expert < count; ++expert) {
        const auto slice = per_expert
                               ? describe_weight_bank(matrix, {0, 1, rows, rows})
                               : describe_weight_bank(
                                     bank_geometry,
                                     {static_cast<std::uint64_t>(expert) * rows, 1, rows, rows});
        ExpertFileEntry entry;
        entry.image_bytes = layout.entry_bytes;
        const auto object_index = objects[per_expert ? expert : 0].index;
        chunk(entry, slice.codes, object_index, layout.codes.offset);
        chunk(entry, slice.high, object_index, layout.high.offset);
        chunk(entry, slice.scales, object_index, layout.scales.offset);
        bank.experts.push_back(std::move(entry));
    }
    return bank;
}

} // namespace ninfer::artifact
