#include "artifact/expert_reader.h"

#include "artifact/file_io.h"
#include "artifact/framing.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace ninfer::artifact {
namespace {

// Direct reads need aligned offsets, lengths and buffers; artifact chunks are page sized in
// practice, and anything that is not falls back to the buffered path for that segment.
[[nodiscard]] bool direct_aligned(ReadMode mode, std::uint64_t file_offset, std::size_t bytes,
                                  const std::byte* destination) {
    return mode == ReadMode::Direct && file_offset % kPayloadAlignment == 0 &&
           bytes % kPayloadAlignment == 0 &&
           reinterpret_cast<std::uintptr_t>(destination) % kPayloadAlignment == 0;
}

void read_segment(InputFile& file, ReadMode mode, const ReadSegment& segment,
                  std::span<std::byte> target) {
    if (direct_aligned(mode, segment.file_offset, target.size(), target.data())) {
        (void)file.read_direct(segment.file_offset, target);
    } else {
        file.read_exact(segment.file_offset, target);
    }
}

} // namespace

ArtifactExpertReader::ArtifactExpertReader(std::filesystem::path path)
    : reader_(std::move(path)), files_(reader_.directory().files.size()),
      mode_(expert_read_mode(reader_.path(), reader_.file_bytes())) {}

InputFile& ArtifactExpertReader::file(std::size_t file_index) const {
    if (file_index >= files_.size()) {
        throw std::invalid_argument("expert read names an absent artifact file");
    }
    auto& handle = files_[file_index];
    if (!handle) { handle = reader_.open_part(file_index); }
    return *handle;
}

std::vector<ReadSegment> ArtifactExpertReader::resolve(std::uint32_t object_index,
                                                       std::uint64_t offset,
                                                       std::uint64_t bytes) const {
    const auto& directory = reader_.directory();
    if (object_index >= directory.objects.size()) {
        throw std::invalid_argument("expert chunk names an absent artifact object");
    }
    const auto& object     = directory.objects[object_index];
    const auto object_size = object_bytes(object);
    if (offset > object_size || bytes > object_size - offset) {
        throw std::invalid_argument("expert chunk lies outside its object");
    }
    return reader_.segments(checked_add(object_offset(object), offset, "expert chunk"), bytes);
}

void ArtifactExpertReader::read_entry(const ExpertFileEntry& entry,
                                      std::span<std::byte> destination) const {
    if (std::getenv("NINFER_HOST_DEBUG")) {
        std::fprintf(stderr, "[reader] image=%llu dest=%zu at=%p chunks=%zu first_object=%u\n",
                     static_cast<unsigned long long>(entry.image_bytes), destination.size(),
                     static_cast<const void*>(destination.data()), entry.chunks.size(),
                     entry.chunks.empty() ? 0u : entry.chunks.front().file_index);
    }
    if (destination.size() < entry.image_bytes) {
        throw std::invalid_argument("expert entry image exceeds its destination");
    }
    for (const auto& chunk : entry.chunks) {
        if (chunk.image_offset > entry.image_bytes ||
            chunk.bytes > entry.image_bytes - chunk.image_offset) {
            throw std::invalid_argument("expert chunk lies outside its entry image");
        }
        // Each chunk names its own object, and the object names the files it spans: a bank whose
        // experts were written into different parts resolves per chunk, never per bank.
        for (const auto& segment : resolve(chunk.file_index, chunk.file_offset, chunk.bytes)) {
            auto& input = file(segment.file_index);
            read_segment(
                input, mode_, segment,
                destination.subspan(static_cast<std::size_t>(chunk.image_offset) +
                                        static_cast<std::size_t>(segment.destination_offset),
                                    static_cast<std::size_t>(segment.bytes)));
        }
    }
}

bool ArtifactExpertReader::read_range(std::uint32_t object_index, std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    if (destination.empty()) { return false; }
    try {
        // A coalesced range is a whole run of chunks, so it may cross a part boundary even when no
        // single chunk does. Resolving it through the object's logical payload reads each piece
        // from the file that holds it.
        for (const auto& segment :
             resolve(object_index, offset, static_cast<std::uint64_t>(destination.size()))) {
            auto& input = file(segment.file_index);
            read_segment(input, mode_, segment,
                         destination.subspan(static_cast<std::size_t>(segment.destination_offset),
                                             static_cast<std::size_t>(segment.bytes)));
        }
    } catch (...) {
        // A source that cannot answer this range leaves the caller on read_entry; saying so is the
        // whole contract, and throwing would turn a fallback into a failure.
        return false;
    }
    return true;
}

void ArtifactExpertReader::read_entry_cached(const ExpertFileEntry& entry,
                                             std::span<std::byte> destination) const {
    if (destination.size() < entry.image_bytes) {
        throw std::invalid_argument("expert entry image exceeds its destination");
    }
    for (const auto& chunk : entry.chunks) {
        if (chunk.image_offset > entry.image_bytes ||
            chunk.bytes > entry.image_bytes - chunk.image_offset) {
            throw std::invalid_argument("expert chunk lies outside its entry image");
        }
        // read_exact always goes through the buffered handle, whatever NINFER_EXPERT_IO selected
        // for the streaming tier. A hit is a memcpy out of the standby list.
        for (const auto& segment : resolve(chunk.file_index, chunk.file_offset, chunk.bytes)) {
            file(segment.file_index)
                .read_exact(
                    segment.file_offset,
                    destination.subspan(static_cast<std::size_t>(chunk.image_offset) +
                                            static_cast<std::size_t>(segment.destination_offset),
                                        static_cast<std::size_t>(segment.bytes)));
        }
    }
}

std::optional<ObjectMapping>
ArtifactExpertReader::object_mapping(std::uint32_t object_index) const {
    try {
        const auto& directory = reader_.directory();
        if (object_index >= directory.objects.size()) { return std::nullopt; }
        const auto bytes = object_bytes(directory.objects[object_index]);
        if (!bytes) { return std::nullopt; }
        const auto segments = resolve(object_index, 0, bytes);
        // One segment means one file holds the whole object, which is the only case in which one
        // base can name it.
        if (segments.size() != 1) { return std::nullopt; }
        auto& input      = file(segments.front().file_index);
        const auto* base = input.map_view();
        if (base == nullptr) { return std::nullopt; }
        if (segments.front().file_offset > input.bytes() ||
            bytes > input.bytes() - segments.front().file_offset) {
            return std::nullopt;
        }
        return ObjectMapping{base, segments.front().file_offset, bytes};
    } catch (const std::exception&) { return std::nullopt; }
}

const std::byte* ArtifactExpertReader::object_base(std::uint32_t object_index) const {
    const auto mapping = object_mapping(object_index);
    if (!mapping) { return nullptr; }
    return mapping->base + mapping->offset;
}

std::optional<ObjectMapping> object_mapping_of(const ExpertBlobReader& reader,
                                               std::uint32_t object_index) {
    const auto* artifact_reader = dynamic_cast<const ArtifactExpertReader*>(&reader);
    if (artifact_reader == nullptr) { return std::nullopt; }
    return artifact_reader->object_mapping(object_index);
}

bool ArtifactExpertReader::spans(const ExpertFileEntry& entry,
                                 std::vector<ExpertBlobSpan>& out) const {
    out.clear();
    // port (sm_89): Q60's measurement switch. On this machine every artifact object maps, so
    // spans() always succeeds and the uploader never takes the staged path - which is why the
    // coalesced read measured zero: it had no consumer, not a defect. This makes the reader answer
    // "I cannot name them" so that path runs, and the coalesced form can be compared against the
    // per-expert one.
    static const bool no_spans = std::getenv("NINFER_NO_SPANS") != nullptr;
    if (no_spans) { return false; }
    out.reserve(entry.chunks.size());
    // Only name slices that are already mapped and lie inside one file. Anything else falls back to
    // read_entry, which is the path this replaces rather than the one it depends on.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_EXPERT_SPANS");
        return value == nullptr || *value == '\0' || std::strtol(value, nullptr, 10) != 0;
    }();
    if (!enabled) { return false; }
    for (const auto& chunk : entry.chunks) {
        std::vector<ReadSegment> segments;
        try {
            segments = resolve(chunk.file_index, chunk.file_offset, chunk.bytes);
        } catch (const std::exception&) { return false; }
        if (segments.size() != 1) { return false; }
        auto& input      = file(segments.front().file_index);
        const auto* base = input.map_view();
        if (base == nullptr) { return false; }
        const auto& segment = segments.front();
        if (segment.file_offset > input.bytes() ||
            segment.bytes > input.bytes() - segment.file_offset) {
            return false;
        }
        const auto* source      = base + segment.file_offset;
        const auto image_offset = chunk.image_offset + segment.destination_offset;
        if (!out.empty()) {
            auto& last = out.back();
            if (static_cast<const std::byte*>(last.source) + last.bytes == source &&
                last.image_offset + last.bytes == image_offset) {
                last.bytes += segment.bytes;
                continue;
            }
        }
        out.push_back({source, segment.bytes, image_offset});
    }
    return !out.empty();
}

} // namespace ninfer::artifact
