#include "artifact/expert_reader.h"

#include "artifact/file_io.h"
#include "artifact/framing.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace ninfer::artifact {

ArtifactExpertReader::ArtifactExpertReader(std::filesystem::path path)
    : reader_(std::move(path)), files_(reader_.directory().files.size()),
      mode_(expert_read_mode(reader_.path(), reader_.file_bytes())) {}

void ArtifactExpertReader::read_entry(const ExpertFileEntry& entry,
                                      std::span<std::byte> destination) const {
    if (std::getenv("NINFER_HOST_DEBUG")) {
        std::fprintf(stderr, "[reader] image=%llu dest=%zu at=%p chunks=%zu first_file=%u\n",
                     static_cast<unsigned long long>(entry.image_bytes), destination.size(),
                     static_cast<const void*>(destination.data()), entry.chunks.size(),
                     entry.chunks.empty() ? 0u : entry.chunks.front().file_index);
    }
    if (destination.size() < entry.image_bytes) {
        throw std::invalid_argument("expert entry image exceeds its destination");
    }
    for (const auto& chunk : entry.chunks) {
        if (chunk.file_index >= files_.size()) {
            throw std::invalid_argument("expert chunk names an absent artifact file");
        }
        if (chunk.image_offset > entry.image_bytes ||
            chunk.bytes > entry.image_bytes - chunk.image_offset) {
            throw std::invalid_argument("expert chunk lies outside its entry image");
        }
        auto& file = files_[chunk.file_index];
        if (!file) { file = reader_.open_file(chunk.file_index); }
        auto span = destination.subspan(static_cast<std::size_t>(chunk.image_offset),
                                        static_cast<std::size_t>(chunk.bytes));
        // Direct reads need aligned offsets, lengths and buffers; artifact chunks are page sized in
        // practice, and anything that is not falls back to the buffered path for that chunk.
        const bool aligned =
            mode_ == ReadMode::Direct && chunk.file_offset % kPayloadAlignment == 0 &&
            span.size() % kPayloadAlignment == 0 &&
            reinterpret_cast<std::uintptr_t>(span.data()) % kPayloadAlignment == 0;
        if (aligned) {
            (void)file->read_direct(chunk.file_offset, span);
        } else {
            file->read_exact(chunk.file_offset, span);
        }
    }
}

void ArtifactExpertReader::read_entry_cached(const ExpertFileEntry& entry,
                                            std::span<std::byte> destination) const {
    if (destination.size() < entry.image_bytes) {
        throw std::invalid_argument("expert entry image exceeds its destination");
    }
    for (const auto& chunk : entry.chunks) {
        if (chunk.file_index >= files_.size()) {
            throw std::invalid_argument("expert chunk names an absent artifact file");
        }
        if (chunk.image_offset > entry.image_bytes ||
            chunk.bytes > entry.image_bytes - chunk.image_offset) {
            throw std::invalid_argument("expert chunk lies outside its entry image");
        }
        auto& file = files_[chunk.file_index];
        if (!file) { file = reader_.open_file(chunk.file_index); }
        // read_exact always goes through the buffered handle, whatever NINFER_EXPERT_IO selected
        // for the streaming tier. A hit is a memcpy out of the standby list.
        file->read_exact(chunk.file_offset,
                         destination.subspan(static_cast<std::size_t>(chunk.image_offset),
                                             static_cast<std::size_t>(chunk.bytes)));
    }
}

const std::byte* ArtifactExpertReader::object_base(std::uint32_t object_index) const {
    if (object_index >= files_.size()) { return nullptr; }
    auto& file = files_[object_index];
    if (!file) {
        try {
            file = reader_.open_file(object_index);
        } catch (const std::exception&) {
            return nullptr;
        }
    }
    return file ? file->map_view() : nullptr;
}

bool ArtifactExpertReader::spans(const ExpertFileEntry& entry,
                                 std::vector<ExpertBlobSpan>& out) const {
    out.clear();
    out.reserve(entry.chunks.size());
    // Only name slices that are already mapped. A file that cannot be mapped here falls back to
    // read_entry, which is the path this replaces rather than the one it depends on.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_EXPERT_SPANS");
        return value == nullptr || *value == '\0' || std::strtol(value, nullptr, 10) != 0;
    }();
    if (!enabled) { return false; }
    for (const auto& chunk : entry.chunks) {
        if (chunk.file_index >= files_.size()) { return false; }
        auto& file = files_[chunk.file_index];
        if (!file) {
            try {
                file = reader_.open_file(chunk.file_index);
            } catch (const std::exception&) {
                return false;
            }
        }
        if (!file) { return false; }
        const auto* base = file->map_view();
        if (base == nullptr) { return false; }
        if (chunk.file_offset > file->bytes() || chunk.bytes > file->bytes() - chunk.file_offset) {
            return false;
        }
        const auto* source = base + chunk.file_offset;
        if (!out.empty()) {
            auto& last = out.back();
            if (static_cast<const std::byte*>(last.source) + last.bytes == source &&
                last.image_offset + last.bytes == chunk.image_offset) {
                last.bytes += chunk.bytes;
                continue;
            }
        }
        out.push_back({source, chunk.bytes, chunk.image_offset});
    }
    return !out.empty();
}

} // namespace ninfer::artifact
