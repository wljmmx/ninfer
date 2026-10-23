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
      mode_(expert_read_mode(reader_.path())) {}

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

} // namespace ninfer::artifact
