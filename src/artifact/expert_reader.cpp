#include "artifact/expert_reader.h"

#include "artifact/file_io.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace ninfer::artifact {

ArtifactExpertReader::ArtifactExpertReader(std::filesystem::path path)
    : reader_(std::move(path)), files_(reader_.directory().files.size()) {}

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
        file->read_exact(
            chunk.file_offset,
            destination.subspan(static_cast<std::size_t>(chunk.image_offset),
                                static_cast<std::size_t>(chunk.bytes)));
    }
}

} // namespace ninfer::artifact
