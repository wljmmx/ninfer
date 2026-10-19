#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// port (sm_89): where one expert image lives in the artifact file set.
//
// The image mirrors the cache entry layout, so a fetched expert uploads with one copy.
struct ExpertFileChunk {
    std::uint32_t file_index  = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t bytes       = 0;
    std::uint64_t image_offset = 0;

    friend bool operator==(const ExpertFileChunk&, const ExpertFileChunk&) = default;
};

struct ExpertFileEntry {
    std::uint64_t image_bytes = 0;
    std::vector<ExpertFileChunk> chunks;

    friend bool operator==(const ExpertFileEntry&, const ExpertFileEntry&) = default;
};

struct ExpertFileBank {
    std::vector<ExpertFileEntry> experts;
};

// Synchronous artifact reader used by the file tier. Implementations block the uploading
// host thread; they never run inside capture.
class ExpertBlobReader {
public:
    virtual ~ExpertBlobReader() = default;
    // `destination` is exactly entry.image_bytes.
    virtual void read_entry(const ExpertFileEntry& entry, std::span<std::byte> destination) const = 0;
};

} // namespace ninfer
