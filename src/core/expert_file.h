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

// port (sm_89): one slice of an entry image that a reader can point at in place.
struct ExpertBlobSpan {
    const void* source     = nullptr;
    std::uint64_t bytes    = 0;
    std::uint64_t image_offset = 0;
};

// Synchronous artifact reader used by the file tier. Implementations block the uploading
// host thread; they never run inside capture.
class ExpertBlobReader {
public:
    virtual ~ExpertBlobReader() = default;
    // `destination` is exactly entry.image_bytes.
    virtual void read_entry(const ExpertFileEntry& entry, std::span<std::byte> destination) const = 0;
    // port (sm_89): a source that already holds the bytes in place (the Host replica) can name
    // them instead of copying through a staging buffer.
    virtual bool spans(const ExpertFileEntry& entry, std::vector<ExpertBlobSpan>& out) const {
        (void)entry;
        (void)out;
        return false;
    }
    // port (sm_89): base of a whole object, so a bank that lives in one object can be addressed in
    // place. Null when the source cannot name it, which keeps the copying path as the fallback.
    virtual const std::byte* object_base(std::uint32_t object_index) const {
        (void)object_index;
        return nullptr;
    }
    // port (sm_89): read through the OS page cache instead of around it. The Device uploader reads
    // an expert once and drops it, so bypassing the cache is right there; a tier that reads the same
    // expert again on the next step wants the opposite, and on this machine the difference is 8.6 ms
    // against 0.2 ms for the same 1.8 MiB. Sources that have no second path keep the plain read.
    virtual void read_entry_cached(const ExpertFileEntry& entry,
                                   std::span<std::byte> destination) const {
        read_entry(entry, destination);
    }
};

// port (sm_89): a pinned RAM tier the Device uploader can read from, when one exists. The host tier
// already holds the working set in locked memory for the CPU half; an upload that reaches the same
// expert should take it from there rather than from the pageable artifact mapping, which forces the
// driver through its bounce buffer. Queried per expert because the two tiers see the same experts at
// different moments.
class ExpertArenaSource {
public:
    virtual ~ExpertArenaSource() = default;
    // Pointer to the start of one expert's image inside the arena, or null when the arena does not
    // hold it (or is not the backing for that bank). `bytes` receives the image length.
    virtual const std::byte* arena_image(std::uint32_t bank, std::uint32_t expert,
                                         std::uint64_t& bytes) const = 0;
    // port (sm_89): the copies that were handed out by arena_image() have landed.
    //
    // Handing out that pointer is the moment a slot stops being "a slot I may reuse" and becomes "bytes
    // a copy in flight is reading". Without this call the arena cannot tell the difference, and its
    // coldest-first eviction can free the slot under the copy - which reads another expert's weights
    // with no error anywhere. A tier that never runs copies simply does not override it.
    virtual void uploads_settled() {}
};

} // namespace ninfer
