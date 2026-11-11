#pragma once

#include "artifact/file_io.h"
#include "artifact/reader.h"
#include "core/expert_file.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::artifact {

// port (sm_89): where one object's bytes live in the artifact file set.
//
// The expert tier's in-place addressing adds object-relative offsets to one base, so it needs the
// mapping the object starts in and the object's offset inside that mapping. An object that crosses
// a part boundary has no single base - its remaining bytes are in another file's mapping - so
// `base` is null then and the caller keeps the copying path.
struct ObjectMapping {
    const std::byte* base = nullptr;
    std::uint64_t offset  = 0;
    std::uint64_t bytes   = 0;
};

// A mapping ask for a source typed as the interface: only the artifact reader can answer it, so a
// retained Host replica or a test double gets nullopt and keeps the copying path.
[[nodiscard]] std::optional<ObjectMapping> object_mapping_of(const ExpertBlobReader& reader,
                                                             std::uint32_t object_index);

// port (sm_89): runtime file source for the expert cache. The load-time Reader is gone by the
// time a layer streams, so this owner keeps its own cold directory plus one positional read
// handle per file. Reads are buffered (the OS page cache is the second tier) and complete before
// `read_entry` returns, which is the contract the cache uploader expects.
//
// An `ExpertFileChunk` names an OBJECT and a byte offset inside that object - the same convention
// the retained Host replica uses (host_expert_reader.h resolves the same index through
// model_.host_bytes(ObjectHandle)). This reader resolves the object through the directory's logical
// payload, so a chunk in a continuation part reads that part, and an object that crosses a part
// boundary is read from both. An object index is never a file index: the artifact's file table, not
// the caller, decides which file holds a byte.
class ArtifactExpertReader final : public ExpertBlobReader {
public:
    explicit ArtifactExpertReader(std::filesystem::path path);

    void read_entry(const ExpertFileEntry& entry, std::span<std::byte> destination) const override;
    // port (sm_89): the host tier reads the same expert again and again across steps, so it goes
    // through the buffered handle and lets the OS keep the artifact hot.
    void read_entry_cached(const ExpertFileEntry& entry,
                           std::span<std::byte> destination) const override;

    // port (sm_89): map the object on demand and hand back its base, so a bank that occupies one
    // object can be read where it already is. Without this the host tier copies every expert it
    // needs into its arena, which measured 20 ms per layer against 0.5 ms of CPU work.
    [[nodiscard]] const std::byte* object_base(std::uint32_t object_index) const override;

    // port (sm_89): the same answer with the file offset that base belongs to, so a bank whose
    // experts are separate objects can still be addressed in place when they share one file.
    [[nodiscard]] std::optional<ObjectMapping> object_mapping(std::uint32_t object_index) const;

    // port (sm_89): name the payload slices so the uploader can DMA straight out of the mapped
    // artifact instead of copying each expert image into the staging buffer first. Against the e2
    // per-expert layout a layer that misses 2.8 experts pays 0.5 ms of CPU memcpy it does not need;
    // the bytes are already in the page cache, so the copy only exists because the reader could not
    // name them. Only a chunk that lies inside one file can be named; the caller falls back to
    // read_entry otherwise.
    bool spans(const ExpertFileEntry& entry, std::vector<ExpertBlobSpan>& out) const override;

    // port (sm_89): Q60. One byte range of one object, so the uploader can fetch the range that
    // covers a layer's experts with a single read instead of one read_entry per expert. Both file
    // modes answer it: the direct path when offset, length and destination are aligned, the
    // buffered one otherwise - the same rule read_entry uses per chunk, which is why a coalesced
    // read and a per-entry read return identical bytes.
    bool read_range(std::uint32_t object_index, std::uint64_t offset,
                    std::span<std::byte> destination) const override;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

private:
    // The byte range [offset, offset+bytes) of `object_index`, split the way the artifact's file
    // table splits it: one segment per file, entry or continuation part.
    [[nodiscard]] std::vector<ReadSegment> resolve(std::uint32_t object_index, std::uint64_t offset,
                                                   std::uint64_t bytes) const;
    // The handle for an artifact FILE index, opened on first use.
    [[nodiscard]] InputFile& file(std::size_t file_index) const;

    Reader reader_;
    mutable std::vector<std::unique_ptr<InputFile>> files_;
    // port (sm_89): chosen once from the volume; see expert_read_mode.
    ReadMode mode_ = ReadMode::Buffered;
};

} // namespace ninfer::artifact
