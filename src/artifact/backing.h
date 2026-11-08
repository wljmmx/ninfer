#pragma once

#include "artifact/reader.h"

namespace ninfer::artifact {

struct FileObjectView {
    std::uint64_t bytes = 0;
    // Complete object mapping: destination_offset is object-relative, file_index indexes
    // FileBacking::file(). Both the mapping and file handles borrow the backing owner.
    std::span<const ReadSegment> segments;
};

// port (sm_89): upstream keeps io::ReadOnlyFile here and reads through io_uring/preadv. This tree
// keeps the same object-with-segments view but owns artifact::InputFile handles instead, so every
// read reaches the Windows platform layer: FILE_FLAG_NO_BUFFERING with kPayloadAlignment-aligned
// offsets, lengths and staging buffers. Move-only; borrowers end before destruction.
class FileBacking {
public:
    FileBacking();
    ~FileBacking();
    FileBacking(FileBacking&&) noexcept;
    FileBacking& operator=(FileBacking&&) noexcept;
    FileBacking(const FileBacking&)            = delete;
    FileBacking& operator=(const FileBacking&) = delete;

    [[nodiscard]] FileObjectView object(ObjectHandle handle) const;
    [[nodiscard]] const WeightGeometry& geometry(ObjectHandle handle) const;
    [[nodiscard]] std::size_t file_count() const noexcept;
    [[nodiscard]] const InputFile& file(std::size_t index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend FileBacking prepare_file_backing(const Reader&, std::span<const ObjectHandle>);
};

[[nodiscard]] FileBacking prepare_file_backing(const Reader& reader,
                                               std::span<const ObjectHandle> objects);

// Allocation-free range mapping. Returned segment destinations are relative to this request.
// Use object.segments.size() as an upper bound for output capacity; zero bytes yields zero parts.
[[nodiscard]] std::size_t map_file_range(FileObjectView object, std::uint64_t offset,
                                         std::uint64_t bytes, std::span<ReadSegment> output);

} // namespace ninfer::artifact
