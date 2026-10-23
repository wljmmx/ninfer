#pragma once

#include "artifact/file_io.h"
#include "artifact/reader.h"
#include "core/expert_file.h"

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::artifact {

// port (sm_89): runtime file source for the expert cache. The load-time Reader is gone by the
// time a layer streams, so this owner keeps its own cold directory plus one positional read
// handle per file. Reads are buffered (the OS page cache is the second tier) and complete before
// `read_entry` returns, which is the contract the cache uploader expects.
class ArtifactExpertReader final : public ExpertBlobReader {
public:
    explicit ArtifactExpertReader(std::filesystem::path path);

    void read_entry(const ExpertFileEntry& entry, std::span<std::byte> destination) const override;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

private:
    Reader reader_;
    mutable std::vector<std::unique_ptr<InputFile>> files_;
    // port (sm_89): chosen once from the volume; see expert_read_mode.
    ReadMode mode_ = ReadMode::Buffered;
};

} // namespace ninfer::artifact
