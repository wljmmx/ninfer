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
    // port (sm_89): the host tier reads the same expert again and again across steps, so it goes
    // through the buffered handle and lets the OS keep the artifact hot.
    void read_entry_cached(const ExpertFileEntry& entry,
                           std::span<std::byte> destination) const override;

    // port (sm_89): map the object on demand and hand back its base, so a bank that occupies one
    // object can be read where it already is. Without this the host tier copies every expert it
    // needs into its arena, which measured 20 ms per layer against 0.5 ms of CPU work.
    [[nodiscard]] const std::byte* object_base(std::uint32_t object_index) const override;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

private:
    Reader reader_;
    mutable std::vector<std::unique_ptr<InputFile>> files_;
    // port (sm_89): chosen once from the volume; see expert_read_mode.
    ReadMode mode_ = ReadMode::Buffered;
};

} // namespace ninfer::artifact
