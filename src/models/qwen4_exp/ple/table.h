#pragma once

#include "artifact/backing.h"
#include "core/nf4.h"

#include <array>

namespace ninfer::models::qwen4_exp {

inline constexpr std::uint32_t kPleHeads             = 16;
inline constexpr std::uint32_t kPleRowElements       = 160;
inline constexpr std::uint32_t kPleEmbeddingElements = kPleHeads * kPleRowElements;
inline constexpr std::size_t kPleRowBytes            = kPleRowElements / 2 + kPleRowElements / 16;
inline constexpr std::size_t kPleEmbeddingBytes = kPleEmbeddingElements * sizeof(std::uint16_t);

struct PleHashControls {
    std::array<std::int64_t, 3> multipliers;
    std::array<std::int64_t, kPleHeads> vocab_sizes;
    std::array<std::int64_t, kPleHeads> offsets;
};

struct PleRowSource {
    artifact::FileObjectView object;
    std::uint64_t offset; // Encoded row's byte offset within object.
    const Nf4Bf16Lut* lut;
};

// Immutable disk table in shard order. Reads only framing, divisors and controls at startup.
// A row is 90 encoded bytes / 160 BF16 values; global row IDs span all shards.
// Returned views survive owner moves and borrow it through their last I/O/CPU consumer.
class PleTableSource {
public:
    PleTableSource(const artifact::Reader&, std::span<const artifact::ObjectHandle> shards,
                   artifact::ObjectHandle controls);
    ~PleTableSource();
    PleTableSource(PleTableSource&&) noexcept;
    PleTableSource& operator=(PleTableSource&&) noexcept;
    PleTableSource(const PleTableSource&)            = delete;
    PleTableSource& operator=(const PleTableSource&) = delete;

    [[nodiscard]] PleRowSource row(std::uint32_t global_row) const;
    [[nodiscard]] std::uint64_t rows() const noexcept;
    [[nodiscard]] const PleHashControls& controls() const noexcept;
    [[nodiscard]] std::size_t max_row_segments() const noexcept;
    // port (sm_89): the backing owns one InputFile per container file; reads go through the
    // Windows NO_BUFFERING layer in artifact/file_io.cpp.
    [[nodiscard]] std::size_t file_count() const noexcept;
    [[nodiscard]] const artifact::InputFile& file(std::size_t index) const;
    [[nodiscard]] const artifact::FileBacking& backing() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
