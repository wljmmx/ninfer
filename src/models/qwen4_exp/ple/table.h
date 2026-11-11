#pragma once

#include "artifact/backing.h"
#include "core/nf4.h"

#include <array>

namespace ninfer::models::qwen4_exp {

inline constexpr std::uint32_t kPleHeads             = 16;
inline constexpr std::uint32_t kPleRowElements       = 160;
inline constexpr std::uint32_t kPleEmbeddingElements = kPleHeads * kPleRowElements;
// One encoded NF4/paged row: 80 code bytes then ten E4M3 scale bytes.
inline constexpr std::size_t kPleRowBytes            = kPleRowElements / 2 + kPleRowElements / 16;
// One encoded q4_g64 row_split row: 128 base-code bytes (4 groups of 32) then 8 scale bytes. The
// planes live apart, so a row is two object-relative spans, not one range.
inline constexpr std::size_t kPleQ4RowBytes    = 128 + 8;
inline constexpr std::size_t kPleMaxRowBytes   = kPleQ4RowBytes;
inline constexpr std::uint32_t kPleMaxRowParts = 2;
inline constexpr std::size_t kPleEmbeddingBytes = kPleEmbeddingElements * sizeof(std::uint16_t);

// How a table's rows are encoded on disk. The reader serves exactly the entries below; anything
// else is refused by name at load time (require_supported_ple_table).
enum class PleRowEncoding : std::uint8_t {
    // NF4 codebook with one E4M3 scale per 16-wide group, rows paged 45-per-4096 bytes.
    Nf4PagedG16,
    // q4_g64_fp16 in row_split_k128_v1: signed nibbles against one binary16 scale per 64 lanes.
    Q4G64RowSplit,
};

// The encodings `PleTableSource` serves, so the loader's refusal and the reader cannot drift apart.
[[nodiscard]] constexpr bool supported_ple_table(QType format, QuantLayout layout,
                                                 std::uint64_t width) noexcept {
    if (width != kPleRowElements) return false;
    return (format == QType::NF4_G16_E4M3FN && layout == QuantLayout::RowPagedG16_4K) ||
           (format == QType::Q4_G64_FP16 && layout == QuantLayout::RowSplit);
}

struct PleHashControls {
    std::array<std::int64_t, 3> multipliers;
    std::array<std::int64_t, kPleHeads> vocab_sizes;
    std::array<std::int64_t, kPleHeads> offsets;
};

// One encoded row's source: the object it lives in, the encoding, and the object-relative spans
// the parts of the row occupy (a paged NF4 row is one span; a row_split q4 row is a code-plane
// span followed by a scale-plane span). `lut` is the NF4 divisor table and is null for q4.
struct PleRowSource {
    artifact::FileObjectView object;
    PleRowEncoding encoding;
    const Nf4Bf16Lut* lut = nullptr;
    std::uint32_t parts   = 0;
    std::array<std::uint64_t, kPleMaxRowParts> part_offsets{};
    std::array<std::uint32_t, kPleMaxRowParts> part_bytes{};

    [[nodiscard]] std::size_t encoded_bytes() const noexcept {
        std::size_t total = 0;
        for (std::uint32_t i = 0; i < parts; ++i) total += part_bytes[i];
        return total;
    }
};

// Decodes the concatenated parts of one row into `output`'s BF16 words. Both encodings produce
// exactly `kPleRowElements` words; the seam is the encoded row, never a wider type.
void decode_ple_row(const PleRowSource& source, std::span<const std::byte> encoded,
                    std::span<std::uint16_t> output);

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
    [[nodiscard]] PleRowEncoding encoding() const noexcept;
    // Bytes the encoding spends on one row, and how many object spans a row occupies.
    [[nodiscard]] std::size_t row_encoded_bytes() const noexcept;
    [[nodiscard]] std::uint32_t row_parts() const noexcept;
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
