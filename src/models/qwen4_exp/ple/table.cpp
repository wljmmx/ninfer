#include "models/qwen4_exp/ple/table.h"

#include "ops/linear/q4/q4_row_decode.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace {

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw artifact::ArtifactError(std::string(label) + " overflows");
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* label) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw artifact::ArtifactError(std::string(label) + " overflows");
    return a * b;
}

// RowPagedG16_4K: 4096-byte pages of rows_per_page encoded rows, then the divisor word.
std::uint64_t paged_row_offset(const WeightGeometry& geometry, std::uint64_t row) {
    const auto page = row / geometry.rows_per_page;
    return checked_add(checked_mul(page, geometry.alignment, "PLE page offset"),
                       (row % geometry.rows_per_page) * geometry.row_bytes, "PLE row offset");
}

// The encoding a shard's declared geometry names. One table is one encoding: the reader decodes
// every row of every shard the same way, so a mixed table is refused here instead of half-decoded.
PleRowEncoding row_encoding(const WeightGeometry& geometry) {
    if (geometry.format == QType::NF4_G16_E4M3FN && geometry.layout == QuantLayout::RowPagedG16_4K)
        return PleRowEncoding::Nf4PagedG16;
    if (geometry.format == QType::Q4_G64_FP16 && geometry.layout == QuantLayout::RowSplit)
        return PleRowEncoding::Q4G64RowSplit;
    throw artifact::ArtifactError(
        "PLE table: the declared encoding is neither nf4_g16_e4m3fn/row_paged_g16_4k_v1 nor "
        "q4_g64_fp16/row_split_k128_v1");
}

std::size_t encoded_row_bytes(PleRowEncoding encoding) {
    return encoding == PleRowEncoding::Q4G64RowSplit ? kPleQ4RowBytes : kPleRowBytes;
}

} // namespace

void decode_ple_row(const PleRowSource& source, std::span<const std::byte> encoded,
                    std::span<std::uint16_t> output) {
    switch (source.encoding) {
    case PleRowEncoding::Nf4PagedG16:
        if (source.lut == nullptr) throw std::logic_error("PLE row: NF4 row has no codebook");
        decode_nf4_row(encoded, *source.lut, output);
        return;
    case PleRowEncoding::Q4G64RowSplit: {
        // The parts are concatenated in order, so the code plane is the row's first span and the
        // binary16 scale plane the second.
        const auto code_bytes = std::size_t(source.part_bytes[0]);
        if (source.parts != 2 || encoded.size() < code_bytes)
            throw std::logic_error("PLE row: q4 row is not a code plane followed by its scales");
        ops::q4::decode_q4_g64_row(encoded.first(code_bytes), encoded.subspan(code_bytes), output);
        return;
    }
    }
    throw std::logic_error("PLE row: unknown encoding");
}

struct PleTableSource::Impl {
    struct Shard {
        artifact::FileObjectView object;
        WeightGeometry geometry;
        Nf4Bf16Lut lut;
        std::uint64_t row_end;
        std::uint64_t row_begin;
        PleRowEncoding encoding;
    };

    artifact::FileBacking backing;
    std::vector<Shard> shards;
    PleHashControls controls;
    PleRowEncoding encoding = PleRowEncoding::Nf4PagedG16;
    std::uint64_t rows       = 0;
    std::size_t max_segments = 0;
};

PleTableSource::PleTableSource(const artifact::Reader& reader,
                               std::span<const artifact::ObjectHandle> shard_objects,
                               artifact::ObjectHandle controls)
    : impl_(std::make_unique<Impl>()) {
    using namespace artifact;
    if (shard_objects.empty()) throw ArtifactError("PLE table has no shards");
    for (const auto object : shard_objects) {
        const auto& g = reader.geometry(object);
        if (!supported_ple_table(g.format, g.layout, g.shape.size() == 2 ? g.shape[1] : 0) ||
            g.shape.size() != 2) {
            throw ArtifactError("PLE requires a supported encoding with width 160");
        }
        const auto encoding = row_encoding(g);
        if (impl_->rows == 0) {
            impl_->encoding = encoding;
        } else if (encoding != impl_->encoding) {
            throw ArtifactError("PLE table shards mix encodings");
        }
        impl_->rows = checked_add(impl_->rows, g.shape[0], "PLE rows");
    }
    if (impl_->rows > std::numeric_limits<std::uint32_t>::max())
        throw ArtifactError("PLE table logical extent is inconsistent");
    impl_->backing = prepare_file_backing(reader, shard_objects);
    impl_->shards.reserve(shard_objects.size());
    std::uint64_t row_end = 0;
    for (const auto object : shard_objects) {
        const auto& g = impl_->backing.geometry(object);
        const auto encoding = row_encoding(g);
        Nf4Bf16Lut lut{};
        if (encoding == PleRowEncoding::Nf4PagedG16) {
            std::array<std::byte, 4> word{};
            reader.read_into(checked_add(reader.directory().tensor(object).offset, g.divisor_offset,
                                         "PLE divisor offset"),
                             word);
            const auto divisor = std::bit_cast<float>(read_u32_le(word.data()));
            try {
                lut = make_nf4_bf16_lut(divisor);
            } catch (const std::invalid_argument& e) {
                throw ArtifactError(reader.directory().tensor(object).id + ": " + e.what());
            }
        } else if (g.code_bytes_per_row != 128 || g.scale_bytes_per_row != 8 ||
                   g.group_size != 64) {
            // The decoder above is the G64/q4 row shape; anything else would silently read the
            // wrong plane offsets, so it is refused with the geometry in the message.
            throw ArtifactError(reader.directory().tensor(object).id +
                                ": q4 PLE rows are not G64 rows of 128 code + 8 scale bytes");
        }
        const auto view     = impl_->backing.object(object);
        impl_->max_segments = std::max(impl_->max_segments, view.segments.size());
        row_end += g.shape[0];
        impl_->shards.push_back(
            {view, g, lut, row_end, row_end - g.shape[0], encoding});
    }
    const auto& resource = reader.directory().object(controls);
    if (!std::holds_alternative<ResourceObject>(resource) || object_bytes(resource) != 280)
        throw ArtifactError("invalid PLE hash controls resource");
    const auto bytes = reader.read_object(controls);
    auto word        = [&](std::size_t i) {
        return std::bit_cast<std::int64_t>(read_u64_le(bytes.data() + 8 * i));
    };
    auto& c = impl_->controls;
    for (std::size_t i = 0; i < 3; ++i) c.multipliers[i] = word(i);
    for (std::size_t i = 0; i < kPleHeads; ++i) {
        c.vocab_sizes[i] = word(3 + i);
        c.offsets[i]     = word(3 + kPleHeads + i);
        if (c.vocab_sizes[i] <= 0 || c.offsets[i] < 0 ||
            std::uint64_t(c.offsets[i]) > impl_->rows ||
            std::uint64_t(c.vocab_sizes[i]) > impl_->rows - c.offsets[i])
            throw ArtifactError("PLE hash head exceeds table rows");
    }
}

PleTableSource::~PleTableSource()                                    = default;
PleTableSource::PleTableSource(PleTableSource&&) noexcept            = default;
PleTableSource& PleTableSource::operator=(PleTableSource&&) noexcept = default;

PleRowSource PleTableSource::row(std::uint32_t global_row) const {
    if (global_row >= impl_->rows) throw std::out_of_range("PLE global row exceeds table");
    const auto it = std::upper_bound(
        impl_->shards.begin(), impl_->shards.end(), global_row,
        [](std::uint32_t row, const Impl::Shard& shard) { return row < shard.row_end; });
    const auto local = global_row - it->row_begin;
    PleRowSource out{it->object, it->encoding, nullptr, 0, {}, {}};
    if (it->encoding == PleRowEncoding::Nf4PagedG16) {
        out.lut              = &it->lut;
        out.parts            = 1;
        out.part_offsets[0]  = paged_row_offset(it->geometry, local);
        out.part_bytes[0]    = std::uint32_t(kPleRowBytes);
        return out;
    }
    // q4_g64 in row_split_k128_v1: the base-code plane then the binary16 scale plane, each row's
    // slice a contiguous run because every G64 group is 32 code bytes and 2 scale bytes.
    out.parts           = 2;
    out.part_offsets[0] = checked_mul(local, it->geometry.code_bytes_per_row, "PLE row offset");
    out.part_bytes[0]   = std::uint32_t(it->geometry.code_bytes_per_row);
    out.part_offsets[1] = checked_add(it->geometry.scale_offset,
                                      checked_mul(local, it->geometry.scale_bytes_per_row,
                                                  "PLE row scale offset"),
                                      "PLE row scale offset");
    out.part_bytes[1]   = std::uint32_t(it->geometry.scale_bytes_per_row);
    return out;
}

PleRowEncoding PleTableSource::encoding() const noexcept { return impl_->encoding; }

std::size_t PleTableSource::row_encoded_bytes() const noexcept {
    return encoded_row_bytes(impl_->encoding);
}

std::uint32_t PleTableSource::row_parts() const noexcept {
    return impl_->encoding == PleRowEncoding::Q4G64RowSplit ? 2 : 1;
}

std::uint64_t PleTableSource::rows() const noexcept { return impl_->rows; }

const PleHashControls& PleTableSource::controls() const noexcept { return impl_->controls; }

std::size_t PleTableSource::max_row_segments() const noexcept { return impl_->max_segments; }

std::size_t PleTableSource::file_count() const noexcept { return impl_->backing.file_count(); }

const artifact::InputFile& PleTableSource::file(std::size_t index) const {
    return impl_->backing.file(index);
}

const artifact::FileBacking& PleTableSource::backing() const { return impl_->backing; }

} // namespace ninfer::models::qwen4_exp
