#include "models/qwen4_exp/ple/table.h"

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

} // namespace

struct PleTableSource::Impl {
    struct Shard {
        artifact::FileObjectView object;
        WeightGeometry geometry;
        Nf4Bf16Lut lut;
        std::uint64_t row_end;
    };

    artifact::FileBacking backing;
    std::vector<Shard> shards;
    PleHashControls controls;
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
        if (g.format != QType::NF4_G16_E4M3FN || g.layout != QuantLayout::RowPagedG16_4K ||
            g.shape.size() != 2 || g.shape[1] != kPleRowElements) {
            throw ArtifactError("PLE requires complete NF4 shards with width 160");
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
        std::array<std::byte, 4> word{};
        reader.read_into(checked_add(reader.directory().tensor(object).offset, g.divisor_offset,
                                     "PLE divisor offset"),
                         word);
        const auto divisor  = std::bit_cast<float>(read_u32_le(word.data()));
        const auto view     = impl_->backing.object(object);
        impl_->max_segments = std::max(impl_->max_segments, view.segments.size());
        row_end += g.shape[0];
        try {
            impl_->shards.push_back({view, g, make_nf4_bf16_lut(divisor), row_end});
        } catch (const std::invalid_argument& e) {
            throw ArtifactError(reader.directory().tensor(object).id + ": " + e.what());
        }
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
    const auto local = global_row - (it->row_end - it->geometry.shape[0]);
    return {it->object, paged_row_offset(it->geometry, local), &it->lut};
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
