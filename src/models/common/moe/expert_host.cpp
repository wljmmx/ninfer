#include "models/common/moe/expert_host.h"

#include "core/io/file.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::moe {
namespace {

std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t extent_of(const WeightBankLayout& layout) {
    // Packed per-expert image: codes, then high, then scales, with no shared-array offset.
    return layout.codes.expert_bytes + layout.high.expert_bytes + layout.scales.expert_bytes;
}

} // namespace

ExpertHostTier::ExpertHostTier(const artifact::Reader& reader,
                               std::span<const BankDescription> banks, std::size_t quota_bytes) {
    quota_ = quota_bytes;
    banks_.resize(banks.size());
    last_use_.resize(banks.size());
    live_.resize(banks.size());

    struct Layout {
        const BankDescription* bank;
        std::int32_t matrix;
    };
    std::vector<Layout> order;
    order.reserve(banks.size() * 2);
    for (std::size_t b = 0; b < banks.size(); ++b) {
        order.push_back({&banks[b], 0});
        order.push_back({&banks[b], 1});
    }

    std::uint64_t total = 0;
    const auto unit     = HostReserve::granularity();
    for (const auto& entry : order) {
        const auto& source = entry.matrix == 0 ? entry.bank->gate_up : entry.bank->down;
        const auto& layout = source.layout;
        const auto count   = layout.region.count;
        const auto extent  = extent_of(layout);
        const auto segment = round_up(std::max<std::uint64_t>(extent, 1), unit);
        auto& matrix = banks_[static_cast<std::size_t>(entry.bank - banks.data())][entry.matrix];
        matrix.base        = total;
        matrix.segment     = segment;
        matrix.count       = count;
        total += segment * count;
    }
    reserve_ = HostReserve(static_cast<std::size_t>(total));

    for (std::size_t b = 0; b < banks.size(); ++b) {
        const auto& description = banks[b];
        const auto count        = description.gate_up.layout.region.count;
        last_use_[b].assign(count, 0);
        live_[b].assign(count, 0);
    }

    files_.reserve(reader.directory().files.size());
    for (std::size_t index = 0; index < reader.directory().files.size(); ++index) {
        files_.push_back(
            std::make_unique<io::ReadOnlyFile>(reader.open_file(index, io::ReadMode::Buffered)));
    }

    for (const auto& entry : order) {
        const auto& source   = entry.matrix == 0 ? entry.bank->gate_up : entry.bank->down;
        const auto& layout   = source.layout;
        const auto slot      = static_cast<std::size_t>(entry.bank - banks.data());
        auto& matrix         = banks_[slot][entry.matrix];
        const auto object    = reader.directory().tensor(source.object).offset;
        matrix.parent.geometry = reader.geometry(source.object);
        matrix.parent.geometry.bytes =
            static_cast<std::uint64_t>(matrix.segment) * matrix.count;
        matrix.parent.device = Device::cpu();
        matrix.parent.data   = reserve_.data() + matrix.base;
        // Host divisor metadata is tiny and always resident.
        matrix.divisors.assign(matrix.parent.geometry.divisor_count, 1.0F);
        if (matrix.parent.geometry.divisor_count) {
            std::vector<std::byte> bytes(matrix.parent.geometry.divisor_count * sizeof(float));
            for (const auto& segment :
                 reader.segments(object + matrix.parent.geometry.divisor_offset, bytes.size())) {
                files_[segment.file_index]->read_exact(
                    segment.file_offset,
                    std::span<std::byte>(bytes).subspan(
                        static_cast<std::size_t>(segment.destination_offset),
                        static_cast<std::size_t>(segment.bytes)));
            }
            std::memcpy(matrix.divisors.data(), bytes.data(), bytes.size());
        }
        matrix.parent.weight_divisors = matrix.divisors;
        matrix.offsets.resize(matrix.count);
        matrix.chunks.resize(matrix.count);
        const auto codes_bytes = layout.codes.expert_bytes;
        const auto high_bytes  = layout.high.expert_bytes;
        for (std::uint32_t e = 0; e < matrix.count; ++e) {
            matrix.offsets[e] = std::uint64_t{e} * matrix.segment;
            auto& chunks      = matrix.chunks[e];
            const auto append        = [&](const WeightBankPlane& plane, std::uint64_t image) {
                if (!plane.expert_bytes) { return; }
                const auto logical = object + plane.offset + std::uint64_t{e} * plane.expert_stride;
                ExpertFileChunk chunk;
                chunk.image_offset = image;
                for (const auto& segment : reader.segments(logical, plane.expert_bytes)) {
                    chunk.spans.push_back({static_cast<std::uint32_t>(segment.file_index),
                                           segment.file_offset, segment.bytes});
                }
                chunks.push_back(std::move(chunk));
            };
            append(layout.codes, 0);
            append(layout.high, codes_bytes);
            append(layout.scales, codes_bytes + high_bytes);
        }
    }

    // Diagnostic / prewarm switch: commit and read every expert up front.
    if (const char* preload = std::getenv("NINFER_EXPERT_HOST_PRELOAD");
        preload && *preload != '0') {
        for (std::size_t b = 0; b < banks_.size(); ++b) {
            for (std::uint32_t e = 0; e < last_use_[b].size(); ++e) {
                for (auto& matrix : banks_[b]) { load(matrix, e); }
                live_[b][e] = 1;
            }
        }
        std::fprintf(stderr, "expert host tier: preloaded %.2f GiB\n",
                     static_cast<double>(reserve_.committed_bytes()) / (1024.0 * 1024 * 1024));
    }
}

ExpertHostTier::~ExpertHostTier() = default;

const WeightParent* ExpertHostTier::parent(std::uint32_t bank, bool gate_up) const {
    if (bank >= banks_.size()) { return nullptr; }
    return &banks_[bank][gate_up ? 0 : 1].parent;
}

const std::uint64_t* ExpertHostTier::offsets(std::uint32_t bank, bool gate_up) const {
    if (bank >= banks_.size()) { return nullptr; }
    const auto& matrix = banks_[bank][gate_up ? 0 : 1];
    return matrix.offsets.empty() ? nullptr : matrix.offsets.data();
}

bool ExpertHostTier::resident(const Matrix& matrix, std::uint32_t expert) const {
    return reserve_.resident(matrix.base + std::uint64_t{expert} * matrix.segment, matrix.segment);
}

void ExpertHostTier::load(Matrix& matrix, std::uint32_t expert) {
    const auto base = matrix.base + std::uint64_t{expert} * matrix.segment;
    reserve_.commit(base, matrix.segment);
    auto* destination = reserve_.data() + base;
    for (const auto& chunk : matrix.chunks[expert]) {
        auto cursor = std::span<std::byte>(destination + chunk.image_offset,
                                           matrix.segment > chunk.image_offset
                                               ? static_cast<std::size_t>(matrix.segment -
                                                                          chunk.image_offset)
                                               : 0);
        for (const auto& span : chunk.spans) {
            if (span.file_index >= files_.size() || span.bytes > cursor.size()) {
                throw std::runtime_error("expert host tier span exceeds its segment");
            }
            files_[span.file_index]->read_exact(
                span.file_offset, cursor.subspan(0, static_cast<std::size_t>(span.bytes)));
            cursor = cursor.subspan(static_cast<std::size_t>(span.bytes));
        }
    }
    ++loads_;
}

void ExpertHostTier::release(Matrix& matrix, std::uint32_t expert) {
    reserve_.decommit(matrix.base + std::uint64_t{expert} * matrix.segment, matrix.segment);
}

void ExpertHostTier::enforce_quota(ExpertBankId bank, std::uint32_t keep_expert) {
    if (!quota_) { return; }
    while (reserve_.committed_bytes() > quota_) {
        std::uint64_t best_epoch = UINT64_MAX;
        std::size_t best_bank    = banks_.size();
        std::uint32_t best_expert = 0;
        for (std::size_t b = 0; b < live_.size(); ++b) {
            for (std::uint32_t e = 0; e < live_[b].size(); ++e) {
                if (!live_[b][e]) { continue; }
                if (b == bank.index && e == keep_expert) { continue; }
                if (last_use_[b][e] < best_epoch) {
                    best_epoch  = last_use_[b][e];
                    best_bank   = b;
                    best_expert = e;
                }
            }
        }
        if (best_bank == banks_.size()) { break; }
        for (auto& matrix : banks_[best_bank]) { release(matrix, best_expert); }
        live_[best_bank][best_expert] = 0;
        ++evictions_;
    }
}

void ExpertHostTier::ensure(ExpertBankId bank, std::span<const std::int32_t> experts,
                            const std::int32_t* resident_mask) {
    if (bank.index >= banks_.size()) { return; }
    const auto current = ++epoch_;
    for (const auto id : experts) {
        if (id < 0 || static_cast<std::size_t>(id) >= live_[bank.index].size()) { continue; }
        if (resident_mask && resident_mask[id] != 0) { continue; }
        const auto expert = static_cast<std::uint32_t>(id);
        if (live_[bank.index][expert]) {
            ++hits_;
            last_use_[bank.index][expert] = current;
            continue;
        }
        for (auto& matrix : banks_[bank.index]) { load(matrix, expert); }
        live_[bank.index][expert] = 1;
        last_use_[bank.index][expert] = current;
        enforce_quota(bank, expert);
    }
}

} // namespace ninfer::models::moe
