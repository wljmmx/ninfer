#pragma once

// The payload a *checkpoint* has to carry, and why it cannot be a raw KV chunk.
//
// This model keeps recurrent state alongside the KV pages (state images/slots, the GDN replay
// records), so restoring KV pages alone and skipping prefill would resume from a state that never
// saw those tokens: silently wrong output, no crash, no assertion. A reusable prefix therefore has
// to travel as a checkpoint: the KV spans plus the state spans, plus the frontier they belong to.
//
// Format (little endian, all offsets relative to the payload start):
//     "NFCK" | u16 version | u16 span_count | u32 frontier | u64 total_bytes | digest[8]
//     span_count * { u32 tag | u32 bytes | u64 offset }
//     payload bytes, spans back to back
//
// Namespace: these keys are NOT LMCache keys. LMCache's tier stores one raw block per key
// (model@world@worker@hash@dtype) and knows nothing about our span table, so a checkpoint under
// that name would be a lie. Checkpoints go under "ninfer.ckpt." + the digest, and the two can never
// collide.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::core {

inline constexpr std::uint16_t kCheckpointBlobVersion = 1;
inline constexpr std::size_t kCheckpointHeaderBytes   = 4 + 2 + 2 + 4 + 8 + 8;
inline constexpr std::size_t kCheckpointSpanBytes     = 4 + 4 + 8;

enum class CheckpointSpanTag : std::uint32_t {
    KvPages    = 1,
    StateSlots = 2,
};

struct CheckpointSpan {
    CheckpointSpanTag tag = CheckpointSpanTag::KvPages;
    std::span<const std::byte> bytes;
};

struct CheckpointBlobView {
    std::uint32_t frontier = 0;
    std::uint64_t digest   = 0;
    // One entry per span, pointing into the decoded buffer.
    std::vector<std::pair<CheckpointSpanTag, std::span<const std::byte>>> spans;
};

// 64-bit content digest (FNV-1a over the header fields and every span byte). It is the key suffix,
// so two different checkpoints cannot overwrite each other by accident.
[[nodiscard]] inline std::uint64_t checkpoint_digest(std::uint32_t frontier,
                                                     std::uint64_t total_bytes,
                                                     std::span<const std::byte> payload) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix     = [&hash](std::uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            hash ^= (value >> (i * 8)) & 0xFFULL;
            hash *= 1099511628211ULL;
        }
    };
    mix(frontier);
    mix(total_bytes);
    for (const auto byte : payload) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    return hash;
}

[[nodiscard]] inline std::string checkpoint_key(std::uint64_t digest) {
    char buffer[32]{};
    std::snprintf(buffer, sizeof buffer, "ninfer.ckpt.%016llx", static_cast<unsigned long long>(digest));
    return buffer;
}

namespace detail {

inline void put_u16(std::vector<std::byte>& out, std::uint16_t value) {
    out.push_back(static_cast<std::byte>(value & 0xFF));
    out.push_back(static_cast<std::byte>((value >> 8) & 0xFF));
}
inline void put_u32(std::vector<std::byte>& out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) { out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFF)); }
}
inline void put_u64(std::vector<std::byte>& out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) { out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFF)); }
}
inline std::uint16_t get_u16(const std::byte* p) {
    return static_cast<std::uint16_t>(static_cast<unsigned>(p[0]) |
                                      (static_cast<unsigned>(p[1]) << 8));
}
inline std::uint32_t get_u32(const std::byte* p) {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(static_cast<unsigned>(p[i])) << (8 * i);
    }
    return value;
}
inline std::uint64_t get_u64(const std::byte* p) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(static_cast<unsigned>(p[i])) << (8 * i);
    }
    return value;
}

} // namespace detail

// Builds the blob. Total payload is the concatenation of the spans in the order given.
[[nodiscard]] inline std::vector<std::byte> encode_checkpoint(std::uint32_t frontier,
                                                              std::span<const CheckpointSpan> spans) {
    std::uint64_t total = 0;
    for (const auto& span : spans) { total += span.bytes.size(); }

    std::vector<std::byte> payload;
    payload.reserve(static_cast<std::size_t>(total));
    for (const auto& span : spans) { payload.insert(payload.end(), span.bytes.begin(), span.bytes.end()); }

    const auto digest = checkpoint_digest(frontier, total, payload);

    std::vector<std::byte> out;
    out.reserve(kCheckpointHeaderBytes + spans.size() * kCheckpointSpanBytes + payload.size());
    out.push_back(std::byte{'N'});
    out.push_back(std::byte{'F'});
    out.push_back(std::byte{'C'});
    out.push_back(std::byte{'K'});
    detail::put_u16(out, kCheckpointBlobVersion);
    detail::put_u16(out, static_cast<std::uint16_t>(spans.size()));
    detail::put_u32(out, frontier);
    detail::put_u64(out, total);
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::byte>((digest >> (8 * i)) & 0xFF));
    }
    std::uint64_t offset = 0;
    for (const auto& span : spans) {
        detail::put_u32(out, static_cast<std::uint32_t>(span.tag));
        detail::put_u32(out, static_cast<std::uint32_t>(span.bytes.size()));
        detail::put_u64(out, offset);
        offset += span.bytes.size();
    }
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// Validates everything before handing out spans: a blob whose spans overlap or overrun the buffer
// would let a caller read past the payload and treat garbage as KV.
[[nodiscard]] inline std::optional<CheckpointBlobView> decode_checkpoint(std::span<const std::byte> blob) {
    if (blob.size() < kCheckpointHeaderBytes) { return std::nullopt; }
    const auto* p = blob.data();
    if (!(p[0] == std::byte{'N'} && p[1] == std::byte{'F'} && p[2] == std::byte{'C'} &&
          p[3] == std::byte{'K'})) {
        return std::nullopt;
    }
    if (detail::get_u16(p + 4) != kCheckpointBlobVersion) { return std::nullopt; }
    const auto span_count = detail::get_u16(p + 6);
    const auto frontier   = detail::get_u32(p + 8);
    const auto total      = detail::get_u64(p + 12);
    if (blob.size() != kCheckpointHeaderBytes + static_cast<std::size_t>(span_count) * kCheckpointSpanBytes +
                           static_cast<std::size_t>(total)) {
        return std::nullopt;
    }
    std::uint64_t digest = 0;
    for (int i = 0; i < 8; ++i) {
        digest |= static_cast<std::uint64_t>(static_cast<unsigned>(p[20 + i])) << (8 * i);
    }

    const std::byte* table = p + kCheckpointHeaderBytes;
    const std::byte* body  = table + static_cast<std::size_t>(span_count) * kCheckpointSpanBytes;

    CheckpointBlobView view{.frontier = frontier, .digest = digest, .spans = {}};
    view.spans.reserve(span_count);
    std::uint64_t cursor = 0;
    for (std::uint16_t i = 0; i < span_count; ++i) {
        const std::byte* entry = table + static_cast<std::size_t>(i) * kCheckpointSpanBytes;
        const auto tag         = detail::get_u32(entry);
        const auto bytes       = detail::get_u32(entry + 4);
        const auto offset      = detail::get_u64(entry + 8);
        // Spans must be laid out in order, without gaps or overlap, inside the payload.
        if (offset != cursor || offset + bytes > total) { return std::nullopt; }
        cursor += bytes;
        view.spans.emplace_back(static_cast<CheckpointSpanTag>(tag),
                                std::span<const std::byte>(body + offset, bytes));
    }
    if (cursor != total) { return std::nullopt; }

    // The digest covers the payload, so a corrupted body is rejected rather than decoded.
    const auto expected = checkpoint_digest(frontier, total, std::span<const std::byte>(body, total));
    if (expected != digest) { return std::nullopt; }
    return view;
}

} // namespace ninfer::core

