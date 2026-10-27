// Splitting a state image into transferable shards.
//
// The image is ~64 MB and the transport refuses a single frame somewhere between 7.86 and 10.49 MB,
// so it has to cross in pieces. A shard here is a *contiguous byte range* of the image with a small
// fixed header, and the ranges are chosen on layer boundaries rather than by dividing the byte count:
//
//   * a shard boundary that falls inside a layer is meaningless -- a partial layer can be neither
//     validated nor retried on its own;
//   * layer boundaries survive a dtype change. If the recurrent state ever moves from fp32 to bf16,
//     the layer size halves but the boundaries stay put, so the format does not have to be redone.
//
// The trailing regions (continuation_hidden, and the draft-local planes when speculative decoding is
// on) ride on the last shard when they fit. They are tiny next to the layer body and do not deserve
// a round trip of their own.
//
// Everything is little-endian and fixed width: a shard is meant to be written by one process and
// read by another on a different machine, so nothing about the format may depend on the compiler.

#ifndef NINFER_CORE_STATE_IMAGE_SHARDS_H
#define NINFER_CORE_STATE_IMAGE_SHARDS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::core::state_shards {

inline constexpr std::uint32_t kMagic = 0x4853534Eu;  // "NSSH" little-endian
inline constexpr std::uint16_t kVersion = 1;
inline constexpr std::size_t kHeaderBytes = 40;

enum class Region : std::uint8_t {
    Conv = 0,       // the linear-attention convolution state
    Recurrent = 1,  // the delta-rule recurrent state, by far the largest part
    Tail = 2,       // continuation hidden, draft-local planes
};

const char* region_name(Region region);

// Where the image's parts live, in the order the layout places them.
struct RegionSizes {
    std::uint64_t conv_bytes{0};
    std::uint64_t layer_bytes{0};   // size of one layer of recurrent state
    std::uint32_t layers{0};
    std::uint64_t tail_bytes{0};
};

struct Plan {
    std::uint64_t offset{0};
    std::uint32_t length{0};
    Region region{Region::Recurrent};
    std::uint32_t first_layer{0};
    std::uint32_t layer_count{0};
};

inline std::uint64_t image_bytes(const RegionSizes& sizes) {
    return sizes.conv_bytes + sizes.layer_bytes * sizes.layers + sizes.tail_bytes;
}

// Chooses the shard ranges. Returns an empty plan only when the inputs cannot describe an image at
// all, so a caller never silently gets a half-described split.
//
// Every size here comes from the layout, so a different model simply produces different numbers.
// What must not depend on the model is the *policy*, and that is: cut on layer boundaries while a
// whole layer fits, and fall back to byte-cutting only for a unit that cannot fit at all (a very
// large conv block, or a layer whose recurrent matrix exceeds the frame budget). Refusing to plan
// those cases -- as the first version did -- would make the format work for one model and fail on
// the next one, which is the opposite of what a transfer format is for.
inline std::vector<Plan> plan_shards(const RegionSizes& sizes, std::uint64_t max_payload) {
    std::vector<Plan> plan;
    if (max_payload == 0 || max_payload > 0xFFFFFFFFull) { return plan; }
    const std::uint64_t total = image_bytes(sizes);
    if (total == 0) { return plan; }

    std::uint64_t cursor = 0;
    // Appends shards for `bytes`, splitting further only when `bytes` alone exceeds the budget.
    // The layer counters go on the first piece of a unit, so summing them still counts each layer once.
    const auto emit = [&](std::uint64_t bytes, Region region, std::uint32_t first_layer,
                          std::uint32_t layer_count) {
        std::uint64_t done = 0;
        bool first_piece = true;
        while (done < bytes) {
            const std::uint64_t piece = std::min<std::uint64_t>(max_payload, bytes - done);
            plan.push_back(Plan{cursor + done, static_cast<std::uint32_t>(piece), region,
                                first_piece ? first_layer : 0, first_piece ? layer_count : 0});
            done += piece;
            first_piece = false;
        }
        cursor += bytes;
    };

    if (sizes.conv_bytes > 0) { emit(sizes.conv_bytes, Region::Conv, 0, 0); }

    if (sizes.layers > 0) {
        if (sizes.layer_bytes == 0) { return {}; }
        if (sizes.layer_bytes <= max_payload) {
            const auto per_shard = static_cast<std::uint32_t>(max_payload / sizes.layer_bytes);
            std::uint32_t layer = 0;
            while (layer < sizes.layers) {
                const std::uint32_t count = std::min(per_shard, sizes.layers - layer);
                emit(sizes.layer_bytes * count, Region::Recurrent, layer, count);
                layer += count;
            }
        } else {
            for (std::uint32_t layer = 0; layer < sizes.layers; ++layer) {
                emit(sizes.layer_bytes, Region::Recurrent, layer, 1);
            }
        }
    }

    if (sizes.tail_bytes > 0) { emit(sizes.tail_bytes, Region::Tail, 0, 0); }

    // The plan must tile the image exactly: no gap, no overlap, nothing left over.
    if (cursor != total) { return {}; }
    for (std::size_t i = 1; i < plan.size(); ++i) {
        if (plan[i].offset != plan[i - 1].offset + plan[i - 1].length) { return {}; }
    }
    return plan;
}

// --- header ----------------------------------------------------------------

namespace detail {

inline void put_u16(std::string& out, std::uint16_t value) {
    out.push_back(static_cast<char>(value & 0xFFu));
    out.push_back(static_cast<char>((value >> 8) & 0xFFu));
}

inline void put_u32(std::string& out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xFFu));
    }
}

inline void put_u64(std::string& out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xFFu));
    }
}

inline bool take_u16(const std::string& in, std::size_t& off, std::uint16_t& value) {
    if (off + 2 > in.size()) { return false; }
    value = static_cast<std::uint16_t>(static_cast<std::uint8_t>(in[off]) |
                                       (static_cast<std::uint8_t>(in[off + 1]) << 8));
    off += 2;
    return true;
}

inline bool take_u32(const std::string& in, std::size_t& off, std::uint32_t& value) {
    if (off + 4 > in.size()) { return false; }
    value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(in[off + i])) << (8 * i);
    }
    off += 4;
    return true;
}

inline bool take_u64(const std::string& in, std::size_t& off, std::uint64_t& value) {
    if (off + 8 > in.size()) { return false; }
    value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(in[off + i])) << (8 * i);
    }
    off += 8;
    return true;
}

}  // namespace detail

inline const char* region_name(Region region) {
    switch (region) {
    case Region::Conv: return "conv";
    case Region::Recurrent: return "recurrent";
    case Region::Tail: return "tail";
    default: return "unknown";
    }
}

// magic | version | region | reserved | index | count | total | offset | length | reserved
inline std::string encode_header(const Plan& plan, std::uint32_t index, std::uint32_t count,
                                 std::uint64_t total) {
    std::string out;
    out.reserve(kHeaderBytes);
    detail::put_u32(out, kMagic);
    detail::put_u16(out, kVersion);
    out.push_back(static_cast<char>(plan.region));
    out.push_back('\0');
    detail::put_u32(out, index);
    detail::put_u32(out, count);
    detail::put_u64(out, total);
    detail::put_u64(out, plan.offset);
    detail::put_u32(out, plan.length);
    detail::put_u16(out, static_cast<std::uint16_t>(plan.first_layer));
    detail::put_u16(out, static_cast<std::uint16_t>(plan.layer_count));
    out.append(kHeaderBytes - out.size(), '\0');
    return out;
}

// Parses a header and checks it against what the caller expects. Returns false rather than throwing:
// a shard arriving from the network is untrusted input.
inline bool decode_header(const std::string& bytes, std::uint32_t& index, std::uint32_t& count,
                          std::uint64_t& total, Plan& plan) {
    if (bytes.size() < kHeaderBytes) { return false; }
    std::size_t off = 0;
    std::uint32_t magic = 0;
    std::uint16_t version = 0;
    if (!detail::take_u32(bytes, off, magic) || magic != kMagic) { return false; }
    if (!detail::take_u16(bytes, off, version) || version != kVersion) { return false; }
    plan.region = static_cast<Region>(static_cast<std::uint8_t>(bytes[off]));
    ++off;
    ++off;  // reserved
    if (!detail::take_u32(bytes, off, index)) { return false; }
    if (!detail::take_u32(bytes, off, count)) { return false; }
    if (!detail::take_u64(bytes, off, total)) { return false; }
    if (!detail::take_u64(bytes, off, plan.offset)) { return false; }
    if (!detail::take_u32(bytes, off, plan.length)) { return false; }
    std::uint16_t first = 0;
    std::uint16_t layers = 0;
    if (!detail::take_u16(bytes, off, first) || !detail::take_u16(bytes, off, layers)) {
        return false;
    }
    plan.first_layer = first;
    plan.layer_count = layers;
    return count != 0 && index < count;
}

struct SplitResult {
    bool ok{false};
    std::string error;
    std::vector<std::string> shards;  // header + payload, in plan order
};

// Splits the image. Each shard is self-describing, so the receiver can validate before allocating.
inline SplitResult split(const std::string& image, const RegionSizes& sizes,
                         std::uint64_t max_payload) {
    SplitResult result;
    const std::vector<Plan> plan = plan_shards(sizes, max_payload);
    if (plan.empty()) {
        result.error = "no valid plan for these region sizes";
        return result;
    }
    if (image.size() != image_bytes(sizes)) {
        result.error = "image is " + std::to_string(image.size()) + " B but the regions describe " +
                       std::to_string(image_bytes(sizes)) + " B";
        return result;
    }
    result.shards.reserve(plan.size());
    for (std::size_t i = 0; i < plan.size(); ++i) {
        const Plan& piece = plan[i];
        std::string shard = encode_header(piece, static_cast<std::uint32_t>(i),
                                          static_cast<std::uint32_t>(plan.size()),
                                          image_bytes(sizes));
        shard.append(image, static_cast<std::size_t>(piece.offset), piece.length);
        result.shards.push_back(std::move(shard));
    }
    result.ok = true;
    return result;
}

// Reassembles in order. Rejects gaps, overlaps, duplicates, a wrong total, and a short payload --
// every one of those is a corrupted transfer rather than a recoverable condition.
inline bool join(const std::vector<std::string>& shards, std::string& image, std::string& error) {
    if (shards.empty()) {
        error = "no shards";
        return false;
    }
    struct Parsed {
        Plan plan;
        std::size_t payload_size;
    };
    std::vector<Parsed> parsed;
    parsed.reserve(shards.size());
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < shards.size(); ++i) {
        std::uint32_t index = 0;
        std::uint32_t count = 0;
        std::uint64_t declared_total = 0;
        Plan plan;
        if (!decode_header(shards[i], index, count, declared_total, plan)) {
            error = "shard " + std::to_string(i) + " has an unreadable header";
            return false;
        }
        if (index != i) {
            error = "shard " + std::to_string(i) + " carries index " + std::to_string(index);
            return false;
        }
        if (count != shards.size()) {
            error = "shard " + std::to_string(i) + " expects " + std::to_string(count) +
                    " shards, got " + std::to_string(shards.size());
            return false;
        }
        if (i == 0) {
            total = declared_total;
        } else if (declared_total != total) {
            error = "shards disagree on the total size";
            return false;
        }
        const std::size_t payload = shards[i].size() - kHeaderBytes;
        if (payload != plan.length) {
            error = "shard " + std::to_string(i) + " declares " + std::to_string(plan.length) +
                    " B but carries " + std::to_string(payload);
            return false;
        }
        parsed.push_back(Parsed{plan, payload});
    }

    image.clear();
    image.reserve(static_cast<std::size_t>(total));
    std::uint64_t cursor = 0;
    for (std::size_t i = 0; i < parsed.size(); ++i) {
        if (parsed[i].plan.offset != cursor) {
            error = "shard " + std::to_string(i) + " starts at " + std::to_string(parsed[i].plan.offset) +
                    ", expected " + std::to_string(cursor);
            return false;
        }
        image.append(shards[i], kHeaderBytes, parsed[i].payload_size);
        cursor += parsed[i].payload_size;
    }
    if (cursor != total) {
        error = "reassembled " + std::to_string(cursor) + " B, expected " + std::to_string(total);
        return false;
    }
    return true;
}

}  // namespace ninfer::core::state_shards

#endif  // NINFER_CORE_STATE_IMAGE_SHARDS_H
