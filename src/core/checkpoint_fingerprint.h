// What has to match before a checkpoint may be restored into a process.
//
// A restore that copies the wrong bytes into the right place does not fail -- it produces wrong
// output, later, somewhere else. So the check cannot be "same model name": two runs of the same
// model differ in ways that change the bytes:
//
//   * KV dtype          -- bf16 / int8 / fp8 / k8v4 all change the page size
//   * page and chunk geometry -- how many tokens a page holds
//   * speculative decoding -- adds the draft-local regions to the state image
//   * architecture       -- layer count, hidden size, head geometry
//
// and, less obviously, layer_bytes. A dtype change halves the recurrent layer size while
// image_bytes can land on a value some other configuration also produces; if the fingerprint
// carried only the total, the shard boundaries computed on the producing side would silently not
// line up with the consuming side.
//
// The encoding is fixed-width and little-endian on purpose: it is meant to travel next to the
// payload, be compared by a different process, and go into a cache key, so it cannot depend on the
// compiler's layout rules or on field order in a struct.

#ifndef NINFER_CORE_CHECKPOINT_FINGERPRINT_H
#define NINFER_CORE_CHECKPOINT_FINGERPRINT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::core::fingerprint {

inline constexpr std::uint16_t kVersion = 1;
inline constexpr std::size_t kEncodedBytes = 74;  // 2+4+8+4+4+1+1+2 + 16 + 24 + 4 + 4

// KV plane dtype, as a wire tag rather than a torch/engine enum: the number is part of the format.
// Values mirror KvCacheStorage in the engine exactly (same order, same numbers), so this is a
// wire image of that enum rather than a second taxonomy that can drift from it. An incomplete
// enum here would not fail loudly: two unrepresentable layouts would both decode as Unknown, and
// a checkpoint produced under one would be accepted by a process running the other.
enum class KVDtype : std::uint8_t {
    BF16 = 0,          // KvCacheStorage::BFloat16
    INT8 = 1,          // Int8Group64
    FP8 = 2,           // Fp8E4M3Row256
    NVFP4 = 3,         // Nvfp4Group16
    K8V4 = 4,          // Fp8KeyNvfp4Value  (8-bit key, 4-bit value)
    RK8V4 = 5,         // RotatedInt8KeyInt4ValueGroup64
    RK4V4 = 6,         // RotatedInt4KeyInt4ValueGroup64
    RK4V4E8 = 7,       // 4-bit E8-lattice key, 4-bit value
    RK2V4E8 = 8,       // 2-bit E8-cylinder key, 4-bit value
    RK4V2E8 = 9,       // 4-bit E8-lattice key, 2-bit E8-cylinder value
    Unknown = 255,
};

struct Features {
    bool dflash_local{false};  // draft-local K/V regions present in the state image
    bool vision{false};

    std::uint32_t bits() const {
        return (dflash_local ? 1u : 0u) | (vision ? 2u : 0u);
    }
};

struct CheckpointFingerprint {
    // identity
    std::uint64_t model_name_hash{0};  // FNV-1a of the model name; not a security boundary

    // KV side
    std::uint32_t kv_page_bytes{0};      // bytes in one engine page (all planes)
    std::uint32_t kv_tokens_per_page{0}; // engine page granularity
    KVDtype kv_dtype{KVDtype::Unknown};
    std::uint8_t kv_planes{0};           // 2 = separate K/V, 1 = fused

    // architecture
    std::uint32_t layers{0};
    std::uint32_t hidden{0};
    std::uint32_t kv_heads{0};
    std::uint32_t head_dim{0};

    // state image
    std::uint64_t state_image_bytes{0};
    std::uint64_t state_layer_bytes{0};  // one layer; decides the shard boundaries
    std::uint64_t state_conv_bytes{0};
    std::uint32_t state_layers{0};

    Features features;
};

// FNV-1a. Deliberately not a cryptographic hash: this groups cache entries and catches
// misconfiguration, it does not defend against a crafted payload.
inline std::uint64_t hash_name(std::string_view name) {
    std::uint64_t value = 0xcbf29ce484222325ull;
    for (const char character : name) {
        value ^= static_cast<std::uint8_t>(character);
        value *= 0x100000001b3ull;
    }
    return value;
}

namespace detail {

inline void put_u8(std::string& out, std::uint8_t value) {
    out.push_back(static_cast<char>(value));
}

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

inline bool take_u8(const std::string& in, std::size_t& off, std::uint8_t& value) {
    if (off + 1 > in.size()) { return false; }
    value = static_cast<std::uint8_t>(in[off++]);
    return true;
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

inline std::string encode(const CheckpointFingerprint& print) {
    std::string out;
    out.reserve(kEncodedBytes);
    detail::put_u16(out, kVersion);
    detail::put_u32(out, print.features.bits());
    detail::put_u64(out, print.model_name_hash);
    detail::put_u32(out, print.kv_page_bytes);
    detail::put_u32(out, print.kv_tokens_per_page);
    detail::put_u8(out, static_cast<std::uint8_t>(print.kv_dtype));
    detail::put_u8(out, print.kv_planes);
    detail::put_u16(out, 0);  // reserved, keeps the following u32s aligned
    detail::put_u32(out, print.layers);
    detail::put_u32(out, print.hidden);
    detail::put_u32(out, print.kv_heads);
    detail::put_u32(out, print.head_dim);
    detail::put_u64(out, print.state_image_bytes);
    detail::put_u64(out, print.state_layer_bytes);
    detail::put_u64(out, print.state_conv_bytes);
    detail::put_u32(out, print.state_layers);
    detail::put_u32(out, 0);  // reserved
    if (out.size() < kEncodedBytes) { out.append(kEncodedBytes - out.size(), '\0'); }
    return out;
}

// Parses an encoded fingerprint. A wrong version or a short buffer is a refusal, not a best effort.
inline bool decode(const std::string& bytes, CheckpointFingerprint& print) {
    if (bytes.size() != kEncodedBytes) { return false; }
    std::size_t off = 0;
    std::uint16_t version = 0;
    if (!detail::take_u16(bytes, off, version) || version != kVersion) { return false; }
    std::uint32_t bits = 0;
    if (!detail::take_u32(bytes, off, bits)) { return false; }
    print.features.dflash_local = (bits & 1u) != 0;
    print.features.vision = (bits & 2u) != 0;
    if (!detail::take_u64(bytes, off, print.model_name_hash)) { return false; }
    if (!detail::take_u32(bytes, off, print.kv_page_bytes)) { return false; }
    if (!detail::take_u32(bytes, off, print.kv_tokens_per_page)) { return false; }
    std::uint8_t dtype = 0;
    if (!detail::take_u8(bytes, off, dtype)) { return false; }
    print.kv_dtype = static_cast<KVDtype>(dtype);
    if (!detail::take_u8(bytes, off, print.kv_planes)) { return false; }
    std::uint16_t reserved = 0;
    if (!detail::take_u16(bytes, off, reserved)) { return false; }
    if (!detail::take_u32(bytes, off, print.layers)) { return false; }
    if (!detail::take_u32(bytes, off, print.hidden)) { return false; }
    if (!detail::take_u32(bytes, off, print.kv_heads)) { return false; }
    if (!detail::take_u32(bytes, off, print.head_dim)) { return false; }
    if (!detail::take_u64(bytes, off, print.state_image_bytes)) { return false; }
    if (!detail::take_u64(bytes, off, print.state_layer_bytes)) { return false; }
    if (!detail::take_u64(bytes, off, print.state_conv_bytes)) { return false; }
    if (!detail::take_u32(bytes, off, print.state_layers)) { return false; }
    return true;
}

// Field-by-field, naming the first difference. "They differ" is not actionable; "kv_dtype differs"
// is, and it is the difference between a five-minute fix and a day of wrong-looking output.
inline bool matches(const CheckpointFingerprint& have, const CheckpointFingerprint& want,
                    std::string& difference) {
    difference.clear();
    const auto check = [&difference](bool equal, const char* field) {
        if (equal) { return true; }
        difference = field;
        return false;
    };
    if (!check(have.model_name_hash == want.model_name_hash, "model_name")) { return false; }
    if (!check(have.kv_dtype == want.kv_dtype, "kv_dtype")) { return false; }
    if (!check(have.kv_page_bytes == want.kv_page_bytes, "kv_page_bytes")) { return false; }
    if (!check(have.kv_tokens_per_page == want.kv_tokens_per_page, "kv_tokens_per_page")) {
        return false;
    }
    if (!check(have.kv_planes == want.kv_planes, "kv_planes")) { return false; }
    if (!check(have.layers == want.layers, "layers")) { return false; }
    if (!check(have.hidden == want.hidden, "hidden")) { return false; }
    if (!check(have.kv_heads == want.kv_heads, "kv_heads")) { return false; }
    if (!check(have.head_dim == want.head_dim, "head_dim")) { return false; }
    if (!check(have.state_image_bytes == want.state_image_bytes, "state_image_bytes")) {
        return false;
    }
    // Checked before the total would be: a dtype change can leave the total looking plausible while
    // the shard boundaries no longer line up.
    if (!check(have.state_layer_bytes == want.state_layer_bytes, "state_layer_bytes")) {
        return false;
    }
    if (!check(have.state_conv_bytes == want.state_conv_bytes, "state_conv_bytes")) { return false; }
    if (!check(have.state_layers == want.state_layers, "state_layers")) { return false; }
    if (!check(have.features.dflash_local == want.features.dflash_local,
               "features.dflash_local")) {
        return false;
    }
    if (!check(have.features.vision == want.features.vision, "features.vision")) { return false; }
    return true;
}

// The state half only. A state image can be verified on its own terms -- its size, its per-layer
// size, the presence of draft-local regions -- while the KV half (page bytes, dtype, granularity)
// belongs to whoever holds the KV pool. Splitting the check keeps each caller able to verify what it
// can actually see, instead of one side pretending to know the other's geometry.
inline bool matches_state(const CheckpointFingerprint& have, const CheckpointFingerprint& want,
                          std::string& difference) {
    difference.clear();
    const auto check = [&difference](bool equal, const char* field) {
        if (equal) { return true; }
        difference = field;
        return false;
    };
    if (!check(have.state_image_bytes == want.state_image_bytes, "state_image_bytes")) {
        return false;
    }
    if (!check(have.state_layer_bytes == want.state_layer_bytes, "state_layer_bytes")) {
        return false;
    }
    if (!check(have.state_conv_bytes == want.state_conv_bytes, "state_conv_bytes")) { return false; }
    if (!check(have.state_layers == want.state_layers, "state_layers")) { return false; }
    if (!check(have.features.dflash_local == want.features.dflash_local,
               "features.dflash_local")) {
        return false;
    }
    return true;
}
}  // namespace ninfer::core::fingerprint

#endif  // NINFER_CORE_CHECKPOINT_FINGERPRINT_H
