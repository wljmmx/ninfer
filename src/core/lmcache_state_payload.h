#pragma once

// L7-b: the partial trailing chunk that rides in the LMCache state object.
//
// An LMCache key addresses whole chunks, so a prompt whose length is not a multiple of the chunk can
// only be published up to its last complete chunk. Measured on a 474-token prompt: the capture
// covered 467 tokens, the key ended at 384 (3 chunks), and the entry was restored as 384 -- the last
// 83 tokens were prefilled again on every reuse. The KV group cannot carry the rest (the server
// rejects a chunk index past the key with "Inline tensor group 0 chunk index out of range"), but the
// state object is not addressed per chunk: it is one blob per entry. So the tail travels there.
//
//     [u32 valid_tokens, little endian][state image][one tail chunk, zero padded]
//
// The lengths are not repeated inside the object. They are part of the registered shape (see
// LmcacheBackend::configure), so a client with different geometry computes a different object size
// and answers its own lookup with a miss instead of reading the image from the wrong offset. The
// salt carries a payload version for the same reason: an entry written under the old layout must
// not be decoded under this one.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::core::lmcache {

// The prefix every state object starts with.
inline constexpr std::size_t kStateLengthBytes = 4;

// What the state group registers, and what the bridge checks a payload against.
[[nodiscard]] inline std::size_t state_object_bytes(std::size_t image_bytes,
                                                    std::size_t chunk_bytes) {
    return kStateLengthBytes + image_bytes + chunk_bytes;
}

struct StateObjectView {
    std::uint32_t valid_tokens = 0;
    std::span<const std::byte> image;
    // The whole tail region, in page order, zero past what was carried. The reader cuts the pages
    // it needs out of it; the padding is what makes a blind cut safe.
    std::span<const std::byte> tail;
};

namespace detail {

inline void put_u32_le(std::byte* out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<std::byte>((value >> (8 * i)) & 0xFFu);
    }
}

[[nodiscard]] inline std::uint32_t get_u32_le(const std::byte* in) {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(std::to_integer<unsigned>(in[i])) << (8 * i);
    }
    return value;
}

}  // namespace detail

// `tail` may be shorter than chunk_bytes -- at the end of an export it is whatever pages the capture
// actually held -- and the remainder stays zero.
[[nodiscard]] inline std::vector<std::byte> encode_state_object(std::span<const std::byte> image,
                                                                std::uint32_t valid_tokens,
                                                                std::span<const std::byte> tail,
                                                                std::size_t chunk_bytes) {
    std::vector<std::byte> out(state_object_bytes(image.size(), chunk_bytes), std::byte{0});
    detail::put_u32_le(out.data(), valid_tokens);
    if (!image.empty()) {
        std::copy(image.begin(), image.end(), out.begin() + kStateLengthBytes);
    }
    if (!tail.empty()) {
        const std::size_t carried = std::min(tail.size(), chunk_bytes);
        std::copy_n(tail.begin(), carried, out.begin() + kStateLengthBytes + image.size());
    }
    return out;
}

// The size is exact on purpose: [4][image][chunk]. A buffer of any other length was written under
// another geometry, another version, or is not a state object at all -- reading it anyway would
// take image bytes for tail bytes (wrong data, no error), so it is a miss.
[[nodiscard]] inline std::optional<StateObjectView>
decode_state_object(std::span<const std::byte> object, std::size_t image_bytes,
                    std::size_t chunk_bytes) {
    if (image_bytes == 0 || chunk_bytes == 0) { return std::nullopt; }
    if (object.size() != state_object_bytes(image_bytes, chunk_bytes)) { return std::nullopt; }
    StateObjectView view;
    view.valid_tokens = detail::get_u32_le(object.data());
    view.image        = object.subspan(kStateLengthBytes, image_bytes);
    view.tail         = object.subspan(kStateLengthBytes + image_bytes, chunk_bytes);
    return view;
}

}  // namespace ninfer::core::lmcache
