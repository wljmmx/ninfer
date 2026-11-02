// L7-b: the state object's payload -- [u32 valid_tokens][state image][tail chunk].
//
// The tail is the part of the KV run the cache key cannot address, so it rides in the object that is
// not chunked. What has to hold, and is checked here without a server: the length survives, the
// image lands where the reader expects it, a short tail is zero padded to a whole chunk, and any
// object whose size is not exactly [4][image][chunk] is refused rather than decoded at the wrong
// offsets -- an entry from another geometry must be a miss, never silently wrong KV.

#include "core/lmcache_state_payload.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

using ninfer::core::lmcache::decode_state_object;
using ninfer::core::lmcache::encode_state_object;
using ninfer::core::lmcache::kStateLengthBytes;
using ninfer::core::lmcache::state_object_bytes;

std::vector<std::byte> bytes_of(std::size_t size, unsigned char seed) {
    std::vector<std::byte> out(size);
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = static_cast<std::byte>((seed + i) & 0xFF);
    }
    return out;
}

}  // namespace

int main() {
    int failures = 0;
    const auto expect = [&](bool condition, const char* what) {
        std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
        if (!condition) { ++failures; }
    };

    // A proportional copy of the live geometry: 64 tokens per page, two pages per chunk, a 474-token
    // prompt whose capture stopped at 467 (8 pages) while the key addressed 3 chunks = 384 tokens.
    constexpr std::size_t kPageTokens    = 64;
    constexpr std::size_t kPagesPerChunk = 2;
    constexpr std::size_t kChunkTokens   = kPageTokens * kPagesPerChunk;
    constexpr std::size_t kPageBytes     = 4096;
    constexpr std::size_t kChunkBytes    = kPageBytes * kPagesPerChunk;
    constexpr std::size_t kImageBytes    = 1000;
    constexpr std::uint32_t kValid       = 467;
    constexpr std::size_t kCovered       = 3 * kChunkTokens;

    expect(state_object_bytes(kImageBytes, kChunkBytes) ==
               kStateLengthBytes + kImageBytes + kChunkBytes,
           "the object is [4][image][chunk] with nothing else");

    const auto image = bytes_of(kImageBytes, 0x11);
    const auto tail  = bytes_of(kChunkBytes, 0x77);

    {
        const auto object = encode_state_object(image, kValid, tail, kChunkBytes);
        expect(object.size() == state_object_bytes(kImageBytes, kChunkBytes),
               "the encoder writes exactly the registered size");

        // The header is little endian BY CONSTRUCTION, not by the host's order: an entry written by
        // one build has to decode under the next one.
        expect(object[0] == std::byte{0xD3} && object[1] == std::byte{0x01} &&
                   object[2] == std::byte{0x00} && object[3] == std::byte{0x00},
               "467 (= 0x1D3) is written as D3 01 00 00 (little endian)");

        const auto view = decode_state_object(object, kImageBytes, kChunkBytes);
        expect(view.has_value(), "an object of the registered size decodes");
        if (view) {
            expect(view->valid_tokens == kValid, "valid_tokens survives the round trip");
            expect(view->image.size() == kImageBytes &&
                       std::equal(view->image.begin(), view->image.end(), image.begin()),
                   "the image comes back byte for byte");
            expect(view->tail.size() == kChunkBytes &&
                       std::equal(view->tail.begin(), view->tail.end(), tail.begin()),
                   "the tail comes back byte for byte");
            expect(view->valid_tokens - kCovered == 83,
                   "the tail is what the key could not address");
            expect(view->valid_tokens <= kCovered + kChunkTokens,
                   "the claimed length still fits in the carried chunk");
        }
    }

    // A tail shorter than the region (the last chunk of an export is usually part filled) is padded
    // with zeros, not left uninitialised: the reader cuts whole pages out of the region blind.
    {
        const auto short_tail = bytes_of(kPageBytes, 0x22);
        const auto object     = encode_state_object(image, 400, short_tail, kChunkBytes);
        const auto view       = decode_state_object(object, kImageBytes, kChunkBytes);
        expect(view.has_value() && view->valid_tokens == 400,
               "a short tail does not disturb the length");
        if (view) {
            expect(view->tail.size() == kChunkBytes, "the tail region is always a whole chunk");
            expect(std::equal(short_tail.begin(), short_tail.end(), view->tail.begin()),
                   "the carried pages sit at the front of the region");
            expect(std::all_of(view->tail.begin() + static_cast<std::ptrdiff_t>(kPageBytes),
                               view->tail.end(),
                               [](std::byte b) { return b == std::byte{0}; }),
                   "the rest of the region is zero");
        }
    }

    // No tail at all: the object still has the region, all zero, so the size never varies with the
    // length -- that would be a second, redundant declaration of geometry.
    {
        const auto object = encode_state_object(image, kCovered, {}, kChunkBytes);
        const auto view   = decode_state_object(object, kImageBytes, kChunkBytes);
        expect(view.has_value() && view->valid_tokens == kCovered,
               "a chunk-aligned length round-trips with no tail");
        expect(view && std::all_of(view->tail.begin(), view->tail.end(),
                                   [](std::byte b) { return b == std::byte{0}; }),
               "an empty tail region is zero");
    }

    // The image ends exactly where the tail begins. A one-byte slippage here would restore the image
    // from the wrong offset, which reads as subtly wrong text rather than as a failure.
    {
        const auto object    = encode_state_object(image, kValid, tail, kChunkBytes);
        const auto image_end = kStateLengthBytes + kImageBytes;
        expect(object[image_end - 1] == image.back() && object[image_end] == tail.front(),
               "the image and the tail meet at 4 + image_bytes");
    }

    // Anything that is not exactly the registered size is a miss. The sizes checked are the ones a
    // neighbouring geometry produces: an old object (no header, no tail), an object with a different
    // image, and an object with a different chunk.
    {
        const auto object = encode_state_object(image, kValid, tail, kChunkBytes);
        expect(!decode_state_object(object, kImageBytes + 1, kChunkBytes).has_value(),
               "a different image size is refused");
        expect(!decode_state_object(object, kImageBytes, kChunkBytes * 2).has_value(),
               "a different chunk size is refused");
        expect(!decode_state_object(std::span<const std::byte>(object).first(object.size() - 1),
                                    kImageBytes, kChunkBytes)
                    .has_value(),
               "a truncated object is refused");
        std::vector<std::byte> longer = object;
        longer.push_back(std::byte{0});
        expect(!decode_state_object(longer, kImageBytes, kChunkBytes).has_value(),
               "a longer object is refused");
        expect(!decode_state_object(object, 0, kChunkBytes).has_value(),
               "an object with no image is refused");
        expect(!decode_state_object(object, kImageBytes, 0).has_value(),
               "an object with no tail region is refused");
    }

    // valid_tokens is the reader's business, not the codec's: zero and a length past the prompt are
    // rejected in LmcacheBackend::fetch. What the codec promises is that the number it hands back is
    // the number it was given, including zero.
    {
        const auto object = encode_state_object(image, 0, {}, kChunkBytes);
        const auto view   = decode_state_object(object, kImageBytes, kChunkBytes);
        expect(view.has_value() && view->valid_tokens == 0,
               "a zero length is carried verbatim (the caller decides it is a miss)");
    }

    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
