// P1-b: chunk hashing and the "how much can I skip" answer, both pure CPU.
//
// The tests that matter here are the chain and the tail: a prompt whose tail differs must still hit
// its shared head, and a partial final chunk must never be counted as reusable.

#include "core/chunk_hasher.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    int failures = 0;
    const auto expect = [&](bool condition, const char* what) {
        std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
        if (!condition) { ++failures; }
    };

    using ninfer::core::chunk_hashes;
    using ninfer::core::chunk_hash;
    using ninfer::core::reusable_prefix_tokens;
    using ninfer::core::PrefixIndex;

    const std::vector<std::uint32_t> head{1, 2, 3, 4};
    const std::vector<std::uint32_t> same_head{1, 2, 3, 4};
    const std::vector<std::uint32_t> other_head{1, 2, 9, 9};

    expect(chunk_hash(0, head) == chunk_hash(0, same_head), "the same tokens hash the same");
    expect(chunk_hash(0, head) != chunk_hash(0, other_head), "different tokens hash differently");
    expect(chunk_hash(7, head) != chunk_hash(0, head), "the parent hash is part of the chain");

    {
        const std::vector<std::uint32_t> prompt{1, 2, 3, 4, 5, 6, 7, 8};
        const auto hashes = chunk_hashes(prompt, 4);
        expect(hashes.size() == 2, "8 tokens at chunk 4 is two chunks");
    }
    {
        const std::vector<std::uint32_t> prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        const auto hashes = chunk_hashes(prompt, 4);
        expect(hashes.size() == 2, "a 2-token tail is not a reusable chunk");
    }
    {
        const std::vector<std::uint32_t> a{1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<std::uint32_t> b{1, 2, 3, 4, 8, 7, 6, 5};
        const auto ha = chunk_hashes(a, 4);
        const auto hb = chunk_hashes(b, 4);
        expect(ha[0] == hb[0] && ha[1] != hb[1], "a shared head hits, a different tail does not");
    }

    {
        PrefixIndex index;
        const std::vector<std::uint32_t> prompt{1, 2, 3, 4, 5, 6, 7, 8};
        auto hashes = chunk_hashes(prompt, 4);
        expect(reusable_prefix_tokens(prompt, 4, index) == 0, "nothing stored means nothing to skip");

        index.record(hashes[0]);
        expect(reusable_prefix_tokens(prompt, 4, index) == 4, "one stored chunk skips 4 tokens");

        index.record(hashes[1]);
        expect(reusable_prefix_tokens(prompt, 4, index) == 8, "two stored chunks skip 8 tokens");

        index.forget(hashes[0]);
        expect(reusable_prefix_tokens(prompt, 4, index) == 0,
               "losing the first chunk makes the second unusable: prefixes are contiguous");

        index.record(hashes[0], false);
        expect(reusable_prefix_tokens(prompt, 4, index) == 0, "a half-written chunk is not skipped");
    }

    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
