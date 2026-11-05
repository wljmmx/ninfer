#pragma once

// A KV tier the engine can hand blocks to, shaped after LMCache's protocol: a lookup that answers
// "do you have this prefix", a prefetch that starts the fetch early, a retrieve that waits, and a
// store that writes back. The three stages are separate on purpose - a hit that cannot be fetched
// without stalling the step is worse than a miss that was never asked for.
//
// Nothing here is wired into the model yet. This is the seam the KV work (plan section 44, F2/F3)
// and the prefetcher (D1) both attach to; it is header-only so it costs the build nothing until a
// backend and a caller exist.

#include "ninfer/types.h"
#include "core/checkpoint_fingerprint.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <map>
#include <vector>

namespace ninfer::core {

// A prefix is identified by a rolling hash over its token ids plus its length, so two requests that
// share a prefix name the same block. The length is part of the key: a hash collision between
// different lengths would otherwise hand back a block of the wrong size.
struct KvBlockKey {
    std::uint64_t hash   = 0;
    std::uint32_t tokens = 0;

    friend bool operator==(const KvBlockKey&, const KvBlockKey&) = default;
    friend bool operator<(const KvBlockKey& a, const KvBlockKey& b) noexcept {
        return a.hash < b.hash || (a.hash == b.hash && a.tokens < b.tokens);
    }
};

[[nodiscard]] inline std::uint64_t kv_hash_tokens(std::span<const TokenId> tokens,
                                                  std::uint64_t seed = 1469598103934665603ULL) noexcept {
    std::uint64_t hash = seed;
    for (const auto token : tokens) {
        hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(token));
        hash *= 1099511628211ULL;
        hash ^= hash >> 29;
    }
    return hash;
}

[[nodiscard]] inline KvBlockKey kv_block_key(std::span<const TokenId> prefix) noexcept {
    return KvBlockKey{kv_hash_tokens(prefix), static_cast<std::uint32_t>(prefix.size())};
}

enum class KvTierState { miss, fetching, resident };

// What a backend reports for a prefix. `bytes` is the size of the payload a retrieve would fill.
struct KvTierLookup {
    KvTierState state = KvTierState::miss;
    std::size_t bytes = 0;
};

// Q54's prefix index entry: what a longer request needs to find a block it did not store.
//
// A block is named by the WHOLE prompt that stored it, so a request that extends that prompt
// computes a different key and can never name the block, even though every byte it would skip
// is already on disk. This note is the bridge: it records, for one stored block, the identity
// digest of the prefix at the frontier the payload actually covers. A reader walks its own
// prefix digests against these notes and names the block through `block_hash`/`block_tokens`.
//
// The digest is an INDEX, not a proof: a hit is still gated by the payload's own decode,
// frontier and fingerprint validation, and a digest/frontier disagreement with the payload is
// a refusal. `fingerprint` is the 74-byte canonical CheckpointFingerprint encoding so a reader
// can skip foreign-model notes without opening their payloads; core treats it as opaque bytes.
struct KvTierPrefixNote {
    std::uint64_t block_hash   = 0;
    std::uint32_t block_tokens = 0;
    std::uint32_t frontier     = 0;
    std::uint64_t digest_a     = 0;
    std::uint64_t digest_b     = 0;
    std::uint32_t identity_tag = 0;
    static constexpr std::size_t kFingerprintBytes = fingerprint::kEncodedBytes;
    std::array<std::uint8_t, kFingerprintBytes> fingerprint{};
};

class KvTier {
public:
    virtual ~KvTier() = default;

    // Never blocks on IO: answers from the directory of what this tier holds or is fetching.
    [[nodiscard]] virtual KvTierLookup lookup(const KvBlockKey& key) const = 0;
    // Starts the fetch if this tier holds the block and it is not resident yet. Returns false when
    // the tier does not hold it at all.
    virtual bool prefetch(const KvBlockKey& key) = 0;
    // Waits for the payload and copies it out. Returns false when the tier cannot serve it.
    virtual bool retrieve(const KvBlockKey& key, std::span<std::byte> destination) = 0;
    // Writes a payload back so a later request can hit it.
    virtual bool store(const KvBlockKey& key, std::span<const std::byte> source) = 0;

    // Q54: persists / answers the prefix index. A tier that keeps no index answers false and
    // empty, which leaves every request on the whole-prompt key - exactly the pre-index
    // behaviour, so a backend cannot become a new failure mode by ignoring these.
    virtual bool note_prefix(const KvTierPrefixNote& note) {
        (void)note;
        return false;
    }
    [[nodiscard]] virtual std::vector<KvTierPrefixNote> prefix_notes() const { return {}; }
};

// A local-disk tier: one file per block, named by the hash and the token count, with the directory
// kept in memory. This is the third level of plan section 44 (R6b) in its simplest useful form -
// enough to make the lookup/prefetch/retrieve split real and measurable before any remote tier.
//
// Q54's prefix index lives here as one sidecar per stored block, named `pfx-<hash>.<tokens>`,
// rewritten whenever the block is. A sidecar is fixed-width, little-endian and checksummed; a
// file that is short, wrongly versioned or fails its checksum is skipped, so a torn or foreign
// index can only shrink what this tier answers - never break a lookup that used to succeed.
class DiskKvTier final : public KvTier {
public:
    explicit DiskKvTier(std::filesystem::path directory) : directory_(std::move(directory)) {
        std::error_code ec;
        std::filesystem::create_directories(directory_, ec);
        std::lock_guard lock(mutex_);
        if (!std::filesystem::exists(directory_, ec)) { return; }
        for (const auto& entry : std::filesystem::directory_iterator(directory_, ec)) {
            if (!entry.is_regular_file(ec)) { continue; }
            const auto name = entry.path().filename().string();
            const auto dot  = name.find('.');
            if (dot == std::string::npos) { continue; }
            if (name.rfind(kPrefixName, 0) == 0) {
                // Index sidecar, not a block: a bad one is a skipped entry, not an error. The
                // directory is read at startup, so the fallback to "no index" is silent here and
                // the reader prints what it actually had to work with.
                if (KvTierPrefixNote note; parse_note(entry.path(), note)) {
                    prefix_notes_.push_back(note);
                }
                continue;
            }
            const auto hash   = std::strtoull(name.c_str(), nullptr, 16);
            const auto tokens = static_cast<std::uint32_t>(std::strtoul(name.c_str() + dot + 1, nullptr, 10));
            index_[KvBlockKey{hash, tokens}] = entry.path();
        }
    }

    [[nodiscard]] KvTierLookup lookup(const KvBlockKey& key) const override {
        std::lock_guard lock(mutex_);
        const auto it = index_.find(key);
        if (it == index_.end()) { return {}; }
        std::error_code ec;
        const auto bytes = std::filesystem::file_size(it->second, ec);
        if (ec) { return {}; }
        return KvTierLookup{KvTierState::resident, static_cast<std::size_t>(bytes)};
    }

    bool prefetch(const KvBlockKey& key) override {
        // A local file needs no staging: the first read is the prefetch, and it lands in the page
        // cache the retrieve then hits. Kept as a distinct call so a remote tier can do real work.
        return lookup(key).state == KvTierState::resident;
    }

    bool retrieve(const KvBlockKey& key, std::span<std::byte> destination) override {
        std::filesystem::path path;
        {
            std::lock_guard lock(mutex_);
            const auto it = index_.find(key);
            if (it == index_.end()) { return false; }
            path = it->second;
        }
        std::FILE* file = std::fopen(path.string().c_str(), "rb");
        if (file == nullptr) { return false; }
        const auto read = std::fread(destination.data(), 1, destination.size(), file);
        std::fclose(file);
        return read == destination.size();
    }

    bool store(const KvBlockKey& key, std::span<const std::byte> source) override {
        const auto name = block_name(key);
        const auto path = directory_ / name;
        std::FILE* file = std::fopen(path.string().c_str(), "wb");
        if (file == nullptr) { return false; }
        const auto written = std::fwrite(source.data(), 1, source.size(), file);
        std::fclose(file);
        if (written != source.size()) { return false; }
        std::lock_guard lock(mutex_);
        index_[key] = path;
        return true;
    }

    bool note_prefix(const KvTierPrefixNote& note) override {
        // The sidecar is written AFTER the block: a process that dies between the two leaves a
        // block without an index entry (findable by its whole-prompt key, exactly as before),
        // never an entry without its block.
        if (note.frontier == 0 || note.block_tokens == 0) { return false; }
        const auto path = directory_ / (std::string(kPrefixName) +
                                        block_name({note.block_hash, note.block_tokens}));
        std::FILE* file = std::fopen(path.string().c_str(), "wb");
        if (file == nullptr) { return false; }
        const auto bytes = encode_note(note);
        const auto written = std::fwrite(bytes.data(), 1, bytes.size(), file);
        std::fclose(file);
        if (written != bytes.size()) { return false; }
        const std::lock_guard lock(mutex_);
        std::erase_if(prefix_notes_, [&note](const KvTierPrefixNote& existing) {
            return existing.block_hash == note.block_hash &&
                   existing.block_tokens == note.block_tokens;
        });
        prefix_notes_.push_back(note);
        return true;
    }

    [[nodiscard]] std::vector<KvTierPrefixNote> prefix_notes() const override {
        const std::lock_guard lock(mutex_);
        return prefix_notes_;
    }

private:
    static constexpr std::string_view kPrefixName = "pfx-";
    static constexpr std::uint32_t kNoteMagic   = 0x5844494EU;  // "NIDX"
    static constexpr std::uint32_t kNoteVersion = 1U;
    // magic, version, block_hash, block_tokens, frontier, digest_a, digest_b, identity_tag,
    // fingerprint[74], checksum. Fixed width so a torn file is a short read, not a guess.
    static constexpr std::size_t kNoteBytes = 4 + 4 + 8 + 4 + 4 + 8 + 8 + 4 +
                                              KvTierPrefixNote::kFingerprintBytes + 8;

    static void put_u32(std::span<std::uint8_t> out, std::size_t& offset, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            out[offset + static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFU);
        }
        offset += 4;
    }

    static void put_u64(std::span<std::uint8_t> out, std::size_t& offset, std::uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            out[offset + static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFU);
        }
        offset += 8;
    }

    static bool take_u32(std::span<const std::uint8_t> in, std::size_t& offset, std::uint32_t& value) {
        if (offset + 4 > in.size()) { return false; }
        value = 0;
        for (int i = 0; i < 4; ++i) {
            value |= static_cast<std::uint32_t>(in[offset + static_cast<std::size_t>(i)]) << (8U * i);
        }
        offset += 4;
        return true;
    }

    static bool take_u64(std::span<const std::uint8_t> in, std::size_t& offset, std::uint64_t& value) {
        if (offset + 8 > in.size()) { return false; }
        value = 0;
        for (int i = 0; i < 8; ++i) {
            value |= static_cast<std::uint64_t>(in[offset + static_cast<std::size_t>(i)]) << (8U * i);
        }
        offset += 8;
        return true;
    }

    static std::uint64_t note_checksum(std::span<const std::uint8_t> bytes) {
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        for (const auto byte : bytes.subspan(0, kNoteBytes - 8)) {
            hash ^= byte;
            hash *= 0x100000001b3ULL;
        }
        return hash;
    }

    static std::array<std::uint8_t, kNoteBytes> encode_note(const KvTierPrefixNote& note) {
        std::array<std::uint8_t, kNoteBytes> out{};
        const std::span<std::uint8_t> view(out.data(), out.size());
        std::size_t offset = 0;
        put_u32(view, offset, kNoteMagic);
        put_u32(view, offset, kNoteVersion);
        put_u64(view, offset, note.block_hash);
        put_u32(view, offset, note.block_tokens);
        put_u32(view, offset, note.frontier);
        put_u64(view, offset, note.digest_a);
        put_u64(view, offset, note.digest_b);
        put_u32(view, offset, note.identity_tag);
        std::copy(note.fingerprint.begin(), note.fingerprint.end(), view.begin() + offset);
        offset += KvTierPrefixNote::kFingerprintBytes;
        put_u64(view, offset, note_checksum(view));
        return out;
    }

    static bool parse_note(const std::filesystem::path& path, KvTierPrefixNote& note) {
        std::FILE* file = std::fopen(path.string().c_str(), "rb");
        if (file == nullptr) { return false; }
        std::array<std::uint8_t, kNoteBytes> buffer{};
        const auto read = std::fread(buffer.data(), 1, buffer.size(), file);
        std::fclose(file);
        if (read != buffer.size()) { return false; }
        const std::span<const std::uint8_t> view(buffer.data(), buffer.size());
        std::size_t offset = 0;
        std::uint32_t magic = 0, version = 0;
        if (!take_u32(view, offset, magic) || magic != kNoteMagic) { return false; }
        if (!take_u32(view, offset, version) || version != kNoteVersion) { return false; }
        std::uint64_t stored_checksum = 0, computed_checksum = 0;
        computed_checksum = note_checksum(view);
        offset = kNoteBytes - 8;
        if (!take_u64(view, offset, stored_checksum) || stored_checksum != computed_checksum) {
            return false;
        }
        offset = 8;
        if (!take_u64(view, offset, note.block_hash)) { return false; }
        if (!take_u32(view, offset, note.block_tokens)) { return false; }
        if (!take_u32(view, offset, note.frontier)) { return false; }
        if (!take_u64(view, offset, note.digest_a)) { return false; }
        if (!take_u64(view, offset, note.digest_b)) { return false; }
        if (!take_u32(view, offset, note.identity_tag)) { return false; }
        std::copy(view.begin() + offset, view.begin() + offset + KvTierPrefixNote::kFingerprintBytes,
                  note.fingerprint.begin());
        return note.frontier != 0 && note.block_tokens != 0;
    }

    static std::string block_name(const KvBlockKey& key) {
        char buffer[32];
        std::snprintf(buffer, sizeof buffer, "%016llx.%u", static_cast<unsigned long long>(key.hash),
                      key.tokens);
        return buffer;
    }

    std::filesystem::path directory_;
    mutable std::mutex mutex_;
    std::map<KvBlockKey, std::filesystem::path> index_;
    std::vector<KvTierPrefixNote> prefix_notes_;
};

} // namespace ninfer::core

