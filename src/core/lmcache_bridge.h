// The seam between a checkpoint export and LMCache.
//
// The engine exports KV as a flat run of pages and the state image as one blob; the cache wants
// chunks of `pages_per_chunk` pages and one object per group. This translates between the two and
// owns nothing else -- deliberately: no engine types appear in the interface, so the split can be
// tested without a model.
//
// Every shape here is SUPPLIED, never derived. Models differ in layer count, head width, K/V plane
// count (MLA and fused formats have one) and state image dtype, so a formula that divides a byte
// count back into a shape would be a constant for one model and wrong for the next. The only thing
// this file does with the numbers is check that the shape a caller registered actually describes
// the bytes a caller hands it.
#pragma once

#include "core/lmcache_client.h"
#include "core/lmcache_wire.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <span>
#include <string>
#include <vector>

namespace ninfer::core::lmcache {

// Element size of a torch dtype name, as the server spells them. Returns 0 for a name it does not
// know, which turns into a refused configuration rather than a silently wrong byte count.
[[nodiscard]] inline std::size_t dtype_itemsize(const std::string& dtype) {
    if (dtype == "float64") { return 8; }
    if (dtype == "float32") { return 4; }
    if (dtype == "float16" || dtype == "bfloat16") { return 2; }
    if (dtype == "int8" || dtype == "uint8" || dtype == "float8_e4m3fn" ||
        dtype == "float8_e5m2") {
        return 1;
    }
    return 0;
}

[[nodiscard]] inline std::size_t shape_bytes(const std::vector<std::int64_t>& shape) {
    std::size_t total = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0) { return 0; }
        total *= static_cast<std::size_t>(dimension);
    }
    return total;
}

// One LMCache group, exactly as the server will see it. shape is the layout the payload must match;
// kv_size and window_* are registration-only fields the server reads when it sizes the object.
struct GroupSpec {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::int64_t kv_size{2};       // 2 = separate K/V planes, 1 = MLA / fused / indexer
    std::int64_t window_tokens{0}; // 0 means "the whole chunk"
    std::int64_t window_chunks{0}; // 0 means "every chunk of the prefix"
    bool recurrent{false};

    [[nodiscard]] std::size_t object_bytes() const {
        return shape_bytes(shape) * dtype_itemsize(dtype);
    }

    // The only shape that fits every case: one plane, one token, N elements. The engine's Host page
    // layout is plane-major and a quantized format has FOUR planes per layer (K data, K scales, V
    // data, V scales), while the server can only express one or two -- so any shape that names the
    // planes is a shape that will not fit the next format. Bytes are the invariant; this states them
    // and nothing else.
    //
    // uint8 is used when the byte count is not a whole number of the natural element, because the
    // server only checks prod(shape) x itemsize and an arbitrary stride (page records are aligned,
    // not sized to an element) is not always evenly divisible.
    [[nodiscard]] static GroupSpec exact(std::string dtype, std::size_t bytes) {
        std::size_t itemsize = dtype_itemsize(dtype);
        if (itemsize == 0 || bytes == 0 || bytes % itemsize != 0) {
            dtype = "uint8";
            itemsize = 1;
        }
        GroupSpec spec;
        spec.dtype    = std::move(dtype);
        spec.shape    = {1, 1, static_cast<std::int64_t>(bytes / itemsize)};
        spec.kv_size  = 1;
        return spec;
    }
};

struct BridgeConfig {
    std::string model_name;
    // Part of the object key. Two entries that differ only in KV format share nothing else -- model
    // name and tokens are the same -- so a format fingerprint belongs here. Empty means "no
    // isolation", which is only safe while one format is ever used for a given model.
    std::string cache_salt;
    // Tokens in one engine page, and how many pages go into one cache chunk. Both come from the
    // caller: the page size is a property of the engine, the chunk size a choice made against it.
    std::int64_t page_tokens{64};
    std::int64_t pages_per_chunk{2};
    std::size_t page_bytes{0};
    GroupSpec kv;
    GroupSpec state;

    [[nodiscard]] std::int64_t chunk_tokens() const { return page_tokens * pages_per_chunk; }
};

// The cache identity: the token sequence alone. Reusing a sequence hits an entry an earlier run
// stored, and a shape mismatch there reads as a wrong byte count rather than an error, so callers
// must not reuse one either.
struct BridgeKey {
    std::vector<std::int64_t> tokens;
    std::string request_id;
};

class Bridge {
public:
    Bridge() = default;
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    [[nodiscard]] const std::string& error() const { return error_; }
    [[nodiscard]] std::int64_t instance_id() const { return instance_id_; }
    [[nodiscard]] const BridgeConfig& config() const { return config_; }

    // The bridge refuses a configuration whose shapes do not describe the bytes it will be handed.
    // That is a startup error with a visible reason; the alternative is every payload rejected at
    // request time, where the reason is a timeout.
    bool open(const ClientOptions& options, int connections, const BridgeConfig& config,
              std::int64_t instance_id) {
        config_      = config;
        instance_id_ = instance_id;
        if (config_.page_bytes == 0) {
            error_ = "page_bytes is not known yet";
            return false;
        }
        if (config_.page_tokens <= 0 || config_.pages_per_chunk <= 0) {
            error_ = "page_tokens and pages_per_chunk must be positive";
            return false;
        }
        if (dtype_itemsize(config_.kv.dtype) == 0 || dtype_itemsize(config_.state.dtype) == 0) {
            error_ = "unknown dtype in a group layout";
            return false;
        }
        // '@' separates the fields of LMCache's own cache key, so a salt containing one is rejected
        // server-side by raising -- which produces no reply frame, and the client sees only a receive
        // timeout. Refusing here turns a two-minute stall into a startup error.
        if (config_.cache_salt.find('@') != std::string::npos) {
            error_ = "cache_salt must not contain '@' (it is the server's key separator)";
            return false;
        }
        if (config_.kv.kv_size != 1 && config_.kv.kv_size != 2) {
            error_ = "kv_size must be 1 (single plane) or 2 (separate K/V)";
            return false;
        }
        // The server can only express rank 4 (two planes) or rank 3 (one plane). Anything else
        // would silently lose an axis, so refuse rather than register a different shape.
        const std::size_t kv_rank = config_.kv.kv_size == 2 ? 4 : 3;
        if (config_.kv.shape.size() != kv_rank) {
            error_ = "KV shape rank does not match its kv_size";
            return false;
        }
        if (config_.state.kv_size == 2 && config_.state.shape.size() != 4) {
            error_ = "state shape rank does not match its kv_size";
            return false;
        }
        if (config_.state.kv_size == 1 && config_.state.shape.size() != 3) {
            error_ = "state shape rank does not match its kv_size";
            return false;
        }
        if (config_.kv.object_bytes() != chunk_bytes()) {
            error_ = "KV shape does not describe one chunk of pages (" +
                     std::to_string(config_.kv.object_bytes()) + " vs " +
                     std::to_string(chunk_bytes()) + " B)";
            return false;
        }
        if (config_.state.object_bytes() == 0) {
            error_ = "state shape describes no bytes";
            return false;
        }
        if (!pool_.open(options, static_cast<std::size_t>(connections))) {
            error_ = pool_.error();
            return false;
        }
        if (!pool_.register_context(registration())) {
            error_ = pool_.error();
            return false;
        }
        return true;
    }

    void close() { pool_.close(); }

    [[nodiscard]] std::size_t chunk_bytes() const {
        return static_cast<std::size_t>(config_.pages_per_chunk) * config_.page_bytes;
    }

    // How many chunks a run of pages becomes. Not a constant: it follows from the page count.
    [[nodiscard]] std::size_t chunks_for(std::span<const std::byte> kv) const {
        const std::size_t pages = kv.size() / config_.page_bytes;
        return pages / static_cast<std::size_t>(config_.pages_per_chunk);
    }

    bool publish(const BridgeKey& key, std::span<const std::byte> kv, std::span<const std::byte> state,
                 std::int64_t first_chunk = 0, bool with_state = true) {
        if (!pool_.size() || instance_id_ == 0) {
            error_ = "bridge is not open";
            return false;
        }
        if (kv.empty() || kv.size() % config_.page_bytes != 0 ||
            kv.size() % chunk_bytes() != 0) {
            error_ = "KV bytes are not a whole number of chunks";
            return false;
        }
        // A part that carries no state sends an empty payload; the server's grouped path accepts a
        // group with no chunks and writes nothing for it.
        if (with_state && state.size() != config_.state.object_bytes()) {
            error_ = "state image size does not match its registered shape";
            return false;
        }

        std::vector<std::string> chunks;
        chunks.reserve(kv.size() / chunk_bytes());
        for (std::size_t offset = 0; offset < kv.size(); offset += chunk_bytes()) {
            chunks.emplace_back(reinterpret_cast<const char*>(kv.data() + offset), chunk_bytes());
        }
        const std::string payload =
            with_state ? std::string(reinterpret_cast<const char*>(state.data()), state.size())
                       : std::string();

        // One turn is one call on one connection, and that connection is one ZMQ identity, which the
        // server pins to one worker thread (affinity_pool.py). Sharding a store across the pool was
        // tried on 2026-11-21 and does not work: four concurrent stores for the same key leave the
        // first one waiting for a prepare_store reply that never comes ("receive failed: Resource
        // temporarily unavailable"), so the server serializes a key's store regardless of identity.
        // Making this parallel needs a server-side answer for concurrent partial stores of one key,
        // not a client-side fan-out.
        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        const bool ok = lease->store(make_key(key), chunks, config_.kv.shape, payload,
                                     config_.state.shape, first_chunk, with_state);
        if (!ok) { error_ = lease->error(); }
        return ok;
    }

    bool has(const BridgeKey& key) {
        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        const bool ok = lease->has(make_key(key));
        if (!ok) { error_ = lease->error(); }
        return ok;
    }

    // Rebuilds the flat page run the engine's import expects.
    //
    // MATCH and TRANSFER are timed separately and always printed, because they answer different
    // questions and only one of them is a prefetch target: a match is a hash lookup and should be
    // milliseconds, a transfer is the payload and is proportional to bytes. Reporting the sum (as
    // fetch() used to) hides which one a change moved.
    // Streaming restore, in the two calls the import path needs.
    //
    // `restore_header` answers "how much text does this record cover" without moving the payload: one
    // chunk of KV makes the grouped request legal and the state image arrives whole (156 MB). The
    // caller derives the page count from it, reserves the engine's host extent, and then hands that
    // extent to `restore_into`, which streams every part straight into it. The prefix therefore never
    // exists twice -- which is the whole reason this pair exists, because holding it once cost 2.5 GB
    // at 163k on a machine where WSL already had 13 GB.
    // How many parts this payload is fetched in. One rule, used by every restore entry point: one
    // part per ~128 MB of KV, capped at four because the prefetch reservation that serves the shared
    // state object carries four read locks. NINFER_RESTORE_SPLIT=1 forces the single-message path;
    // unset or 0 derives.
    [[nodiscard]] int split_parts(const BridgeKey& key) const {
        const char* split_text = std::getenv("NINFER_RESTORE_SPLIT");
        const int from_env     = split_text != nullptr ? std::atoi(split_text) : 0;
        if (from_env > 1) { return from_env; }
        if (from_env == 1) { return 1; }
        const std::size_t chunk = static_cast<std::size_t>(config_.chunk_tokens());
        if (chunk == 0 || chunk_bytes() == 0) { return 1; }
        const std::size_t bytes = (key.tokens.size() / chunk) * chunk_bytes();
        constexpr std::size_t kPartTarget = 128ULL * 1024 * 1024;
        const std::size_t wanted          = (bytes + kPartTarget - 1) / kPartTarget;
        return static_cast<int>(std::min<std::size_t>(4, std::max<std::size_t>(1, wanted)));
    }

    bool restore_header(const BridgeKey& key, std::vector<std::byte>* state) {
        const CacheKey cache_key = make_key(key);
        auto lease               = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        const auto state_bytes = static_cast<std::int64_t>(config_.state.object_bytes());
        if (state_bytes <= 0) {
            error_ = "state geometry not known";
            return false;
        }
        RestorePayload got;
        const auto began = std::chrono::steady_clock::now();
        if (!lease->has(cache_key)) {
            error_ = lease->error().empty() ? "header lookup missed" : lease->error();
            std::fprintf(stderr, "[lmcache] header: lookup missed (%s)\n", error_.c_str());
            std::fflush(stderr);
            return false;
        }
        if (!lease->fetch(cache_key, &got, 0, 1, 0, state_bytes)) {
            error_ = lease->error().empty() ? "header fetch failed" : lease->error();
            std::fprintf(stderr, "[lmcache] header: fetch failed (%s)\n", error_.c_str());
            std::fflush(stderr);
            return false;
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        std::fprintf(stderr, "[lmcache] header: state %zu B in %.3f s\n", got.state.size(), seconds);
        std::fflush(stderr);
        if (state != nullptr) {
            state->assign(reinterpret_cast<const std::byte*>(got.state.data()),
                          reinterpret_cast<const std::byte*>(got.state.data()) + got.state.size());
        }
        return true;
    }

    // Writes the payload into `dest`, which the caller sized from the page count. Same part sizing,
    // same lookup rules and the same range arithmetic as restore(); only the destination differs.
    bool restore_into(const BridgeKey& key, std::span<std::byte> dest) {
        const CacheKey cache_key = make_key(key);
        int parts                = split_parts(key);
        if (parts > 1) { return restore_split_into(key, cache_key, parts, dest, nullptr); }
        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        RestorePayload got;
        if (!lease->has(cache_key) || !lease->fetch(cache_key, &got)) {
            error_ = lease->error().empty() ? "not cached" : lease->error();
            return false;
        }
        std::size_t written = 0;
        for (const std::string& chunk : got.kv_chunks) {
            if (written + chunk.size() > dest.size()) { break; }
            std::memcpy(dest.data() + written,
                        reinterpret_cast<const std::byte*>(chunk.data()), chunk.size());
            written += chunk.size();
        }
        if (written != dest.size()) {
            error_ = "restore wrote " + std::to_string(written) + " B of " +
                     std::to_string(dest.size()) + " B";
            return false;
        }
        return true;
    }

    bool restore(const BridgeKey& key, std::vector<std::byte>* kv, std::vector<std::byte>* state) {
        const CacheKey cache_key = make_key(key);
        // Probe: the server binds a worker to the ZMQ identity, so connections should run on
        // different workers. Whether that makes two retrieves of the SAME key overlap -- and whether
        // they trip over each other's session state -- is a measurement, not a reading of the code.
        if (const char* fanout_text = std::getenv("NINFER_RESTORE_FANOUT")) {
            const int fanout = std::atoi(fanout_text);
            if (fanout > 1) { return restore_fanout(cache_key, fanout, kv, state); }
        }
        // Split the payload across connections. The server assembles a restore in ~0.15 s and then
        // pushes 181 MB down one socket; the link carries far more than that when asked in parallel.
        //
        // Sizing follows the payload, not the machine, because the two costs pull opposite ways: an
        // extra part costs a connection and re-sends the state image (156 MB, fixed), while leaving a
        // large run on one socket serialises it. Measured against 243f95a: 503 MB over two parts
        // 8.96 s, over four parts 3.49 s (the state image is re-sent per part, so a part that is too
        // big loses more to the socket than it saves in copies); 2.51 GB over four parts 15.9 s.
        // One part per ~128 MB of KV, capped at four -- the prefetch reservation that serves the
        // shared state object carries four read locks (four parts all read it; a fifth would lose).
        const int parts = split_parts(key);
        if (parts > 1) { return restore_split(key, cache_key, parts, kv, state); }
        if (std::getenv("NINFER_LMCACHE_PROBE") != nullptr) {
            std::fprintf(stderr, "[lmcache] restore parts=%d (chunk=%lld bytes=%zu)\n", parts,
                         static_cast<long long>(config_.chunk_tokens()),
                         static_cast<std::size_t>(chunk_bytes()));
            std::fflush(stderr);
        }
        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        RestorePayload got;
        const auto matched_at      = std::chrono::steady_clock::now();
        const bool matched         = lease->has(cache_key);
        const auto transferred_at  = std::chrono::steady_clock::now();
        if (!matched || !lease->fetch(cache_key, &got)) {
            error_ = lease->error().empty() ? "not cached" : lease->error();
            return false;
        }
        const auto done_at = std::chrono::steady_clock::now();
        const double match_seconds =
            std::chrono::duration<double>(transferred_at - matched_at).count();
        const double transfer_seconds =
            std::chrono::duration<double>(done_at - transferred_at).count();
        std::fprintf(stderr, "[lmcache] restore: match %.3f s, transfer %.3f s\n", match_seconds,
                     transfer_seconds);
        if (kv != nullptr) {
            kv->clear();
            for (const std::string& chunk : got.kv_chunks) {
                const auto* bytes = reinterpret_cast<const std::byte*>(chunk.data());
                kv->insert(kv->end(), bytes, bytes + chunk.size());
            }
        }
        if (state != nullptr) {
            state->assign(reinterpret_cast<const std::byte*>(got.state.data()),
                          reinterpret_cast<const std::byte*>(got.state.data()) + got.state.size());
        }
        return true;
    }

private:
    // Probe only: run N retrieves of the same key on N connections at once and report what each
    // branch actually cost. If the branches overlap, the server really is running them on different
    // worker threads (affinity is the ZMQ identity, not the cache key) and a chunk range split is
    // worth building. If they serialize, the total will be the sum and the split buys nothing.
    bool restore_fanout(const CacheKey& cache_key, int fanout, std::vector<std::byte>* kv,
                        std::vector<std::byte>* state) {
        const auto started = std::chrono::steady_clock::now();
        const auto width   = static_cast<std::size_t>(fanout);
        std::vector<RestorePayload> payloads(width);
        std::vector<double> branch_seconds(width, 0.0);
        std::vector<std::future<bool>> tasks;
        tasks.reserve(width);
        for (std::size_t i = 0; i < width; ++i) {
            tasks.push_back(std::async(std::launch::async, [this, &cache_key, &payloads,
                                                            &branch_seconds, i]() {
                auto lease = pool_.acquire();
                if (!lease) { return false; }
                if (!lease->has(cache_key)) { return false; }
                const auto began = std::chrono::steady_clock::now();
                const bool ok    = lease->fetch(cache_key, &payloads[i]);
                branch_seconds[i] =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
                return ok;
            }));
        }
        int succeeded = 0;
        for (auto& task : tasks) {
            if (task.get()) { ++succeeded; }
        }
        const double total =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::fprintf(stderr, "[lmcache] restore fanout=%d ok=%d/%d total %.3f s | branches",
                     fanout, succeeded, fanout, total);
        for (const double seconds : branch_seconds) { std::fprintf(stderr, " %.3f", seconds); }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
        if (succeeded != fanout) {
            error_ = "restore fanout branch failed";
            return false;
        }
        const RestorePayload& got = payloads[0];
        if (kv != nullptr) {
            kv->clear();
            for (const std::string& chunk : got.kv_chunks) {
                const auto* bytes = reinterpret_cast<const std::byte*>(chunk.data());
                kv->insert(kv->end(), bytes, bytes + chunk.size());
            }
        }
        if (state != nullptr) {
            state->assign(reinterpret_cast<const std::byte*>(got.state.data()),
                          reinterpret_cast<const std::byte*>(got.state.data()) + got.state.size());
        }
        return true;
    }

    // Split one restore into `parts` retrieves that run at the same time on different connections.
    //
    // Each part asks the server for its own slice - KV chunk range, state byte range - and carries
    // its own request_id, because the server's prefetch job is consumed once per request_id: with a
    // shared one only the first part finds the prefetched objects and the rest time out (measured:
    // 4/4 branches failed, 5.008 s each). The token stream is untouched, so all parts name the same
    // cache entry; only the session differs.
    bool restore_split(const BridgeKey& key, const CacheKey& base_key, int parts,
                       std::vector<std::byte>* kv, std::vector<std::byte>* state) {
        // Bytes-in-hand wrapper: size the destination, then run the one streaming implementation.
        // Keeping a single body is the point -- a second importer with its own range arithmetic is
        // how the two paths drift apart, which is what the handoff notes kept warning about.
        const auto chunk_tok      = static_cast<std::int64_t>(config_.chunk_tokens());
        const std::int64_t chunks = chunk_tok > 0 ? base_key.end / chunk_tok : 0;
        const std::size_t want_kv =
            chunks > 0 ? static_cast<std::size_t>(chunks) * chunk_bytes() : 0;
        if (kv != nullptr) {
            kv->assign(want_kv, std::byte{0});
            return restore_split_into(key, base_key, parts,
                                      std::span<std::byte>(kv->data(), kv->size()), state);
        }
        return restore_split_into(key, base_key, parts, {}, state);
    }

    bool restore_split_into(const BridgeKey& key, const CacheKey& base_key, int parts,
                            std::span<std::byte> dest, std::vector<std::byte>* state) {
        const auto width      = static_cast<std::size_t>(parts);
        const auto chunk_tok  = static_cast<std::int64_t>(config_.chunk_tokens());
        if (chunk_tok <= 0) {
            error_ = "chunk tokens not known";
            return false;
        }
        const std::int64_t chunks = base_key.end / chunk_tok;
        const auto state_bytes    = static_cast<std::int64_t>(config_.state.object_bytes());
        if (chunks <= 0 || state_bytes <= 0) {
            error_ = "nothing to split";
            return false;
        }

        struct PartResult {
            bool ok = false;
            std::string error;
            std::int64_t chunk_begin = 0;
            std::int64_t chunk_end   = 0;
        };
        std::vector<PartResult> results(width);
        const auto started = std::chrono::steady_clock::now();

        // Fill the destination in place instead of assembling it afterwards. The old shape held the
        // whole payload (2.5 GB at 163k) next to the destination and then inserted chunk by chunk,
        // and the caller still has to copy the destination into engine pages -- three live copies of
        // the same bytes. At 64 GB with WSL holding 18 GB that is what turned a working retrieve into
        // "bad allocation". Pre-sizing means a part writes into its own disjoint slice as it lands.
        const std::size_t want_kv    = static_cast<std::size_t>(chunks) * chunk_bytes();
        const std::size_t want_state = static_cast<std::size_t>(state_bytes);
        if (!dest.empty() && dest.size() != want_kv) {
            error_ = "restore destination is " + std::to_string(dest.size()) + " B, expected " +
                     std::to_string(want_kv) + " B";
            return false;
        }
        if (state != nullptr) { state->assign(want_state, std::byte{0}); }
        const std::size_t part_chunk_bytes = chunk_bytes();

        // How many parts may hold a payload at once. Bandwidth scales with this, peak memory with it
        // too: 4 x 600 MB in flight was enough to fail the allocation above. Measured with the
        // streaming path at 163k: four parts together restore in 8.45 s against 11.9 s for two at a
        // time, so the default is four -- the part count is already capped at four by the read-lock
        // reservation, which means this is the whole payload and no more.
        std::size_t inflight = 4;
        if (const char* text = std::getenv("NINFER_RESTORE_INFLIGHT")) {
            const int parsed = std::atoi(text);
            if (parsed > 0) { inflight = static_cast<std::size_t>(parsed); }
        }

        // Shared-lookup, per wave. With a distinct request_id per part every part looks up the WHOLE
        // key and takes its own read lock on the same objects; the L1 read lock is what makes an
        // object readable to unsafe_read, and only the reservation that took it can consume it, so
        // the parts that lost the race see KEY_NOT_READABLE ("exists but cannot be read").
        //
        // One lookup per wave, shared by that wave's parts, is the unit that works: a measured
        // reservation carries two read locks (four parts: the first two read the state image, the
        // other two were refused), so a wave must not exceed what one lookup provides, and each wave
        // needs its own -- the locks are consumed, not shared for the whole restore.
        const bool share_lookup = std::getenv("NINFER_RESTORE_SHARE_LOOKUP") != nullptr;

        for (std::size_t base = 0; base < width; base += inflight) {
            const std::size_t last = std::min(width, base + inflight);
            const std::size_t wave = base / inflight;
            // Each wave names itself, so its lookup reserves a fresh set of read locks.
            BridgeKey wave_key = key;
            if (share_lookup) { wave_key.request_id += "-w" + std::to_string(wave); }
            if (share_lookup) {
                auto probe = pool_.acquire();
                if (!probe) {
                    error_ = "no connection available for the wave lookup";
                    if (state != nullptr) { state->clear(); }
                    return false;
                }
                const auto began = std::chrono::steady_clock::now();
                const bool ok    = probe->has(make_key(wave_key));
                const double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
                std::fprintf(stderr, "[lmcache] split wave %zu lookup %.3f s (ok=%d%s%s)\n", wave,
                             seconds, ok ? 1 : 0, probe->error().empty() ? "" : " | ",
                             probe->error().c_str());
                std::fflush(stderr);
                if (!ok) {
                    error_ = probe->error().empty() ? "wave lookup missed" : probe->error();
                    if (state != nullptr) { state->clear(); }
                    return false;
                }
            }
            std::vector<std::future<void>> tasks;
            for (std::size_t i = base; i < last; ++i) {
                // Contiguous, exhaustive cover of both ranges: chunk i owns
                // [chunks*i/parts, chunks*(i+1)/parts) and byte i the same fraction of the state
                // object. Together they reassemble exactly.
                const std::int64_t cb =
                    chunks * static_cast<std::int64_t>(i) / static_cast<std::int64_t>(width);
                const std::int64_t ce =
                    chunks * static_cast<std::int64_t>(i + 1) / static_cast<std::int64_t>(width);
                const std::int64_t sb =
                    state_bytes * static_cast<std::int64_t>(i) / static_cast<std::int64_t>(width);
                const std::int64_t se =
                    state_bytes * static_cast<std::int64_t>(i + 1) / static_cast<std::int64_t>(width);
                results[i].chunk_begin = cb;
                results[i].chunk_end   = ce;
                tasks.push_back(std::async(std::launch::async, [this, wave_key, &results, dest, state,
                                                                i, cb, ce, sb, se, share_lookup,
                                                                part_chunk_bytes]() {
                    BridgeKey slice_key = wave_key;
                    if (!share_lookup) { slice_key.request_id += "-p" + std::to_string(i); }
                    const CacheKey slice = make_key(slice_key);
                    PartResult& out      = results[i];
                    const auto wait_began = std::chrono::steady_clock::now();
                    auto lease           = pool_.acquire();
                    const auto has_began  = std::chrono::steady_clock::now();
                    if (!lease) {
                        out.error = "no connection available";
                        return;
                    }
                    const bool matched = share_lookup ? true : lease->has(slice);
                    const auto fetch_began = std::chrono::steady_clock::now();
                    RestorePayload payload;
                    if (!matched) {
                        out.error = "not cached";
                    } else {
                        out.ok = lease->fetch(slice, &payload, cb, ce, sb, se);
                        if (!out.ok) { out.error = lease->error(); }
                    }
                    const auto done = std::chrono::steady_clock::now();
                    // Copy straight into this part's slice, then drop the payload. The ranges are
                    // disjoint by construction, so no lock is needed.
                    if (out.ok && !dest.empty()) {
                        const std::size_t want = static_cast<std::size_t>(ce - cb) * part_chunk_bytes;
                        const std::size_t at   = static_cast<std::size_t>(cb) * part_chunk_bytes;
                        std::size_t written    = 0;
                        for (const std::string& chunk : payload.kv_chunks) {
                            if (written + chunk.size() > want || at + want > dest.size()) { break; }
                            std::memcpy(dest.data() + at + written,
                                        reinterpret_cast<const std::byte*>(chunk.data()),
                                        chunk.size());
                            written += chunk.size();
                        }
                        if (written != want) {
                            out.ok    = false;
                            out.error = "kv slice is " + std::to_string(written) + " B, expected " +
                                        std::to_string(want) + " B";
                        }
                    }
                    if (out.ok && state != nullptr) {
                        const std::size_t want = static_cast<std::size_t>(se - sb);
                        if (payload.state.size() != want || static_cast<std::size_t>(sb) + want > state->size()) {
                            out.ok    = false;
                            out.error = "state slice is " + std::to_string(payload.state.size()) +
                                        " B, expected " + std::to_string(want) + " B";
                        } else if (want != 0) {
                            std::memcpy(state->data() + sb,
                                        reinterpret_cast<const std::byte*>(payload.state.data()), want);
                        }
                    }
                    const auto secs = [](auto from, auto to) {
                        return std::chrono::duration<double>(to - from).count();
                    };
                    std::fprintf(stderr,
                                 "[lmcache] split part %zu: acquire %.3f s, has %.3f s, fetch %.3f s "
                                 "(ok=%d%s%s)\n",
                                 i, secs(wait_began, has_began), secs(has_began, fetch_began),
                                 secs(fetch_began, done), out.ok ? 1 : 0,
                                 out.error.empty() ? "" : " | ", out.error.c_str());
                    std::fflush(stderr);
                }));
            }
            for (auto& task : tasks) { task.get(); }
        }
        const double total =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

        for (std::size_t i = 0; i < width; ++i) {
            if (results[i].ok) { continue; }
            error_ = "restore part " + std::to_string(i) + " failed: " + results[i].error;
            if (state != nullptr) { state->clear(); }
            return false;
        }

        const bool sized = (dest.empty() || dest.size() == want_kv) &&
                           (state == nullptr || state->size() == want_state);
        std::fprintf(stderr,
                     "[lmcache] restore split=%d total %.3f s | kv %zu/%zu B, state %s%s\n",
                     parts, total, dest.size(), want_kv,
                     state != nullptr ? (std::to_string(state->size()) + "/" +
                                         std::to_string(want_state) + " B").c_str()
                                      : "n/a (streamed)",
                     sized ? "" : "  SIZE MISMATCH");
        std::fflush(stderr);
        if (!sized) {
            error_ = "restore split reassembled the wrong size";
            return false;
        }
        return true;
    }

    [[nodiscard]] RegisterContext registration() const {
        RegisterContext context;
        context.instance_id        = instance_id_;
        context.model_name         = config_.model_name;
        context.world_size         = 1;
        context.block_size         = config_.chunk_tokens();
        context.num_layers         = layers_of(config_.kv);
        context.hidden_dim_size    = hidden_of(config_.kv);
        context.dtype_str          = config_.kv.dtype;
        context.use_mla            = config_.kv.kv_size == 1;
        context.num_physical_slots = config_.chunk_tokens();
        context.group_layouts.push_back(to_group_layout(config_.kv));
        context.group_layouts.push_back(to_group_layout(config_.state));
        return context;
    }

    // The server reconstructs a shape from three numbers: [layers, tokens, hidden] for one plane,
    // [2, layers, tokens, hidden] for two. So the registration is the caller's declared shape taken
    // apart along the axes the server knows about -- not a shape invented here from a byte count.
    // A shape with any other rank cannot be expressed and is refused at open().
    static GroupLayout to_group_layout(const GroupSpec& spec) {
        GroupLayout layout;
        const bool two_planes = spec.kv_size == 2;
        const std::size_t expected_rank = two_planes ? 4 : 3;
        layout.num_layers       = spec.shape.size() == expected_rank ? spec.shape[expected_rank - 3] : 0;
        layout.window_tokens    = spec.shape.size() == expected_rank ? spec.shape[expected_rank - 2] : 0;
        layout.hidden_dim_size  = spec.shape.size() == expected_rank ? spec.shape[expected_rank - 1] : 0;
        layout.dtype_str        = spec.dtype;
        layout.kv_size          = spec.kv_size;
        layout.window_chunks    = spec.window_chunks;
        layout.recurrent        = spec.recurrent;
        // Per-block tokens is metadata about the engine's paging, not about the object; the server
        // uses it only to describe the layout, so it follows the chunk.
        layout.tokens_per_block = layout.window_tokens;
        return layout;
    }

    static std::int64_t layers_of(const GroupSpec& spec) {
        const std::size_t expected_rank = spec.kv_size == 2 ? 4 : 3;
        return spec.shape.size() == expected_rank ? spec.shape[expected_rank - 3] : 0;
    }

    static std::int64_t hidden_of(const GroupSpec& spec) {
        return spec.shape.empty() ? 0 : spec.shape.back();
    }

    [[nodiscard]] CacheKey make_key(const BridgeKey& key) const {
        CacheKey out;
        out.model_name = config_.model_name;
        out.world_size = 1;
        out.worker_id  = 0;
        // Only complete chunks exist as far as the server is concerned: get_hashes() asserts that
        // the range ends on a chunk boundary, and a request that trips the assert produces no reply
        // frame at all. Truncating here is the difference between a miss and a stall.
        const std::size_t chunk = static_cast<std::size_t>(config_.chunk_tokens());
        const std::size_t complete = chunk == 0 ? 0 : key.tokens.size() / chunk * chunk;
        std::vector<std::int64_t> kept(key.tokens.begin(),
                                       key.tokens.begin() + static_cast<std::ptrdiff_t>(complete));
        out.token_bytes = pack_tokens(kept);
        out.start       = 0;
        out.end         = static_cast<std::int64_t>(complete);
        out.request_id  = key.request_id;
        out.cache_salt  = config_.cache_salt;
        return out;
    }

    Pool pool_;
    BridgeConfig config_{};
    std::int64_t instance_id_{0};
    std::string error_;
};

}  // namespace ninfer::core::lmcache
