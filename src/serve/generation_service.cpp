#include <charconv>
#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "serve/translate.h"

#if NINFER_HAVE_LMCACHE
#include "core/controller_mode.h"
#include "core/lmcache_bridge.h"
#include "core/lmcache_layout.h"
#include "core/lmcache_prefix_registry.h"
#include "core/lmcache_state_payload.h"
#include "core/lmcache_wal.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <thread>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::serve {

#if NINFER_HAVE_LMCACHE
namespace {
// The tag the program publishes is the KvCacheStorage numbering. Naming it in the log is what makes
// "which format was this entry stored under" answerable without decoding the key.
[[nodiscard]] std::string_view kv_format_name(std::uint32_t tag) {
    switch (tag) {
    case 0: return "bf16";
    case 1: return "int8-g64";
    case 2: return "fp8-e4m3";
    case 3: return "nvfp4";
    case 4: return "fp8k-nvfp4v";
    case 5: return "rk8v4";
    case 6: return "rk4v4";
    case 7: return "rk4v4-e8";
    case 8: return "rk2v4-e8";
    case 9: return "rk4v2-e8";
    default: return "unknown";
    }
}

// The engine's whole view of the cache: one lookup, no transport types. Kept separate from Bridge so
// the engine side never sees a key, a connection or a chunk.
class LmcacheBackend final : public ninfer::CheckpointCacheBackend {
public:
    // Second step of a two-step handoff. The backend has to exist before the engine because the
    // engine takes it in its options; it cannot be configured until the pools exist, because it
    // needs page_bytes and the image size. fetch() stays silent until this succeeds, so nothing is
    // ever registered against a zero-size layout.
    bool configure(const ninfer::CacheGeometry& geometry, const std::string& model_name,
                   int connections) {
        if (geometry.page_bytes == 0 || geometry.page_tokens <= 0 ||
            geometry.state_image_bytes == 0) {
            error_ = "geometry is not ready";
            return false;
        }
        // A page count is meaningless without saying what a page holds. Refusing here is the
        // difference between a startup line that names the format and a key that cannot.
        if (geometry.kv_planes == 0) {
            error_ = "KV plane count is not known";
            return false;
        }
        ninfer::core::lmcache::BridgeConfig config;
        config.model_name = model_name;
        config.page_tokens = geometry.page_tokens;
        // The chunk is the cache's addressing unit, and it is a SERVER setting (--chunk-size). The
        // server does not expose it on any endpoint (its chunk_size_provider only feeds coordinator
        // registration), so the two have to be configured to agree. Two pages is the default the
        // restore path was measured at; NINFER_LMCACHE_CHUNK_TOKENS overrides it, and the value is
        // logged so a mismatch is visible without reading the code.
        std::int64_t chunk_tokens = geometry.page_tokens * 2;
        if (const char* value = std::getenv("NINFER_LMCACHE_CHUNK_TOKENS")) {
            const auto parsed = std::strtoll(value, nullptr, 10);
            if (parsed > 0) { chunk_tokens = parsed; }
        }
        if (chunk_tokens <= 0 || chunk_tokens % geometry.page_tokens != 0) {
            error_ = "chunk tokens must be a whole number of pages";
            return false;
        }
        config.pages_per_chunk = chunk_tokens / geometry.page_tokens;
        config.page_bytes      = geometry.page_bytes;
        // Opaque bytes, one plane (v6 1p): a quantized format has four physical planes per layer and
        // the server can express two, so the shape states the byte count and nothing else.
        config.kv =
            ninfer::core::lmcache::GroupSpec::exact("uint8", 2 * geometry.page_bytes);
        // L7-b: the state object is where a partial trailing chunk travels, because the key addresses
        // whole chunks only and the KV group is sized to a whole chunk per object:
        //   [u32 valid_tokens][state image][one tail chunk, zero padded to chunk_bytes]
        // Without it a 474-token prompt was restored as 384. The extra 4 + chunk_bytes are part of
        // the registered shape, which is also what publish() checks the payload against.
        const std::size_t chunk_bytes =
            static_cast<std::size_t>(config.pages_per_chunk) * geometry.page_bytes;
        config.state = ninfer::core::lmcache::GroupSpec::exact(
            "uint8",
            ninfer::core::lmcache::state_object_bytes(geometry.state_image_bytes, chunk_bytes));
        config.state.window_chunks = 1;
        config.state.recurrent     = true;
        // The key is the model and the tokens; the geometry is what changes when the format or the
        // page layout does, so it is what isolates one from the other. Sizes alone are not enough:
        // a 4-bit layout at one head width and an 8-bit one at half of it can both describe 136 B
        // per vector, and then two formats would share a key and one would answer for the other.
        // The tag and the plane count close that; both are read from the live layout.
        // ':' and not '@': LMCache forbids '@' in a salt because it separates the fields of its own
        // cache key, and it answers a violating request by raising -- no reply frame at all, so the
        // client's only symptom is a receive timeout. Measured: a two-minute stall.
        // ':l7b1' is the payload version: the state object now carries a length header and a tail
        // chunk, so an entry written by the old client must not be read by this one (and vice versa).
        // Naming it in the salt is what keeps the two apart without inspecting a payload.
        config.cache_salt = model_name + ":" + std::to_string(geometry.page_bytes) + ":" +
                            std::to_string(geometry.page_tokens) + ":" +
                            std::to_string(geometry.state_image_bytes) + ":k" +
                            std::to_string(geometry.kv_dtype_tag) + "p" +
                            std::to_string(geometry.kv_planes) + ":l7b1";
        page_bytes_    = geometry.page_bytes;
        page_tokens_   = static_cast<std::size_t>(geometry.page_tokens);
        image_bytes_   = geometry.state_image_bytes;
        chunk_bytes_   = chunk_bytes;
        // A lookup resolves complete chunks only. Below one chunk there is nothing to ask about --
        // and the server answers such a request with no frames at all, so the client sits on its
        // receive timeout. Measured: a 22-token prompt held a request for two minutes.
        chunk_tokens_ = static_cast<std::size_t>(chunk_tokens);

        ninfer::core::lmcache::ClientOptions client;
        if (const char* endpoint = std::getenv("NINFER_LMCACHE_ENDPOINT");
            endpoint != nullptr && *endpoint != '\0') {
            client.endpoint = endpoint;
        }
        // Restore is best-effort: a healthy round trip is about 1.3 s, so a much longer wait only
        // means the server is not answering -- and there it answers with silence, not an error. The
        // default 120 s would turn one bad payload into a two-minute request.
        client.receive_timeout_ms = 5000;
        // A fresh id per process start: the server does not replace the layout of an id it already
        // knows.
        const std::int64_t instance =
            static_cast<std::int64_t>(std::chrono::system_clock::now().time_since_epoch().count());
        bridge_ = std::make_unique<ninfer::core::lmcache::Bridge>();
        if (!bridge_->open(client, connections, config, instance)) {
            error_ = bridge_->error();
            bridge_.reset();
            return false;
        }
        ready_.store(true);
        // 13.4: jobs the previous process accepted and did not confirm. Replaying after ready means
        // the same publish path the writer thread uses; a record is removed only after publish
        // returns, so a crash during replay leaves it for the next start.
        if (const char* wal_dir = std::getenv("NINFER_LMCACHE_WAL_DIR");
            wal_dir != nullptr && *wal_dir != '\0' && wal_.open(wal_dir)) {
            const auto pending = wal_.pending();
            std::size_t replayed = 0;
            for (const auto& record : pending) {
                if (!publish_job(record.tokens, record.checkpoint)) { continue; }
                wal_.complete(record.path);
                ++replayed;
            }
            if (replayed != 0) {
                std::fprintf(stderr, "[lmcache] wal: replayed %zu pending write(s)\n", replayed);
                std::fflush(stderr);
            }
        }
        return true;
    }

    [[nodiscard]] const std::string& error() const { return error_; }
    [[nodiscard]] std::size_t chunk_tokens() const { return chunk_tokens_; }

    [[nodiscard]] std::optional<ninfer::CachedCheckpoint>
    fetch(std::span<const TokenId> prompt_tokens) override {
        if (!ready_.load() || bridge_ == nullptr || prompt_tokens.empty()) { return std::nullopt; }
        // Fewer tokens than one chunk means no complete chunk to look up. The server replies to such
        // a lookup with nothing at all, so asking costs a full receive timeout.
        if (chunk_tokens_ != 0 && prompt_tokens.size() < chunk_tokens_) { return std::nullopt; }

        // Q54: the cache key is the whole prompt, so a request that extends a stored prefix computes
        // a different one. Query the request's own prefixes instead, longest first. The registry
        // holds only lengths this process has stored or restored, so this is a handful of lookups,
        // not a scan of every possible frontier.
        std::vector<std::size_t> attempts;
        attempts.push_back(prompt_tokens.size());
        std::vector<std::int64_t> sequence(prompt_tokens.begin(), prompt_tokens.end());
        for (const auto frontier : registry_.candidates(sequence, 3)) {
            if (frontier >= chunk_tokens_ && frontier < prompt_tokens.size()) {
                attempts.push_back(frontier);
            }
        }

        for (const std::size_t length : attempts) {
            const auto attempt = prompt_tokens.first(length);
            if (auto restored = restore_prefix(attempt)) {
                // The run that came from the cache is not captured: record its coverage so the next
                // request in this process can find it without another round trip.
                const std::vector<std::int64_t> prefix(attempt.begin(), attempt.end());
                registry_.record(prefix, restored->frontier);
                if (restored->valid_tokens != 0 && restored->valid_tokens <= attempt.size()) {
                    registry_.record(prefix, restored->valid_tokens);
                }
                return restored;
            }
        }
        return std::nullopt;
    }

    // One lookup at one frontier. Split out of fetch() because Q54 makes a hit consist of several
    // candidate lengths; everything below is the single-prompt behaviour that was measured.
    [[nodiscard]] std::optional<ninfer::CachedCheckpoint>
    restore_prefix(std::span<const TokenId> prompt_tokens) {
        ninfer::core::lmcache::BridgeKey key;
        key.tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
        // The server keys its token session by request_id and does not reset the rolling hash when
        // the token buffer is replaced, so this has to be unique per call, not per process.
        key.request_id = "svc-" + std::to_string(instance_tag_) + "-" + std::to_string(++sequence_);

        std::vector<std::byte> kv;
        std::vector<std::byte> state;
        // Timed, because "the cache took 0.9 s" and "the cache took 0.2 s" lead to different next
        // steps, and the serve log's queue_wait_seconds covers the whole admission path (fetch, import,
        // Device copies) without saying how much of it is the cache.
        const auto fetch_started = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> guard(bridge_mutex_);
        ++fetches_;
        if (!bridge_->restore(key, &kv, &state)) {
            // Q70/13.4: the first half of the hit-rate shutdown. A lookup that reached the bridge
            // and came back empty is a miss; an early return above is not counted, because it never
            // asked.
            ++misses_;
            return std::nullopt;
        }
        ++hits_;
        const double fetch_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - fetch_started).count();
        std::fprintf(stderr, "[lmcache] fetch %zu KV byte(s) + %zu state byte(s) in %.3f s\n",
                     kv.size(), state.size(), fetch_seconds);
        if (page_bytes_ == 0 || kv.size() % page_bytes_ != 0) { return std::nullopt; }

        const std::size_t chunk_bytes_total = chunk_bytes_;
        if (chunk_bytes_total == 0) { return std::nullopt; }
        // L7-b: [u32 valid_tokens][state image][tail chunk]. The size is exact, so an object that
        // does not have it is not one this client wrote (or one written before the salt version
        // changed) -- a miss, not a guess at where the image starts.
        const auto object =
            ninfer::core::lmcache::decode_state_object(state, image_bytes_, chunk_bytes_total);
        if (!object) { return std::nullopt; }
        const std::uint32_t valid_tokens = object->valid_tokens;
        // Zero is "no length recorded", and a length longer than the prompt describes a prefix this
        // request does not have. Adopting either would claim coverage that is not there.
        if (valid_tokens == 0 || valid_tokens > prompt_tokens.size()) { return std::nullopt; }

        ninfer::CachedCheckpoint out;
        out.page_bytes = page_bytes_;
        // Derived from the bytes, never from the prompt. The record may cover less than the prompt
        // (a shorter capture), and a frontier that disagrees with the page count produces an address
        // whose coverage and pages do not line up -- that is wrong data, not a miss. Page count and
        // frontier computed from the same number cannot disagree.
        // Q64 2d-2b-4: the same arithmetic the store side uses, from the same header. The restore side
        // used to re-derive the frontier from the byte count and re-do the page rounding by hand, which
        // is how "coverage" could mean 384 here and 467 there in one build.
        const auto plan = ninfer::core::lmcache::plan_restore(
            kv.size(), valid_tokens, object->tail.size(), chunk_bytes_total, chunk_tokens_,
            page_bytes_, page_tokens_);
        if (std::getenv("NINFER_KVT_DEBUG") != nullptr) {
            std::fprintf(stderr,
                         "[kvt-debug] restore prompt=%zu valid=%u kv_bytes=%zu chunks=%zu "
                         "frontier=%zu tail_pages=%zu tail_bytes=%zu page=%zu\n",
                         prompt_tokens.size(), valid_tokens, kv.size(), plan.chunks, plan.frontier,
                         plan.tail_pages, object->tail.size(), page_bytes_);
            std::fflush(stderr);
        }
        if (plan.chunks == 0) { return std::nullopt; }
        out.frontier = static_cast<std::uint32_t>(plan.frontier);
        out.kv_bytes = std::move(kv);
        if (plan.tail_pages != 0) {
            const auto tail = object->tail.first(plan.tail_pages * page_bytes_);
            out.kv_bytes.insert(out.kv_bytes.end(), tail.begin(), tail.end());
        }
        out.valid_tokens = valid_tokens;
        // One element: the image travels whole (v6 1n). The sharded layout exists for transport
        // sizing, not for import. The header and the tail stay here -- neither is part of the image.
        out.state_shards.emplace_back(reinterpret_cast<const char*>(object->image.data()),
                                      object->image.size());
        return out;
    }

    // port (sm_89): Q64 2d-2h. The publish is a one-second round trip (measured: 1.02 s transfer for a
    // 469-token checkpoint), and it used to run inside the request that produced it. A cache write
    // does not make the request any better, so with NINFER_LMCACHE_ASYNC_WRITE it goes to a writer
    // thread and the request returns as soon as the checkpoint is copied.
    //
    // The copy is the price: the checkpoint owns its bytes, so handing it to the thread costs one
    // memcpy of kv_bytes (~2.8 MB, about a millisecond). That is the reason this can be done without
    // touching the engine's buffers or their lifetimes.
    //
    // Switch off (the default) is the old call, on the old thread, byte for byte.
    void store(std::span<const TokenId> prompt_tokens,
               const ninfer::CachedCheckpoint& checkpoint) override {
        if (!ready_.load() || bridge_ == nullptr || prompt_tokens.empty()) { return; }
        if (write_disabled_.load(std::memory_order_relaxed)) { return; }
        if (checkpoint.page_bytes == 0 || checkpoint.kv_bytes.empty()) { return; }
        ++stores_;
        // Q70/13.4: the second half of the hit-rate shutdown. A write costs bytes and a round trip;
        // it only pays if something reads it back. Eight lookups with zero restores is not a ratio
        // threshold, it is the state "this process has never been served by the cache"; continuing
        // to fill it in that state is cost with no measured return. The switch is opt-in so the
        // default cache still fills before its first reader arrives.
        if (std::getenv("NINFER_LMCACHE_AUTO_OFF") != nullptr && fetches_.load() >= 8 &&
            hits_.load() == 0) {
            bool expected = false;
            if (write_disabled_.compare_exchange_strong(expected, true)) {
                std::fprintf(stderr,
                             "[lmcache] writes disabled: %llu fetch(es), 0 hit(s), %llu store(s) - "
                             "the cache has never answered a lookup\n",
                             static_cast<unsigned long long>(fetches_.load()),
                             static_cast<unsigned long long>(stores_.load()));
                std::fflush(stderr);
            }
            return;
        }
        std::call_once(writer_once_, [this] {
            // Read once, at the first store: a cache either writes behind the request or in front of
            // it, and flipping mid-run would make the two indistinguishable in a log.
            async_write_ = std::getenv("NINFER_LMCACHE_ASYNC_WRITE") != nullptr;
            if (async_write_) { writer_ = std::thread([this] { write_loop(); }); }
        });
        if (!async_write_) {
            const auto wal_path = wal_.append(prompt_tokens, checkpoint);
            if (publish_job(prompt_tokens, checkpoint)) { wal_.complete(wal_path); }
            return;
        }
        {
            std::lock_guard<std::mutex> guard(write_mutex_);
            // A writer that cannot keep up must not grow without bound: a dropped write costs one
            // future miss, while an unbounded queue costs the process. Counted, never silent.
            if (writes_.size() >= kMaxPendingWrites) {
                ++dropped_writes_;
                return;
            }
            // 13.4: the record is written before the job becomes queued, so "accepted" and
            // "recoverable" are the same instant. A crash between here and the publish leaves a
            // file that the next start replays.
            writes_.push_back(Job{std::vector<TokenId>(prompt_tokens.begin(), prompt_tokens.end()),
                                  checkpoint, wal_.append(prompt_tokens, checkpoint)});
        }
        write_cv_.notify_one();
    }

    // Q64: the explicit exit. This is the same body the destructor used to run inline; the only
    // change is that it is now reachable without ending the process, which is what verifying
    // "writer stopped" required. Draining is still not attempted: a stopped writer drops its last
    // queued writes by design, and the counters say how many.
    void stop() override {
        std::size_t pending = 0;
        {
            std::lock_guard<std::mutex> guard(write_mutex_);
            if (write_stop_) { return; }
            pending     = writes_.size();
            write_stop_ = true;
        }
        write_cv_.notify_all();
        if (writer_.joinable()) { writer_.join(); }
        // Printed once, on the way out: a writer that silently lost its last writes looks exactly
        // like a cache that had nothing to say.
        if (async_write_ && (written_ != 0 || dropped_writes_ != 0 || pending != 0)) {
            std::fprintf(stderr, "[lmcache] writer stopped: written=%llu dropped=%llu pending=%llu\n",
                         static_cast<unsigned long long>(written_),
                         static_cast<unsigned long long>(dropped_writes_),
                         static_cast<unsigned long long>(pending));
            std::fprintf(stderr, "[lmcache] lookups fetches=%llu hits=%llu misses=%llu stores=%llu\n",
                         static_cast<unsigned long long>(fetches_.load()),
                         static_cast<unsigned long long>(hits_.load()),
                         static_cast<unsigned long long>(misses_.load()),
                         static_cast<unsigned long long>(stores_.load()));
            std::fflush(stderr);
        }
    }

    ~LmcacheBackend() override { stop(); }

    struct Job {
        std::vector<TokenId> tokens;
        ninfer::CachedCheckpoint checkpoint;
        std::filesystem::path wal_path;
    };

    void write_loop() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(write_mutex_);
                write_cv_.wait(lock, [this] { return write_stop_ || !writes_.empty(); });
                if (writes_.empty()) {
                    if (write_stop_) { return; }
                    continue;
                }
                job = std::move(writes_.front());
                writes_.pop_front();
            }
            if (publish_job(job.tokens, job.checkpoint)) {
                wal_.complete(job.wal_path);
                ++written_;
            }
        }
    }

    // The body that used to be store(): unchanged, so the async path is the same code, on another
    // thread. Returns whether the server accepted the publish, which is also whether the WAL record
    // may be removed.
    bool publish_job(std::span<const TokenId> prompt_tokens,
                     const ninfer::CachedCheckpoint& checkpoint) {
        if (!ready_.load() || bridge_ == nullptr || prompt_tokens.empty()) { return false; }
        if (checkpoint.page_bytes == 0 || checkpoint.kv_bytes.empty()) { return false; }

        // Only whole chunks can be addressed, so the key and the KV group stay chunk-aligned. The
        // partial trailing chunk is not thrown away any more: it rides in the state object (L7-b).
        //
        // Two counts decide the payload, and they are not the same number: how many chunks the export
        // holds, and how many the prompt can address. The key is truncated from the prompt, so the
        // KV payload has to be cut to the same count -- otherwise the envelope carries more tensors
        // than the key has chunks and the server refuses the whole store. Measured on a 474-token
        // prompt: the checkpoint covered 467 tokens (8 pages, 4 chunks) while the key ended at 384
        // (3), and the server answered "Inline tensor group 0 chunk index out of range (chunks=3)".
        // The publish was lost silently; only a cross-process retrieve would have shown it.
        const std::size_t chunk_bytes_total = chunk_bytes_;
        if (chunk_bytes_total == 0 || chunk_tokens_ == 0 || page_bytes_ == 0 || page_tokens_ == 0) {
            return false;
        }
        // Q64 2d-2b-4: the layout arithmetic lives in core/lmcache_layout.h so the store side and the
        // restore side cannot drift apart. This call computes exactly what the inline block did.
        const auto plan = ninfer::core::lmcache::plan_payload(
            checkpoint.kv_bytes.size(), checkpoint.valid_tokens, checkpoint.frontier,
            prompt_tokens.size(), chunk_bytes_total, chunk_tokens_, page_bytes_, page_tokens_);
        if (plan.chunks == 0 || plan.valid_tokens == 0) { return false; }
        const std::size_t payload_chunks = plan.payload_chunks;
        const std::size_t addressable    = prompt_tokens.size() / chunk_tokens_;
        const std::size_t chunks         = plan.chunks;
        const std::size_t complete       = plan.complete;
        const std::size_t covered        = plan.covered;
        const std::size_t tail_begin     = plan.tail_begin;
        const std::size_t tail_capacity  = plan.tail_capacity;
        const std::uint32_t valid_tokens = plan.valid_tokens;

        if (valid_tokens > covered) {
            std::fprintf(stderr,
                         "[lmcache] publish covers %zu of %zu token(s) (%zu of %zu chunk(s)) "
                         "+ the %zu-token tail\n",
                         static_cast<std::size_t>(valid_tokens), prompt_tokens.size(), chunks,
                         payload_chunks, static_cast<std::size_t>(valid_tokens) - covered);
        } else if (addressable < payload_chunks) {
            // Not silent: the key truncated the prefix, and this export had nothing past the last
            // chunk to carry. The tail is then prefilled again on reuse.
            std::fprintf(stderr,
                         "[lmcache] publish covers %zu of %zu token(s) (%zu of %zu chunk(s)); "
                         "no tail past the last chunk\n",
                         covered, prompt_tokens.size(), chunks, payload_chunks);
        }

        ninfer::core::lmcache::BridgeKey key;
        key.tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
        key.request_id =
            "put-" + std::to_string(instance_tag_) + "-" + std::to_string(++sequence_);

        std::string state;
        for (const std::string& shard : checkpoint.state_shards) { state += shard; }
        if (state.size() != image_bytes_) { return false; }

        // [u32 valid_tokens][state image][tail chunk]. The tail region is zero when the real length
        // does not reach past the last addressable chunk, so a reader can slice it blind.
        const std::size_t carried = valid_tokens > covered ? tail_capacity : 0;
        const std::span<const std::byte> tail =
            carried == 0 ? std::span<const std::byte>{}
                         : std::span<const std::byte>(checkpoint.kv_bytes.data() + tail_begin,
                                                      carried);
        const auto payload = ninfer::core::lmcache::encode_state_object(
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(state.data()),
                                       state.size()),
            valid_tokens, tail, chunk_bytes_total);

        std::lock_guard<std::mutex> guard(bridge_mutex_);
        const bool written = bridge_->publish(
            key, std::span<const std::byte>(checkpoint.kv_bytes.data(), complete),
            std::span<const std::byte>(payload.data(), payload.size()));
        if (written) {
            // Q54: remember which prefix lengths this entry answers for. The frontier and the real
            // committed length are different numbers (388 vs 397 in the measured case), and a longer
            // request has to be able to name either one.
            const std::vector<std::int64_t> sequence(prompt_tokens.begin(), prompt_tokens.end());
            if (valid_tokens != 0 && valid_tokens <= sequence.size()) {
                registry_.record(sequence, valid_tokens);
            }
            if (checkpoint.frontier != 0 && checkpoint.frontier <= sequence.size()) {
                registry_.record(sequence, checkpoint.frontier);
            }
        }
        return written;
    }

private:
    std::unique_ptr<ninfer::core::lmcache::Bridge> bridge_;
    // Q54: which prefix lengths this process knows the cache can answer. Written on store and on a
    // successful restore, read by fetch before it spends a lookup timeout.
    ninfer::core::lmcache::PrefixRegistry registry_;
    // 13.4: write replay. Inert unless NINFER_LMCACHE_WAL_DIR names a directory.
    ninfer::core::lmcache::CheckpointWriteAheadLog wal_;
    // Q64 2d-2h, step 1: one lock for the bridge, because it owns a ZMQ socket and a socket is not
    // thread-safe. Publishing moves to a writer thread in step 2; that is only safe if every user of
    // the bridge - the restore on the request path and the publish on either - takes this first.
    // Today both callers are on one thread, so this changes nothing observable, which is exactly why
    // it is landing ahead of the thread rather than with it.
    std::mutex bridge_mutex_;
    // 2d-2h: the writer and its queue. Every field below is inert unless NINFER_LMCACHE_ASYNC_WRITE
    // was set at the first store; with the switch off the thread is never created and the queue never
    // holds anything, so the synchronous path is the one that was measured.
    static constexpr std::size_t kMaxPendingWrites = 8;
    std::once_flag writer_once_;
    std::thread writer_;
    std::mutex write_mutex_;
    std::condition_variable write_cv_;
    std::deque<Job> writes_;
    bool write_stop_   = false;
    bool async_write_  = false;
    std::uint64_t written_        = 0;
    std::uint64_t dropped_writes_ = 0;
    // 13.4: what the cache was asked and what it returned, so "stop writing" is a decision about
    // this process's traffic instead of a timer.
    std::atomic<std::uint64_t> fetches_{0};
    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> misses_{0};
    std::atomic<std::uint64_t> stores_{0};
    std::atomic<bool> write_disabled_{false};
    std::atomic<bool> ready_{false};
    std::atomic<std::uint64_t> sequence_{0};
    std::size_t page_bytes_ = 0;
    std::size_t page_tokens_ = 0;
    std::size_t image_bytes_ = 0;
    std::size_t chunk_bytes_ = 0;
    std::size_t chunk_tokens_ = 0;
    std::int64_t instance_tag_ =
        static_cast<std::int64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    std::string error_;
};

}  // namespace

struct GenerationService::CheckpointCache {
    std::shared_ptr<LmcacheBackend> backend;
};
#else
struct GenerationService::CheckpointCache {};
#endif

GenerationService::~GenerationService() {
    stop_checkpoint_cache();
    checkpoint_cache_.reset();
}

void GenerationService::stop_checkpoint_cache() {
#if NINFER_HAVE_LMCACHE
    if (checkpoint_cache_ && checkpoint_cache_->backend) { checkpoint_cache_->backend->stop(); }
#endif
}

bool GenerationService::attach_checkpoint_cache() {
#if !NINFER_HAVE_LMCACHE
    return false;
#else
    // The switch was read at construction, where the backend had to exist so the engine could take
    // it in its options. No holder means it was off then.
    if (!checkpoint_cache_ || !checkpoint_cache_->backend) { return false; }

    const ninfer::CacheGeometry geometry = cache_geometry();
    int connections = 4;
    if (const char* value = std::getenv("NINFER_LMCACHE_CONNECTIONS")) {
        const int parsed = std::atoi(value);
        if (parsed > 0) { connections = parsed; }
    }
    if (!checkpoint_cache_->backend->configure(geometry, load_summary().model_name, connections)) {
        std::fprintf(stderr, "[lmcache] could not attach (%s); the cache stays off\n",
                     checkpoint_cache_->backend->error().c_str());
        checkpoint_cache_.reset();
        return false;
    }
    const std::size_t chunk = checkpoint_cache_->backend->chunk_tokens();
    const std::size_t chunk_pages = chunk / static_cast<std::size_t>(geometry.page_tokens);
    const std::size_t chunk_bytes = chunk_pages * geometry.page_bytes;
    std::fprintf(stderr,
                 "[lmcache] attached | page %zu B, %lld token(s), chunk %zu token(s) "
                 "(= %zu page(s)) -- the server must run --chunk-size %zu;\n"
                 "          kv %s (tag %u, %u plane(s)), state %zu B "
                 "(= %zu B image + %zu B length + %zu B tail chunk), %d connection(s)\n",
            geometry.page_bytes, static_cast<long long>(geometry.page_tokens), chunk,
            chunk_pages, chunk,
            kv_format_name(geometry.kv_dtype_tag).data(), geometry.kv_dtype_tag,
            geometry.kv_planes,
            ninfer::core::lmcache::state_object_bytes(geometry.state_image_bytes, chunk_bytes),
            geometry.state_image_bytes, ninfer::core::lmcache::kStateLengthBytes, chunk_bytes,
            connections);
    return true;
#endif
}

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

ApiError request_error_to_api_error(const ninfer::RequestError& exception,
                                    std::string_view constraint_param,
                                    std::span<const std::string> tool_schema_params) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    const auto tool_param = [&] {
        const std::string_view pointer = exception.pointer();
        const auto split               = pointer.find('/', 1);
        std::size_t index              = 0;
        if (pointer.starts_with('/') && split != std::string_view::npos) {
            const auto number = pointer.substr(1, split - 1);
            const auto parsed =
                std::from_chars(number.data(), number.data() + number.size(), index);
            if (parsed.ec == std::errc{} && parsed.ptr == number.data() + number.size() &&
                index < tool_schema_params.size() && !tool_schema_params[index].empty() &&
                (pointer.substr(split) == "/parameters" ||
                 pointer.substr(split).starts_with("/parameters/")))
                return tool_schema_params[index] + std::string(pointer.substr(split + 11));
        }
        return "tools" + exception.pointer();
    };
    switch (exception.kind()) {
    case ninfer::RequestErrorKind::InvalidToolConstraint:
        error.status = 400;
        error.code   = "invalid_tool_constraint";
        error.param  = tool_param();
        break;
    case ninfer::RequestErrorKind::InvalidGrammar:
    case ninfer::RequestErrorKind::InvalidChoice:
    case ninfer::RequestErrorKind::InvalidRegex:
    case ninfer::RequestErrorKind::ConstraintDeadEnd:
        error.status = 400;
        error.param  = constraint_param;
        error.code =
            exception.kind() == ninfer::RequestErrorKind::InvalidGrammar  ? "invalid_grammar"
            : exception.kind() == ninfer::RequestErrorKind::InvalidChoice ? "invalid_choice"
            : exception.kind() == ninfer::RequestErrorKind::InvalidRegex  ? "invalid_regex"
                                                                          : "constraint_dead_end";
        break;
    case ninfer::RequestErrorKind::InvalidJsonSchema:
    case ninfer::RequestErrorKind::UnsupportedJsonSchema:
    case ninfer::RequestErrorKind::UnsatisfiableJsonSchema:
        error.status = 400;
        error.param  = exception.source() == RequestErrorSource::Tools
                           ? tool_param()
                           : std::string(constraint_param) + exception.pointer();
        error.code = exception.kind() == ninfer::RequestErrorKind::InvalidJsonSchema
                         ? "invalid_json_schema"
                     : exception.kind() == ninfer::RequestErrorKind::UnsupportedJsonSchema
                         ? "unsupported_json_schema"
                         : "unsatisfiable_json_schema";
        break;
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    return media;
}

[[noreturn]] void
throw_request_error(const ninfer::RequestError& exception,
                    std::string_view constraint_param               = "structured_outputs.grammar",
                    std::span<const std::string> tool_schema_params = {}) {
    throw ApiException(request_error_to_api_error(exception, constraint_param, tool_schema_params));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninfer::GenerationStart start) override {
        deliver([&] {
            if (sink_->on_start) { sink_->on_start(start); }
        });
    }

    void progress(ninfer::PromptProgress progress) override {
        deliver([&] {
            if (sink_->on_progress) { sink_->on_progress(progress); }
        });
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        deliver([&] {
            if (sink_->on_timing) { sink_->on_timing(timing); }
        });
    }

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        deliver([&] {
            if (delta.channel == ninfer::OutputChannel::Reasoning) {
                if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
            } else {
                if (sink_->on_content) { sink_->on_content(delta.text); }
            }
        });
    }

    [[nodiscard]] bool disconnected() const noexcept { return disconnected_; }

private:
    template <class Callback>
    void deliver(Callback&& callback) {
        if (disconnected_) { return; }
        try {
            callback();
        } catch (const ClientDisconnected&) { disconnected_ = true; }
    }

    const StreamSink* sink_ = nullptr;
    bool disconnected_      = false;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, StartupObserver startup_observer)
    : options_(std::move(options)) {
    // The unified startup configuration already built the engine options; serving only adds the
    // pieces it owns (the startup observer, and the optional external checkpoint cache below).
    ninfer::EngineOptions engine_options = options_.launch.engine;
    engine_options.startup_observer      = std::move(startup_observer);
    // appendix IL: the idle demotion (9b) is ON by default through the CLI; the environment still
    // wins when it is set, so the recorded experiments stay reproducible.
    if (options_.idle_reclaim_ms != 0 && std::getenv("NINFER_KV_IDLE_RECLAIM_MS") == nullptr) {
        const std::string text = std::to_string(options_.idle_reclaim_ms);
#ifdef _WIN32
        (void)_putenv_s("NINFER_KV_IDLE_RECLAIM_MS", text.c_str());
#else
        (void)setenv("NINFER_KV_IDLE_RECLAIM_MS", text.c_str(), 1);
#endif
    }
    // Two-step handoff. The engine takes the backend through its options, so it has to exist before
    // the engine is built; but it cannot be configured until the pools exist, because it needs the
    // page size and the image size. Hand over an empty one now, fill it in at attach time.
    if (const char* enabled = std::getenv("NINFER_LMCACHE_RESTORE");
        enabled != nullptr && *enabled != '\0' && std::strcmp(enabled, "0") != 0) {
#if NINFER_HAVE_LMCACHE
        auto backend                   = std::make_shared<LmcacheBackend>();
        checkpoint_cache_              = std::make_unique<CheckpointCache>();
        checkpoint_cache_->backend     = backend;
        engine_options.checkpoint_cache = backend;
#endif
    }
    // v8 section 12, first phase: apply the deployment mode as the controller's INITIAL value.
    // This must run before the engine exists - no worker thread may have read the mode yet - and
    // after this point the mode lives in the controller, not in the environment or the argv.
    // NINFER_CONTROLLER_MODE is the same initial-value channel for launches without the flag,
    // and it is parsed (and logged) by the mode's own first read.
    if (!options_.launch.controller_mode.empty() && options_.launch.controller_mode != "auto") {
        (void)ninfer::set_controller_mode_initial(options_.launch.controller_mode);
    }
    engine_           = std::make_unique<ninfer::Engine>(std::move(engine_options));
    request_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.launch.engine.max_concurrency) + options_.launch.engine.max_pending_requests);
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.launch.engine.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(request_capacity_, started, deadline);
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           ninfer::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled,
                                           ContextCacheHints context_cache) const {
    return prepare_impl(request, consumer_mode, std::move(observation), std::move(is_cancelled),
                        std::move(context_cache),
                        options_.launch.engine.context_cache.enabled ? CacheParticipation::ReadWrite
                                                    : CacheParticipation::Disabled,
                        DeadlinePolicy::ClientPendingTimeout);
}

PreparedRequest GenerationService::prepare_impl(const GenerationRequest& request,
                                                GenerationConsumerMode consumer_mode,
                                                ninfer::GenerationObservationOptions observation,
                                                std::function<bool()> is_cancelled,
                                                ContextCacheHints context_cache,
                                                CacheParticipation cache_participation,
                                                DeadlinePolicy deadline_policy) const {
    PreparedRequest prepared;
    prepared.constraint_param               = request.constraint_param;
    for (const auto& tool : request.tools) prepared.tool_schema_params.push_back(tool.schema_param);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    ninfer::RequestOptions request_options  = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    prepared.thinking_budget     = request_options.execution.thinking.budget;
    prepared.reasoning_effort    = semantics.reasoning_effort;
    prepared.preserve_thinking   = semantics.preserve_thinking;
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.launch.engine.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime(deadline_policy);

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        std::vector<PromptCacheMarker> protocol_markers = std::move(input.context_cache.markers);
        const bool protocol_allows_engine_automatic =
            input.context_cache.allow_engine_automatic_shared_prefixes;
        input.context_cache = std::move(context_cache);
        input.context_cache.markers.insert(input.context_cache.markers.end(),
                                           std::make_move_iterator(protocol_markers.begin()),
                                           std::make_move_iterator(protocol_markers.end()));
        input.context_cache.allow_engine_automatic_shared_prefixes =
            input.context_cache.allow_engine_automatic_shared_prefixes &&
            protocol_allows_engine_automatic;
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.enable_thinking = prompt.summary().starts_in_reasoning;
        if (!prepared.enable_thinking) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
        }
        prepared.prompt_tokens           = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation             = prompt.preparation_stats();
        prepared.service_prepare_seconds = std::max(
            0.0, std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count() -
                     prepared.preparation.seconds);
        if (observation.first_token) {
            observation.first_token = [callback = std::move(observation.first_token),
                                       seconds  = prepared.service_prepare_seconds](
                                          ninfer::GenerationFirstTokenObservation first) {
                first.prepare_seconds += seconds;
                callback(first);
            };
        }
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              consumer_mode == GenerationConsumerMode::Streaming
                                                  ? ninfer::OutputConsumerMode::Streaming
                                                  : ninfer::OutputConsumerMode::Aggregate,
                                              std::move(observation), prepared.lifetime->deadline);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception, request.constraint_param, prepared.tool_schema_params);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.launch.engine.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.launch.engine.pending_timeout_ms);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

void GenerationService::wait_for_admission(PreparedRequest& prepared,
                                          std::function<bool()> is_cancelled) const {
    try {
        prepared.generation.wait_for_admission(ninfer::CancellationView(is_cancelled));
    } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception, prepared.constraint_param, prepared.tool_schema_params);
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || sink != nullptr) {
        cancellation = ninfer::CancellationView(
            [external = std::move(is_cancelled), sink, observed = output_sink.get()]() {
                return (observed && observed->disconnected()) || (external && external()) ||
                       (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
            });
    }

    ninfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception, prepared.constraint_param, prepared.tool_schema_params);
    }
    GenerationOutcome outcome;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.generated_token_ids = std::move(result.generated_token_ids);
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.constraint          = result.constraint;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);

    outcome.metrics.prepare_seconds =
        prepared.service_prepare_seconds + result.timings.prepare_seconds;
    outcome.metrics.ttft_seconds =
        outcome.metrics.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds          = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds         = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds          = result.timings.decode_seconds;
    outcome.metrics.prompt_wall_seconds     = result.timings.prompt_wall_seconds;
    outcome.metrics.generation_wall_seconds = result.timings.generation_wall_seconds;
    outcome.metrics.total_seconds = prepared.service_prepare_seconds + result.timings.total_seconds;
    outcome.metrics.engine_timing = result.engine_timing;
    outcome.metrics.first_output_timing         = std::move(result.first_output_timing);
    outcome.metrics.scheduling                  = result.scheduling;
    outcome.metrics.admission                   = result.admission;
    outcome.metrics.engine_request_id           = result.engine_request_id;
    outcome.metrics.computed_prefill_tokens     = result.computed_prefill_tokens;
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
    return outcome;
}

void GenerationService::warmup() {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.text     = "hi";
    content.type_raw = "text";
    turn.content.push_back(std::move(content));
    request.messages.push_back(std::move(turn));
    request.max_tokens = 4;
    PreparedRequest prepared =
        prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                     CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup);
    run(prepared, nullptr);
}

} // namespace ninfer::serve

