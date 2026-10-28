// The engine-side LMCache client: one DEALER connection, plus a pool of them.
//
// Two things here are not stylistic, they are measured constraints (v6 1c / 1n):
//
//  * The server's concurrency unit is the CONNECTION, not the instance. Its affinity key is
//    hash(prefix_frames[0]), which is the ZMQ identity frame, and the default pool has one worker
//    slot. So parallelism requires N connections AND --max-workers N; either alone still serialises.
//  * Every call of one request must stay on the same connection. A request's request_id also keys
//    the server's token session, whose rolling hash state is not reset by set_tokens, so ids must be
//    unique per request and never shared across connections mid-request.
//
// The payload shapes are likewise fixed by measurement: a store is ONE turn carrying every KV chunk
// plus the whole state image, with the state addressed to the trailing chunk and its group registered
// with window_chunks=1 (v6 1n). There is no 10 MB frame ceiling -- that number came from a payload
// the server rejected, not from a transport limit.
#pragma once

#include "core/lmcache_wire.h"
#include "core/msgpack_wire.h"

#include <zmq.h>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::core::lmcache {

enum Operation : std::int64_t {
    kLookup          = 0x08,
    kWaitPrefetch    = 0x0A,
    kRegisterContext = 0x0E,
    kPrepareStore    = 0x10,
    kCommitStore     = 0x11,
    kPrepareRetrieve = 0x12,
    kCommitRetrieve  = 0x13,
};

struct ClientOptions {
    std::string endpoint{"tcp://127.0.0.1:6555"};
    // A store carries the whole snapshot (~77 MB at 640 tokens), so the send side needs room to
    // drain. The receive side needs the same, because a rejected payload produces no reply at all
    // and the only signal is this timeout.
    int send_timeout_ms{120000};
    int receive_timeout_ms{120000};
    int linger_ms{0};
    int reconnect_ivl_ms{100};
};

// What one restore hands back: the KV chunks in chunk order, plus the state image.
struct RestorePayload {
    std::vector<std::string> kv_chunks;
    std::string state;
    // The shapes the server sized the objects to. They come from the registration, so a mismatch
    // here means the layout the server used is not the layout that was sent.
    std::vector<std::int64_t> kv_shape;
    std::vector<std::int64_t> state_shape;
};

class Connection {
public:
    Connection() = default;
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    ~Connection() { close(); }

    bool open(const ClientOptions& options) {
        close();
        options_ = options;
        context_ = zmq_ctx_new();
        if (context_ == nullptr) {
            error_ = zmq_strerror(zmq_errno());
            return false;
        }
        socket_ = zmq_socket(context_, ZMQ_DEALER);
        if (socket_ == nullptr) {
            error_ = zmq_strerror(zmq_errno());
            return false;
        }
        const int linger = options.linger_ms;
        const int send_timeout = options.send_timeout_ms;
        const int receive_timeout = options.receive_timeout_ms;
        const int reconnect = options.reconnect_ivl_ms;
        zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &send_timeout, sizeof(send_timeout));
        zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &receive_timeout, sizeof(receive_timeout));
        zmq_setsockopt(socket_, ZMQ_RECONNECT_IVL, &reconnect, sizeof(reconnect));
        if (zmq_connect(socket_, options.endpoint.c_str()) < 0) {
            error_ = zmq_strerror(zmq_errno());
            return false;
        }
        return true;
    }

    void close() {
        if (socket_ != nullptr) {
            zmq_close(socket_);
            socket_ = nullptr;
        }
        if (context_ != nullptr) {
            zmq_ctx_term(context_);
            context_ = nullptr;
        }
    }

    [[nodiscard]] bool good() const { return socket_ != nullptr; }
    [[nodiscard]] const std::string& error() const { return error_; }

    // One registration per process. The instance id must be fresh on every start: re-registering an
    // existing id does not replace its layout, so a recycled id silently keeps the old shapes.
    bool register_context(const RegisterContext& context, std::int64_t* null_block_id = nullptr) {
        std::string body;
        if (!call(kRegisterContext, {encode_register_context(context)}, &body)) { return false; }
        std::size_t offset = 0;
        msgpack::Value value;
        if (!msgpack::decode(body, offset, value) || value.kind != msgpack::Value::Kind::Map) {
            error_ = "register reply is not a map";
            return false;
        }
        registered_ = context;
        if (null_block_id != nullptr) {
            const msgpack::Value* found = value.find("null_block_id");
            *null_block_id = found != nullptr ? found->integer : -1;
        }
        return true;
    }

    // Take the registration without sending it. The server keeps layout metadata per instance, not
    // per connection, so one REGISTER covers every connection of the pool -- and only one REGISTER
    // should be sent, because the call is serialised server-side.
    void adopt_registration(const RegisterContext& context) { registered_ = context; }

    // One turn: every KV chunk, then the state image addressed to the trailing chunk.
    //
    // kv_shape is one chunk's shape and is the same for every chunk; state_shape is the whole image.
    // Both are validated by the server against the registered layout, which is why they are passed
    // in rather than guessed here.
    bool store(const CacheKey& key, const std::vector<std::string>& kv_chunks,
               const std::vector<std::int64_t>& kv_shape, const std::string& state,
               const std::vector<std::int64_t>& state_shape) {
        if (kv_chunks.empty()) {
            error_ = "store needs at least one KV chunk";
            return false;
        }
        if (!registered_) {
            error_ = "store before register_context";
            return false;
        }
        {
            std::string body;
            msgpack::Writer instance;
            instance.integer(registered_->instance_id);
            msgpack::Writer skipped;
            skipped.array_header(0);
            if (!call(kPrepareStore, {encode_cache_key(key), instance.take(), skipped.take()},
                      &body)) {
                return false;
            }
        }

        std::vector<TensorGroupWire> groups(2);
        groups[0].chunk_indices.reserve(kv_chunks.size());
        groups[0].tensors.reserve(kv_chunks.size());
        for (std::size_t i = 0; i < kv_chunks.size(); ++i) {
            groups[0].chunk_indices.push_back(static_cast<std::int64_t>(i));
            groups[0].tensors.push_back({kv_dtype(), kv_shape, kv_chunks[i]});
        }
        // The state describes the whole prefix, so it is stored once, under the trailing chunk. The
        // server only requires chunk indices to be increasing and below the chunk count, not to
        // cover it -- but a retrieve is a miss unless each group covers its in-window suffix, which
        // is why the group is registered with window_chunks=1.
        groups[1].chunk_indices = {static_cast<std::int64_t>(kv_chunks.size()) - 1};
        groups[1].tensors.push_back({state_dtype(), state_shape, state});

        const std::string envelope = encode_grouped_tensor_envelope(groups);
        std::string body;
        msgpack::Writer instance;
        instance.integer(registered_->instance_id);
        // cpu_data is declared `bytes`, so the envelope has to arrive as a msgpack bin. Sending the
        // map itself puts an object where the server expects bytes and it says so.
        msgpack::Writer data;
        data.binary(envelope);
        return call(kCommitStore, {encode_cache_key(key), instance.take(), data.take()}, &body) &&
               read_scalar_bool(body);
    }

    // lookup takes a read lock and starts the prefetch; it answers with no body. Skipping it is not
    // detected by store -- commit_store still returns true -- and only shows up as a retrieve miss.
    bool has(const CacheKey& key) {
        // Two payloads: the key and tp_size. tp_size is a legacy field the server ignores, but the
        // payload count is checked first, and a mismatch raises -- with no reply at all.
        msgpack::Writer tp_size;
        tp_size.integer(1);
        if (!call(kLookup, {encode_cache_key(key), tp_size.take()}, nullptr)) { return false; }
        std::string body;
        msgpack::Writer request;
        request.string(key.request_id);
        // Short on purpose. This waits for a prefetch to finish, and on a miss there is no prefetch
        // to finish -- the request then sits here until the timeout expires, which on a best-effort
        // path is indistinguishable from a hang. A healthy prefetch of one chunk completes in
        // milliseconds; the retrieve below is what actually carries the payload.
        msgpack::Writer timeout;
        timeout.real(0.5);
        return call(kWaitPrefetch, {request.take(), timeout.take()}, &body);
    }

    // Only call this after has() returned true: it moves the whole payload across.
    bool fetch(const CacheKey& key, RestorePayload* out) {
        std::string body;
        {
            msgpack::Writer instance;
            instance.integer(registered_->instance_id);
            if (!call(kPrepareRetrieve, {encode_cache_key(key), instance.take()}, &body)) {
                return false;
            }
        }
        if (!read_field_bool(body, "success")) {
            error_ = "prepare_retrieve missed";
            return false;
        }
        std::size_t offset = 0;
        msgpack::Value value;
        if (!msgpack::decode(body, offset, value) || value.kind != msgpack::Value::Kind::Map) {
            error_ = "prepare_retrieve reply is not a map";
            return false;
        }
        const msgpack::Value* data = value.find("data");
        if (data == nullptr) {
            error_ = "prepare_retrieve reply carries no data";
            return false;
        }

        std::vector<DecodedGroup> groups;
        if (!decode_grouped_tensor_envelope(data->bytes, groups) || groups.size() != 2) {
            error_ = "returned payload is not a two-group envelope";
            return false;
        }
        if (groups[1].tensors.size() != 1) {
            error_ = "state group should carry exactly one tensor";
            return false;
        }
        if (out != nullptr) {
            out->kv_chunks.clear();
            out->kv_chunks.reserve(groups[0].tensors.size());
            for (const DecodedTensor& tensor : groups[0].tensors) {
                out->kv_chunks.push_back(tensor.data);
            }
            out->state       = groups[1].tensors[0].data;
            out->kv_shape    = groups[0].tensors[0].shape;
            out->state_shape = groups[1].tensors[0].shape;
        }

        msgpack::Writer instance;
        instance.integer(registered_->instance_id);
        return call(kCommitRetrieve, {encode_cache_key(key), instance.take()}, &body) &&
               read_scalar_bool(body);
    }

private:
    [[nodiscard]] std::string kv_dtype() const { return registered_ ? registered_->dtype_str : ""; }

    [[nodiscard]] std::string state_dtype() const {
        if (!registered_ || registered_->group_layouts.size() < 2) { return ""; }
        return registered_->group_layouts[1].dtype_str;
    }

    // commit_store and commit_retrieve answer with a bare bool, not a map: {"success": ...} is only
    // the shape of prepare_retrieve's reply.
    static bool read_scalar_bool(const std::string& body) {
        std::size_t offset = 0;
        msgpack::Value value;
        if (!msgpack::decode(body, offset, value)) { return false; }
        return value.kind == msgpack::Value::Kind::Bool && value.boolean;
    }

    static bool read_field_bool(const std::string& body, const char* field) {
        std::size_t offset = 0;
        msgpack::Value value;
        if (!msgpack::decode(body, offset, value) || value.kind != msgpack::Value::Kind::Map) {
            return false;
        }
        const msgpack::Value* found = value.find(field);
        return found != nullptr && found->kind == msgpack::Value::Kind::Bool && found->boolean;
    }

    // Returns true when the reply echoes uid and operation. A void operation (lookup) answers with
    // the two echo frames and nothing else, which is success, not a truncated reply.
    bool call(std::int64_t operation, const std::vector<std::string>& payloads, std::string* body) {
        if (socket_ == nullptr) {
            error_ = "connection is not open";
            return false;
        }
        const std::int64_t uid = ++uid_;
        std::vector<std::string> frames;
        frames.reserve(payloads.size() + 2);
        {
            msgpack::Writer writer;
            writer.integer(uid);
            frames.push_back(writer.take());
        }
        {
            msgpack::Writer writer;
            writer.integer(operation);
            frames.push_back(writer.take());
        }
        for (const std::string& payload : payloads) { frames.push_back(payload); }

        for (std::size_t i = 0; i < frames.size(); ++i) {
            const int flags = (i + 1 < frames.size()) ? ZMQ_SNDMORE : 0;
            if (zmq_send(socket_, frames[i].data(), frames[i].size(), flags) < 0) {
                error_ = "send failed: ";
                error_ += zmq_strerror(zmq_errno());
                return false;
            }
        }

        std::vector<std::string> reply;
        while (true) {
            zmq_msg_t message;
            zmq_msg_init(&message);
            if (zmq_msg_recv(&message, socket_, 0) < 0) {
                // The usual cause is a payload the server rejected: a raising handler answers with
                // no frames at all, so this timeout is the only symptom and it looks like EAGAIN.
                error_ = "receive failed: ";
                error_ += zmq_strerror(zmq_errno());
                zmq_msg_close(&message);
                return false;
            }
            reply.emplace_back(static_cast<const char*>(zmq_msg_data(&message)),
                               zmq_msg_size(&message));
            const bool more = zmq_msg_more(&message) != 0;
            zmq_msg_close(&message);
            if (!more) { break; }
        }

        if (reply.size() < 2) {
            error_ = "reply has fewer than two frames";
            return false;
        }
        {
            std::size_t offset = 0;
            msgpack::Value got;
            if (!msgpack::decode(reply[0], offset, got) || got.integer != uid) {
                error_ = "reply uid does not echo the request";
                return false;
            }
        }
        {
            std::size_t offset = 0;
            msgpack::Value got;
            if (!msgpack::decode(reply[1], offset, got) || got.integer != operation) {
                error_ = "reply operation does not echo the request";
                return false;
            }
        }
        if (body != nullptr) { *body = reply.size() >= 3 ? reply[2] : std::string(); }
        return true;
    }

    ClientOptions options_{};
    void* context_{nullptr};
    void* socket_{nullptr};
    std::int64_t uid_{0};
    std::optional<RegisterContext> registered_;
    std::string error_;
};

// A fixed set of connections, handed out one lease at a time.
//
// The size should match the server's --max-workers: connections beyond that wrap around onto shared
// slots and serialise again, and fewer connections than slots leaves the server's workers idle.
class Pool {
public:
    Pool() = default;
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    bool open(const ClientOptions& options, std::size_t connections) {
        connections_.clear();
        busy_.assign(connections, false);
        free_count_ = connections;
        for (std::size_t i = 0; i < connections; ++i) {
            auto connection = std::make_unique<Connection>();
            if (!connection->open(options)) {
                error_ = connection->error();
                connections_.clear();
                return false;
            }
            connections_.push_back(std::move(connection));
        }
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        connections_.clear();
        busy_.clear();
        free_count_ = 0;
    }

    // Sends one REGISTER on the first connection and shares the layout with the rest. The metadata
    // is per instance on the server, so a second REGISTER would only re-create the same entry.
    bool register_context(const RegisterContext& context, std::int64_t* null_block_id = nullptr) {
        if (connections_.empty()) {
            error_ = "pool is not open";
            return false;
        }
        if (!connections_[0]->register_context(context, null_block_id)) {
            error_ = connections_[0]->error();
            return false;
        }
        for (std::size_t i = 1; i < connections_.size(); ++i) {
            connections_[i]->adopt_registration(context);
        }
        return true;
    }

    [[nodiscard]] const std::string& error() const { return error_; }
    [[nodiscard]] std::size_t size() const { return connections_.size(); }

    class Lease {
    public:
        Lease() = default;
        Lease(Pool* pool, std::size_t slot) : pool_(pool), slot_(slot) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept : pool_(other.pool_), slot_(other.slot_) { other.pool_ = nullptr; }
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();
                pool_ = other.pool_;
                slot_ = other.slot_;
                other.pool_ = nullptr;
            }
            return *this;
        }
        ~Lease() { release(); }

        [[nodiscard]] Connection* operator->() const {
            return pool_ != nullptr ? pool_->connections_[slot_].get() : nullptr;
        }
        [[nodiscard]] Connection& operator*() const { return *operator->(); }
        [[nodiscard]] explicit operator bool() const { return pool_ != nullptr; }

    private:
        void release() {
            if (pool_ != nullptr) {
                pool_->give_back(slot_);
                pool_ = nullptr;
            }
        }
        Pool* pool_{nullptr};
        std::size_t slot_{0};
    };

    // Blocks until a connection is free. One lease covers a whole request, so acquire it once and
    // keep it across store/lookup/fetch -- the server's token session and affinity both depend on
    // that.
    Lease acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return free_count_ > 0; });
        for (std::size_t i = 0; i < busy_.size(); ++i) {
            if (!busy_[i]) {
                busy_[i] = true;
                --free_count_;
                return Lease(this, i);
            }
        }
        return {};
    }

    // Fails instead of blocking. For paths that must not wait -- a prefetch that would otherwise
    // hold a worker while the request it serves is already running.
    std::optional<Lease> try_acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_count_ == 0) { return std::nullopt; }
        for (std::size_t i = 0; i < busy_.size(); ++i) {
            if (!busy_[i]) {
                busy_[i] = true;
                --free_count_;
                return Lease(this, i);
            }
        }
        return std::nullopt;
    }

private:
    void give_back(std::size_t slot) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            busy_[slot] = false;
            ++free_count_;
        }
        cv_.notify_one();
    }

    std::vector<std::unique_ptr<Connection>> connections_;
    std::vector<bool> busy_;
    std::size_t free_count_{0};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::string error_;
};

}  // namespace ninfer::core::lmcache
