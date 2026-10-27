// The five calls of LMCache's engine_driven protocol, end to end, against a live server.
//
//   register -> prepare_store -> commit_store -> prepare_retrieve -> commit_retrieve
//
// The point is not that the calls return. It is that bytes written through this path come back
// unchanged, including the state image that travels as its own recurrent group rather than being
// smuggled into the KV payload. So the check at the end is a byte comparison, not a success flag.
//
// Frame layout per call is [request_uid, operation, *payloads], each frame one msgpack value, and
// the reply echoes the uid and operation back. Those two echoes are the cheapest proof that a reply
// belongs to the request we think it does.

#include "core/lmcache_wire.h"
#include "core/msgpack_wire.h"

#include <zmq.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
    if (!condition) {
        ++failures;
    }
}

// --- transport -------------------------------------------------------------

constexpr int kRegisterContext = 0x0E;
constexpr int kPrepareStore = 0x10;
constexpr int kCommitStore = 0x11;
constexpr int kPrepareRetrieve = 0x12;
constexpr int kCommitRetrieve = 0x13;
constexpr int kLookup = 0x08;
constexpr int kWaitPrefetchStatus = 0x0A;

constexpr std::int64_t kInstanceId = 4242;

struct Reply {
    bool received{false};
    std::vector<std::string> frames;
    std::string error;
};

class Session {
public:
    bool connect_to(const char* endpoint) {
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
        const int linger = 0;
        const int timeout_ms = 30000;  // a store carries ~2.6 MB; do not cut it off early
        const int reconnect = 100;
        zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
        zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &timeout_ms, sizeof(timeout_ms));
        zmq_setsockopt(socket_, ZMQ_RECONNECT_IVL, &reconnect, sizeof(reconnect));
        if (zmq_connect(socket_, endpoint) < 0) {
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

    // Returns the response body (frame 2), or an empty Value with ok=false on any protocol error.
    bool call(const char* name, std::int64_t operation, const std::vector<std::string>& payloads,
              ninfer::core::msgpack::Value& body) {
        const std::int64_t uid = ++uid_;
        std::vector<std::string> frames;
        frames.reserve(payloads.size() + 2);
        {
            ninfer::core::msgpack::Writer writer;
            writer.integer(uid);
            frames.push_back(writer.take());
        }
        {
            ninfer::core::msgpack::Writer writer;
            writer.integer(operation);
            frames.push_back(writer.take());
        }
        for (const std::string& payload : payloads) {
            frames.push_back(payload);
        }

        std::printf("-> %s (uid %lld, %zu payload frame(s)): ", name,
                    static_cast<long long>(uid), payloads.size());
        for (std::size_t i = 2; i < frames.size(); ++i) {
            std::printf("%zu B ", frames[i].size());
        }
        std::printf("\n");

        for (std::size_t i = 0; i < frames.size(); ++i) {
            const int flags = (i + 1 < frames.size()) ? ZMQ_SNDMORE : 0;
            if (zmq_send(socket_, frames[i].data(), frames[i].size(), flags) < 0) {
                error_ = zmq_strerror(zmq_errno());
                return false;
            }
        }

        Reply reply;
        while (true) {
            zmq_msg_t message;
            zmq_msg_init(&message);
            if (zmq_msg_recv(&message, socket_, 0) < 0) {
                error_ = zmq_strerror(zmq_errno());
                zmq_msg_close(&message);
                return false;
            }
            reply.frames.emplace_back(static_cast<const char*>(zmq_msg_data(&message)),
                                      zmq_msg_size(&message));
            const bool more = zmq_msg_more(&message) != 0;
            zmq_msg_close(&message);
            if (!more) {
                break;
            }
        }
        reply.received = true;

        if (reply.frames.size() < 2) {
            error_ = "reply has fewer than two frames";
            return false;
        }
        std::size_t offset = 0;
        ninfer::core::msgpack::Value got_uid;
        if (!ninfer::core::msgpack::decode(reply.frames[0], offset, got_uid) ||
            got_uid.integer != uid) {
            error_ = "reply uid does not echo the request";
            return false;
        }
        offset = 0;
        ninfer::core::msgpack::Value got_operation;
        if (!ninfer::core::msgpack::decode(reply.frames[1], offset, got_operation) ||
            got_operation.integer != operation) {
            error_ = "reply operation does not echo the request";
            return false;
        }
        if (reply.frames.size() < 3) {
            // A void operation (lookup) answers with the two echo frames and nothing else. That is
            // a successful call, not a malformed one.
            body = ninfer::core::msgpack::Value{};
            body.kind = ninfer::core::msgpack::Value::Kind::Nil;
            return true;
        }
        offset = 0;
        if (!ninfer::core::msgpack::decode(reply.frames[2], offset, body)) {
            error_ = "reply body is not valid msgpack";
            return false;
        }
        return true;
    }

    const std::string& error() const { return error_; }

private:
    void* context_{nullptr};
    void* socket_{nullptr};
    std::int64_t uid_{0};
    std::string error_;
};

// --- helpers ---------------------------------------------------------------

std::string make_payload(std::size_t size, unsigned char seed) {
    std::string bytes(size, '\0');
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<char>((i * 31u + seed * 17u + (i >> 8)) & 0xFFu);
    }
    return bytes;
}

// Compares and reports where they first differ -- an offset is far more useful than "not equal".
// Counting is left to the caller so a mismatch is reported once, not twice.
bool same_bytes(const std::string& left, const std::string& right, const char* what) {
    if (left.size() != right.size()) {
        std::printf("   %s: size %zu vs %zu\n", what, left.size(), right.size());
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i] != right[i]) {
            std::printf("   %s: first difference at byte %zu (0x%02X vs 0x%02X)\n", what, i,
                        static_cast<unsigned char>(left[i]),
                        static_cast<unsigned char>(right[i]));
            return false;
        }
    }
    return true;
}

bool read_bool(const ninfer::core::msgpack::Value& body, bool& value) {
    if (body.kind != ninfer::core::msgpack::Value::Kind::Bool) {
        return false;
    }
    value = body.boolean;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* endpoint = "tcp://127.0.0.1:6555";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            endpoint = argv[++i];
        } else {
            std::printf("usage: lmcache_ops [--server tcp://127.0.0.1:6555]\n");
            return 2;
        }
    }

    // Geometry here is the placeholder used for the encoding tests: one 256-token chunk of
    // attention KV (two planes) plus a state plane. Real numbers have to come from the engine's KV
    // layout; what this verifies is the protocol, not that these sizes are the right ones.
    const std::int64_t kLayers = 40;
    const std::int64_t kSlots = 256;
    const std::int64_t kHidden = 32;
    const std::int64_t kChunkTokens = 256;

    std::vector<std::int64_t> tokens(kChunkTokens);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<std::int64_t>(i * 7 + 3);
    }

    ninfer::core::lmcache::CacheKey key;
    key.model_name = "qwen3.6-35b-a3b";
    key.world_size = 1;
    key.worker_id = 0;
    key.token_bytes = ninfer::core::lmcache::pack_tokens(tokens);
    key.start = 0;
    key.end = kChunkTokens;
    key.request_id = "req-ops-0001";
    key.num_kv_readers = 1;

    ninfer::core::lmcache::RegisterContext registration;
    registration.instance_id = kInstanceId;
    registration.model_name = key.model_name;
    registration.world_size = 1;
    registration.block_size = kChunkTokens;
    registration.num_layers = kLayers;
    registration.hidden_dim_size = kHidden;
    registration.dtype_str = "bfloat16";
    registration.use_mla = false;
    registration.num_physical_slots = kSlots;
    {
        ninfer::core::lmcache::GroupLayout kv;
        kv.num_layers = kLayers;
        kv.hidden_dim_size = kHidden;
        kv.dtype_str = "bfloat16";
        kv.tokens_per_block = kChunkTokens;
        kv.kv_size = 2;
        registration.group_layouts.push_back(kv);

        ninfer::core::lmcache::GroupLayout state;
        state.num_layers = kLayers;
        state.hidden_dim_size = kHidden;
        state.dtype_str = "float32";
        state.tokens_per_block = kChunkTokens;
        state.kv_size = 1;
        state.recurrent = true;
        registration.group_layouts.push_back(state);
    }

    // Payloads sized from the registered layouts: two-plane KV in bf16, single-plane state in f32.
    const std::size_t kv_bytes =
        static_cast<std::size_t>(2 * kLayers * kSlots * kHidden) * sizeof(std::uint16_t);
    const std::size_t state_bytes =
        static_cast<std::size_t>(kLayers * kSlots * kHidden) * sizeof(std::uint32_t);
    const std::string kv_payload = make_payload(kv_bytes, 0x11);
    const std::string state_payload = make_payload(state_bytes, 0x5C);

    std::vector<ninfer::core::lmcache::TensorGroupWire> groups(2);
    groups[0].chunk_indices = {0};
    groups[0].tensors.push_back({"bfloat16", {2, kLayers, kSlots, kHidden}, kv_payload});
    groups[1].chunk_indices = {0};
    groups[1].tensors.push_back({"float32", {kLayers, kSlots, kHidden}, state_payload});
    const std::string envelope =
        ninfer::core::lmcache::encode_grouped_tensor_envelope(groups);

    std::printf("endpoint   : %s\n", endpoint);
    std::printf("key        : %s, tokens %lld..%lld (1 chunk)\n", key.model_name.c_str(),
                static_cast<long long>(key.start), static_cast<long long>(key.end));
    std::printf("groups     : KV %zu B + state %zu B, envelope %zu B\n", kv_bytes, state_bytes,
                envelope.size());
    std::printf("\n");

    Session session;
    if (!session.connect_to(endpoint)) {
        std::printf("FAIL connect: %s\n", session.error().c_str());
        return 1;
    }

    ninfer::core::msgpack::Value body;

    // 1. register ------------------------------------------------------------
    {
        const std::string payload =
            ninfer::core::lmcache::encode_register_context(registration);
        if (!session.call("register", kRegisterContext, {payload}, body)) {
            std::printf("FAIL register: %s\n", session.error().c_str());
            session.close();
            return 1;
        }
        const auto* shm_name = body.find("shm_name");
        const auto* pool_size = body.find("pool_size");
        expect(body.kind == ninfer::core::msgpack::Value::Kind::Map,
               "register answers with a map of the expected fields");
        if (shm_name != nullptr && pool_size != nullptr) {
            std::printf("   shm_name=\"%s\" pool_size=%lld\n", shm_name->bytes.c_str(),
                        static_cast<long long>(pool_size->integer));
        }
    }

    // 2. prepare_store -------------------------------------------------------
    // The third payload is skipped_chunks. It is empty here, but it must be present: the server
    // counts payloads against payload_types, and a default argument does not make it optional.
    {
        std::string key_bytes = ninfer::core::lmcache::encode_cache_key(key);
        ninfer::core::msgpack::Writer instance_writer;
        instance_writer.integer(kInstanceId);
        ninfer::core::msgpack::Writer skipped_writer;
        skipped_writer.array_header(0);

        if (!session.call("prepare_store", kPrepareStore,
                          {key_bytes, instance_writer.take(), skipped_writer.take()}, body)) {
            std::printf("FAIL prepare_store: %s\n", session.error().c_str());
            session.close();
            return 1;
        }
        expect(body.kind == ninfer::core::msgpack::Value::Kind::Map,
               "prepare_store answers with a response map");
        const auto* context = body.find("context");
        const auto* slots = context != nullptr ? context->find("slots") : nullptr;
        if (slots == nullptr) {
            std::printf("   context.slots: (absent)\n");
        } else {
            std::printf("   context.slots: %zu\n", slots->items.size());
        }
    }

    // 3. commit_store --------------------------------------------------------
    {
        std::string key_bytes = ninfer::core::lmcache::encode_cache_key(key);
        ninfer::core::msgpack::Writer instance_writer;
        instance_writer.integer(kInstanceId);
        ninfer::core::msgpack::Writer payload_writer;
        payload_writer.binary(envelope);

        if (!session.call("commit_store", kCommitStore,
                          {key_bytes, instance_writer.take(), payload_writer.take()}, body)) {
            std::printf("FAIL commit_store: %s\n", session.error().c_str());
            session.close();
            return 1;
        }
        bool stored = false;
        expect(read_bool(body, stored) && stored, "commit_store reports the chunk was written");
    }

    // 4. lookup --------------------------------------------------------------
    // prepare_retrieve reads *prefetched* objects, so a lookup has to come first: it submits the
    // prefetch that takes the read locks. Without it the retrieve is a miss even though the store
    // reported success -- which is exactly what happened the first time this ran.
    {
        std::string key_bytes = ninfer::core::lmcache::encode_cache_key(key);
        ninfer::core::msgpack::Writer tp_writer;
        tp_writer.integer(1);  // tp_size: legacy field, ignored by the server

        if (!session.call("lookup", kLookup, {key_bytes, tp_writer.take()}, body)) {
            std::printf("FAIL lookup: %s\n", session.error().c_str());
            session.close();
            return 1;
        }
        expect(body.kind == ninfer::core::msgpack::Value::Kind::Nil,
               "lookup is a void call and answers with no body");
    }

    // 5. wait_prefetch_status ------------------------------------------------
    {
        ninfer::core::msgpack::Writer request_writer;
        request_writer.string(key.request_id);
        ninfer::core::msgpack::Writer timeout_writer;
        timeout_writer.real(10.0);

        if (!session.call("wait_prefetch_status", kWaitPrefetchStatus,
                          {request_writer.take(), timeout_writer.take()}, body)) {
            std::printf("FAIL wait_prefetch_status: %s\n", session.error().c_str());
            session.close();
            return 1;
        }
        const bool has_count = body.kind == ninfer::core::msgpack::Value::Kind::Int;
        const std::int64_t count = has_count ? body.integer : -1;
        std::printf("   prefetch chunks: %lld\n", static_cast<long long>(count));
        expect(!has_count || count == 1, "the prefetch reports the one chunk that was stored");
    }

    // 6. prepare_retrieve ----------------------------------------------------
    std::string returned;
    {
        std::string key_bytes = ninfer::core::lmcache::encode_cache_key(key);
        ninfer::core::msgpack::Writer instance_writer;
        instance_writer.integer(kInstanceId);

        if (!session.call("prepare_retrieve", kPrepareRetrieve,
                          {key_bytes, instance_writer.take()}, body)) {
            std::printf("FAIL prepare_retrieve: %s\n", session.error().c_str());
            session.close();
            return 1;
        }
        const auto* success = body.find("success");
        const auto* data = body.find("data");
        expect(success != nullptr && success->kind == ninfer::core::msgpack::Value::Kind::Bool &&
                   success->boolean,
               "prepare_retrieve reports a hit");
        if (data != nullptr) {
            returned = data->bytes;
            std::printf("   returned %zu B\n", returned.size());
        }
    }

    // 7. commit_retrieve -----------------------------------------------------
    {
        std::string key_bytes = ninfer::core::lmcache::encode_cache_key(key);
        ninfer::core::msgpack::Writer instance_writer;
        instance_writer.integer(kInstanceId);
        if (!session.call("commit_retrieve", kCommitRetrieve,
                          {key_bytes, instance_writer.take()}, body)) {
            std::printf("FAIL commit_retrieve: %s\n", session.error().c_str());
            session.close();
            return 1;
        }
        bool committed = false;
        expect(read_bool(body, committed) && committed, "commit_retrieve completes");
    }

    session.close();

    // 8. the payloads, byte for byte ----------------------------------------
    std::printf("\n");
    std::vector<ninfer::core::lmcache::DecodedGroup> decoded;
    if (!ninfer::core::lmcache::decode_grouped_tensor_envelope(returned, decoded)) {
        std::printf("FAIL returned payload is not a GroupedTensorEnvelope\n");
        ++failures;
    } else {
        expect(decoded.size() == 2, "the returned envelope carries both groups");
        if (decoded.size() == 2) {
            expect(decoded[0].tensors.size() == 1 && decoded[1].tensors.size() == 1,
                   "each group carries exactly one chunk");
            if (decoded[0].tensors.size() == 1 && decoded[1].tensors.size() == 1) {
                const auto& got_kv = decoded[0].tensors[0];
                const auto& got_state = decoded[1].tensors[0];
                expect(got_kv.dtype == "bfloat16" && got_state.dtype == "float32",
                       "dtypes come back as sent, so the groups did not get crossed");
                const bool kv_same = same_bytes(got_kv.data, kv_payload, "KV payload");
                expect(kv_same, "KV payload is byte-identical after the round trip");
                const bool state_same = same_bytes(got_state.data, state_payload, "state payload");
                expect(state_same, "state payload is byte-identical after the round trip");
            }
        }
    }

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
