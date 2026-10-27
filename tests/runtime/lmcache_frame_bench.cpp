// Move a real number of 6 MiB frames through LMCache and time it.
//
// The restore estimate (about 1 s for 77 MB) came from timing a 5.24 MB turn and dividing.
// That is arithmetic, not measurement of the shape the restore actually uses: ten or so
// back-to-back frames of 6 MiB through a single-worker server.
//
// LMCache only accepts typed tensors, so a frame of arbitrary bytes has to be dressed as one. The
// layout is chosen to make one chunk exactly 6 MiB: kv_size 1 x layers 1 x slots 128 x hidden
// 24576 x bf16 = 6,291,456 bytes. The bytes are not KV data and are never read as such; what is
// being measured is the transport, and the tensors are only the envelope it insists on.

#include "core/lmcache_wire.h"
#include "core/msgpack_wire.h"

#include <zmq.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kRegister = 0x0E;
constexpr int kPrepareStore = 0x10;
constexpr int kCommitStore = 0x11;
constexpr int kLookup = 0x08;
constexpr int kWaitPrefetch = 0x0A;
constexpr int kPrepareRetrieve = 0x12;
constexpr int kCommitRetrieve = 0x13;

// A fresh id per run: re-registering an existing instance does not replace its layout, so reusing
// the id kept the shape from an earlier attempt and every frame failed validation against it.
// Both the instance id and the element count are settable, because the open question is whether one
// frame can carry a whole 64 MiB state image or whether that has to be split.
constexpr std::int64_t kSlots = 128;
constexpr std::int64_t kLayers = 1;
std::int64_t g_instance = 6152;
std::int64_t g_hidden    = 12288;  // 2 * 1 * 128 * 12288 * 2 = 6,291,456
// The single-group path validates against the TOP-LEVEL layout_desc, which is
// [2, num_layers, num_physical_slots, hidden] when use_mla is false -- the leading 2 is the K/V
// plane count. Forgetting it was the last failure: shape right, byte count half of what was expected.
std::size_t frame_bytes() {
    return static_cast<std::size_t>(2 * kSlots * kLayers * g_hidden) * sizeof(std::uint16_t);
}

struct Session {
    void* context{nullptr};
    void* socket{nullptr};
    std::int64_t uid{0};

    bool open(const char* endpoint) {
        context = zmq_ctx_new();
        socket  = context != nullptr ? zmq_socket(context, ZMQ_DEALER) : nullptr;
        if (socket == nullptr) { return false; }
        const int linger = 0;
        const int timeout = 60000;
        zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
        zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
        return zmq_connect(socket, endpoint) == 0;
    }

    void close() {
        if (socket) { zmq_close(socket); }
        if (context) { zmq_ctx_term(context); }
    }

    bool call(std::int64_t operation, const std::vector<std::string>& payloads,
              std::string* body = nullptr) {
        std::vector<std::string> frames;
        ninfer::core::msgpack::Writer uid_writer;
        uid_writer.integer(++uid);
        frames.push_back(uid_writer.take());
        ninfer::core::msgpack::Writer op_writer;
        op_writer.integer(operation);
        frames.push_back(op_writer.take());
        for (const std::string& payload : payloads) { frames.push_back(payload); }

        for (std::size_t i = 0; i < frames.size(); ++i) {
            const int flags = (i + 1 < frames.size()) ? ZMQ_SNDMORE : 0;
            if (zmq_send(socket, frames[i].data(), frames[i].size(), flags) < 0) { return false; }
        }
        std::vector<std::string> reply;
        while (true) {
            zmq_msg_t message;
            zmq_msg_init(&message);
            if (zmq_msg_recv(&message, socket, 0) < 0) {
                zmq_msg_close(&message);
                return false;
            }
            reply.emplace_back(static_cast<const char*>(zmq_msg_data(&message)),
                               zmq_msg_size(&message));
            const bool more = zmq_msg_more(&message) != 0;
            zmq_msg_close(&message);
            if (!more) { break; }
        }
        if (body != nullptr && reply.size() >= 3) { *body = reply[2]; }
        return true;
    }
};

std::string register_payload() {
    ninfer::core::lmcache::RegisterContext context;
    context.instance_id        = g_instance;
    context.model_name         = "frame-bench";
    context.world_size         = 1;
    context.block_size         = kSlots;
    context.num_layers         = kLayers;
    context.hidden_dim_size    = g_hidden;
    context.dtype_str          = "bfloat16";
    context.use_mla            = false;
    context.num_physical_slots = kSlots;
    ninfer::core::lmcache::GroupLayout group;
    group.num_layers      = kLayers;
    group.hidden_dim_size = g_hidden;
    group.dtype_str       = "bfloat16";
    group.tokens_per_block = kSlots;
    group.kv_size          = 2;
    context.group_layouts.push_back(group);
    return ninfer::core::lmcache::encode_register_context(context);
}

// The server keys its token session by request_id and `set_tokens` replaces the buffer without
// resetting the rolling hash state, so a reused request_id keeps hashing against the tokens of the
// first request that used it. Every store/lookup call therefore needs a request_id unique per run.
std::string request_id(int index) {
    return "frame-" + std::to_string(g_instance) + "-" + std::to_string(index);
}

std::string key_payload(int index) {
    ninfer::core::lmcache::CacheKey key;
    key.model_name = "frame-bench";
    key.world_size = 1;
    key.worker_id  = 0;
    std::vector<std::int64_t> tokens(static_cast<std::size_t>(kSlots));
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        // The key is derived from the tokens alone, so two runs that reuse the same token sequence
        // hit each other's entries: a 16 MiB store then "succeeded" while retrieve handed back the
        // 6 MiB image an earlier run had stored under the same key. Offset by instance to keep runs
        // independent, but keep the ids inside int32 range: a 7.2e9 token was accepted by store and
        // then unreadable ("key exists but cannot be read").
        const std::int64_t base = (g_instance % 1000) * 100000;
        tokens[i]            = static_cast<std::int64_t>(base + index * 1000 + static_cast<int>(i));
    }
    key.token_bytes = ninfer::core::lmcache::pack_tokens(tokens);
    key.start       = 0;
    key.end         = kSlots;
    key.request_id  = request_id(index);
    return ninfer::core::lmcache::encode_cache_key(key);
}

// Single-group transfers take TensorEnvelope {version, tensors}, not the grouped form. The two
// envelopes have disjoint field names on purpose, so sending the wrong one does not half-decode:
// the server raises, and a raising handler answers with no body at all.
std::string envelope(const std::string& bytes) {
    ninfer::core::msgpack::Writer writer;
    writer.map_header(2);
    writer.key("version");
    writer.integer(1);
    writer.key("tensors");
    writer.array_header(1);
    writer.map_header(3);
    writer.key("dtype");
    writer.string("bfloat16");
    writer.key("shape");
    writer.array_header(4);
    writer.integer(2);  // the K/V plane count: the server sizes a non-MLA layout as
                        // [2, layers, slots, hidden], and a leading 1 fails that check
    writer.integer(kLayers);
    writer.integer(kSlots);
    writer.integer(g_hidden);
    writer.key("data");
    writer.binary(bytes);
    return writer.take();
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* endpoint = "tcp://127.0.0.1:6555";
    int frames = 10;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            endpoint = argv[++i];
        } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            frames = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            g_hidden = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--instance") == 0 && i + 1 < argc) {
            g_instance = std::atoll(argv[++i]);
        }
    }

    std::printf("frame bytes: %zu  frames: %d  total: %.1f MB  instance: %lld\n", frame_bytes(),
                frames, static_cast<double>(frame_bytes()) * frames / 1048576.0,
                static_cast<long long>(g_instance));

    Session session;
    if (!session.open(endpoint)) {
        std::printf("FAIL connect\n");
        return 1;
    }
    // Warm up, so the first call's cost does not land in the measurement.
    std::string register_body;
    if (!session.call(kRegister, {register_payload()}, &register_body)) {
        std::printf("FAIL register (no reply)\n");
        return 1;
    }
    std::printf("register reply: %zu B\n", register_body.size());

    std::vector<std::string> payloads;
    for (int i = 0; i < frames; ++i) {
        payloads.push_back(std::string(frame_bytes(), static_cast<char>(0xA7)));
    }

    const auto store_start = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) {
        ninfer::core::msgpack::Writer instance_writer;
        instance_writer.integer(g_instance);
        ninfer::core::msgpack::Writer skipped_writer;
        skipped_writer.array_header(0);
        std::string prep_body;
        if (!session.call(kPrepareStore, {key_payload(i), instance_writer.take(), skipped_writer.take()}, &prep_body)) {
            std::printf("FAIL prepare_store %d (prep %zu B)\n", i, prep_body.size());
            return 1;
        }
        ninfer::core::msgpack::Writer payload_writer;
        payload_writer.integer(g_instance);
        const std::string body = envelope(payloads[i]);
        std::string commit_body;
        // cpu_data is declared `bytes`, so the envelope must arrive as a msgpack bin, not as the
        // map it is made of. Sending the envelope directly put a map where a bytes was expected, and
        // the server said so: Expected `bytes`, got `object`.
        ninfer::core::msgpack::Writer data_writer;
        data_writer.binary(body);
        if (!session.call(kCommitStore, {key_payload(i), payload_writer.take(), data_writer.take()},
                          &commit_body)) {
            std::printf("FAIL commit_store %d (prep %zu B, commit %zu B)\n", i, prep_body.size(), commit_body.size());
            return 1;
        }
    }
    const auto store_end = std::chrono::steady_clock::now();

    const auto retrieve_start = std::chrono::steady_clock::now();
    std::size_t bytes_back = 0;
    for (int i = 0; i < frames; ++i) {
        ninfer::core::msgpack::Writer tp_writer;
        tp_writer.integer(1);
        if (!session.call(kLookup, {key_payload(i), tp_writer.take()})) {
            std::printf("FAIL lookup %d\n", i);
            return 1;
        }
        ninfer::core::msgpack::Writer request_writer;
        request_writer.string(request_id(i));
        ninfer::core::msgpack::Writer timeout_writer;
        timeout_writer.real(20.0);
        if (!session.call(kWaitPrefetch, {request_writer.take(), timeout_writer.take()})) {
            std::printf("FAIL wait_prefetch %d\n", i);
            return 1;
        }
        ninfer::core::msgpack::Writer instance_writer;
        instance_writer.integer(g_instance);
        std::string body;
        if (!session.call(kPrepareRetrieve, {key_payload(i), instance_writer.take()}, &body)) {
            std::printf("FAIL prepare_retrieve %d\n", i);
            return 1;
        }
        std::size_t offset = 0;
        ninfer::core::msgpack::Value value;
        if (!ninfer::core::msgpack::decode(body, offset, value)) {
            std::printf("FAIL decode %d\n", i);
            return 1;
        }
        const auto* data = value.find("data");
        if (data != nullptr) { bytes_back += data->bytes.size(); }
        ninfer::core::msgpack::Writer commit_writer;
        commit_writer.integer(g_instance);
        if (!session.call(kCommitRetrieve, {key_payload(i), commit_writer.take()})) {
            std::printf("FAIL commit_retrieve %d\n", i);
            return 1;
        }
    }
    const auto retrieve_end = std::chrono::steady_clock::now();
    session.close();

    const auto ms = [](auto from, auto to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };
    const double store_ms = ms(store_start, store_end);
    const double retrieve_ms = ms(retrieve_start, retrieve_end);
    const double total_mb = static_cast<double>(frame_bytes()) * frames / 1048576.0;

    std::printf("\nstore    %8.1f ms   (%.1f MB -> %.1f MB/s)\n", store_ms, total_mb,
                total_mb / (store_ms / 1000.0));
    std::printf("retrieve %8.1f ms   (%.1f MB back -> %.1f MB/s, %zu B on the wire)\n",
                retrieve_ms, total_mb, total_mb / (retrieve_ms / 1000.0), bytes_back);
    std::printf("per frame: store %.1f ms, retrieve %.1f ms\n", store_ms / frames,
                retrieve_ms / frames);
    return 0;
}
