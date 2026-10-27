// How expensive is a lookup on its own?
//
// The restore design leans on one claim: lookup is cheap enough to run on every request, so the
// expensive transfer can be gated behind it. That claim decides the whole shape of the integration
// (1j in the plan), so it needs a number rather than an assumption.
//
// A lookup on a missing key is the worst case for usefulness and the common case for a cold cache:
// it hashes the tokens, submits a prefetch that finds nothing, and returns. That is what this
// measures, in round-trip terms, over the real transport.

#include "core/lmcache_wire.h"
#include "core/msgpack_wire.h"

#include <zmq.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kLookup = 0x08;
constexpr int kWaitPrefetch = 0x0A;

struct Connection {
    void* context{nullptr};
    void* socket{nullptr};

    bool open(const char* endpoint) {
        context = zmq_ctx_new();
        socket  = context != nullptr ? zmq_socket(context, ZMQ_DEALER) : nullptr;
        if (socket == nullptr) { return false; }
        const int linger = 0;
        const int timeout = 10000;
        zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
        zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
        return zmq_connect(socket, endpoint) == 0;
    }

    void close() {
        if (socket != nullptr) { zmq_close(socket); }
        if (context != nullptr) { zmq_ctx_term(context); }
    }

    bool call(std::int64_t uid, std::int64_t operation,
              const std::vector<std::string>& payloads) {
        std::vector<std::string> frames;
        frames.reserve(payloads.size() + 2);
        ninfer::core::msgpack::Writer writer;
        writer.integer(uid);
        frames.push_back(writer.take());
        ninfer::core::msgpack::Writer op_writer;
        op_writer.integer(operation);
        frames.push_back(op_writer.take());
        for (const std::string& payload : payloads) { frames.push_back(payload); }

        for (std::size_t i = 0; i < frames.size(); ++i) {
            const int flags = (i + 1 < frames.size()) ? ZMQ_SNDMORE : 0;
            if (zmq_send(socket, frames[i].data(), frames[i].size(), flags) < 0) { return false; }
        }
        while (true) {
            zmq_msg_t message;
            zmq_msg_init(&message);
            if (zmq_msg_recv(&message, socket, 0) < 0) {
                zmq_msg_close(&message);
                return false;
            }
            const bool more = zmq_msg_more(&message) != 0;
            zmq_msg_close(&message);
            if (!more) { break; }
        }
        return true;
    }
};

std::string key_for(int index) {
    ninfer::core::lmcache::CacheKey key;
    key.model_name = "lookup-bench";
    key.world_size = 1;
    key.worker_id  = 0;
    std::vector<std::int64_t> tokens(128);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<std::int64_t>(i * 13 + index);
    }
    key.token_bytes = ninfer::core::lmcache::pack_tokens(tokens);
    key.start       = 0;
    key.end         = 128;
    key.request_id  = "bench-" + std::to_string(index);
    return ninfer::core::lmcache::encode_cache_key(key);
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) { return 0.0; }
    std::sort(values.begin(), values.end());
    const std::size_t index =
        std::min(values.size() - 1,
                 static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1)));
    return values[index];
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* endpoint = "tcp://127.0.0.1:6555";
    int rounds = 200;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            endpoint = argv[++i];
        } else if (std::strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
            rounds = std::atoi(argv[++i]);
        }
    }

    Connection connection;
    if (!connection.open(endpoint)) {
        std::printf("FAIL connect %s\n", endpoint);
        return 1;
    }

    std::vector<double> lookup_ms;
    std::vector<double> wait_ms;
    lookup_ms.reserve(static_cast<std::size_t>(rounds));
    wait_ms.reserve(static_cast<std::size_t>(rounds));

    std::int64_t uid = 1;
    for (int round = 0; round < rounds; ++round) {
        const std::string key = key_for(round);
        ninfer::core::msgpack::Writer tp_writer;
        tp_writer.integer(1);  // tp_size, ignored by the server

        auto started = std::chrono::steady_clock::now();
        if (!connection.call(uid++, kLookup, {key, tp_writer.take()})) {
            std::printf("FAIL lookup on round %d\n", round);
            connection.close();
            return 1;
        }
        lookup_ms.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());

        ninfer::core::msgpack::Writer request_writer;
        request_writer.string("bench-" + std::to_string(round));
        ninfer::core::msgpack::Writer timeout_writer;
        timeout_writer.real(5.0);
        started = std::chrono::steady_clock::now();
        if (!connection.call(uid++, kWaitPrefetch, {request_writer.take(), timeout_writer.take()})) {
            std::printf("FAIL wait_prefetch_status on round %d\n", round);
            connection.close();
            return 1;
        }
        wait_ms.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    }
    connection.close();

    const auto report = [](const char* label, std::vector<double> samples) {
        if (samples.empty()) { return; }
        double total = 0.0;
        for (const double value : samples) { total += value; }
        std::printf("%-22s min %6.3f  median %6.3f  p95 %6.3f  max %6.3f  mean %6.3f ms\n", label,
                    percentile(samples, 0.0), percentile(samples, 0.5), percentile(samples, 0.95),
                    percentile(samples, 1.0), total / static_cast<double>(samples.size()));
    };
    std::printf("rounds: %d  endpoint: %s\n\n", rounds, endpoint);
    report("lookup (miss)", lookup_ms);
    report("wait_prefetch_status", wait_ms);
    return 0;
}
