#include "serve/http_transport.h"
#include "serve/request.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <ws2tcpip.h>
#elif defined(__linux__)
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <unistd.h>
#endif

namespace {

using ninfer::serve::ClientDisconnected;
using ninfer::serve::SseTransport;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int test_sse_transport() {
    using namespace std::chrono_literals;
    int failures = 0;

    std::vector<std::string> writes;
    bool writable = true;
    bool write_ok = true;
    httplib::DataSink sink;
    sink.write = [&](const char* data, std::size_t size) {
        if (!write_ok) { return false; }
        writes.emplace_back(data, size);
        return true;
    };
    sink.is_writable = [&] { return writable; };

    std::atomic<bool> cancelled{false};
    const SseTransport::Clock::time_point start{100s};
    SseTransport transport(sink, cancelled, 5s, start);

    failures += check(!transport.poll(start + 4999ms) && writes.empty() && !cancelled,
                      "SSE transport emitted a heartbeat before the quiet interval");
    failures +=
        check(!transport.poll(start + 5s) &&
                  writes == std::vector<std::string>{std::string(SseTransport::kHeartbeatComment)},
              "SSE transport did not emit the standard comment heartbeat on time");
    failures += check(!transport.poll(start + 9999ms) && writes.size() == 1,
                      "SSE heartbeat did not reset the quiet interval");

    transport.write("data: token\n\n", start + 10s);
    failures += check(!transport.poll(start + 14999ms) && writes.size() == 2,
                      "normal SSE output did not reset the heartbeat interval");
    failures += check(!transport.poll(start + 15s) && writes.size() == 3 &&
                          writes.back() == SseTransport::kHeartbeatComment,
                      "quiet SSE output did not resume heartbeats after a data event");

    transport.write(std::vector<std::string>{"data: one\n\n", "data: two\n\n"}, start + 16s);
    failures +=
        check(writes.size() == 5 && writes[3] == "data: one\n\n" && writes[4] == "data: two\n\n",
              "SSE transport changed ordered multi-event output");

    writable = false;
    failures += check(transport.poll(start + 16s) && cancelled,
                      "unwritable SSE transport did not cancel its request");
    bool cancelled_write_threw = false;
    try {
        transport.write("data: unreachable\n\n", start + 16s);
    } catch (const ClientDisconnected&) { cancelled_write_threw = true; }
    failures += check(cancelled_write_threw && writes.size() == 5,
                      "cancelled SSE transport accepted another event");

    httplib::DataSink failed_sink;
    failed_sink.write       = [](const char*, std::size_t) { return false; };
    failed_sink.is_writable = [] { return true; };
    std::atomic<bool> heartbeat_cancelled{false};
    SseTransport failed_heartbeat(failed_sink, heartbeat_cancelled, 5s, start);
    failures += check(failed_heartbeat.poll(start + 5s) && heartbeat_cancelled,
                      "failed SSE heartbeat did not cancel its request");

    write_ok = false;
    writable = true;
    std::atomic<bool> event_cancelled{false};
    SseTransport failed_event(sink, event_cancelled, 5s, start);
    bool failed_event_threw = false;
    try {
        failed_event.write("data: failed\n\n", start);
    } catch (const ClientDisconnected&) { failed_event_threw = true; }
    failures += check(failed_event_threw && event_cancelled,
                      "failed SSE event write did not cancel its request");

    std::atomic<bool> releaser_cancelled{true};
    SseTransport released(sink, releaser_cancelled, 5s, start);
    failures += check(released.poll(start),
                      "response resource cancellation was not observed by SSE transport");
    return failures;
}

int test_sse_response_headers() {
    httplib::Response response;
    ninfer::serve::prepare_sse_response(response);
    return check(response.get_header_value("Cache-Control") == "no-cache" &&
                     response.get_header_value("X-Accel-Buffering") == "no",
                 "SSE response lost its no-cache or anti-buffering header");
}

int test_prompt_json_member_order() {
    httplib::Request request;
    request.body =
        R"({"model":"claude-local","tools":[{"name":"probe","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}}}],"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"probe","input":{"zeta":"last","alpha":{"yankee":2,"bravo":true}}}]}],"max_tokens":32})";

    const auto parsed = ninfer::serve::parse_json_body(request);
    return check(
        parsed.at("tools").at(0).at("input_schema").at("properties").dump() ==
                R"({"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}})" &&
            parsed.at("messages").at(0).at("content").at(0).at("input").dump() ==
                R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})",
        "HTTP request JSON changed prompt-bearing object member order");
}

int test_schema_number_precision() {
    int failures = 0;
    const std::vector<std::pair<std::string, std::string>> cases{
        {R"({"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"number","minimum":0.10000000000000001}}}})",
         "response_format.json_schema.schema/minimum"},
        {R"({"text":{"format":{"type":"json_schema","name":"x","schema":{"type":"number","minimum":1e-400}}}})",
         "text.format.schema/minimum"},
        {R"({"output_config":{"format":{"type":"json_schema","schema":{"type":"number","maximum":0.10000000000000001}}}})",
         "output_config.format.schema/maximum"},
        {R"({"tools":[{"type":"function","function":{"name":"x","strict":true,"parameters":{"type":"object","properties":{"x":{"minimum":0.10000000000000001}}}}}]})",
         "tools/0/function/parameters/properties/x/minimum"},
        {R"({"tools":[{"type":"namespace","name":"n","tools":[{"type":"function","name":"x","strict":true,"parameters":{"type":"object","properties":{"x":{"minimum":0.10000000000000001}}}}]}]})",
         "tools/0/tools/0/parameters/properties/x/minimum"},
        {R"({"tools":[{"name":"x","strict":true,"input_schema":{"type":"object","properties":{"x":{"minimum":0.10000000000000001}}}}]})",
         "tools/0/input_schema/properties/x/minimum"},
        {R"({"temperature":0.10000000000000001,"tools":[{"name":"x","strict":false,"input_schema":{"minimum":0.10000000000000001}}]})",
         ""},
        {R"({"response_format":{"json_schema":{"schema":{"type":"number","default":0.10000000000000001,"minimum":0.1}}}})",
         ""},
    };
    for (const auto& [body, expected] : cases) {
        httplib::Request request;
        request.body = body;
        try {
            (void)ninfer::serve::parse_json_body(request);
            failures += check(expected.empty(), "HTTP rounded a schema number before validation");
        } catch (const ninfer::serve::ApiException& error) {
            failures +=
                check(!expected.empty() && error.error().code == "unsupported_json_schema" &&
                          error.error().param == expected,
                      "schema precision error lost its protocol location");
        }
    }
    return failures;
}

#if defined(__linux__) || defined(_WIN32)
#if defined(_WIN32)
using TestSocket                    = SOCKET;
constexpr TestSocket kInvalidSocket = INVALID_SOCKET;
int close_test_socket(TestSocket socket) { return ::closesocket(socket); }
#else
using TestSocket                    = int;
constexpr TestSocket kInvalidSocket = -1;
int close_test_socket(TestSocket socket) { return ::close(socket); }
#endif

class Socket final {
public:
    explicit Socket(TestSocket descriptor = kInvalidSocket) : descriptor_(descriptor) {}

    ~Socket() {
        if (descriptor_ != kInvalidSocket) { close_test_socket(descriptor_); }
    }

    Socket(const Socket&)            = delete;
    Socket& operator=(const Socket&) = delete;

    [[nodiscard]] TestSocket get() const noexcept { return descriptor_; }

private:
    TestSocket descriptor_ = kInvalidSocket;
};

template <class T>
bool socket_option_equals(TestSocket socket, int level, int option, T expected) {
    T actual{};
#if defined(_WIN32)
    int size = sizeof(actual);
    if (::getsockopt(socket, level, option, reinterpret_cast<char*>(&actual), &size) != 0) {
        return false;
    }
#else
    socklen_t size = sizeof(actual);
    if (::getsockopt(socket, level, option, &actual, &size) != 0) { return false; }
#endif
    return size == sizeof(actual) && actual == expected;
}

// Binds a configured listener to an ephemeral loopback port and returns the bound address.
sockaddr_in listen_on_loopback(TestSocket listener) {
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener, 4) != 0) {
        return {};
    }
#if defined(_WIN32)
    int size = sizeof(address);
#else
    socklen_t size = sizeof(address);
#endif
    if (::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) != 0) { return {}; }
    return address;
}

// Completes a server-first close so the listening port carries the closed connection.
bool leave_server_closed_connection(TestSocket listener, const sockaddr_in& address) {
    Socket client(::socket(AF_INET, SOCK_STREAM, 0));
    if (client.get() == kInvalidSocket ||
        ::connect(client.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        return false;
    }
    Socket accepted(::accept(listener, nullptr, nullptr));
    if (accepted.get() == kInvalidSocket) { return false; }
    char byte = 0;
#if defined(_WIN32)
    const int send_flag = SD_SEND;
#else
    const int send_flag = SHUT_WR;
#endif
    return ::shutdown(accepted.get(), send_flag) == 0 && ::recv(client.get(), &byte, 1, 0) == 0 &&
           ::shutdown(client.get(), send_flag) == 0 && ::recv(accepted.get(), &byte, 1, 0) == 0;
}

int last_socket_error() {
#if defined(_WIN32)
    return ::WSAGetLastError();
#else
    return errno;
#endif
}
#endif

#if defined(__linux__)
int test_inherited_socket_liveness() {
    int failures = 0;
    Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
    if (listener.get() == kInvalidSocket) {
        return check(false, "failed to create HTTP listener test socket");
    }
    ninfer::serve::configure_http_server_socket(listener.get());

    const auto address = listen_on_loopback(listener.get());
    if (address.sin_port == 0) { return check(false, "failed to bind HTTP listener test socket"); }

    Socket client(::socket(AF_INET, SOCK_STREAM, 0));
    if (client.get() == kInvalidSocket ||
        ::connect(client.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        return check(false, "failed to connect HTTP liveness test socket");
    }
    Socket accepted(::accept(listener.get(), nullptr, nullptr));
    if (accepted.get() == kInvalidSocket) {
        return check(false, "failed to accept HTTP liveness test socket");
    }

    failures += check(socket_option_equals(accepted.get(), SOL_SOCKET, SO_KEEPALIVE, 1),
                      "accepted HTTP socket did not inherit SO_KEEPALIVE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPIDLE, 10),
                      "accepted HTTP socket did not inherit TCP_KEEPIDLE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPINTVL, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPINTVL");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPCNT, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPCNT");
    failures += check(
        socket_option_equals<unsigned int>(accepted.get(), IPPROTO_TCP, TCP_USER_TIMEOUT, 15000U),
        "accepted HTTP socket did not inherit TCP_USER_TIMEOUT");
    return failures;
}

using SocketSetup = void (*)(ninfer::serve::socket_t);

void reuse_address_only(ninfer::serve::socket_t socket) {
    const int enabled = 1;
    (void)::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
}

int test_listener_reuse() {
    const SocketSetup configure = ninfer::serve::configure_http_server_socket;

    struct Case {
        SocketSetup previous, next;
    };

    int failures = 0;
    for (const Case c : {Case{configure, reuse_address_only}, Case{reuse_address_only, configure},
                         Case{configure, configure}}) {
        sockaddr_in address{};
        {
            Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
            if (listener.get() == kInvalidSocket) {
                failures += check(false, "failed to create HTTP listener test socket");
                continue;
            }
            c.previous(listener.get());
            address = listen_on_loopback(listener.get());
            if (address.sin_port == 0 ||
                !leave_server_closed_connection(listener.get(), address)) {
                failures += check(false, "failed to establish a closed HTTP connection");
                continue;
            }
        }
        Socket successor(::socket(AF_INET, SOCK_STREAM, 0));
        c.next(successor.get());
        failures += check(
            ::bind(successor.get(), reinterpret_cast<const sockaddr*>(&address),
                   sizeof(address)) == 0,
            "closed listener prevented immediate port reuse");
    }
    Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
    configure(listener.get());
    const auto address = listen_on_loopback(listener.get());
    if (address.sin_port == 0) { return failures + check(false, "failed to bind active listener"); }
    Socket duplicate(::socket(AF_INET, SOCK_STREAM, 0));
    configure(duplicate.get());
    const int result =
        ::bind(duplicate.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address));
    failures += check(result == -1 && last_socket_error() == EADDRINUSE,
                      "HTTP listener allowed another server to share its active port");
    return failures;
}
#elif defined(_WIN32)
int test_windows_listener_privacy() {
    int failures = 0;
    {
        Socket probe(::socket(AF_INET, SOCK_STREAM, 0));
        if (probe.get() == kInvalidSocket) {
            return check(false, "failed to create HTTP listener test socket");
        }
        ninfer::serve::configure_http_server_socket(probe.get());
        failures += check(socket_option_equals(probe.get(), SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1),
                          "HTTP listener did not request SO_EXCLUSIVEADDRUSE");
        failures += check(
            !socket_option_equals(probe.get(), SOL_SOCKET, SO_REUSEADDR, 1),
            "HTTP listener enabled SO_REUSEADDR, which lets another process take its port");
    }
    for (const bool closed_connection : {false, true}) {
        sockaddr_in address{};
        {
            Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
            if (listener.get() == kInvalidSocket) {
                failures += check(false, "failed to create HTTP listener test socket");
                continue;
            }
            ninfer::serve::configure_http_server_socket(listener.get());
            address = listen_on_loopback(listener.get());
            if (address.sin_port == 0 ||
                (closed_connection && !leave_server_closed_connection(listener.get(), address))) {
                failures += check(false, "failed to prepare a closed HTTP listener");
                continue;
            }
        }
        Socket successor(::socket(AF_INET, SOCK_STREAM, 0));
        ninfer::serve::configure_http_server_socket(successor.get());
        failures += check(
            ::bind(successor.get(), reinterpret_cast<const sockaddr*>(&address),
                   sizeof(address)) == 0,
            closed_connection
                ? "closed connection prevented port reuse after listener shutdown"
                : "closed listener prevented immediate port reuse");
    }
    {
        Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
        if (listener.get() == kInvalidSocket) {
            return failures + check(false, "failed to create HTTP listener test socket");
        }
        ninfer::serve::configure_http_server_socket(listener.get());
        const auto address = listen_on_loopback(listener.get());
        if (address.sin_port == 0) {
            return failures + check(false, "failed to bind active listener");
        }
        Socket duplicate(::socket(AF_INET, SOCK_STREAM, 0));
        const int enabled = 1;
        (void)::setsockopt(duplicate.get(), SOL_SOCKET, SO_REUSEADDR,
                           reinterpret_cast<const char*>(&enabled), sizeof(enabled));
        const int result =
            ::bind(duplicate.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address));
        failures += check(result != 0, "HTTP listener allowed another socket to take its active port");
    }
    return failures;
}
#endif

} // namespace

int main() {
#if defined(_WIN32)
    WSADATA winsock{};
    if (::WSAStartup(MAKEWORD(2, 2), &winsock) != 0) {
        std::cerr << "failed to initialize Winsock\n";
        return 1;
    }
#endif
    int failures = test_sse_transport() + test_sse_response_headers() +
                   test_prompt_json_member_order() + test_schema_number_precision();
#if defined(__linux__)
    failures += test_inherited_socket_liveness();
    failures += test_listener_reuse();
#elif defined(_WIN32)
    failures += test_windows_listener_privacy();
#endif
    if (failures == 0) { std::cout << "ok\n"; }
#if defined(_WIN32)
    ::WSACleanup();
#endif
    return failures == 0 ? 0 : 1;
}
