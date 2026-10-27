#pragma once

// A RESP2 client, the transport LMCache's remote tier actually speaks.
//
// Why this exists (P1-a of the plan): LMCache's data plane has two shapes, and only one of them is
// reachable from a C++ engine. The in-process connector needs vLLM; the remote backend is plain
// Redis/Valkey over RESP2, where a chunk is a string key and the value is the raw KV bytes
// (lmcache/v1/storage_backend/connector/redis_connector.py, RESPConnector._put/_get:
// `client.set(key.to_string(), memory_obj.byte_array)` and the mirror on the way back). So the
// whole integration surface is: AUTH, SET, GET, EXISTS, DEL on binary-safe bulk strings.
//
// Deliberately narrow: one connection, no pipelining, no cluster redirection, no RESP3. The
// batched forms exist in LMCache's client for throughput; they are P5 work, and adding them before
// a single round trip is measured would be guessing.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace ninfer::core {

// Encodes one command as a RESP array of bulk strings. Binary safe: the value may contain NUL and
// CRLF, which is why the length prefix is used instead of any terminator.
[[nodiscard]] inline std::string resp_encode(std::span<const std::string_view> args) {
    std::string out;
    out.reserve(64);
    out += '*';
    out += std::to_string(args.size());
    out += "\r\n";
    for (const auto arg : args) {
        out += '$';
        out += std::to_string(arg.size());
        out += "\r\n";
        out.append(arg.data(), arg.size());
        out += "\r\n";
    }
    return out;
}

struct RespReply {
    char type = 0;          // '+', '-', ':', '$', '*'
    std::string text;       // simple string, error message, or bulk payload
    long long number = 0;   // ':' replies
    bool null = false;      // null bulk / null array
};

class RespClient {
public:
    RespClient() = default;
    ~RespClient() { close(); }
    RespClient(const RespClient&)            = delete;
    RespClient& operator=(const RespClient&) = delete;

    // Connects with a receive/send timeout. Returns false and fills error() on failure.
    bool connect_to(std::string host, std::uint16_t port, int timeout_ms = 2000) {
        close();
        error_.clear();
#ifdef _WIN32
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            error_ = "WSAStartup failed";
            return false;
        }
        wsa_started_ = true;
#endif
        socket_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (!valid()) {
            error_ = "socket() failed";
            return false;
        }
        timeval tv{};
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ::setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof tv);
        ::setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&tv), sizeof tv);

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port   = htons(port);
        if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
            error_ = "bad address: " + host;
            close();
            return false;
        }
        if (::connect(socket_, reinterpret_cast<const sockaddr*>(&address), sizeof address) != 0) {
            error_ = "connect failed";
            close();
            return false;
        }
        host_       = std::move(host);
        port_       = port;
        timeout_ms_ = timeout_ms;
        io_failed_  = false;
        int one = 1;
        ::setsockopt(socket_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
        return true;
    }

    void close() {
        if (valid()) {
#ifdef _WIN32
            ::closesocket(socket_);
#else
            ::close(socket_);
#endif
            socket_ = invalid_socket();
        }
#ifdef _WIN32
        if (wsa_started_) {
            WSACleanup();
            wsa_started_ = false;
        }
#endif
    }

    [[nodiscard]] const std::string& error() const { return error_; }
    // Timeouts sized per call: a 12 MiB checkpoint needs far longer than the 2 s that suits a small
    // block, and the socket timeout is otherwise fixed at connect.
    void set_timeout(int timeout_ms) {
        timeout_ms_ = timeout_ms;
        if (!valid()) { return; }
        timeval tv{};
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ::setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof tv);
        ::setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&tv), sizeof tv);
    }

    bool command(std::span<const std::string_view> args, RespReply& reply) {
        if (!send_command(args, reply)) {
            // A dropped connection is not an answer. Reconnect once and ask again; a real server
            // error comes back as a complete reply and is never retried here.
            if (!io_failed_ || host_.empty()) { return false; }
            if (!connect_to(host_, port_, timeout_ms_)) { return false; }
            io_failed_ = false;
            return send_command(args, reply);
        }
        return true;
    }

    bool send_command(std::span<const std::string_view> args, RespReply& reply) {
        const auto frame = resp_encode(args);
        if (!send_all(frame)) { return false; }
        return read_reply(reply);
    }

    bool ping() {
        RespReply reply;
        const std::string_view args[]{"PING"};
        return command(args, reply) && reply.type == '+';
    }

    bool auth(const std::string& username, const std::string& password) {
        RespReply reply;
        if (username.empty()) {
            const std::string_view args[]{"AUTH", password};
            return command(args, reply) && reply.type == '+';
        }
        const std::string_view args[]{"AUTH", username, password};
        return command(args, reply) && reply.type == '+';
    }

    // Writes a binary value. Returns the number of bytes stored, or nullopt on failure.
    std::optional<std::size_t> set(std::string_view key, std::span<const std::byte> value) {
        RespReply reply;
        const std::string_view args[]{"SET", key, {reinterpret_cast<const char*>(value.data()), value.size()}};
        if (!command(args, reply) || reply.type != '+') { return std::nullopt; }
        return value.size();
    }

    // Reads into the caller's buffer. Returns bytes read, or nullopt on a miss or a size mismatch:
    // a partial copy would hand the caller a silently truncated KV block.
    std::optional<std::size_t> get(std::string_view key, std::span<std::byte> destination) {
        RespReply reply;
        const std::string_view args[]{"GET", key};
        if (!command(args, reply) || reply.type != '$' || reply.null) { return std::nullopt; }
        if (reply.text.size() != destination.size()) {
            error_ = "size mismatch on GET: " + std::to_string(reply.text.size());
            return std::nullopt;
        }
        std::memcpy(destination.data(), reply.text.data(), destination.size());
        return destination.size();
    }

    // Size of a stored value without transferring it: a checkpoint blob is read back into a
    // buffer of exactly this size, so a short buffer never has to be guessed at.
    std::optional<std::size_t> length(std::string_view key) {
        RespReply reply;
        const std::string_view args[]{"STRLEN", key};
        if (!command(args, reply) || reply.type != ':') { return std::nullopt; }
        return static_cast<std::size_t>(reply.number);
    }

    // The lookup stage of the tier protocol: never transfers the payload.
    std::optional<bool> exists(std::string_view key) {
        RespReply reply;
        const std::string_view args[]{"EXISTS", key};
        if (!command(args, reply) || reply.type != ':') { return std::nullopt; }
        return reply.number != 0;
    }

    std::optional<bool> del(std::string_view key) {
        RespReply reply;
        const std::string_view args[]{"DEL", key};
        if (!command(args, reply) || reply.type != ':') { return std::nullopt; }
        return reply.number != 0;
    }

private:
#ifdef _WIN32
    using socket_type = SOCKET;
    static constexpr socket_type invalid_socket() { return INVALID_SOCKET; }
#else
    using socket_type = int;
    static constexpr socket_type invalid_socket() { return -1; }
#endif

    [[nodiscard]] bool valid() const { return socket_ != invalid_socket(); }

    bool send_all(std::string_view bytes) {
        std::size_t sent = 0;
        while (sent < bytes.size()) {
            const auto chunk = ::send(socket_, bytes.data() + sent,
                                      static_cast<int>(bytes.size() - sent), 0);
            if (chunk <= 0) {
                error_ = "send failed";
                io_failed_ = true;
                return false;
            }
            sent += static_cast<std::size_t>(chunk);
        }
        return true;
    }

    bool fill() {
        if (begin_ < end_) { return true; }
        const auto got = ::recv(socket_, buffer_, sizeof buffer_, 0);
        if (got <= 0) {
            error_ = got == 0 ? "connection closed" : "recv failed";
            io_failed_ = true;
            return false;
        }
        begin_ = 0;
        end_   = static_cast<std::size_t>(got);
        return true;
    }

    bool read_byte(char& value) {
        if (!fill()) { return false; }
        value = buffer_[begin_++];
        return true;
    }

    bool read_line(std::string& line) {
        line.clear();
        for (;;) {
            char c = 0;
            if (!read_byte(c)) { return false; }
            if (c == '\r') {
                char next = 0;
                if (!read_byte(next)) { return false; }
                if (next != '\n') { line.push_back(c); line.push_back(next); continue; }
                return true;
            }
            line.push_back(c);
        }
    }

    bool read_exact(std::size_t count, std::string& out) {
        out.clear();
        out.reserve(count);
        while (out.size() < count) {
            if (!fill()) { return false; }
            const auto take = std::min(count - out.size(), end_ - begin_);
            out.append(buffer_ + begin_, take);
            begin_ += take;
        }
        return true;
    }

    bool read_reply(RespReply& reply) {
        reply = RespReply{};
        char type = 0;
        if (!read_byte(type)) { return false; }
        reply.type = type;
        std::string line;
        switch (type) {
            case '+':
            case '-':
                if (!read_line(line)) { return false; }
                reply.text = std::move(line);
                if (type == '-') { error_ = reply.text; }
                return type == '+';
            case ':':
                if (!read_line(line)) { return false; }
                reply.number = std::strtoll(line.c_str(), nullptr, 10);
                return true;
            case '$': {
                if (!read_line(line)) { return false; }
                const auto length = std::strtoll(line.c_str(), nullptr, 10);
                if (length < 0) { reply.null = true; return true; }
                if (!read_exact(static_cast<std::size_t>(length), reply.text)) { return false; }
                char cr = 0, lf = 0;
                return read_byte(cr) && read_byte(lf);
            }
            default:
                error_ = std::string("unexpected reply type: ") + type;
                return false;
        }
    }

    socket_type socket_ = invalid_socket();
#ifdef _WIN32
    bool wsa_started_ = false;
#endif
    std::string error_;
    std::string host_;
    std::uint16_t port_ = 0;
    int timeout_ms_     = 0;
    bool io_failed_     = false;
    char buffer_[64 * 1024]{};
    std::size_t begin_ = 0;
    std::size_t end_   = 0;
};

} // namespace ninfer::core




