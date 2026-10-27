// P1-a: the RESP transport, checked against a real Redis (the LMCache remote tier) when one is
// configured. The encoding checks always run; the round trip runs only when NINFER_RESP_PORT is set,
// so a machine without a Redis still passes the suite (return 77 = skipped in CTest).

#include "core/resp.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main() {
    int failures = 0;
    const auto expect = [&](bool condition, const char* what) {
        std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
        if (!condition) { ++failures; }
    };

    {
        const std::string_view args[]{"SET", "k", "v"};
        expect(ninfer::core::resp_encode(args) == "*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n",
               "a three-argument command encodes as a RESP array of bulk strings");
    }
    {
        // A value that carries CRLF and NUL: the length prefix is what keeps it intact.
        const std::string payload("\r\n\0\0\0", 5);
        const std::string_view args[]{"SET", "bin", {payload.data(), payload.size()}};
        std::string expected = "*3\r\n$3\r\nSET\r\n$3\r\nbin\r\n$5\r\n";
        expected.append(payload);
        expected.append("\r\n", 2);
        expect(ninfer::core::resp_encode(args) == expected,
               "a value with CRLF and NUL keeps its length prefix and its bytes");
    }

    const char* port_text = std::getenv("NINFER_RESP_PORT");
    if (port_text == nullptr || *port_text == '\0') {
        std::printf("SKIP no server: set NINFER_RESP_PORT to run the round trip\n");
        return failures == 0 ? 77 : 1;
    }
    const char* host_text = std::getenv("NINFER_RESP_HOST");
    const std::string host = (host_text != nullptr && *host_text != '\0') ? host_text : "127.0.0.1";
    const auto port = static_cast<std::uint16_t>(std::atoi(port_text));

    ninfer::core::RespClient client;
    if (!client.connect_to(host, port)) {
        std::printf("FAIL connect to %s:%u: %s\n", host.c_str(), static_cast<unsigned>(port),
                    client.error().c_str());
        return 1;
    }
    expect(client.ping(), "PING answers +PONG");

    const std::string key = "ninfer:selftest:" + std::to_string(port);
    std::vector<std::byte> payload(1u << 20);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<std::byte>((i * 31u + 7u) & 0xFFu);
    }
    const auto written = client.set(key, payload);
    expect(written.has_value() && *written == payload.size(), "SET stores a 1 MiB binary value");
    expect(client.exists(key).value_or(false), "EXISTS sees the stored key");

    std::vector<std::byte> read_back(payload.size());
    const auto got = client.get(key, read_back);
    expect(got.has_value() && *got == payload.size(), "GET returns the full payload");
    expect(std::memcmp(read_back.data(), payload.data(), payload.size()) == 0,
           "the bytes survive the round trip");

    std::vector<std::byte> wrong_size(payload.size() / 2);
    expect(!client.get(key, wrong_size).has_value(),
           "GET into a too-small buffer refuses instead of truncating");

    std::vector<std::byte> missing(64);
    expect(!client.get("ninfer:selftest:absent", missing).has_value(), "GET of an absent key misses");

    expect(client.del(key).value_or(false), "DEL removes the key");
    expect(client.exists(key).value_or(true) == false, "EXISTS then reports the key gone");

    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
