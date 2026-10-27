// P1-b-2d-7: the frame layout this engine has to speak to LMCache's ZMQ server.
//
// The exchange is REQ/REP in shape but DEALER/ROUTER on the wire: the client sends
// [request_uid, operation, *payloads] and the server answers [request_uid, operation, *response].
// Every one of those is its own ZMQ frame, and each frame holds a *separately* msgpack-encoded
// scalar -- the uid and the operation are not packed together with each other or with the
// payloads. Get that wrong and the server neither answers nor complains into the socket: the
// frame is simply dropped, so a hang is the only symptom. Hence this probe: one ping, decoded
// field by field, so a hang becomes a diagnosis instead of a mystery.
//
// The msgpack subset here is deliberately hand written. It is nine value forms on the way out
// (nil, bool, positive fixint, unsigned, negative int, str, bin, array) and it has to stay small:
// every byte this engine puts on that socket is on the critical path of a cache lookup.

#include <zmq.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

// --- msgpack ---------------------------------------------------------------

void put_u8(std::string& out, std::uint8_t value) {
    out.push_back(static_cast<char>(value));
}

void put_be(std::string& out, std::uint64_t value, int bytes) {
    for (int i = bytes - 1; i >= 0; --i) {
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xFFu));
    }
}

std::string encode_nil() {
    return std::string(1, static_cast<char>(0xC0));
}

std::string encode_bool(bool value) {
    return std::string(1, static_cast<char>(value ? 0xC3 : 0xC2));
}

std::string encode_int(std::int64_t value) {
    std::string out;
    if (value >= 0) {
        if (value < 0x80) {
            put_u8(out, static_cast<std::uint8_t>(value));
        } else if (value <= 0xFF) {
            put_u8(out, 0xCC);
            put_u8(out, static_cast<std::uint8_t>(value));
        } else if (value <= 0xFFFF) {
            put_u8(out, 0xCD);
            put_be(out, static_cast<std::uint64_t>(value), 2);
        } else if (value <= 0xFFFFFFFFll) {
            put_u8(out, 0xCE);
            put_be(out, static_cast<std::uint64_t>(value), 4);
        } else {
            put_u8(out, 0xCF);
            put_be(out, static_cast<std::uint64_t>(value), 8);
        }
        return out;
    }
    if (value >= -32) {
        put_u8(out, static_cast<std::uint8_t>(value & 0xFF));
    } else if (value >= -128) {
        put_u8(out, 0xD0);
        put_u8(out, static_cast<std::uint8_t>(value & 0xFF));
    } else if (value >= -32768) {
        put_u8(out, 0xD1);
        put_be(out, static_cast<std::uint64_t>(value) & 0xFFFFu, 2);
    } else if (value >= -2147483648ll) {
        put_u8(out, 0xD2);
        put_be(out, static_cast<std::uint64_t>(value) & 0xFFFFFFFFu, 4);
    } else {
        put_u8(out, 0xD3);
        put_be(out, static_cast<std::uint64_t>(value), 8);
    }
    return out;
}

std::string encode_str(std::string_view value) {
    std::string out;
    const auto size = value.size();
    if (size < 32) {
        put_u8(out, static_cast<std::uint8_t>(0xA0u | size));
    } else if (size <= 0xFF) {
        put_u8(out, 0xD9);
        put_u8(out, static_cast<std::uint8_t>(size));
    } else if (size <= 0xFFFF) {
        put_u8(out, 0xDA);
        put_be(out, size, 2);
    } else {
        put_u8(out, 0xDB);
        put_be(out, size, 4);
    }
    out.append(value.data(), value.size());
    return out;
}

std::string encode_bin(std::string_view value) {
    std::string out;
    const auto size = value.size();
    if (size <= 0xFF) {
        put_u8(out, 0xC4);
        put_u8(out, static_cast<std::uint8_t>(size));
    } else if (size <= 0xFFFF) {
        put_u8(out, 0xC5);
        put_be(out, size, 2);
    } else {
        put_u8(out, 0xC6);
        put_be(out, size, 4);
    }
    out.append(value.data(), value.size());
    return out;
}

struct Value {
    enum class Kind { Nil, Bool, Int, Str, Bin, Array, Unsupported };
    Kind kind{Kind::Unsupported};
    bool boolean{false};
    std::int64_t integer{0};
    std::string bytes;         // payload of a Str or a Bin
    std::vector<Value> items;  // elements of an Array
};

const char* kind_name(Value::Kind kind) {
    switch (kind) {
    case Value::Kind::Nil: return "nil";
    case Value::Kind::Bool: return "bool";
    case Value::Kind::Int: return "int";
    case Value::Kind::Str: return "str";
    case Value::Kind::Bin: return "bin";
    case Value::Kind::Array: return "array";
    default: return "unsupported";
    }
}

bool read_be(const std::string& in, std::size_t& off, int bytes, std::uint64_t& value) {
    if (off + static_cast<std::size_t>(bytes) > in.size()) {
        return false;
    }
    value = 0;
    for (int i = 0; i < bytes; ++i) {
        value = (value << 8) | static_cast<std::uint8_t>(in[off + static_cast<std::size_t>(i)]);
    }
    off += static_cast<std::size_t>(bytes);
    return true;
}

bool decode(const std::string& in, std::size_t& off, Value& out) {
    if (off >= in.size()) {
        return false;
    }
    const auto tag = static_cast<std::uint8_t>(in[off]);

    if (tag <= 0x7F) {  // positive fixint
        out.kind = Value::Kind::Int;
        out.integer = tag;
        ++off;
        return true;
    }
    if (tag >= 0xE0) {  // negative fixint
        out.kind = Value::Kind::Int;
        out.integer = static_cast<std::int8_t>(tag);
        ++off;
        return true;
    }
    if ((tag & 0xF0) == 0xA0) {  // fixstr
        const std::size_t size = tag & 0x0F;
        ++off;
        if (off + size > in.size()) {
            return false;
        }
        out.kind = Value::Kind::Str;
        out.bytes = in.substr(off, size);
        off += size;
        return true;
    }
    if ((tag & 0xF0) == 0x90) {  // fixarray
        const std::size_t count = tag & 0x0F;
        ++off;
        out.kind = Value::Kind::Array;
        out.items.clear();
        for (std::size_t i = 0; i < count; ++i) {
            Value item;
            if (!decode(in, off, item)) {
                return false;
            }
            out.items.push_back(std::move(item));
        }
        return true;
    }

    ++off;
    std::uint64_t raw = 0;
    std::size_t size = 0;
    switch (tag) {
    case 0xC0:
        out.kind = Value::Kind::Nil;
        return true;
    case 0xC2:
    case 0xC3:
        out.kind = Value::Kind::Bool;
        out.boolean = tag == 0xC3;
        return true;
    case 0xCC:
    case 0xCD:
    case 0xCE:
    case 0xCF:
        if (!read_be(in, off, 1 << (tag - 0xCC), raw)) {
            return false;
        }
        out.kind = Value::Kind::Int;
        out.integer = static_cast<std::int64_t>(raw);
        return true;
    case 0xD0:
        if (!read_be(in, off, 1, raw)) {
            return false;
        }
        out.kind = Value::Kind::Int;
        out.integer = static_cast<std::int8_t>(static_cast<std::uint8_t>(raw));
        return true;
    case 0xD1:
    case 0xD2:
    case 0xD3: {
        const int bytes = 2 << (tag - 0xD1);
        if (!read_be(in, off, bytes, raw)) {
            return false;
        }
        out.kind = Value::Kind::Int;
        if (bytes == 8) {
            out.integer = static_cast<std::int64_t>(raw);
        } else {
            // Sign extend from the width that was encoded, so -40000 does not read back as
            // 4294927296.
            const std::uint64_t sign_bit = std::uint64_t{1} << (bytes * 8 - 1);
            const std::uint64_t extend_mask = ~((std::uint64_t{1} << (bytes * 8)) - 1);
            out.integer = (raw & sign_bit) != 0
                              ? static_cast<std::int64_t>(raw | extend_mask)
                              : static_cast<std::int64_t>(raw);
        }
        return true;
    }
    case 0xD9:
    case 0xDA:
    case 0xDB: {
        const int bytes = tag == 0xD9 ? 1 : (tag == 0xDA ? 2 : 4);
        if (!read_be(in, off, bytes, raw)) {
            return false;
        }
        size = static_cast<std::size_t>(raw);
        if (off + size > in.size()) {
            return false;
        }
        out.kind = Value::Kind::Str;
        out.bytes = in.substr(off, size);
        off += size;
        return true;
    }
    case 0xC4:
    case 0xC5:
    case 0xC6: {
        const int bytes = tag == 0xC4 ? 1 : (tag == 0xC5 ? 2 : 4);
        if (!read_be(in, off, bytes, raw)) {
            return false;
        }
        size = static_cast<std::size_t>(raw);
        if (off + size > in.size()) {
            return false;
        }
        out.kind = Value::Kind::Bin;
        out.bytes = in.substr(off, size);
        off += size;
        return true;
    }
    case 0xDC:
    case 0xDD: {
        const int bytes = tag == 0xDC ? 2 : 4;
        if (!read_be(in, off, bytes, raw)) {
            return false;
        }
        out.kind = Value::Kind::Array;
        out.items.clear();
        for (std::uint64_t i = 0; i < raw; ++i) {
            Value item;
            if (!decode(in, off, item)) {
                return false;
            }
            out.items.push_back(std::move(item));
        }
        return true;
    }
    default:
        out.kind = Value::Kind::Unsupported;
        out.integer = tag;
        return true;
    }
}

std::string hex_of(std::string_view bytes, std::size_t limit) {
    std::string out;
    char buffer[4];
    const std::size_t shown = bytes.size() < limit ? bytes.size() : limit;
    for (std::size_t i = 0; i < shown; ++i) {
        std::snprintf(buffer, sizeof(buffer), "%02X", static_cast<std::uint8_t>(bytes[i]));
        out.append(buffer);
        if (i + 1 < shown) {
            out.push_back(' ');
        }
    }
    if (shown < bytes.size()) {
        out.append(" ...");
    }
    return out;
}

// --- self test -------------------------------------------------------------

int failures = 0;

void expect(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
    if (!condition) {
        ++failures;
    }
}

// Each case pins both directions: the exact bytes msgspec would emit, and that they read back.
void check_int(std::int64_t value, const std::string& wire, const char* what) {
    const std::string encoded = encode_int(value);
    expect(encoded == wire, what);
    std::size_t off = 0;
    Value decoded;
    const bool ok = decode(encoded, off, decoded) && off == encoded.size() &&
                    decoded.kind == Value::Kind::Int && decoded.integer == value;
    expect(ok, "  ...and reads back as the same integer");
}

void self_test() {
    // 22 is the legacy operation id for ping, so its encoding is not incidental.
    check_int(22, std::string(1, '\x16'), "positive fixint: 22 encodes as 0x16");
    check_int(-1, std::string(1, '\xFF'), "negative fixint: -1 encodes as 0xff");
    // A non-negative value widens into the unsigned tags, which is what msgspec emits; the signed
    // ones only appear below zero. Both directions have to be right or the uid frame is misread.
    check_int(300, std::string("\xCD\x01\x2C", 3), "uint16: 300 encodes as 0xcd 0x01 0x2c");
    check_int(-1000, std::string("\xD1\xFC\x18", 3), "int16: -1000 encodes as 0xd1 0xfc 0x18");
    check_int(-40000, std::string("\xD2\xFF\xFF\x63\xC0", 5),
              "int32: -40000 encodes as 0xd2 0xffff63c0");

    {
        const std::string wire(1, static_cast<char>(0xC0));
        expect(encode_nil() == wire, "nil encodes as 0xc0");
        std::size_t off = 0;
        Value decoded;
        expect(decode(wire, off, decoded) && decoded.kind == Value::Kind::Nil,
               "  ...and reads back as nil");
    }
    {
        expect(encode_bool(true) == std::string(1, '\xC3'), "bool true encodes as 0xc3");
        expect(encode_bool(false) == std::string(1, '\xC2'), "bool false encodes as 0xc2");
    }
    {
        expect(encode_str("ping") == std::string("\xA4ping", 5), "str 'ping' encodes as 0xa4 70 69 6e 67");
        std::size_t off = 0;
        Value decoded;
        expect(decode(encode_str("ping"), off, decoded) && decoded.kind == Value::Kind::Str &&
                   decoded.bytes == "ping",
               "  ...and reads back as the same string");
    }
    {
        const std::string payload("\x00\x01\xFE\xFF", 4);
        expect(encode_bin(payload) == std::string("\xC4\x04\x00\x01\xFE\xFF", 6),
               "bin of four bytes encodes as 0xc4 0x04 and is not treated as text");
        std::size_t off = 0;
        Value decoded;
        expect(decode(encode_bin(payload), off, decoded) && decoded.kind == Value::Kind::Bin &&
                   decoded.bytes == payload,
               "  ...and reads back byte for byte");
    }
    {
        // The frame the probe actually sends, encoded as one array, must round trip as three
        // values -- this is the last chance to catch an encoder that cannot describe a request.
        std::string request(1, static_cast<char>(0x93));
        request += encode_int(1);
        request += encode_int(22);
        request += encode_nil();
        std::size_t off = 0;
        Value decoded;
        const bool ok = decode(request, off, decoded) && decoded.items.size() == 3 &&
                        decoded.items[1].integer == 22 && decoded.items[2].kind == Value::Kind::Nil;
        expect(ok, "a three-element request round trips as uid, operation, payload");
    }
}

// --- zmq -------------------------------------------------------------------

struct Reply {
    std::vector<std::string> parts;
    bool received{false};
    std::string error;
};

bool send_frames(void* socket, const std::vector<std::string>& parts) {
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const int flags = (i + 1 < parts.size()) ? ZMQ_SNDMORE : 0;
        if (zmq_send(socket, parts[i].data(), parts[i].size(), flags) < 0) {
            return false;
        }
    }
    return true;
}

Reply receive_frames(void* socket) {
    Reply reply;
    while (true) {
        zmq_msg_t message;
        zmq_msg_init(&message);
        if (zmq_msg_recv(&message, socket, 0) < 0) {
            reply.error = zmq_strerror(zmq_errno());
            zmq_msg_close(&message);
            return reply;
        }
        reply.parts.emplace_back(static_cast<const char*>(zmq_msg_data(&message)),
                                 zmq_msg_size(&message));
        const bool more = zmq_msg_more(&message) != 0;
        zmq_msg_close(&message);
        if (!more || reply.parts.size() >= 64) {
            break;
        }
    }
    reply.received = true;
    return reply;
}

// One request/response turn, with the whole frame list printed so a mismatch is visible rather
// than inferred from a timeout.
Reply call(void* socket, const std::string& label, const std::vector<std::string>& frames) {
    std::printf("send %s: %zu frames", label.c_str(), frames.size());
    for (const std::string& frame : frames) {
        std::printf(" [%s]", hex_of(frame, 24).c_str());
    }
    std::printf("\n");
    if (!send_frames(socket, frames)) {
        Reply reply;
        reply.error = zmq_strerror(zmq_errno());
        return reply;
    }
    Reply reply = receive_frames(socket);
    if (!reply.received) {
        std::printf("recv %s: nothing (%s)\n", label.c_str(), reply.error.c_str());
        return reply;
    }
    std::printf("recv %s: %zu frames", label.c_str(), reply.parts.size());
    for (const std::string& frame : reply.parts) {
        std::printf(" [%s]", hex_of(frame, 32).c_str());
    }
    std::printf("\n");
    return reply;
}

// The contract: the first frame echoes the uid, the second echoes the operation, the rest is the
// response. Anything else means the frame layout is wrong even if frames arrived.
bool check_envelope(const Reply& reply, std::int64_t uid, std::int64_t operation, Value& body) {
    if (!reply.received || reply.parts.size() < 2) {
        return false;
    }
    std::size_t off = 0;
    Value got_uid;
    if (!decode(reply.parts[0], off, got_uid) || got_uid.kind != Value::Kind::Int ||
        got_uid.integer != uid) {
        std::printf("     uid frame does not echo %lld\n", static_cast<long long>(uid));
        return false;
    }
    off = 0;
    Value got_operation;
    if (!decode(reply.parts[1], off, got_operation) || got_operation.kind != Value::Kind::Int ||
        got_operation.integer != operation) {
        std::printf("     operation frame does not echo %lld\n", static_cast<long long>(operation));
        return false;
    }
    if (reply.parts.size() < 3) {
        // A nil response is sent as a bare prefix, so an absent third frame is legitimate.
        body = Value{};
        return true;
    }
    off = 0;
    return decode(reply.parts[2], off, body);
}

int probe(const char* endpoint) {
    std::printf("connecting to %s\n", endpoint);
    void* context = zmq_ctx_new();
    if (context == nullptr) {
        std::printf("FAIL zmq_ctx_new: %s\n", zmq_strerror(zmq_errno()));
        return 1;
    }
    void* socket = zmq_socket(context, ZMQ_DEALER);
    if (socket == nullptr) {
        std::printf("FAIL zmq_socket: %s\n", zmq_strerror(zmq_errno()));
        zmq_ctx_term(context);
        return 1;
    }
    const int linger = 0;
    const int timeout = 5000;
    const int reconnect = 100;
    zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
    zmq_setsockopt(socket, ZMQ_RECONNECT_IVL, &reconnect, sizeof(reconnect));

    if (zmq_connect(socket, endpoint) < 0) {
        std::printf("FAIL connect: %s\n", zmq_strerror(zmq_errno()));
        zmq_close(socket);
        zmq_ctx_term(context);
        return 1;
    }

    // ping carries one payload: the caller's instance id, or nil for an untracked prober.
    const Reply ping_reply = call(socket, "ping", {encode_int(1), encode_int(22), encode_nil()});
    Value ping_body;
    const bool ping_ok = check_envelope(ping_reply, 1, 22, ping_body);
    expect(ping_ok, "ping answers with the uid and operation it was given");
    expect(ping_ok && ping_body.kind == Value::Kind::Bool && ping_body.boolean,
           "  ...and the response body is bool true");

    // A second turn proves the socket still pairs requests with responses, which is what the four
    // engine-driven operations will depend on. It also returns the chunk size the cache will use.
    const Reply chunk_reply = call(socket, "get_chunk_size", {encode_int(2), encode_int(21)});
    Value chunk_body;
    const bool chunk_ok = check_envelope(chunk_reply, 2, 21, chunk_body);
    expect(chunk_ok && chunk_body.kind == Value::Kind::Int,
           "get_chunk_size answers after ping, with an int body");
    if (chunk_ok && chunk_body.kind == Value::Kind::Int) {
        std::printf("     server chunk size = %lld tokens\n",
                    static_cast<long long>(chunk_body.integer));
    }

    zmq_close(socket);
    zmq_ctx_term(context);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const char* endpoint = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--server" && i + 1 < argc) {
            endpoint = argv[++i];
        } else if (argument == "--help") {
            std::printf("usage: zmq_ping [--server tcp://127.0.0.1:6555]\n");
            return 0;
        } else {
            std::printf("unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    self_test();
    if (endpoint != nullptr) {
        const int rc = probe(endpoint);
        std::printf("\n%s\n", rc == 0 ? "VERDICT: frame layout matches the server"
                                      : "VERDICT: frame layout does not match the server");
        return rc;
    }
    std::printf("\nmsgpack only (no --server given)\n");
    return failures == 0 ? 0 : 1;
}
