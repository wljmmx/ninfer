// MessagePack for the wire this engine speaks: LMCache's ZMQ transport.
//
// Scope is deliberate. This is not a general MessagePack library -- it covers the value forms the
// engine_driven protocol actually uses (nil, bool, int, str, bin, array, map) and nothing else.
// Anything wider would be unused code on the path of every cache RPC.
//
// Two details are load-bearing and easy to get wrong:
//   * Every frame holds exactly ONE msgpack value. The request uid, the operation, and each payload
//     are separate frames, each separately encoded -- not one array.
//   * Structs and dataclasses on the server side are encoded as MAPS keyed by field name. So a key
//     is a 10-entry map, not a tuple, and the decoder matches on names rather than position.
//
// Encoding is byte-compatible with msgspec (the server's own encoder): non-negative integers widen
// into the unsigned tags, which is why 300 is 0xcd 0x01 0x2c and not 0xd1 0x01 0x2c.

#ifndef NINFER_CORE_MSGPACK_WIRE_H
#define NINFER_CORE_MSGPACK_WIRE_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::core::msgpack {

// --- writer ----------------------------------------------------------------

class Writer {
public:
    void nil() { buffer_.push_back(static_cast<char>(0xC0)); }

    void boolean(bool value) { buffer_.push_back(static_cast<char>(value ? 0xC3 : 0xC2)); }

    void integer(std::int64_t value) {
        if (value >= 0) {
            const auto magnitude = static_cast<std::uint64_t>(value);
            if (magnitude < 0x80) {
                put(static_cast<std::uint8_t>(magnitude));
            } else if (magnitude <= 0xFF) {
                put(0xCC);
                put(static_cast<std::uint8_t>(magnitude));
            } else if (magnitude <= 0xFFFF) {
                put(0xCD);
                put_be(magnitude, 2);
            } else if (magnitude <= 0xFFFFFFFFull) {
                put(0xCE);
                put_be(magnitude, 4);
            } else {
                put(0xCF);
                put_be(magnitude, 8);
            }
            return;
        }
        if (value >= -32) {
            put(static_cast<std::uint8_t>(value & 0xFF));
        } else if (value >= -128) {
            put(0xD0);
            put(static_cast<std::uint8_t>(value & 0xFF));
        } else if (value >= -32768) {
            put(0xD1);
            put_be(static_cast<std::uint64_t>(value) & 0xFFFFu, 2);
        } else if (value >= -2147483648ll) {
            put(0xD2);
            put_be(static_cast<std::uint64_t>(value) & 0xFFFFFFFFu, 4);
        } else {
            put(0xD3);
            put_be(static_cast<std::uint64_t>(value), 8);
        }
    }

    void string(std::string_view text) {
        const auto size = text.size();
        if (size < 32) {
            put(static_cast<std::uint8_t>(0xA0u | size));
        } else if (size <= 0xFF) {
            put(0xD9);
            put(static_cast<std::uint8_t>(size));
        } else if (size <= 0xFFFF) {
            put(0xDA);
            put_be(size, 2);
        } else {
            put(0xDB);
            put_be(size, 4);
        }
        buffer_.append(text.data(), text.size());
    }

    void binary(const void* data, std::size_t size) {
        if (size <= 0xFF) {
            put(0xC4);
            put(static_cast<std::uint8_t>(size));
        } else if (size <= 0xFFFF) {
            put(0xC5);
            put_be(size, 2);
        } else {
            put(0xC6);
            put_be(size, 4);
        }
        buffer_.append(static_cast<const char*>(data), size);
    }

    void binary(std::string_view data) { binary(data.data(), data.size()); }

    // float64. Only needed for the one place the protocol takes a float (a prefetch timeout), but
    // trying to pass an integer there fails validation, so it is worth having right.
    void real(double value) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "double is not 64 bit");
        std::memcpy(&bits, &value, sizeof(bits));
        put(0xCB);
        put_be(bits, 8);
    }

    void array_header(std::size_t count) {
        if (count < 16) {
            put(static_cast<std::uint8_t>(0x90u | count));
        } else if (count <= 0xFFFF) {
            put(0xDC);
            put_be(count, 2);
        } else {
            put(0xDD);
            put_be(count, 4);
        }
    }

    void map_header(std::size_t count) {
        if (count < 16) {
            put(static_cast<std::uint8_t>(0x80u | count));
        } else if (count <= 0xFFFF) {
            put(0xDE);
            put_be(count, 2);
        } else {
            put(0xDF);
            put_be(count, 4);
        }
    }

    // A map entry is a key and a value written back to back; this only exists so call sites read
    // as "this field is named X" instead of a bare string followed by a value.
    void key(std::string_view name) { string(name); }

    const std::string& bytes() const { return buffer_; }
    std::string take() { return std::move(buffer_); }
    void clear() { buffer_.clear(); }
    std::size_t size() const { return buffer_.size(); }

private:
    void put(std::uint8_t value) { buffer_.push_back(static_cast<char>(value)); }

    void put_be(std::uint64_t value, int bytes) {
        for (int i = bytes - 1; i >= 0; --i) {
            buffer_.push_back(static_cast<char>((value >> (8 * i)) & 0xFFu));
        }
    }

    std::string buffer_;
};

// --- reader ----------------------------------------------------------------

struct Value {
    enum class Kind { Nil, Bool, Int, Str, Bin, Array, Map, Unsupported };

    Kind kind{Kind::Unsupported};
    bool boolean{false};
    std::int64_t integer{0};
    std::string bytes;                              // payload of a Str or a Bin
    std::vector<Value> items;                       // elements of an Array
    std::vector<std::pair<std::string, Value>> entries;  // key/value pairs of a Map

    // Map lookup by key name; the server's responses are maps keyed by field name.
    const Value* find(std::string_view name) const {
        for (const auto& entry : entries) {
            if (entry.first == name) {
                return &entry.second;
            }
        }
        return nullptr;
    }

    const char* kind_name() const {
        switch (kind) {
        case Kind::Nil: return "nil";
        case Kind::Bool: return "bool";
        case Kind::Int: return "int";
        case Kind::Str: return "str";
        case Kind::Bin: return "bin";
        case Kind::Array: return "array";
        case Kind::Map: return "map";
        default: return "unsupported";
        }
    }
};

namespace detail {

inline bool read_be(const std::string& in, std::size_t& off, int bytes, std::uint64_t& value) {
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

}  // namespace detail

inline bool decode(const std::string& in, std::size_t& off, Value& out);
inline bool decode_array(const std::string& in, std::size_t& off, std::size_t count, Value& out);
inline bool decode_map(const std::string& in, std::size_t& off, std::size_t count, Value& out);

// Returns false on truncation. An unrecognised tag yields Kind::Unsupported with the tag in
// integer, which lets a caller print what it saw instead of silently misreading the frame.
inline bool decode(const std::string& in, std::size_t& off, Value& out) {
    if (off >= in.size()) {
        return false;
    }
    const auto tag = static_cast<std::uint8_t>(in[off]);

    if (tag <= 0x7F) {
        out.kind = Value::Kind::Int;
        out.integer = tag;
        ++off;
        return true;
    }
    if (tag >= 0xE0) {
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
        ++off;  // step past the header, or the first element reads the header again and recurses
        return decode_array(in, off, tag & 0x0F, out);
    }
    if ((tag & 0xF0) == 0x80) {  // fixmap
        ++off;  // same: without this the key decode re-enters on the same byte
        return decode_map(in, off, tag & 0x0F, out);
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
        if (!detail::read_be(in, off, 1 << (tag - 0xCC), raw)) {
            return false;
        }
        out.kind = Value::Kind::Int;
        out.integer = static_cast<std::int64_t>(raw);
        return true;
    case 0xD0:
        if (!detail::read_be(in, off, 1, raw)) {
            return false;
        }
        out.kind = Value::Kind::Int;
        out.integer = static_cast<std::int8_t>(static_cast<std::uint8_t>(raw));
        return true;
    case 0xD1:
    case 0xD2:
    case 0xD3: {
        const int bytes = 2 << (tag - 0xD1);
        if (!detail::read_be(in, off, bytes, raw)) {
            return false;
        }
        out.kind = Value::Kind::Int;
        if (bytes == 8) {
            out.integer = static_cast<std::int64_t>(raw);
        } else {
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
        if (!detail::read_be(in, off, bytes, raw)) {
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
        if (!detail::read_be(in, off, bytes, raw)) {
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
        if (!detail::read_be(in, off, tag == 0xDC ? 2 : 4, raw)) {
            return false;
        }
        return decode_array(in, off, static_cast<std::size_t>(raw), out);
    }
    case 0xDE:
    case 0xDF: {
        if (!detail::read_be(in, off, tag == 0xDE ? 2 : 4, raw)) {
            return false;
        }
        return decode_map(in, off, static_cast<std::size_t>(raw), out);
    }
    default:
        out.kind = Value::Kind::Unsupported;
        out.integer = tag;
        return true;
    }
}

inline bool decode_array(const std::string& in, std::size_t& off, std::size_t count, Value& out) {
    out.kind = Value::Kind::Array;
    out.items.clear();
    out.items.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        Value item;
        if (!decode(in, off, item)) {
            return false;
        }
        out.items.push_back(std::move(item));
    }
    return true;
}

inline bool decode_map(const std::string& in, std::size_t& off, std::size_t count, Value& out) {
    out.kind = Value::Kind::Map;
    out.entries.clear();
    out.entries.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        Value key;
        if (!decode(in, off, key) || key.kind != Value::Kind::Str) {
            return false;
        }
        Value value;
        if (!decode(in, off, value)) {
            return false;
        }
        out.entries.emplace_back(std::move(key.bytes), std::move(value));
    }
    return true;
}

}  // namespace ninfer::core::msgpack

#endif  // NINFER_CORE_MSGPACK_WIRE_H
