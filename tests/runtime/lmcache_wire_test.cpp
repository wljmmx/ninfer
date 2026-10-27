// The encoders in core/lmcache_wire.h, checked against the server's own decoder.
//
// A byte-for-byte comparison against msgspec output would only prove the two encoders agree on
// today's field order. What actually matters is that LMCache can *decode* what we emit into the
// types its handlers expect, and that the values survive. So this writes the payloads to disk and
// hands them to msgspec's typed decoder (tests/runtime/verify_lmcache_wire.py).
//
// The primitive cases below are pinned to msgspec's bytes for a different reason: a rounding error
// in the integer widening turns a uid into a different number, and that failure looks like a hang.

#include "core/lmcache_wire.h"

#include <cstdio>
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

std::string encode_int(std::int64_t value) {
    ninfer::core::msgpack::Writer writer;
    writer.integer(value);
    return writer.take();
}

std::string encode_string(std::string_view text) {
    ninfer::core::msgpack::Writer writer;
    writer.string(text);
    return writer.take();
}

void primitive_checks() {
    // Byte-compatible with msgspec: non-negative widens into the unsigned tags.
    expect(encode_int(22) == std::string(1, '\x16'), "22 encodes as 0x16");
    expect(encode_int(300) == std::string("\xCD\x01\x2C", 3), "300 encodes as 0xcd 0x01 0x2c");
    expect(encode_int(-1000) == std::string("\xD1\xFC\x18", 3), "-1000 encodes as 0xd1 0xfc 0x18");
    expect(encode_string("ping") == std::string("\xA4ping", 5), "\"ping\" encodes as 0xa4 70 69 6e 67");

    {
        ninfer::core::msgpack::Writer writer;
        writer.nil();
        expect(writer.bytes() == std::string(1, '\xC0'), "nil encodes as 0xc0");
    }
    {
        ninfer::core::msgpack::Writer writer;
        writer.boolean(true);
        expect(writer.bytes() == std::string(1, '\xC3'), "true encodes as 0xc3");
    }
    {
        ninfer::core::msgpack::Writer writer;
        writer.array_header(0);
        expect(writer.bytes() == std::string(1, '\x90'),
               "an empty array is 0x90 -- this is the skipped_chunks payload and omitting it is "
               "not allowed");
    }
    {
        ninfer::core::msgpack::Writer writer;
        writer.binary(std::string("\x00\x01\xFE\xFF", 4));
        expect(writer.bytes() == std::string("\xC4\x04\x00\x01\xFE\xFF", 6),
               "bin keeps its bytes and is not treated as text");
    }

    // Round trip through our own reader, which is what parses the server's replies.
    {
        ninfer::core::msgpack::Writer writer;
        writer.map_header(2);
        writer.key("ok");
        writer.boolean(true);
        writer.key("n");
        writer.integer(7);
        const std::string encoded = writer.take();
        std::size_t offset = 0;
        ninfer::core::msgpack::Value value;
        const bool decoded = ninfer::core::msgpack::decode(encoded, offset, value);
        const auto* ok = value.find("ok");
        const auto* n = value.find("n");
        expect(decoded && offset == encoded.size() && value.kind == ninfer::core::msgpack::Value::Kind::Map &&
                   ok != nullptr && ok->kind == ninfer::core::msgpack::Value::Kind::Bool &&
                   ok->boolean && n != nullptr && n->integer == 7,
               "a map round trips and is readable by field name");
    }
}

bool write_file(const std::string& path, const std::string& bytes) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        std::printf("FAIL cannot open %s\n", path.c_str());
        ++failures;
        return false;
    }
    const std::size_t written = std::fwrite(bytes.data(), 1, bytes.size(), file);
    std::fclose(file);
    if (written != bytes.size()) {
        std::printf("FAIL short write to %s\n", path.c_str());
        ++failures;
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // Unbuffered: an abnormal exit discards stdout's buffer, which would hide where a crash happened.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string out_dir = argc > 1 ? argv[1] : ".";
    primitive_checks();

    // Two chunks worth of tokens (chunk size is 256), so the key covers a range rather than one id.
    std::vector<std::int64_t> tokens(512);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<std::int64_t>(i * 7 + 3);  // deterministic, not sequential
    }

    ninfer::core::lmcache::CacheKey key;
    key.model_name = "qwen3.6-35b-a3b";
    key.world_size = 1;
    key.worker_id = 0;
    key.token_bytes = ninfer::core::lmcache::pack_tokens(tokens);
    key.start = 0;
    key.end = static_cast<std::int64_t>(tokens.size());
    key.request_id = "req-wire-0001";
    key.cache_salt = "";
    key.num_kv_readers = 1;
    const std::string key_bytes = ninfer::core::lmcache::encode_cache_key(key);
    write_file(out_dir + "/wire_key.bin", key_bytes);
    std::printf("ok   key encoded: %zu bytes for %zu tokens\n", key_bytes.size(), tokens.size());

    // Group 0 is attention KV (two planes); group 1 is the state image (one plane, recurrent).
    ninfer::core::lmcache::RegisterContext context;
    context.instance_id = 4242;
    context.model_name = "qwen3.6-35b-a3b";
    context.world_size = 1;
    context.block_size = 256;
    context.num_layers = 40;
    context.hidden_dim_size = 32;
    context.dtype_str = "bfloat16";
    context.use_mla = false;
    context.num_physical_slots = 256;
    {
        ninfer::core::lmcache::GroupLayout kv;
        kv.num_layers = 40;
        kv.hidden_dim_size = 32;
        kv.dtype_str = "bfloat16";
        kv.tokens_per_block = 256;
        kv.window_tokens = 0;
        kv.kv_size = 2;
        kv.window_chunks = 0;
        kv.recurrent = false;
        context.group_layouts.push_back(kv);

        ninfer::core::lmcache::GroupLayout state;
        state.num_layers = 40;
        state.hidden_dim_size = 32;
        state.dtype_str = "float32";
        state.tokens_per_block = 256;
        state.window_tokens = 0;
        state.kv_size = 1;
        state.window_chunks = 0;
        state.recurrent = true;
        context.group_layouts.push_back(state);
    }
    const std::string register_bytes = ninfer::core::lmcache::encode_register_context(context);
    write_file(out_dir + "/wire_register.bin", register_bytes);
    std::printf("ok   register payload encoded: %zu bytes, %zu groups\n",
                register_bytes.size(), context.group_layouts.size());

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
