// The two payloads this engine has to produce for LMCache's engine_driven protocol.
//
// Both are structs on the server side, which msgspec encodes as a MAP keyed by field name. That is
// the part worth stating plainly: this is not a positional tuple, so a field spelled wrong is an
// "unknown field" error rather than a silently shifted value, and the field set must match the
// server version exactly. The names below are copied from custom_types.py, not invented.
//
// Nothing here talks to the socket. Keeping encoding separate from transport is what lets the
// encoders be checked against the server's own decoder without a server running.

#ifndef NINFER_CORE_LMCACHE_WIRE_H
#define NINFER_CORE_LMCACHE_WIRE_H

#include "core/msgpack_wire.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::core::lmcache {

// IPCCacheServerKey. The cache identity is (model, world_size, worker_id, token range) -- there is
// no free-form key. token_bytes is packed big-endian uint32, one word per token id.
struct CacheKey {
    std::string model_name;
    std::int64_t world_size{1};
    std::optional<std::int64_t> worker_id{};  // nullopt means "all workers"
    std::string token_bytes;                  // packed BE uint32
    std::int64_t start{0};
    std::int64_t end{0};
    std::string request_id;
    std::string cache_salt;
    std::int64_t num_kv_readers{1};
};

// Field order matters only for byte-for-byte similarity with msgspec; decoding is by name.
inline std::string encode_cache_key(const CacheKey& key) {
    msgpack::Writer writer;
    writer.map_header(10);
    writer.key("model_name");
    writer.string(key.model_name);
    writer.key("world_size");
    writer.integer(key.world_size);
    writer.key("worker_id");
    if (key.worker_id.has_value()) {
        writer.integer(*key.worker_id);
    } else {
        writer.nil();
    }
    writer.key("token_bytes");
    writer.binary(key.token_bytes);
    writer.key("start");
    writer.integer(key.start);
    writer.key("end");
    writer.integer(key.end);
    writer.key("request_id");
    writer.string(key.request_id);
    writer.key("cache_salt");
    writer.string(key.cache_salt);
    // request_configs is dict[str, Any] | None; we send none rather than an empty map so the
    // server takes its default path.
    writer.key("request_configs");
    writer.nil();
    writer.key("num_kv_readers");
    writer.integer(key.num_kv_readers);
    return writer.take();
}

// Pack token ids the way the server expects: big-endian uint32, four bytes each.
inline std::string pack_tokens(const std::vector<std::int64_t>& tokens) {
    std::string packed;
    packed.reserve(tokens.size() * 4);
    for (const std::int64_t token : tokens) {
        const auto value = static_cast<std::uint32_t>(token);
        packed.push_back(static_cast<char>((value >> 24) & 0xFFu));
        packed.push_back(static_cast<char>((value >> 16) & 0xFFu));
        packed.push_back(static_cast<char>((value >> 8) & 0xFFu));
        packed.push_back(static_cast<char>(value & 0xFFu));
    }
    return packed;
}

// GroupLayout. A recurrent group is the one whose pages hold state snapshots rather than per-token
// attention KV -- that is where our state image travels.
struct GroupLayout {
    std::int64_t num_layers{0};
    std::int64_t hidden_dim_size{0};
    std::string dtype_str;
    std::int64_t tokens_per_block{0};
    std::int64_t window_tokens{0};
    std::int64_t kv_size{2};
    std::int64_t window_chunks{0};
    bool recurrent{false};
};

// RegisterEngineDrivenContextPayload.
struct RegisterContext {
    std::int64_t instance_id{0};
    std::string model_name;
    std::int64_t world_size{1};
    std::int64_t block_size{0};
    std::int64_t num_layers{0};
    std::int64_t hidden_dim_size{0};
    std::string dtype_str;
    bool use_mla{false};
    std::optional<std::int64_t> num_physical_slots{};
    std::vector<GroupLayout> group_layouts;
};

inline std::string encode_register_context(const RegisterContext& context) {
    msgpack::Writer writer;
    writer.map_header(10);
    writer.key("instance_id");
    writer.integer(context.instance_id);
    writer.key("model_name");
    writer.string(context.model_name);
    writer.key("world_size");
    writer.integer(context.world_size);
    writer.key("block_size");
    writer.integer(context.block_size);
    writer.key("num_layers");
    writer.integer(context.num_layers);
    writer.key("hidden_dim_size");
    writer.integer(context.hidden_dim_size);
    writer.key("dtype_str");
    writer.string(context.dtype_str);
    writer.key("use_mla");
    writer.boolean(context.use_mla);
    writer.key("num_physical_slots");
    if (context.num_physical_slots.has_value()) {
        writer.integer(*context.num_physical_slots);
    } else {
        writer.nil();
    }
    writer.key("group_layouts");
    writer.array_header(context.group_layouts.size());
    for (const GroupLayout& layout : context.group_layouts) {
        writer.map_header(8);
        writer.key("num_layers");
        writer.integer(layout.num_layers);
        writer.key("hidden_dim_size");
        writer.integer(layout.hidden_dim_size);
        writer.key("dtype_str");
        writer.string(layout.dtype_str);
        writer.key("tokens_per_block");
        writer.integer(layout.tokens_per_block);
        writer.key("window_tokens");
        writer.integer(layout.window_tokens);
        writer.key("kv_size");
        writer.integer(layout.kv_size);
        writer.key("window_chunks");
        writer.integer(layout.window_chunks);
        writer.key("recurrent");
        writer.boolean(layout.recurrent);
    }
    return writer.take();
}

// One typed chunk of one group: the server checks dtype and shape against the registered layout,
// so these are not decorative -- a mismatch is rejected before anything is stored.
struct TensorFrameWire {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string data;
};

// A group's tensors plus the absolute chunk position each one fills.
struct TensorGroupWire {
    std::vector<std::int64_t> chunk_indices;
    std::vector<TensorFrameWire> tensors;
};

// GroupedTensorEnvelope. Multi-group transfers use this one; the single-group case uses
// TensorEnvelope{version, tensors} instead, and the two must not be confused: they have different
// field names so neither decodes as the other.
inline std::string encode_grouped_tensor_envelope(const std::vector<TensorGroupWire>& groups) {
    msgpack::Writer writer;
    writer.map_header(2);
    writer.key("version");
    writer.integer(1);
    writer.key("groups");
    writer.array_header(groups.size());
    for (const TensorGroupWire& group : groups) {
        writer.map_header(2);
        writer.key("chunk_indices");
        writer.array_header(group.chunk_indices.size());
        for (const std::int64_t index : group.chunk_indices) {
            writer.integer(index);
        }
        writer.key("tensors");
        writer.array_header(group.tensors.size());
        for (const TensorFrameWire& frame : group.tensors) {
            writer.map_header(3);
            writer.key("dtype");
            writer.string(frame.dtype);
            writer.key("shape");
            writer.array_header(frame.shape.size());
            for (const std::int64_t dimension : frame.shape) {
                writer.integer(dimension);
            }
            writer.key("data");
            writer.binary(frame.data);
        }
    }
    return writer.take();
}

// Pull the tensors back out of an envelope the server handed us, so a round trip can be checked
// payload by payload rather than by hoping two encodings agree byte for byte.
struct DecodedTensor {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string data;
};

struct DecodedGroup {
    std::vector<std::int64_t> chunk_indices;
    std::vector<DecodedTensor> tensors;
};

inline bool decode_grouped_tensor_envelope(const std::string& bytes,
                                           std::vector<DecodedGroup>& groups) {
    std::size_t offset = 0;
    msgpack::Value root;
    if (!msgpack::decode(bytes, offset, root) || offset != bytes.size() ||
        root.kind != msgpack::Value::Kind::Map) {
        return false;
    }
    const msgpack::Value* group_array = root.find("groups");
    if (group_array == nullptr || group_array->kind != msgpack::Value::Kind::Array) {
        return false;
    }
    groups.clear();
    groups.reserve(group_array->items.size());
    for (const msgpack::Value& group_value : group_array->items) {
        if (group_value.kind != msgpack::Value::Kind::Map) {
            return false;
        }
        DecodedGroup group;
        if (const msgpack::Value* indices = group_value.find("chunk_indices")) {
            for (const msgpack::Value& index : indices->items) {
                group.chunk_indices.push_back(index.integer);
            }
        }
        const msgpack::Value* tensors = group_value.find("tensors");
        if (tensors == nullptr || tensors->kind != msgpack::Value::Kind::Array) {
            return false;
        }
        for (const msgpack::Value& frame : tensors->items) {
            DecodedTensor tensor;
            if (const msgpack::Value* dtype = frame.find("dtype")) {
                tensor.dtype = dtype->bytes;
            }
            if (const msgpack::Value* shape = frame.find("shape")) {
                for (const msgpack::Value& dimension : shape->items) {
                    tensor.shape.push_back(dimension.integer);
                }
            }
            if (const msgpack::Value* data = frame.find("data")) {
                tensor.data = data->bytes;
            }
            group.tensors.push_back(std::move(tensor));
        }
        groups.push_back(std::move(group));
    }
    return true;
}

}  // namespace ninfer::core::lmcache

#endif  // NINFER_CORE_LMCACHE_WIRE_H
