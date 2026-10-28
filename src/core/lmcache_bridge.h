// The seam between a checkpoint export and LMCache.
//
// The engine exports KV as a flat run of pages and the state image as one blob; the cache wants
// chunks of `pages_per_chunk` pages and one object per group. This translates between the two and
// owns nothing else -- deliberately: no engine types appear in the interface, so the split can be
// tested without a model.
//
// Every shape here is SUPPLIED, never derived. Models differ in layer count, head width, K/V plane
// count (MLA and fused formats have one) and state image dtype, so a formula that divides a byte
// count back into a shape would be a constant for one model and wrong for the next. The only thing
// this file does with the numbers is check that the shape a caller registered actually describes
// the bytes a caller hands it.
#pragma once

#include "core/lmcache_client.h"
#include "core/lmcache_wire.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninfer::core::lmcache {

// Element size of a torch dtype name, as the server spells them. Returns 0 for a name it does not
// know, which turns into a refused configuration rather than a silently wrong byte count.
[[nodiscard]] inline std::size_t dtype_itemsize(const std::string& dtype) {
    if (dtype == "float64") { return 8; }
    if (dtype == "float32") { return 4; }
    if (dtype == "float16" || dtype == "bfloat16") { return 2; }
    if (dtype == "int8" || dtype == "uint8" || dtype == "float8_e4m3fn" ||
        dtype == "float8_e5m2") {
        return 1;
    }
    return 0;
}

[[nodiscard]] inline std::size_t shape_bytes(const std::vector<std::int64_t>& shape) {
    std::size_t total = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0) { return 0; }
        total *= static_cast<std::size_t>(dimension);
    }
    return total;
}

// One LMCache group, exactly as the server will see it. shape is the layout the payload must match;
// kv_size and window_* are registration-only fields the server reads when it sizes the object.
struct GroupSpec {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::int64_t kv_size{2};       // 2 = separate K/V planes, 1 = MLA / fused / indexer
    std::int64_t window_tokens{0}; // 0 means "the whole chunk"
    std::int64_t window_chunks{0}; // 0 means "every chunk of the prefix"
    bool recurrent{false};

    [[nodiscard]] std::size_t object_bytes() const {
        return shape_bytes(shape) * dtype_itemsize(dtype);
    }

    // The only shape that fits every case: one plane, one token, N elements. The engine's Host page
    // layout is plane-major and a quantized format has FOUR planes per layer (K data, K scales, V
    // data, V scales), while the server can only express one or two -- so any shape that names the
    // planes is a shape that will not fit the next format. Bytes are the invariant; this states them
    // and nothing else.
    //
    // uint8 is used when the byte count is not a whole number of the natural element, because the
    // server only checks prod(shape) x itemsize and an arbitrary stride (page records are aligned,
    // not sized to an element) is not always evenly divisible.
    [[nodiscard]] static GroupSpec exact(std::string dtype, std::size_t bytes) {
        std::size_t itemsize = dtype_itemsize(dtype);
        if (itemsize == 0 || bytes == 0 || bytes % itemsize != 0) {
            dtype = "uint8";
            itemsize = 1;
        }
        GroupSpec spec;
        spec.dtype    = std::move(dtype);
        spec.shape    = {1, 1, static_cast<std::int64_t>(bytes / itemsize)};
        spec.kv_size  = 1;
        return spec;
    }
};

struct BridgeConfig {
    std::string model_name;
    // Part of the object key. Two entries that differ only in KV format share nothing else -- model
    // name and tokens are the same -- so a format fingerprint belongs here. Empty means "no
    // isolation", which is only safe while one format is ever used for a given model.
    std::string cache_salt;
    // Tokens in one engine page, and how many pages go into one cache chunk. Both come from the
    // caller: the page size is a property of the engine, the chunk size a choice made against it.
    std::int64_t page_tokens{64};
    std::int64_t pages_per_chunk{2};
    std::size_t page_bytes{0};
    GroupSpec kv;
    GroupSpec state;

    [[nodiscard]] std::int64_t chunk_tokens() const { return page_tokens * pages_per_chunk; }
};

// The cache identity: the token sequence alone. Reusing a sequence hits an entry an earlier run
// stored, and a shape mismatch there reads as a wrong byte count rather than an error, so callers
// must not reuse one either.
struct BridgeKey {
    std::vector<std::int64_t> tokens;
    std::string request_id;
};

class Bridge {
public:
    Bridge() = default;
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    [[nodiscard]] const std::string& error() const { return error_; }
    [[nodiscard]] std::int64_t instance_id() const { return instance_id_; }
    [[nodiscard]] const BridgeConfig& config() const { return config_; }

    // The bridge refuses a configuration whose shapes do not describe the bytes it will be handed.
    // That is a startup error with a visible reason; the alternative is every payload rejected at
    // request time, where the reason is a timeout.
    bool open(const ClientOptions& options, int connections, const BridgeConfig& config,
              std::int64_t instance_id) {
        config_      = config;
        instance_id_ = instance_id;
        if (config_.page_bytes == 0) {
            error_ = "page_bytes is not known yet";
            return false;
        }
        if (config_.page_tokens <= 0 || config_.pages_per_chunk <= 0) {
            error_ = "page_tokens and pages_per_chunk must be positive";
            return false;
        }
        if (dtype_itemsize(config_.kv.dtype) == 0 || dtype_itemsize(config_.state.dtype) == 0) {
            error_ = "unknown dtype in a group layout";
            return false;
        }
        if (config_.kv.kv_size != 1 && config_.kv.kv_size != 2) {
            error_ = "kv_size must be 1 (single plane) or 2 (separate K/V)";
            return false;
        }
        // The server can only express rank 4 (two planes) or rank 3 (one plane). Anything else
        // would silently lose an axis, so refuse rather than register a different shape.
        const std::size_t kv_rank = config_.kv.kv_size == 2 ? 4 : 3;
        if (config_.kv.shape.size() != kv_rank) {
            error_ = "KV shape rank does not match its kv_size";
            return false;
        }
        if (config_.state.kv_size == 2 && config_.state.shape.size() != 4) {
            error_ = "state shape rank does not match its kv_size";
            return false;
        }
        if (config_.state.kv_size == 1 && config_.state.shape.size() != 3) {
            error_ = "state shape rank does not match its kv_size";
            return false;
        }
        if (config_.kv.object_bytes() != chunk_bytes()) {
            error_ = "KV shape does not describe one chunk of pages (" +
                     std::to_string(config_.kv.object_bytes()) + " vs " +
                     std::to_string(chunk_bytes()) + " B)";
            return false;
        }
        if (config_.state.object_bytes() == 0) {
            error_ = "state shape describes no bytes";
            return false;
        }
        if (!pool_.open(options, static_cast<std::size_t>(connections))) {
            error_ = pool_.error();
            return false;
        }
        if (!pool_.register_context(registration())) {
            error_ = pool_.error();
            return false;
        }
        return true;
    }

    void close() { pool_.close(); }

    [[nodiscard]] std::size_t chunk_bytes() const {
        return static_cast<std::size_t>(config_.pages_per_chunk) * config_.page_bytes;
    }

    // How many chunks a run of pages becomes. Not a constant: it follows from the page count.
    [[nodiscard]] std::size_t chunks_for(std::span<const std::byte> kv) const {
        const std::size_t pages = kv.size() / config_.page_bytes;
        return pages / static_cast<std::size_t>(config_.pages_per_chunk);
    }

    bool publish(const BridgeKey& key, std::span<const std::byte> kv, std::span<const std::byte> state) {
        if (!pool_.size() || instance_id_ == 0) {
            error_ = "bridge is not open";
            return false;
        }
        if (kv.empty() || kv.size() % config_.page_bytes != 0 ||
            kv.size() % chunk_bytes() != 0) {
            error_ = "KV bytes are not a whole number of chunks";
            return false;
        }
        if (state.size() != config_.state.object_bytes()) {
            error_ = "state image size does not match its registered shape";
            return false;
        }

        std::vector<std::string> chunks;
        chunks.reserve(kv.size() / chunk_bytes());
        for (std::size_t offset = 0; offset < kv.size(); offset += chunk_bytes()) {
            chunks.emplace_back(reinterpret_cast<const char*>(kv.data() + offset), chunk_bytes());
        }
        const std::string payload(reinterpret_cast<const char*>(state.data()), state.size());

        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        const bool ok = lease->store(make_key(key), chunks, config_.kv.shape, payload,
                                     config_.state.shape);
        if (!ok) { error_ = lease->error(); }
        return ok;
    }

    bool has(const BridgeKey& key) {
        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        const bool ok = lease->has(make_key(key));
        if (!ok) { error_ = lease->error(); }
        return ok;
    }

    // Rebuilds the flat page run the engine's import expects.
    bool restore(const BridgeKey& key, std::vector<std::byte>* kv, std::vector<std::byte>* state) {
        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        RestorePayload got;
        const CacheKey cache_key = make_key(key);
        if (!lease->has(cache_key) || !lease->fetch(cache_key, &got)) {
            error_ = lease->error().empty() ? "not cached" : lease->error();
            return false;
        }
        if (kv != nullptr) {
            kv->clear();
            for (const std::string& chunk : got.kv_chunks) {
                const auto* bytes = reinterpret_cast<const std::byte*>(chunk.data());
                kv->insert(kv->end(), bytes, bytes + chunk.size());
            }
        }
        if (state != nullptr) {
            state->assign(reinterpret_cast<const std::byte*>(got.state.data()),
                          reinterpret_cast<const std::byte*>(got.state.data()) + got.state.size());
        }
        return true;
    }

private:
    [[nodiscard]] RegisterContext registration() const {
        RegisterContext context;
        context.instance_id        = instance_id_;
        context.model_name         = config_.model_name;
        context.world_size         = 1;
        context.block_size         = config_.chunk_tokens();
        context.num_layers         = layers_of(config_.kv);
        context.hidden_dim_size    = hidden_of(config_.kv);
        context.dtype_str          = config_.kv.dtype;
        context.use_mla            = config_.kv.kv_size == 1;
        context.num_physical_slots = config_.chunk_tokens();
        context.group_layouts.push_back(to_group_layout(config_.kv));
        context.group_layouts.push_back(to_group_layout(config_.state));
        return context;
    }

    // The server reconstructs a shape from three numbers: [layers, tokens, hidden] for one plane,
    // [2, layers, tokens, hidden] for two. So the registration is the caller's declared shape taken
    // apart along the axes the server knows about -- not a shape invented here from a byte count.
    // A shape with any other rank cannot be expressed and is refused at open().
    static GroupLayout to_group_layout(const GroupSpec& spec) {
        GroupLayout layout;
        const bool two_planes = spec.kv_size == 2;
        const std::size_t expected_rank = two_planes ? 4 : 3;
        layout.num_layers       = spec.shape.size() == expected_rank ? spec.shape[expected_rank - 3] : 0;
        layout.window_tokens    = spec.shape.size() == expected_rank ? spec.shape[expected_rank - 2] : 0;
        layout.hidden_dim_size  = spec.shape.size() == expected_rank ? spec.shape[expected_rank - 1] : 0;
        layout.dtype_str        = spec.dtype;
        layout.kv_size          = spec.kv_size;
        layout.window_chunks    = spec.window_chunks;
        layout.recurrent        = spec.recurrent;
        // Per-block tokens is metadata about the engine's paging, not about the object; the server
        // uses it only to describe the layout, so it follows the chunk.
        layout.tokens_per_block = layout.window_tokens;
        return layout;
    }

    static std::int64_t layers_of(const GroupSpec& spec) {
        const std::size_t expected_rank = spec.kv_size == 2 ? 4 : 3;
        return spec.shape.size() == expected_rank ? spec.shape[expected_rank - 3] : 0;
    }

    static std::int64_t hidden_of(const GroupSpec& spec) {
        return spec.shape.empty() ? 0 : spec.shape.back();
    }

    [[nodiscard]] CacheKey make_key(const BridgeKey& key) const {
        CacheKey out;
        out.model_name  = config_.model_name;
        out.world_size  = 1;
        out.worker_id   = 0;
        out.token_bytes = pack_tokens(key.tokens);
        out.start       = 0;
        out.end         = static_cast<std::int64_t>(key.tokens.size());
        out.request_id  = key.request_id;
        out.cache_salt  = config_.cache_salt;
        return out;
    }

    Pool pool_;
    BridgeConfig config_{};
    std::int64_t instance_id_{0};
    std::string error_;
};

}  // namespace ninfer::core::lmcache
