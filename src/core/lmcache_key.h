#pragma once

// The key string LMCache's storage backends use, reproduced exactly rather than approximated.
//
// Source of truth: lmcache/utils.py, CacheEngineKey.to_string():
//     f"{model_name}@{world_size}@{worker_id}@{chunk_hash:x}@{dtype_str}"
// `:x` means lowercase hex with no zero padding, and `dtype_str` comes from
// TORCH_DTYPE_TO_STR_DTYPE ("half", "bfloat16", "float", "int8", ...), not from torch's repr.
//
// Two things this header deliberately does NOT do:
//   1. It does not invent the chunk hash. In a vLLM deployment the hash of block N is chained over
//      (hash of block N-1, token ids, extras) by the producer; a consumer that picks its own hash
//      simply never hits. The caller passes the hash its producer uses.
//   2. It does not claim that quantized KV is interchangeable. LMCache's RESP tier stores whatever
//      bytes the engine handed it; if our dtype string named "rk4v4" and the payload were a layout
//      LMCache's other users cannot decode, the key would lie. Only the plain dtypes are offered here.

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace ninfer::core {

// Lowercase hex, no padding: what Python's f"{value:x}" produces.
[[nodiscard]] inline std::string lmcache_hex_hash(std::uint64_t hash) {
    char buffer[17]{};
    std::snprintf(buffer, sizeof buffer, "%llx", static_cast<unsigned long long>(hash));
    return buffer;
}

struct LmcacheKeyParts {
    std::string_view model;
    std::uint32_t world_size = 1;
    std::uint32_t worker_id  = 0;
    std::uint64_t chunk_hash = 0;
    // Names LMCache knows: "half", "bfloat16", "float", "int8", ...
    std::string_view dtype = "bfloat16";
};

[[nodiscard]] inline std::string lmcache_key_string(const LmcacheKeyParts& parts) {
    std::string key;
    key.reserve(parts.model.size() + 40);
    key.append(parts.model);
    key += '@';
    key += std::to_string(parts.world_size);
    key += '@';
    key += std::to_string(parts.worker_id);
    key += '@';
    key += lmcache_hex_hash(parts.chunk_hash);
    key += '@';
    key.append(parts.dtype);
    return key;
}

} // namespace ninfer::core
