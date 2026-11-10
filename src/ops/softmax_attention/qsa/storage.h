#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::qsa {

// The cache storages this Op family implements. `qsa_prepare` writes and `qsa_sparse_gqa` reads the
// same representation, so the two lists cannot drift: both ask this table.
//
// The rk family (RotatedInt8KeyInt4ValueGroup64, RotatedInt4KeyInt4ValueGroup64, RK4V4E8, RK2V4E8,
// RK4V2E8) is ported in the dense attention path only. Its planes mix an int4/int2 value packing
// with the E8 lattice extents, and neither Op here has the matching encoder or reader yet, so every
// one of them is refused by name instead of falling through a generic default. Adding one means
// adding its encoder to `prepare_cache.cuh` and its decoder to `sparse_gqa_cache.cuh`, then moving
// it into this table.
[[nodiscard]] inline constexpr bool qsa_storage_supported(KvCacheStorage storage) noexcept {
    return storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Int8Group64 ||
           storage == KvCacheStorage::Fp8E4M3Row256 || storage == KvCacheStorage::Nvfp4Group16 ||
           storage == KvCacheStorage::Fp8KeyNvfp4Value;
}

[[nodiscard]] inline constexpr const char* qsa_storage_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8-g64";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8-e4m3-row256";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4-g16";
    case KvCacheStorage::Fp8KeyNvfp4Value: return "fp8-k-nvfp4-v";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64: return "rk8v4";
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64: return "rk4v4";
    case KvCacheStorage::RK4V4E8: return "rk4v4-e8";
    case KvCacheStorage::RK2V4E8: return "rk2v4-e8";
    case KvCacheStorage::RK4V2E8: return "rk4v2-e8";
    }
    return "unknown";
}

// One error shape for both Ops: which storage, which Op, and what the family does implement.
[[noreturn]] inline void throw_unsupported_storage(const char* op, KvCacheStorage storage) {
    throw std::invalid_argument(std::string(op) + ": KV storage '" + qsa_storage_name(storage) +
                                "' is not implemented by the QSA Op family (bf16, int8-g64, "
                                "fp8-e4m3-row256, nvfp4-g16, fp8-k-nvfp4-v are)");
}

} // namespace ninfer::ops::detail::qsa
