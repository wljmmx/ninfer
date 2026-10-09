#pragma once

#include "core/dtype.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

/** Physical data/scale planes for one K or V vector. */
struct PagedKVVectorLayout {
    DType data_dtype                  = DType::BF16;
    std::int32_t data_leading_extent  = 0;
    DType scale_dtype                 = DType::U8;
    std::int32_t scale_leading_extent = 0;

    [[nodiscard]] constexpr bool has_scale() const noexcept { return scale_leading_extent != 0; }

    [[nodiscard]] std::size_t physical_bytes() const {
        return static_cast<std::size_t>(data_leading_extent) * dtype_size(data_dtype) +
               static_cast<std::size_t>(scale_leading_extent) * dtype_size(scale_dtype);
    }

    friend bool operator==(const PagedKVVectorLayout&, const PagedKVVectorLayout&) = default;
};

/** Resolved physical plane schema for one paged K/V layer. */
struct PagedKVStorageLayout {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    std::int32_t head_dim  = 0;
    PagedKVVectorLayout key;
    PagedKVVectorLayout value;

    [[nodiscard]] constexpr std::size_t planes_per_layer() const noexcept {
        return 2ULL + static_cast<std::size_t>(key.has_scale()) +
               static_cast<std::size_t>(value.has_scale());
    }

    [[nodiscard]] constexpr std::size_t logical_vector_bytes() const noexcept {
        return static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    }

    [[nodiscard]] constexpr std::size_t logical_bytes_per_token_head() const noexcept {
        return 2ULL * logical_vector_bytes();
    }

    [[nodiscard]] std::size_t physical_bytes_per_token_head() const {
        return key.physical_bytes() + value.physical_bytes();
    }

    friend bool operator==(const PagedKVStorageLayout&, const PagedKVStorageLayout&) = default;
};

[[nodiscard]] inline PagedKVStorageLayout paged_kv_storage_layout(KvCacheStorage storage,
                                                                  std::int32_t head_dim) {
    if (head_dim <= 0) { throw std::invalid_argument("KV-cache head dimension must be positive"); }

    const auto symmetric = [=](PagedKVVectorLayout vector) {
        return PagedKVStorageLayout{storage, head_dim, vector, vector};
    };
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {storage,
                head_dim,
                {DType::BF16, head_dim, DType::U8, 0},
                {DType::FP16, head_dim, DType::U8, 0}};
    case KvCacheStorage::Int8Group64:
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::I8, 256, DType::FP16, 4}); }
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        if (head_dim == kD256KVCacheHeadDim) {
            return symmetric({DType::FP8_E4M3FN, 256, DType::FP16, 1});
        }
        break;
    case KvCacheStorage::Nvfp4Group16:
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::U8, 128, DType::U8, 16}); }
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::FP8_E4M3FN, 256, DType::FP16, 1},
                    {DType::U8, 128, DType::U8, 16}};
        }
        break;
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        // rk8v4: 8-bit Hadamard K + 4-bit V, G64 scales.
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::I8, 256, DType::FP16, 4},
                    {DType::U8, 128, DType::FP16, 4}};
        }
        break;
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
    case KvCacheStorage::RK4V4E8:
        // rk4v4 / rk4v4-e8: 4-bit K + 4-bit V (both packed as U8, half extent), G64 scales.
        if (head_dim == kD256KVCacheHeadDim) {
            return symmetric({DType::U8, 128, DType::FP16, 4});
        }
        break;
    case KvCacheStorage::RK4V2E8:
        // rk4v2-e8: 4-bit E8-lattice K (128 B) + 2-bit E8-cylinder V (64 B), G64 scales.
        // 13 KiB/token — same footprint as rk2v4-e8 but spends the low bits on V.
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::U8, 128, DType::FP16, 4},
                    {DType::U8, 64, DType::FP16, 4}};
        }
        break;
    case KvCacheStorage::RK2V4E8:
        // rk2v4-e8: 2-bit E8-cylinder K (quarter extent) + 4-bit V, G64 scales.
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::U8, 64, DType::FP16, 4},
                    {DType::U8, 128, DType::FP16, 4}};
        }
        break;
    }
    throw std::invalid_argument("unsupported paged KV-cache storage geometry");
}

} // namespace ninfer
