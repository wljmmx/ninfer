#pragma once

// Overflow-safe 64x64 -> 128 unsigned multiplication. The engine uses exact products to
// saturate resource budgets and to compare loss-per-unit ratios without floating point;
// MSVC has no __int128, so the product is expressed as an explicit high/low word pair.

#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace ninfer {

struct WideProduct {
    std::uint64_t low  = 0;
    std::uint64_t high = 0;
};

[[nodiscard]] inline WideProduct wide_mul(std::uint64_t a, std::uint64_t b) noexcept {
#ifdef _MSC_VER
    std::uint64_t high    = 0;
    const std::uint64_t low = _umul128(a, b, &high);
    return WideProduct{low, high};
#else
    const unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
    return WideProduct{static_cast<std::uint64_t>(product),
                      static_cast<std::uint64_t>(product >> 64U)};
#endif
}

// Saturating 64x64 product: UINT64_MAX when the exact product does not fit.
[[nodiscard]] inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) noexcept {
    const WideProduct product = wide_mul(a, b);
    return product.high != 0 ? ~static_cast<std::uint64_t>(0) : product.low;
}

// Lexicographic comparison of two exact 64x64 products.
[[nodiscard]] inline int wide_mul_compare(std::uint64_t a_left, std::uint64_t a_right,
                                          std::uint64_t b_left, std::uint64_t b_right) noexcept {
    const WideProduct a = wide_mul(a_left, a_right);
    const WideProduct b = wide_mul(b_left, b_right);
    if (a.high != b.high) { return a.high < b.high ? -1 : 1; }
    if (a.low != b.low) { return a.low < b.low ? -1 : 1; }
    return 0;
}

} // namespace ninfer
