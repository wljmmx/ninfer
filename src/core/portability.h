#pragma once

// port (route B / P1): MSVC exposes the SIMD intrinsics only through <immintrin.h>, whereas GCC's
// headers drag it in. The CPU MoE kernels call _cvtsh_ss (F16C) and friends directly.
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif

// port (route B / P1): spellings the CPU kernels borrow from GCC/Clang.
// __builtin_unreachable is not available on MSVC; __assume(0) is the documented equivalent and
// carries the same "this point is unreachable" promise to the optimizer.
#if defined(_MSC_VER) && !defined(__builtin_unreachable)
#define __builtin_unreachable() __assume(0)
#endif

// port: _cvtsh_ss (F16C, half -> float, scalar) is a GCC/Clang builtin. MSVC only exposes the packed
// _mm_cvtph_ps, so widen one lane through it: identical value, same rounding (both are IEEE binary16
// to binary32 conversions).
#if defined(_MSC_VER) && !defined(_cvtsh_ss)
#define _cvtsh_ss(value) \
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(value))))
#endif
