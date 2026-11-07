#pragma once

namespace ninfer::ops::detail::hc {
inline constexpr int kWidth = 2560, kStreams = 4, kRank = 320;
inline constexpr int kResidualWidth = kWidth * kStreams;
} // namespace ninfer::ops::detail::hc
