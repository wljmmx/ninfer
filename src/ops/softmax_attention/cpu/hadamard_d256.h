#pragma once

// port (sm_89): host mirror of the normalized D256 Hadamard (ops/kv_cache/hadamard_d256.cuh),
// in the same butterfly/rounding order. The GPU transform runs on one warp per row: lane l
// carries dimensions l + 32*r in values[r], first hadamard_d32_columns_inplace walks the five
// lane strides (dimension bits 0..4, shfl pairs), then the in-register loop walks spans 1/2/4
// (dimension bits 5..7), and the 2^-4 normalization is applied once at the end
// (hadamard_d256.cuh:46-65). In dimension space that is exactly one radix-2 butterfly per bit
// taken in ascending bit order — every element sees the same add/sub operands in the same
// sequence, so the two forms below are bit-identical to the GPU transform for every fp32 input.
// Phase A verified the warp-layout transcription against the production append dumps bit for
// bit (.temp/a-check known_answer); the same order is reproduced here from the source alone.
//
// Frame contract: K pages store the rotated frame (append rotates before quantizing), the
// scoring kernels rotate Q with this transform at entry, V stays unrotated. The inverse
// transform is never part of scoring or of this header.

#include <cstdint>

namespace ninfer::ops::detail::softmax_cpu {

// Warp-lane transcription of normalized_hadamard_d256_inplace (hadamard_d256.cuh:46-65):
// values[r][lane] holds dimension lane + 32*r. Kept for parity testing against the GPU shape.
inline void hadamard_d256_warp_inplace(float (&values)[8][32]) noexcept {
    for (int stride = 1; stride <= 16; stride <<= 1) {  // hadamard_d32_columns_inplace
        for (int r = 0; r < 8; ++r) {
            float next[32];
            for (int lane = 0; lane < 32; ++lane) {
                const float value = values[r][lane];
                const float peer  = values[r][lane ^ stride];
                next[lane]        = (lane & stride) == 0 ? value + peer : peer - value;
            }
            for (int lane = 0; lane < 32; ++lane) { values[r][lane] = next[lane]; }
        }
    }
    for (int span = 1; span < 8; span <<= 1) {  // in-register butterfly over the 8 fragments
        for (int base = 0; base < 8; base += 2 * span) {
            for (int offset = 0; offset < span; ++offset) {
                for (int lane = 0; lane < 32; ++lane) {
                    const float low  = values[base + offset][lane];
                    const float high = values[base + offset + span][lane];
                    values[base + offset][lane]        = low + high;
                    values[base + offset + span][lane] = low - high;
                }
            }
        }
    }
    for (int r = 0; r < 8; ++r) {  // final 2^-4 normalization
        for (int lane = 0; lane < 32; ++lane) { values[r][lane] *= 0x1p-4f; }
    }
}

// Dimension-order form: values[d] is dimension d. Same stage order (bit 0..7 butterflies,
// pairs (d, d|stride) with low+high / low-high), same single 2^-4 scaling — bit-identical to
// hadamard_d256_warp_inplace and to the GPU transform. This is the form the host partial
// consumes (Q rows arrive in dimension order).
inline void hadamard_d256_inplace(float (&values)[256]) noexcept {
    for (int stride = 1; stride < 256; stride <<= 1) {
        for (int d = 0; d < 256; ++d) {
            if ((d & stride) != 0) { continue; }
            const float low  = values[d];
            const float high = values[d | stride];
            values[d]         = low + high;
            values[d | stride] = low - high;
        }
    }
    for (int d = 0; d < 256; ++d) { values[d] *= 0x1p-4f; }
}

} // namespace ninfer::ops::detail::softmax_cpu
