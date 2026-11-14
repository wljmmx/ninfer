#pragma once

// Per-column magnitude of whatever a mixer produced for a verify rectangle, gated by
// NINFER_VERIFY_COLUMN_PROBE. The rectangle's columns are positions F, F+1, ... and the question
// this answers is whether a non-anchor column was computed from the state its predecessor produced:
// a column whose output norm collapses to the noise floor was not carried forward, while a column
// with an ordinary norm was - so "the state chain is broken" and "the head mishandles columns" stop
// looking the same. Diagnostic only: one host read and one line per (label, layer), never in a
// captured pass, and the whole path is off unless the variable is set.

#include "core/device.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

inline void verify_column_probe(const char* label, const Tensor& tensor, std::int32_t elements,
                                std::int32_t column_stride_bytes, std::int32_t width,
                                std::int32_t layer, cudaStream_t stream) {
    static const bool on = std::getenv("NINFER_VERIFY_COLUMN_PROBE") != nullptr;
    if (!on || width <= 0 || elements <= 0) { return; }
    static std::set<std::string> seen;
    if (!seen.insert(std::string(label) + ":" + std::to_string(layer)).second) { return; }
    if (tensor.data == nullptr) { return; }
    const bool bf16      = tensor.dtype == DType::BF16;
    const std::size_t w  = static_cast<std::size_t>(tensor.dtype == DType::BF16 ? 2 : 4);
    std::vector<std::uint8_t> host(static_cast<std::size_t>(elements) * w *
                                   static_cast<std::size_t>(width));
    for (std::int32_t column = 0; column < width; ++column) {
        const auto status = cudaMemcpyAsync(
            host.data() + static_cast<std::size_t>(column) * elements * w,
            static_cast<const std::uint8_t*>(tensor.data) +
                static_cast<std::size_t>(column) * static_cast<std::size_t>(column_stride_bytes),
            static_cast<std::size_t>(elements) * w, cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess) { return; }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return; }
    std::fprintf(stderr, "[verify-probe] %s layer=%d:", label, layer);
    for (std::int32_t column = 0; column < width; ++column) {
        double sum = 0.0;
        for (std::int32_t i = 0; i < elements; ++i) {
            float value = 0.0F;
            const auto* bytes =
                host.data() + (static_cast<std::size_t>(column) * elements + i) * w;
            if (bf16) {
                std::uint16_t bits16 = 0;
                std::memcpy(&bits16, bytes, sizeof(bits16));
                const std::uint32_t bits = static_cast<std::uint32_t>(bits16) << 16U;
                std::memcpy(&value, &bits, sizeof(value));
            } else {
                std::memcpy(&value, bytes, sizeof(value));
            }
            sum += double(value) * double(value);
        }
        std::fprintf(stderr, " c%d=%.4f", column, std::sqrt(sum / double(elements)));
    }
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

} // namespace ninfer::models::qwen3_5::execution
