#include "models/qwen3_5/execution/diagnostics.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

void nan_probe_tensor(const char* tag, const Tensor& value, std::size_t layer, cudaStream_t stream) {
    if (std::getenv("NINFER_NAN_PROBE") == nullptr || value.data == nullptr) { return; }
    if (value.dtype != DType::BF16) { return; }
    const auto rows = value.ne[0], cols = value.ne[1];
    std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * cols);
    if (cudaMemcpyAsync(host.data(), value.data, host.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return; }
    std::size_t bad = 0, elements = 0;
    int first = -1;
    std::vector<std::size_t> row_hits(static_cast<std::size_t>(rows), 0);
    std::vector<std::size_t> col_hits(static_cast<std::size_t>(cols), 0);
    for (int c = 0; c < cols; ++c) {
        std::size_t token_bad = 0;
        for (int r = 0; r < rows; ++r) {
            const auto bits = host[static_cast<std::size_t>(c) * rows + r];
            if ((bits & 0x7f80U) == 0x7f80U && (bits & 0x7fU) != 0U) {
                ++token_bad;
                ++elements;
                ++row_hits[static_cast<std::size_t>(r)];
            }
        }
        if (token_bad != 0) {
            ++bad;
            col_hits[static_cast<std::size_t>(c)] = token_bad;
            if (first < 0) { first = c; }
        }
    }
    std::size_t bad_rows = 0, first_bad_row = static_cast<std::size_t>(-1);
    for (std::size_t r = 0; r < row_hits.size(); ++r) {
        if (row_hits[r] != 0) {
            ++bad_rows;
            if (first_bad_row == static_cast<std::size_t>(-1)) { first_bad_row = r; }
        }
    }
    std::fprintf(stderr,
                 "[nan-probe] layer=%zu %s columns=%d bad_columns=%zu first=%d elements=%zu "
                 "bad_rows=%zu first_row=%lld\n",
                 layer, tag, cols, bad, first, elements, bad_rows,
                 static_cast<long long>(first_bad_row));
    if (std::getenv("NINFER_NAN_PROBE") != nullptr &&
        std::string(std::getenv("NINFER_NAN_PROBE")) == "2") {
        std::fprintf(stderr, "[nan-probe]   %s per-column:", tag);
        for (int c = 0; c < cols; ++c) {
            std::fprintf(stderr, " %d:%zu", c, col_hits[static_cast<std::size_t>(c)]);
        }
        std::fprintf(stderr, "\n");
    }
    if (std::getenv("NINFER_NAN_PROBE") != nullptr &&
        std::string(std::getenv("NINFER_NAN_PROBE")) == "3") {
        std::fprintf(stderr, "[nan-probe]   %s sentinel-per-column (0x8080):", tag);
        for (int c = 0; c < cols; ++c) {
            std::size_t hits = 0;
            for (int r = 0; r < rows; ++r) {
                if (host[static_cast<std::size_t>(c) * rows + r] == 0x8080U) { ++hits; }
            }
            std::fprintf(stderr, " %d:%zu", c, hits);
        }
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
}


bool qsa_dump_wants_layer(int layer) {
    const char* value = std::getenv("NINFER_QSA_DUMP_LAYER");
    if (value == nullptr || *value == '\0') { return layer == 0; }
    const long selected = std::strtol(value, nullptr, 10);
    return selected < 0 || static_cast<int>(selected) == layer;
}

namespace {

std::uint64_t fnv1a64(const void* data, std::size_t bytes) {
    const auto* bytes_in = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash   = 1469598103934665603ULL;
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= bytes_in[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

void qsa_dump_tensor(const char* variable, const char* name, const Tensor& value, int layer,
                     std::int32_t columns, cudaStream_t stream) {
    const char* directory = std::getenv(variable);
    if (directory == nullptr || *directory == '\0' || !qsa_dump_wants_layer(layer)) { return; }
    // A BF16 cache has no scale planes: an absent tensor is simply not dumped.
    if (value.data == nullptr) { return; }
    if (!value.is_contiguous()) {
        throw std::logic_error("QSA dump needs a contiguous device tensor");
    }
    std::vector<std::uint8_t> host(value.bytes());
    if (cudaMemcpyAsync(host.data(), value.data, host.size(), cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return; }
    std::string path(directory);
    path += "\\";
    path += name;
    path += "_layer";
    path += std::to_string(layer);
    if (columns > 0) {
        path += "_T";
        path += std::to_string(columns);
    }
    path += ".bin";
    if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
        (void)std::fwrite(host.data(), 1, host.size(), file);
        (void)std::fclose(file);
    }
    std::fprintf(stderr, "[qsa] dump %s layer=%d bytes=%zu fnv=%016llx path=%s\n", name, layer,
                 host.size(), static_cast<unsigned long long>(fnv1a64(host.data(), host.size())),
                 path.c_str());
    std::fflush(stderr);
}
} // namespace ninfer::models::qwen3_5::execution
