#pragma once

// Debug-only K/V capture for offline quantization analysis (K outlier structure and
// rotation comparison). Disabled unless NINFER_KV_DUMP names an output file, and never
// active while the stream is capturing a CUDA graph, so the default build path and every
// graph capture/replay are unaffected. Records the raw BF16 K/V rows exactly as the
// append kernels receive them -- before rotation and before quantization -- so offline
// analysis can apply any candidate rotation itself.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ninfer::ops::detail {

inline const char* kv_dump_path() {
    static const char* const path = std::getenv("NINFER_KV_DUMP");
    return path;
}

inline bool kv_dump_enabled() { return kv_dump_path() != nullptr; }

inline bool kv_dump_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) { return true; }
    return status != cudaStreamCaptureStatusNone;
}

// One record per append call, followed by dim*kv_heads*tokens BF16 values for K and the
// same count for V. Layout is contiguous [dim, kv_heads, tokens].
struct KvDumpRecord {
    std::uint32_t magic;        // 0x4b564454 "KVDT"
    std::uint32_t call_index;
    std::int32_t  dim;
    std::int32_t  kv_heads;
    std::int32_t  tokens;
    std::int32_t  reserved;
};

inline void kv_dump_capture(const Tensor& k, const Tensor& v, cudaStream_t stream) {
    const char* const path = kv_dump_path();
    if (path == nullptr) { return; }
    if (kv_dump_stream_capturing(stream)) { return; }
    if (k.data == nullptr || v.data == nullptr) { return; }

    const std::int32_t dim    = k.ne[0];
    const std::int32_t heads  = k.ne[1];
    const std::int32_t tokens = k.ne[2];
    if (dim <= 0 || heads <= 0 || tokens <= 0) { return; }

    const std::size_t plane = static_cast<std::size_t>(dim) * heads * tokens * 2u;
    std::vector<unsigned char> host(2u * plane);
    if (cudaMemcpyAsync(host.data(), k.data, plane, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        return;
    }
    if (cudaMemcpyAsync(host.data() + plane, v.data, plane, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return; }

    static std::uint32_t call_index = 0;
    const KvDumpRecord record{0x4b564454u, call_index++, dim, heads, tokens, 0};
    std::FILE* file = std::fopen(path, "ab");
    if (file == nullptr) { return; }
    std::fwrite(&record, sizeof(record), 1, file);
    std::fwrite(host.data(), 1, 2u * plane, file);
    std::fclose(file);
}

} // namespace ninfer::ops::detail