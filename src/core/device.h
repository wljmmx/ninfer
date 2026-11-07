#pragma once

#include "core/memory.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer {

// Only Device allocations use this error; pinned Host allocation failures stay distinct, so a
// caller can tell "the GPU is full" apart from "the system is out of memory".
class DeviceAllocationError : public std::runtime_error {
public:
    DeviceAllocationError(const char* operation, std::size_t requested_bytes = 0);

    [[nodiscard]] std::size_t requested_bytes() const noexcept { return requested_bytes_; }

private:
    std::size_t requested_bytes_;
};

// One reading of the Device's free and total memory. Startup takes them around each step whose
// footprint is not an explicit layout, so the growth is measured rather than assumed.
struct DeviceMemorySnapshot {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
};

[[nodiscard]] DeviceMemorySnapshot device_memory_snapshot(int device);
// Positive when memory was consumed between the two readings.
[[nodiscard]] std::int64_t device_memory_growth(DeviceMemorySnapshot before,
                                                DeviceMemorySnapshot after);

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;
    // port (sm_89): the bench Ops tag operands with the resident device, so carry it here.
    // Aggregate initialisation that omits it keeps the previous behaviour.
    Device resident = Device{};

    [[nodiscard]] constexpr DeviceExecutionView
    on_stream(cudaStream_t target_stream) const noexcept {
        return {target_stream, multiprocessor_count, resident};
    }

    [[nodiscard]] Device device() const noexcept { return resident; }
};

// port (sm_89): the bench Ops name the same view CudaExecutionView.
using CudaExecutionView = DeviceExecutionView;

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    int multiprocessor_count() const noexcept;
    // The Device's free/total memory right now; the stream state is the caller's to settle.
    [[nodiscard]] DeviceMemorySnapshot memory_snapshot() const {
        return device_memory_snapshot(device);
    }
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    const char* sync_mode() const;
    void synchronize() const;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    // Capture a record node observable by the host while later graph nodes are still running.
    // Only call during stream capture; this event must outlive every graph referencing it.
    void record_external(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace ninfer
