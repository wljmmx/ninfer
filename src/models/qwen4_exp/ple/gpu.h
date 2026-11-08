#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen4_exp/ple/prepare.h"

namespace ninfer::models::qwen4_exp {

struct PlePreparationMemory {
    std::size_t ordinary_host_bytes;
    std::size_t pinned_host_bytes;
    std::size_t device_bytes;
};

// Program-owned fixed Host/Device E slots and their Host preparation service. CUDA calls
// belong to the Program thread bound to context.device. Table/context outlive this owner.
// One ordered consumer stream uses each ticket. Its last E reader precedes record_consumed.
class PleGpuPreparation {
public:
    PleGpuPreparation(const PleTableSource&, std::int32_t eos, const DeviceContext&,
                      PlePrepareOptions);
    ~PleGpuPreparation();
    PleGpuPreparation(const PleGpuPreparation&)            = delete;
    PleGpuPreparation& operator=(const PleGpuPreparation&) = delete;

    // Host control/inspection. This owner's windows are released only through try_release.
    [[nodiscard]] PlePrepareService& host() noexcept;
    void upload(PleTicket, CudaExecutionView transfer);
    // Enqueues the current upload's dependency, then borrows Device E. No Host GPU wait.
    [[nodiscard]] Tensor embedding(PleTicket, CudaExecutionView consumer);
    void record_consumed(PleTicket, CudaExecutionView consumer);
    // Transfer completion also remains queryable after cancellation, for retiring its storage.
    [[nodiscard]] bool upload_complete(PleTicket) const;
    // Returns false while Host/CUDA users remain. An unsealed window must first be cancelled.
    // Host release is asynchronous; a later prepare may briefly report no free slot.
    [[nodiscard]] bool try_release(PleTicket);

    // Stable address for building a per-slot consumer graph. Runtime access still requires
    // embedding(ticket, consumer) before launch and record_consumed after its last reader.
    [[nodiscard]] Tensor device_slot(std::uint32_t slot, std::uint32_t tokens,
                                     std::uint32_t batch) const;
    [[nodiscard]] PlePreparationMemory memory() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
