#include "models/qwen4_exp/ple/gpu.h"

#include "core/arena.h"
#include "core/device_scope.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::models::qwen4_exp {

struct PleGpuPreparation::Impl {
    struct Slot {
        PinnedHostBuffer host;
        DeviceBuffer device;
        CudaCompletionEvent uploaded, consumed;
        std::uint64_t generation = 0;
        std::uint32_t tokens = 0, batch = 0;
        cudaStream_t transfer = nullptr, consumer = nullptr;
        bool active = false, upload_recorded = false, consumption_recorded = false;
        std::exception_ptr failure;

        Slot(const DeviceContext& context, std::size_t bytes)
            : host(bytes), device(bytes), uploaded(context), consumed(context) {}
    };

    Device device;
    std::uint32_t max_positions;
    std::vector<Slot> slots;
    std::unique_ptr<PlePrepareService> service;

    Impl(const PleTableSource& table, std::int32_t eos, const DeviceContext& context,
         PlePrepareOptions options)
        : device(Device::cuda(context.device)), max_positions(options.max_positions) {
        if (!max_positions || max_positions > UINT32_MAX / kPleHeads || !options.windows)
            throw std::invalid_argument("invalid PLE GPU capacity");
        DeviceScope scope(device);
        const auto bytes = std::size_t(max_positions) * kPleEmbeddingBytes;
        slots.reserve(options.windows);
        std::vector<MemorySpan> outputs;
        outputs.reserve(options.windows);
        for (std::uint32_t i = 0; i < options.windows; ++i) {
            slots.emplace_back(context, bytes);
            outputs.push_back({slots.back().host.data(), bytes, Device::cpu(), true});
        }
        service = std::make_unique<PlePrepareService>(table, eos, std::move(options), outputs);
    }

    ~Impl() {
        try {
            DeviceScope scope(device);
            for (auto& slot : slots) {
                if (!slot.active) continue;
                // Also covers a failed event record after successful asynchronous submission.
                if (slot.transfer) CUDA_CHECK(cudaStreamSynchronize(slot.transfer));
                if (slot.consumer) CUDA_CHECK(cudaStreamSynchronize(slot.consumer));
            }
            service.reset();
        } catch (...) { std::terminate(); }
    }

    Slot& current(PleTicket ticket) {
        (void)service->poll(ticket);
        auto& slot = slots.at(ticket.slot);
        if (!slot.active || slot.generation != ticket.generation)
            throw std::logic_error("PLE ticket has no current GPU upload");
        if (slot.failure) std::rethrow_exception(slot.failure);
        return slot;
    }

    void validate(CudaExecutionView execution) const {
        if (execution.device() != device || !execution.stream)
            throw std::invalid_argument(
                "PLE transfer/consumer belongs to another execution device");
    }

    [[noreturn]] void submission_failed(PleTicket ticket, Slot& slot) {
        const auto error = std::current_exception();
        slot.failure     = error;
        service->cancel(ticket);
        if (slot.transfer) CUDA_CHECK(cudaStreamSynchronize(slot.transfer));
        if (slot.consumer && slot.consumer != slot.transfer)
            CUDA_CHECK(cudaStreamSynchronize(slot.consumer));
        slot.active = slot.upload_recorded = slot.consumption_recorded = false;
        slot.transfer = slot.consumer = nullptr;
        slot.failure                  = {};
        std::rethrow_exception(error);
    }
};

PleGpuPreparation::PleGpuPreparation(const PleTableSource& table, std::int32_t eos,
                                     const DeviceContext& context, PlePrepareOptions options)
    : impl_(std::make_unique<Impl>(table, eos, context, std::move(options))) {}

PleGpuPreparation::~PleGpuPreparation() = default;

PlePrepareService& PleGpuPreparation::host() noexcept { return *impl_->service; }

void PleGpuPreparation::upload(PleTicket ticket, CudaExecutionView transfer) {
    auto& p = *impl_;
    p.validate(transfer);
    const auto input = p.service->embedding(ticket);
    auto& slot       = p.slots.at(ticket.slot);
    if (slot.active) throw std::logic_error("PLE slot already has a GPU upload");
    slot.generation = ticket.generation;
    slot.tokens     = input.window.tokens;
    slot.batch      = input.window.batch;
    slot.active     = true;
    slot.transfer   = transfer.stream;
    try {
        CUDA_CHECK(cudaMemcpyAsync(slot.device.p, input.values.data, input.values.bytes,
                                   cudaMemcpyHostToDevice, transfer.stream));
        slot.uploaded.record(transfer.stream);
        slot.upload_recorded = true;
    } catch (...) { p.submission_failed(ticket, slot); }
}

Tensor PleGpuPreparation::embedding(PleTicket ticket, CudaExecutionView consumer) {
    auto& p = *impl_;
    p.validate(consumer);
    auto& slot = p.current(ticket);
    if (!slot.upload_recorded || slot.consumption_recorded ||
        (slot.consumer && slot.consumer != consumer.stream))
        throw std::logic_error("invalid PLE GPU consumer ordering");
    (void)p.service->embedding(ticket);
    slot.consumer = consumer.stream;
    try {
        slot.uploaded.wait(consumer.stream);
    } catch (...) { p.submission_failed(ticket, slot); }
    return device_slot(ticket.slot, slot.tokens, slot.batch);
}

void PleGpuPreparation::record_consumed(PleTicket ticket, CudaExecutionView consumer) {
    auto& p = *impl_;
    p.validate(consumer);
    auto& slot = p.current(ticket);
    if (slot.consumer != consumer.stream || slot.consumption_recorded)
        throw std::logic_error("PLE consumer completion has no matching acquisition");
    try {
        slot.consumed.record(consumer.stream);
    } catch (...) { p.submission_failed(ticket, slot); }
    slot.consumption_recorded = true;
}

bool PleGpuPreparation::upload_complete(PleTicket ticket) const {
    const auto& slot = impl_->current(ticket);
    return slot.upload_recorded && slot.uploaded.ready();
}

bool PleGpuPreparation::try_release(PleTicket ticket) {
    auto& p          = *impl_;
    const auto state = p.service->poll(ticket);
    if (state.status == PlePrepareStatus::Preparing || !state.drained) return false;
    auto& slot = p.slots.at(ticket.slot);
    if (slot.active) {
        if (slot.failure) std::rethrow_exception(slot.failure);
        if (slot.generation != ticket.generation)
            throw std::logic_error("PLE GPU slot generation differs");
        if (!slot.upload_recorded || !slot.uploaded.ready() ||
            (slot.consumer && (!slot.consumption_recorded || !slot.consumed.ready())))
            return false;
    }
    p.service->release(ticket);
    slot.active = slot.upload_recorded = slot.consumption_recorded = false;
    slot.transfer = slot.consumer = nullptr;
    return true;
}

Tensor PleGpuPreparation::device_slot(std::uint32_t slot, std::uint32_t tokens,
                                      std::uint32_t batch) const {
    auto& p = *impl_;
    if (!tokens || !batch || batch > 8 || std::uint64_t(tokens) * batch > p.max_positions)
        throw std::invalid_argument("invalid PLE device view shape");
    const auto bytes = std::size_t(tokens) * batch * kPleEmbeddingBytes;
    return Tensor(MemorySpan{p.slots.at(slot).device.p, bytes, p.device, false}, DType::BF16,
                  {static_cast<std::int32_t>(kPleEmbeddingElements),
                   static_cast<std::int32_t>(tokens), static_cast<std::int32_t>(batch)});
}

PlePreparationMemory PleGpuPreparation::memory() const noexcept {
    const auto& p    = *impl_;
    const auto bytes = std::size_t(p.max_positions) * p.slots.size() * kPleEmbeddingBytes;
    return {p.service->memory_bytes() + p.slots.size() * sizeof(Impl::Slot), bytes, bytes};
}

} // namespace ninfer::models::qwen4_exp
