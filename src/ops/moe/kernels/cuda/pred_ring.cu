// port (sm_89): P1 - the prediction descriptor ring, written on the Device.
//
// The prediction is produced where the router lives (a kernel), and the host consumes it one layer
// later. S3 already kept the ids on the Device, but as a single buffer: the host could read it, and
// could not tell whether the producer had since written a different prediction, nor that it had
// missed one. This kernel writes an *entry* - who produced it, for whom, how many ids, under which
// sequence number - and bumps one monotonic head word, which is exactly the shape
// device_admit_ring() established for "the Device decides, the host applies" (kernels/cuda/
// ring_admission.cu). The difference is the counter's reader: that ring is drained inside a layer
// that is about to synchronise anyway, so it calls cudaStreamSynchronize; this one is drained behind
// the layer's own existing synchronise, so it adds none.
#include "ninfer/ops/moe.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {
namespace {

// One block: thread 0 owns the ticket, the block copies the ids. A prediction is tens of ids, so the
// cost is one atomic and one coalesced store - the point is the shape of the hand-off, not throughput.
__global__ void pred_ring_write_kernel(const std::int32_t* ids, int count, std::uint32_t source_bank,
                                       std::uint32_t target_bank, std::uint32_t layer,
                                       PredRingHeader* headers, std::int32_t* entry_ids,
                                       std::uint32_t id_capacity, std::uint32_t depth,
                                       unsigned long long* head) {
    __shared__ unsigned long long sequence;
    __shared__ unsigned int slot;
    if (threadIdx.x == 0) {
        sequence = atomicAdd(head, 1ull);
        slot     = static_cast<unsigned int>(sequence % static_cast<unsigned long long>(depth));
        PredRingHeader header;
        header.source_bank = source_bank;
        header.target_bank = target_bank;
        header.layer       = layer;
        header.count       = static_cast<std::uint32_t>(count);
        header.capacity    = id_capacity;
        // A prediction wider than an entry is refused as a whole: writing the first `id_capacity`
        // ids would place a set the layer never produced, and a half-prediction reads as a wrong one.
        header.truncated = (count < 0 || static_cast<std::uint32_t>(count) > id_capacity) ? 1u : 0u;
        header.sequence  = sequence;
        headers[slot]    = header;
    }
    __syncthreads();
    if (headers[slot].truncated != 0 || ids == nullptr) { return; }
    std::int32_t* destination = entry_ids + static_cast<std::size_t>(slot) * id_capacity;
    for (int i = static_cast<int>(threadIdx.x); i < count; i += static_cast<int>(blockDim.x)) {
        destination[i] = ids[i];
    }
}

} // namespace

void moe_pred_ring_write(const std::int32_t* ids, int count, std::uint32_t source_bank,
                         std::uint32_t target_bank, std::uint32_t layer, PredRingHeader* headers,
                         std::int32_t* entry_ids, std::uint32_t id_capacity, std::uint32_t depth,
                         unsigned long long* head, cudaStream_t stream) {
    if (head == nullptr || headers == nullptr || entry_ids == nullptr || depth == 0 ||
        id_capacity == 0) {
        return;
    }
    pred_ring_write_kernel<<<1, 128, 0, stream>>>(ids, count, source_bank, target_bank, layer,
                                                 headers, entry_ids, id_capacity, depth, head);
}

} // namespace ninfer::ops
