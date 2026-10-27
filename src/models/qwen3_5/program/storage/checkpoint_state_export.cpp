// State image export: the other half of a checkpoint.
//
// KV pages alone are not a resumable checkpoint. This model keeps recurrent state (the linear
// attention conv/recurrent slices plus, when speculative decoding is on, the draft-local K/V), and a
// restore that skipped prefill without it would resume from a state that never existed.
//
// All the pieces were already present: HostStatePool has view()/writable_view() and the layout's
// image_bytes, and StateImageStore can demote a Device replica to Host. What was missing was any way
// to name the Host slot from outside the store -- see StateImageStore::host_slot(). This file is the
// wiring, not a new mechanism.
//
// Export is read-only in spirit: when a Host replica already exists we copy it and stop. When the
// state is Device-only we materialise one and keep the source replica (keep_source_replica = true),
// so an export never costs the model its Device copy.

#include "models/qwen3_5/program/program_impl.h"

#include <cstring>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

std::optional<CheckpointStateExport> ProgramImpl::export_state(CheckpointHandle handle) {
    if (!valid_checkpoint(handle)) { return std::nullopt; }
    const CheckpointState& record = checkpoint(handle);
    if (!state_store || !host_state_images) { return std::nullopt; }
    if (!state_store->valid(record.state)) { return std::nullopt; }

    CheckpointStateExport out;
    if (!state_store->host_resident(record.state)) {
        auto transfer = state_store->begin_device_to_host(record.state, device.transfer_stream);
        if (!transfer) { return std::nullopt; }
        if (cudaStreamSynchronize(device.transfer_stream) != cudaSuccess) {
            state_store->abort_transfer(std::move(*transfer));
            return std::nullopt;
        }
        state_store->publish_transfer(std::move(*transfer), /*keep_source_replica=*/true);
        out.materialised_host_copy = true;
        // The transfer only publishes on success; if it did not, there is nothing to copy.
        if (!state_store->host_resident(record.state)) { return std::nullopt; }
    }

    const qwen3_5::HostStateSlotHandle slot = state_store->host_slot(record.state);
    const qwen3_5::HostStateImageConstView view = host_state_images->view(slot);
    if (view.data == nullptr || view.layout == nullptr) { return std::nullopt; }
    const qwen3_5::StateImageHostLayout& layout = *view.layout;
    if (layout.image_bytes == 0) { return std::nullopt; }

    out.image_bytes = layout.image_bytes;
    out.layers      = layout.spec.linear.layers;
    out.conv_channels  = layout.spec.linear.conv_channels;
    out.value_heads    = layout.spec.linear.value_heads;
    out.value_head_dim = layout.spec.linear.value_head_dim;
    out.key_head_dim   = layout.spec.linear.key_head_dim;
    out.hidden         = layout.spec.hidden;
    out.has_dflash_local = layout.spec.dflash_local.has_value();

    out.bytes.assign(view.data, view.data + layout.image_bytes);
    return out;
}

bool ProgramImpl::state_bytes(StateImageHandle state, std::vector<std::byte>& out) const {
    out.clear();
    if (!state_store || !host_state_images) { return false; }
    if (!state_store->valid(state) || !state_store->host_resident(state)) { return false; }
    const qwen3_5::HostStateSlotHandle slot = state_store->host_slot(state);
    const qwen3_5::HostStateImageConstView view = host_state_images->view(slot);
    if (view.data == nullptr || view.layout == nullptr) { return false; }
    out.assign(view.data, view.data + view.layout->image_bytes);
    return true;
}
} // namespace ninfer::models::qwen3_5::detail
