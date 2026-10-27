// State image import: the inverse of export_state, and the point where a fingerprint earns its keep.
//
// The order is deliberate. Fingerprint first, then the shard join, then any allocation. A restore
// that allocated a Host slot before deciding the payload was for a different configuration would
// have to unwind it, and the unwind path is exactly where a partial restore hides.
//
// The result is a StateImageHandle that owns a published Host replica. Attaching it to a checkpoint
// record is the caller's job (it also needs the KV prefix and the frontier), which is why this
// returns the handle rather than a checkpoint.

#include "core/checkpoint_fingerprint.h"
#include "core/state_image_shards.h"
#include "models/qwen3_5/program/program_impl.h"

#include <cstring>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

std::optional<StateImageHandle> ProgramImpl::import_state(
    const std::vector<std::string>& shards, const core::fingerprint::CheckpointFingerprint& want,
    std::string& error) {
    error.clear();
    if (!state_store || !host_state_images) {
        error = "no state storage in this program";
        return std::nullopt;
    }

    // 1. Would this payload even belong here?
    core::fingerprint::CheckpointFingerprint have;
    host_state_fingerprint(have);
    if (!core::fingerprint::matches_state(have, want, error)) {
        error = "checkpoint fingerprint does not match this process: " + error;
        return std::nullopt;
    }

    // 2. Reassemble. The join already refuses gaps, overlaps and short payloads, so a failure here
    //    is a corrupt transfer rather than a configuration problem.
    std::string image;
    if (!core::state_shards::join(shards, image, error)) {
        error = "state shards: " + error;
        return std::nullopt;
    }
    if (image.size() != want.state_image_bytes) {
        error = "state image is " + std::to_string(image.size()) + " B, fingerprint says " +
                std::to_string(want.state_image_bytes) + " B";
        return std::nullopt;
    }

    // 3. Only now touch the pool.
    const auto slot = host_state_images->allocate();
    if (!slot) {
        error = "no free Host state slot";
        return std::nullopt;
    }
    const auto view = host_state_images->writable_view(*slot);
    if (view.data == nullptr || view.layout == nullptr ||
        view.layout->image_bytes != image.size()) {
        (void)host_state_images->release(*slot);
        error = "Host state slot does not fit this image";
        return std::nullopt;
    }
    std::memcpy(view.data, image.data(), image.size());
    if (!host_state_images->publish(*slot)) {
        (void)host_state_images->release(*slot);
        error = "Host state slot could not be published";
        return std::nullopt;
    }

    // 4. Give the filled slot an object to belong to.
    std::optional<StateImageHandle> handle = state_store->reserve_logical_destination();
    if (!handle) {
        (void)host_state_images->release(*slot);
        error = "no free state handle";
        return std::nullopt;
    }
    if (!state_store->adopt_host_replica(*handle, *slot)) {
        (void)state_store->release(*handle);
        (void)host_state_images->release(*slot);
        error = "could not attach the Host replica to a state handle";
        return std::nullopt;
    }
    return handle;
}

void ProgramImpl::host_state_fingerprint(core::fingerprint::CheckpointFingerprint& out) const {
    out = {};
    if (!host_state_images) { return; }
    const qwen3_5::StateImageHostLayout& layout = host_state_images->layout();
    out.state_image_bytes = layout.image_bytes;
    out.state_layer_bytes = layout.linear_recurrent_layer_bytes;
    out.state_conv_bytes  = layout.linear_conv_layer_bytes * layout.spec.linear.layers;
    out.state_layers      = layout.spec.linear.layers;
    out.features.dflash_local = layout.spec.dflash_local.has_value();
}

} // namespace ninfer::models::qwen3_5::detail
