// Restoring a checkpoint: KV pages and state image into one record.
//
// The two halves are imported by different code paths because they live in different stores, but
// they become a checkpoint together or not at all: a record holding KV without its state resumes
// from a state that never existed, which is exactly the trap this whole item exists to avoid.
//
// Order: state first, then KV pages, then the record. The state import validates the fingerprint and
// can refuse cheaply; doing it before allocating pages means a mismatched payload costs nothing.
//
// What this deliberately does not do: build a PrefixShortlistKey. An imported checkpoint is
// addressable by handle and can be exported and read, but prefix lookup will not find it until a
// caller supplies the sequence digests -- those come from the token stream at restore time, not from
// the bytes. Stated here rather than left as a surprise later.

#include "core/checkpoint_fingerprint.h"
#include "core/state_image_shards.h"
#include "models/qwen3_5/program/program_impl.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

std::optional<CheckpointHandle> ProgramImpl::import_checkpoint(
    const CheckpointExport& kv, const std::vector<std::string>& state_shards,
    const ninfer::core::fingerprint::CheckpointFingerprint& want, std::string& error) {
    error.clear();
    if (!text_kv_pages || !text_kv_addresses || !host_kv_extents) {
        error = "no KV storage in this program";
        return std::nullopt;
    }
    if (kv.page_count == 0 || kv.page_bytes == 0) {
        error = "checkpoint export carries no pages";
        return std::nullopt;
    }
    if (kv.kv_bytes.size() != kv.page_bytes * static_cast<std::size_t>(kv.page_count)) {
        error = "KV bytes do not match page_count x page_bytes";
        return std::nullopt;
    }

    // 1. State first: it validates the fingerprint and refuses before anything is allocated.
    const auto state = import_state(state_shards, want, error);
    if (!state) { return std::nullopt; }
    // adopt_host_replica already left it CheckpointImmutable, and freeze() cannot help here:
    // it demands a Device slot, which a restored image does not have.
    state_store->retain_checkpoint_reference(*state);

    // 2. Device first, because a logical page is born in VRAM: can_pin_source() and
    //    can_pin_active_source() both require a Device replica, so a page without one can never be
    //    given a Host extent. Restoring straight into Host memory is not a path this store has.
    auto device_pages = text_kv_pages->physical_pool().reserve(kv.page_count);
    if (!device_pages) {
        error = "no free Device pages for the restored checkpoint";
        return std::nullopt;
    }
    std::vector<LogicalKVPageHandle> pages(kv.page_count);
    text_kv_pages->materialize(*device_pages, std::span<LogicalKVPageHandle>(pages));

    // 3. Write the bytes into a Host extent (now legal -- the pages have Device replicas), then
    //    copy them down. The Host copy stays as the demoted tier; without this the Device pages
    //    would be published but empty, and readers would take them as authoritative.
    {
        auto reservation = host_kv_extents->prepare(*text_kv_pages, pages);
        if (!reservation) {
            error = "no Host extent for the restored pages";
            return std::nullopt;
        }
        const auto view = host_kv_extents->writable_view(*reservation);
        const std::size_t expected = kv.page_bytes * static_cast<std::size_t>(kv.page_count);
        const std::size_t available =
            static_cast<std::size_t>(view.page_count()) * view.layout().page_stride;
        if (view.data() == nullptr || available < expected) {
            host_kv_extents->abort(*reservation);
            error = "Host extent is smaller than the restored payload";
            return std::nullopt;
        }
        std::memcpy(view.data(), kv.kv_bytes.data(), expected);
        const auto capability = host_kv_extents->publish(std::move(*reservation));

        std::vector<DeviceKVPageHandle> device_handles;
        device_handles.reserve(pages.size());
        for (const LogicalKVPageHandle page : pages) {
            device_handles.push_back(text_kv_pages->physical(page));
        }
        const auto source = host_kv_extents->view(capability);
        (void)text_kv_pages->physical_pool().copy_from_host(source,
                                                           std::span<const DeviceKVPageHandle>(
                                                               device_handles.data(),
                                                               device_handles.size()),
                                                           device.transfer_stream);
        if (cudaStreamSynchronize(device.transfer_stream) != cudaSuccess) {
            error = "Device copy for the restored pages failed";
            return std::nullopt;
        }
    }
    // 4. Map them into a fresh inactive address.
    std::optional<KVAddressSpaceHandle> address = text_kv_addresses->create_inactive();
    if (!address) {
        error = "no free KV address space";
        return std::nullopt;
    }
    if (!text_kv_addresses->adopt_pages(*address, pages)) {
        (void)text_kv_addresses->release(*address);
        error = "could not map the restored pages";
        return std::nullopt;
    }
    // A fresh address commits nothing, and a checkpoint requirement may not exceed the committed
    //    frontier -- so the frontier has to be committed before it can be protected.
    // commit_frontier() only accepts an active address and a checkpoint address is inactive by
    // design; the pages stay alive via the read reference adopt_pages() took, and the frontier

    // 5. The record.
    const auto handle = reserve_checkpoint();
    if (!handle) {
        (void)text_kv_addresses->release(*address);
        error = "no free checkpoint slot";
        return std::nullopt;
    }
    auto history  = std::make_shared<KVHistory>();
    history->text = *address;
    checkpoints[handle->index].value =
        CheckpointState{.kv       = std::move(history),
                        .state    = *state,
                        .frontier = kv.frontier};
    checkpoints[handle->index].reserved = false;
    return handle;
}

// Diagnostic entry point, kept out of the dump hook so that hook stays one call long. Importing a
// checkpoint's own export and comparing the bytes back is the cheapest way to prove that
// create_host_backed_page, adopt_pages, import_state and import_checkpoint agree with the export.
void ProgramImpl::verify_import_round_trip(CheckpointHandle handle) {
    const auto source = export_checkpoint(handle);
    const auto source_state = export_state(handle);
    if (!source || !source_state) {
        std::fprintf(stderr, "[ckpt] import check: nothing to export\n");
        return;
    }

    ninfer::core::fingerprint::CheckpointFingerprint print;
    host_state_fingerprint(print);
    ninfer::core::state_shards::RegionSizes sizes;
    sizes.conv_bytes  = print.state_conv_bytes;
    sizes.layer_bytes = print.state_layer_bytes;
    sizes.layers      = print.state_layers;
    const std::uint64_t covered = sizes.conv_bytes + sizes.layer_bytes * sizes.layers;
    sizes.tail_bytes = print.state_image_bytes > covered ? print.state_image_bytes - covered : 0;

    const std::string image(reinterpret_cast<const char*>(source_state->bytes.data()),
                            source_state->bytes.size());
    const auto split = ninfer::core::state_shards::split(image, sizes, 6ull * 1024 * 1024);
    if (!split.ok) {
        std::fprintf(stderr, "[ckpt] import check: split failed: %s\n", split.error.c_str());
        return;
    }

    std::string why;
    const auto imported = import_checkpoint(*source, split.shards, print, why);
    if (!imported) {
        std::fprintf(stderr, "[ckpt] import check: REFUSED %s\n", why.c_str());
        return;
    }
    const auto again = export_checkpoint(*imported);
    const bool same = again.has_value() && again->kv_bytes == source->kv_bytes &&
                      again->frontier == source->frontier;
    std::fprintf(stderr, "[ckpt] import check: %s (%u pages, %zu B)\n",
                 same ? "KV round-trip OK" : "KV MISMATCH",
                 again ? again->page_count : 0u,
                 again ? again->kv_bytes.size() : 0);
}
} // namespace ninfer::models::qwen3_5::detail
