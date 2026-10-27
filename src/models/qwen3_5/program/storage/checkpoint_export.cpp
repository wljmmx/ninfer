#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Export copies a checkpoint's KV prefix out, page by page, from whichever replica still holds it.
// The three cases are all real:
//   Device  the authoritative bytes; a Host copy may never have been made.
//   Host    the page was demoted; its Device replica may be gone.
//   Spill   the Host extent was given back to the file tier; both other copies are gone.
// Refusing the Spill case (as the first version did) makes export depend on eviction luck.
//
// The Device case borrows a Host extent for the copy and aborts it: copy_to_host only writes into a
// HostKVAllocationView, and leaving the extent published would attach a Host replica to every page
// on what is meant to be a read-only operation.
std::optional<CheckpointExport> ProgramImpl::export_checkpoint(CheckpointHandle handle) {
    if (!valid_checkpoint(handle)) { return std::nullopt; }
    const CheckpointState& record = checkpoint(handle);
    if (!record.kv || !text_kv_pages || !text_kv_addresses || !host_kv_extents) {
        return std::nullopt;
    }
    const KVAddressSpaceHandle address = record.kv->text;
    if (!text_kv_addresses->valid(address)) { return std::nullopt; }

    const std::uint32_t pages = text_kv_addresses->page_count(address);
    if (pages == 0) { return std::nullopt; }
    const std::size_t stride = host_kv_extents->page_layout(*text_kv_pages).page_stride;
    if (stride == 0) { return std::nullopt; }

    CheckpointExport out;
    out.frontier   = record.frontier;
    out.page_count = pages;
    out.page_bytes = stride;
    out.epochs.reserve(pages);
    out.kv_bytes.resize(stride * static_cast<std::size_t>(pages));

    std::byte* destination = out.kv_bytes.data();
    for (std::uint32_t index = 0; index < pages; ++index) {
        const LogicalKVPageHandle page = text_kv_addresses->logical_page(address, index);
        std::byte* slot = destination + static_cast<std::size_t>(index) * stride;
        bool copied = false;

        // A page can hold both replicas -- an imported checkpoint does, because it was written from
        // a Host extent and then copied down. Prefer the Host copy when it exists: the Device path
        // borrows a temporary extent, and prepare() refuses a page that is already Host-resident,
        // so taking the Device branch first would fail on a page that has perfectly good bytes.
        if (index == 0) {
            std::fprintf(stderr, "[ckpt] export page0: host=%d device=%d spilled=%d\n",
                         text_kv_pages->host_resident(page) ? 1 : 0,
                         text_kv_pages->device_resident(page) ? 1 : 0,
                         text_kv_pages->spilled(page) ? 1 : 0);
        }
        if (text_kv_pages->host_resident(page)) {
            const auto view = host_kv_extents->page_view(*text_kv_pages, page);
            if (!view || !view->valid() || view->page_count() != 1 ||
                view->layout().page_stride != stride) {
                return std::nullopt;
            }
            std::memcpy(slot, view->data(), stride);
            copied = true;
        } else if (text_kv_pages->device_resident(page)) {
            const std::array<LogicalKVPageHandle, 1> one{page};
            const std::array<DeviceKVPageHandle, 1> device_page{text_kv_pages->physical(page)};
            auto reservation = host_kv_extents->prepare(*text_kv_pages, one);
            if (!reservation) { return std::nullopt; }
            const auto host = host_kv_extents->writable_view(*reservation);
            (void)text_kv_pages->physical_pool().copy_to_host(
                std::span<const DeviceKVPageHandle>(device_page), host, device.transfer_stream);
            if (cudaStreamSynchronize(device.transfer_stream) != cudaSuccess) {
                host_kv_extents->abort(*reservation);
                return std::nullopt;
            }
            std::memcpy(slot, host.data(), stride);
            host_kv_extents->abort(*reservation);
            copied = true;
        } else if (text_kv_pages->spilled(page)) {
            const auto replica = text_kv_pages->host_replica(page);
            KVSpillFile* file  = host_kv_extents->spill_file();
            if (!file || !replica.spill_slot.has_value() || file->extent_bytes() != stride) {
                return std::nullopt;
            }
            file->read(*replica.spill_slot, std::span<std::byte>(slot, stride));
            copied = true;
        }

        if (!copied) { return std::nullopt; }  // no replica at all: refuse rather than emit a hole
        out.epochs.push_back(text_kv_pages->content_epoch(page));
    }
    return out;
}

CheckpointTierGeometry ProgramImpl::checkpoint_geometry() const {
    CheckpointTierGeometry out;
    // Same stride export_checkpoint reports per page. Asking the layout rather than a model constant
    // is what keeps the transport and the pages from disagreeing.
    if (text_kv_pages && host_kv_extents) {
        out.page_bytes = host_kv_extents->page_layout(*text_kv_pages).page_stride;
    }
    if (state_store) { out.state_image_bytes = state_store->image_bytes(); }
    return out;
}

} // namespace ninfer::models::qwen3_5::detail
