#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Export copies a checkpoint's KV prefix out. The Device replica is the authoritative one: the Host
// copy is optional (a demotion may never have happened, and after a spill the Host extent is gone).
// So the Device path is tried first and the Host path is the fallback, not the other way round --
// requiring a Host copy would make export depend on the demotion policy, which is not its business.
//
// The Device path borrows a Host extent for the copy and then aborts it: the extent is scratch, and
// leaving it published would attach a Host replica to every page just because someone asked for a
// dump, quietly growing the arena on a read-only operation.
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

    std::vector<LogicalKVPageHandle> logical;
    std::vector<DeviceKVPageHandle> devices;
    logical.reserve(pages);
    devices.reserve(pages);
    bool all_device = true;
    for (std::uint32_t index = 0; index < pages; ++index) {
        const LogicalKVPageHandle page = text_kv_addresses->logical_page(address, index);
        logical.push_back(page);
        if (text_kv_pages->device_resident(page)) {
            devices.push_back(text_kv_pages->physical(page));
        } else {
            all_device = false;
        }
    }

    if (all_device) {
        auto reservation = host_kv_extents->prepare(*text_kv_pages, logical);
        if (!reservation) { return std::nullopt; }
        const auto host = host_kv_extents->writable_view(*reservation);
        const auto work = text_kv_pages->physical_pool().copy_to_host(devices, host,
                                                                      device.transfer_stream);
        (void)work;
        if (cudaStreamSynchronize(device.transfer_stream) != cudaSuccess) {
            host_kv_extents->abort(*reservation);
            return std::nullopt;
        }
        for (std::uint32_t index = 0; index < pages; ++index) {
            std::memcpy(out.kv_bytes.data() + static_cast<std::size_t>(index) * stride,
                        host.data() + static_cast<std::size_t>(index) * stride, stride);
            out.epochs.push_back(text_kv_pages->content_epoch(logical[index]));
        }
        host_kv_extents->abort(*reservation);
        return out;
    }

    // Host fallback: the page no longer has a Device replica, so the Host one is all there is.
    for (std::uint32_t index = 0; index < pages; ++index) {
        const LogicalKVPageHandle page = logical[index];
        if (!host_kv_extents->ensure_host_resident(*text_kv_pages, page)) { return std::nullopt; }
        const auto view = host_kv_extents->page_view(*text_kv_pages, page);
        if (!view || !view->valid() || view->page_count() != 1) { return std::nullopt; }
        if (view->layout().page_stride != stride) { return std::nullopt; }
        std::memcpy(out.kv_bytes.data() + static_cast<std::size_t>(index) * stride, view->data(),
                    stride);
        out.epochs.push_back(text_kv_pages->content_epoch(page));
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::detail
