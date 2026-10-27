#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <cstring>

namespace ninfer::models::qwen3_5::detail {

// Export copies the Host bytes of a checkpoint's KV prefix. It deliberately does NOT demote: a
// demote is a scheduled transfer with its own transaction, and a checkpoint whose pages are still
// device-only has no Host bytes to copy yet. Returning nullopt in that case is the honest answer --
// the alternative (silently skipping pages) would produce a blob that looks complete.
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

    for (std::uint32_t index = 0; index < pages; ++index) {
        const LogicalKVPageHandle page = text_kv_addresses->logical_page(address, index);
        // Host residency is a precondition, not something this call forces: forcing it here would
        // start a transfer this layer cannot wait for.
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
