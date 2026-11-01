#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

bool ProgramImpl::checkpoint_matches(CheckpointHandle handle, const RequestBasePlan& base) const {
    if (!valid_checkpoint(handle) || !base.impl_ || !base.impl_->allow_prefix_reuse) {
        return false;
    }
    const auto& record = checkpoint(handle);
    if (record.frontier > base.summary().prompt_tokens ||
        record.key.identity_tag != base.impl_->prefix_identity_tag || !record.identity ||
        base.impl_->prefix_digests.at(record.frontier) != record.key.digests) {
        return false;
    }
    return prefix_matches(*base.impl_->prompt, record.identity->ledger,
                          record.identity->prefix_identity, record.frontier);
}

std::optional<SourceCandidate>
ProgramImpl::inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> handle,
                            bool consume_source, std::span<const CheckpointHandle> private_points,
                            std::span<const CheckpointHandle> retired_points) {
    // port (sm_89): Q54. A prefix this process has never seen can still be on disk. This runs before
    // anything else because everything below is written in terms of a handle: with one, the
    // existing candidate construction, transfer planning and identity checks all apply unchanged,
    // and the disk source is indistinguishable from an in-process capture.
    //
    // A refusal is not a miss: if the tier had the block and the import declined it (a shape that
    // does not match this model, a page count that does not cover the frontier), the reason is
    // printed rather than silently treated as a cold prefix. That distinction is what separates
    // "nothing to reuse" from "reuse is broken here".
    bool from_tier = false;
    if (!handle && kv_tier != nullptr && base.impl_ && base.impl_->prompt &&
        !base.impl_->prompt->token_ids.empty()) {
        std::string why;
        if (const auto from_disk = tier_try_load(
                std::span<const TokenId>(base.impl_->prompt->token_ids), &why)) {
            handle    = from_disk;
            from_tier = true;
        } else if (!why.empty()) {
            std::fprintf(stderr, "[kvt] tier refused the prefix: %s\n", why.c_str());
            std::fflush(stderr);
        }
    }
    if (!base.impl_) { return std::nullopt; }
    // port (sm_89): an import from the tier does not have to pass the captured-path fingerprint.
    //
    // That check exists to prove a checkpoint belongs to THIS request: it compares the record's
    // ledger and digests against the base plan. An import has no such identity by design - the
    // external-cache entry point takes identity as null by default and its comment says so ("it has
    // bytes, not a fingerprint"). Its identity is established the other way round: the lookup key
    // was this prompt, so the bytes were found BY this request, not offered to it.
    //
    // Requiring both made the import unusable: the record was created, the match refused it, and the
    // pages plus the state slot stayed owned by a record nothing would ever adopt.
    if (handle && !from_tier && !checkpoint_matches(*handle, base)) { return std::nullopt; }
    SourceCandidate candidate{.checkpoint    = handle,
                              .reused_tokens = handle ? checkpoint(*handle).frontier : 0};
    const auto& prompt       = *base.impl_->prompt;
    candidate.remaining_work = runtime::make_prefill_work(
        candidate.reused_tokens, base.summary().prompt_tokens - candidate.reused_tokens, 0, 0,
        prefill_chunk);
    for (const auto& item : prompt.vision_items) {
        if (std::any_of(item.token_spans.begin(), item.token_spans.end(), [&](const auto& span) {
                return span.begin + span.count > candidate.reused_tokens;
            })) {
            ++candidate.remaining_work.vision_items;
            candidate.remaining_work.vision_patches += item.patch_count;
        }
    }
    if (!handle) { return candidate; }
    const auto& record       = checkpoint(*handle);
    // port (sm_89): a checkpoint that came from the tier is always safe to consume. Its authoritative
    // copy is on disk, so keeping an in-process one only costs a state slot the request then cannot
    // reserve - measured as "cannot reserve its first legal unit: state=0" on a 1-slot store, where
    // the import had taken the only slot to hold a copy of what the disk already has.
    const bool take_source   = consume_source || from_tier;
    candidate.consume_source = take_source;
    candidate.take_private   = take_source;
    for (const auto point : retired_points) {
        if (point == handle || !can_release_checkpoint(point)) { return std::nullopt; }
        if (std::find(candidate.retired_points.begin(), candidate.retired_points.end(), point) ==
            candidate.retired_points.end()) {
            candidate.retired_points.push_back(point);
        }
    }
    const auto retires = [&](CheckpointHandle point) {
        return std::find(candidate.retired_points.begin(), candidate.retired_points.end(), point) !=
               candidate.retired_points.end();
    };
    auto state_references = state_store->checkpoint_references(record.state);
    for (const auto point : candidate.retired_points) {
        if (checkpoint(point).state == record.state) { --state_references; }
    }
    auto occupied_states = state_store->device_occupied();
    for (std::size_t i = 0; i < candidate.retired_points.size(); ++i) {
        const auto state = checkpoint(candidate.retired_points[i]).state;
        if (std::any_of(candidate.retired_points.begin(), candidate.retired_points.begin() + i,
                        [&](auto point) { return checkpoint(point).state == state; })) {
            continue;
        }
        const auto references = static_cast<std::uint32_t>(
            std::count_if(candidate.retired_points.begin(), candidate.retired_points.end(),
                          [&](auto point) { return checkpoint(point).state == state; }));
        if (references == state_store->checkpoint_references(state) &&
            state_store->device_resident(state) &&
            state_store->can_release_after_checkpoint_references(state, references)) {
            --occupied_states;
        }
    }
    const auto& rewrite = prompt.identity.rewrite_checkpoint;
    const auto desired  = rewrite ? rewrite->recovery_frontier : base.summary().prompt_tokens;
    for (const auto point : private_points) {
        if (!valid_checkpoint(point) || retires(point) || !checkpoint_matches(point, base)) {
            continue;
        }
        const auto& kept = checkpoint(point);
        if (kept.frontier > record.frontier) { continue; }
        const bool input =
            kept.role == runtime::CheckpointRole::InputReplay &&
            (kept.frontier == desired || (point == *handle && kept.frontier >= desired));
        if (input || kept.role == runtime::CheckpointRole::LongAnchor) {
            candidate.private_points.push_back(point);
        }
    }
    const bool keep_source =
        std::find(candidate.private_points.begin(), candidate.private_points.end(), *handle) !=
        candidate.private_points.end();
    // port (sm_89): candidate.consume_source, not the parameter. The parameter says what the CALLER
    // wants; `from_tier` says this record was imported a few lines above and its disk copy makes
    // consuming it free. Reading the parameter here made every one of the five conditions print as
    // satisfied while move_state stayed 0 - the conditions were being evaluated against a flag the
    // code had already decided to override.
    const bool consuming = candidate.consume_source;
    candidate.move_state = consuming && !keep_source && checkpoints[handle->index].pins == 0 &&
                           state_references == 1 && state_store->source_pins(record.state) == 0;
    candidate.split_state =
        !candidate.move_state && occupied_states == state_store->device_capacity() &&
        state_store->device_resident(record.state) && state_store->host_resident(record.state) &&
        state_store->source_pins(record.state) == 0;
    if (!candidate.move_state && !candidate.split_state &&
        occupied_states == state_store->device_capacity() &&
        state_store->device_resident(record.state) && state_store->source_pins(record.state) == 0 &&
        checkpoints[handle->index].pins == 0) {
        if (host_context_arena &&
            host_context_arena->can_allocate(state_images->host_layout().image_bytes)) {
            candidate.backup_state = candidate.split_state = true;
            candidate.transfers.push_back(state_transfer_requirement(
                state_images->host_layout(), runtime::ContextTransferDirection::DeviceToHost));
        } else if (consuming && state_references == 1) {
            // One physical slot cannot retain its immutable recovery image and a writer.
            // Consume this optional point while preserving its already computed execution input.
            candidate.move_state = true;
        }
    }
    const auto movable = [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                             KVAddressSpaceHandle address, std::uint32_t frontier) {
        if (addresses.active(address) || checkpoints[handle->index].pins) { return false; }
        if (!addresses.can_destructive_truncate_inactive(address, frontier, true)) { return false; }
        if (frontier % kPagedKVPageSize) {
            const auto tail = addresses.logical_page(address, kv_pages_for_frontier(frontier) - 1U);
            if (pages.address_references(tail) != 1 || pages.source_pins(tail)) { return false; }
        }
        return true;
    };
    const auto backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                                      ? record.frontier - 1U
                                      : record.backend_frontier;
    candidate.move_history =
        consuming &&
        movable(*text_kv_addresses, *text_kv_pages, record.kv->text, record.frontier) &&
        (!record.kv->backend ||
         movable(*backend_kv_addresses, *backend_kv_pages, *record.kv->backend, backend_frontier));
    if (candidate.move_history) {
        for (std::uint32_t index = 0; index < checkpoints.size(); ++index) {
            const auto& slot = checkpoints[index];
            if (!slot.value || index == handle->index || slot.value->kv != record.kv ||
                retires({this, index, slot.generation})) {
                continue;
            }
            if (slot.value->frontier > record.frontier ||
                slot.value->backend_frontier > backend_frontier) {
                candidate.move_history = false;
                break;
            }
        }
    }
    if (!state_store->device_resident(record.state)) {
        candidate.transfers.push_back(state_transfer_requirement(
            state_images->host_layout(), runtime::ContextTransferDirection::HostToDevice));
    } else if (!candidate.move_state && !candidate.split_state &&
               is_masked_draft_backend(speculative_backend)) {
        candidate.transfers.push_back(state_transfer_requirement(
            state_images->host_layout(), runtime::ContextTransferDirection::DeviceToDevice, true));
    }
    const auto scan = [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                          KVAddressSpaceHandle address, std::uint32_t frontier,
                          runtime::ContextResourceClass resource) {
        const auto count      = kv_pages_for_frontier(frontier);
        const auto& physical  = pages.physical_pool();
        std::uint32_t missing = 0, run_pages = 0;
        TransferWork restore_work;
        const auto finish_run = [&] {
            const auto work = physical.host_transfer_run_work(run_pages);
            restore_work.payload_bytes += work.payload_bytes;
            restore_work.copy_operations += work.copy_operations;
            run_pages = 0;
        };
        std::optional<HostKVPageReplica> previous;
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto page = addresses.logical_page(address, index);
            if (pages.device_resident(page)) { continue; }
            const auto replica = pages.host_replica(page);
            if (previous && (previous->extent != replica.extent ||
                             previous->page_offset + 1 != replica.page_offset)) {
                finish_run();
            }
            previous = replica;
            ++run_pages;
            ++missing;
        }
        if (missing) {
            finish_run();
            candidate.transfers.push_back(kv_transfer_requirement(
                resource, runtime::ContextTransferDirection::HostToDevice, missing, restore_work));
        }
        if (!candidate.move_history && frontier % kPagedKVPageSize) {
            candidate.transfers.push_back(
                kv_transfer_requirement(resource, runtime::ContextTransferDirection::DeviceToDevice,
                                        1, physical.device_copy_work(1)));
        }
    };
    scan(*text_kv_addresses, *text_kv_pages, record.kv->text, record.frontier,
         runtime::ContextResourceClass::MainKV);
    if (record.kv->backend) {
        const auto frontier = speculative_backend == SpeculativeBackend::Mtp && record.frontier
                                  ? record.frontier - 1U
                                  : record.backend_frontier;
        scan(*backend_kv_addresses, *backend_kv_pages, *record.kv->backend, frontier,
             runtime::ContextResourceClass::BackendKV);
    }
    if (from_tier) {
        std::fprintf(stderr,
                     "[kvt] candidate consume=%d move_state=%d split=%d move_history=%d "
                     "transfers=%zu reused=%u | keep_source=%d pins=%u state_refs=%u "
                     "state_source_pins=%d\n",
                     candidate.consume_source ? 1 : 0, candidate.move_state ? 1 : 0,
                     candidate.split_state ? 1 : 0, candidate.move_history ? 1 : 0,
                     candidate.transfers.size(), candidate.reused_tokens, keep_source ? 1 : 0,
                     checkpoints[handle->index].pins, state_references,
                     state_store->source_pins(record.state));
        std::fflush(stderr);
    }
    return candidate;
}
} // namespace ninfer::models::qwen3_5::detail
