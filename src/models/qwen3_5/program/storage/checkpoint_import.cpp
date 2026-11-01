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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

// A cache and the third tier store the image as one blob (neither has a reason to shard it), while
// the import path joins self-describing shards and refuses anything without a header. Splitting here
// keeps that requirement local: the caller hands over what it has, and the format it has to be in
// stays a property of this path rather than something every producer must know. A shard that already
// carries a header also arrives as one element; split() refusing it is the signal to pass it through
// untouched rather than a failure.
std::vector<std::string> with_shard_headers(
    const ninfer::core::fingerprint::CheckpointFingerprint& want,
    const std::vector<std::string>& shards) {
    std::vector<std::string> split = shards;
    if (split.size() == 1 && !split[0].empty()) {
        ninfer::core::state_shards::RegionSizes sizes;
        sizes.conv_bytes  = want.state_conv_bytes;
        sizes.layer_bytes = want.state_layer_bytes;
        sizes.layers      = want.state_layers;
        const std::uint64_t covered = sizes.conv_bytes + sizes.layer_bytes * sizes.layers;
        sizes.tail_bytes = want.state_image_bytes > covered ? want.state_image_bytes - covered : 0;
        const auto parts = ninfer::core::state_shards::split(split[0], sizes, 6ull * 1024 * 1024);
        if (parts.ok) { split = parts.shards; }
    }
    return split;
}

} // namespace

std::optional<TierSource> ProgramImpl::import_checkpoint_source_current_layout(
    const CheckpointExport& kv, const std::vector<std::string>& state_shards, std::string* error,
    bool pin_pages) {
    ninfer::core::fingerprint::CheckpointFingerprint want;
    host_state_fingerprint(want);
    std::string local;
    auto source = import_checkpoint_source(kv, with_shard_headers(want, state_shards), want, local,
                                           pin_pages);
    if (!source && error != nullptr) { *error = std::move(local); }
    return source;
}

std::optional<CheckpointHandle> ProgramImpl::import_checkpoint_current_layout(
    const CheckpointExport& kv, const std::vector<std::string>& state_shards,
    std::shared_ptr<const PreparedCaptureBacking> identity, const PrefixShortlistKey* key,
    std::string* error, bool pin_pages) {
    std::string why;
    auto source = import_checkpoint_source_current_layout(kv, state_shards, &why, pin_pages);
    if (!source) {
        if (error != nullptr) { *error = std::move(why); }
        return std::nullopt;
    }
    return adopt_checkpoint_source(std::move(*source), std::move(identity), key, why);
}

std::optional<TierSource> ProgramImpl::import_checkpoint_source(
    const CheckpointExport& kv, const std::vector<std::string>& state_shards,
    const ninfer::core::fingerprint::CheckpointFingerprint& want, std::string& error,
    bool pin_pages) {
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

    // Coverage and pages have to describe the same amount of text. A record whose frontier says 384
    // while its payload holds 8 pages would commit an address whose pages and coverage disagree --
    // that is wrong KV, not a miss, and nothing downstream would notice.
    //
    // The rule is "exactly enough pages for the coverage", not "pages x tokens == coverage": the
    // engine's own export is routinely NOT page aligned (a capture at 467 tokens holds 8 pages, the
    // last one partly filled), so demanding an exact product refuses records this engine produced.
    const auto geometry = checkpoint_geometry();
    const auto page_tokens = static_cast<std::uint64_t>(geometry.page_tokens);
    if (page_tokens > 0) {
        const std::uint64_t needed =
            (static_cast<std::uint64_t>(kv.frontier) + page_tokens - 1) / page_tokens;
        if (needed != kv.page_count) {
            if (error.empty()) {
                error = "frontier " + std::to_string(kv.frontier) + " needs " +
                        std::to_string(needed) + " page(s) of " + std::to_string(page_tokens) +
                        " token(s); the payload holds " + std::to_string(kv.page_count);
            }
            return std::nullopt;
        }
    }

    // 1. State first: it validates the fingerprint and refuses before anything is allocated.
    const auto state = import_state(state_shards, want, error);
    if (!state) { return std::nullopt; }
    // adopt_host_replica already left it CheckpointImmutable, and freeze() cannot help here:
    // it demands a Device slot, which a restored image does not have.
    //
    // No checkpoint reference is taken here, deliberately. A reference exists so a RECORD's image
    // survives as long as the record does; an import has no record yet, and the image is held by
    // this handle until someone either adopts it into one or hands it to a request.

    // 2. Device first, because a logical page is born in VRAM: can_pin_source() and
    //    can_pin_active_source() both require a Device replica, so a page without one can never be
    //    given a Host extent. Restoring straight into Host memory is not a path this store has.
    auto device_pages = text_kv_pages->physical_pool().reserve(kv.page_count);
    if (!device_pages) {
        // The state image was adopted first, so it is the one thing to give back here. Releasing it
        // returns the Host slot; leaving it behind would make the state pool shrink by one image per
        // failed import.
        (void)state_store->release(*state);
        error = "no free Device pages for the restored checkpoint";
        return std::nullopt;
    }
    std::vector<LogicalKVPageHandle> pages(kv.page_count);
    text_kv_pages->materialize(*device_pages, std::span<LogicalKVPageHandle>(pages));

    // 3. Write the bytes into a Host extent (now legal -- the pages have Device replicas), then
    //    copy them down. The Host copy stays as the demoted tier; without this the Device pages
    //    would be published but empty, and readers would take them as authoritative.
    // Declared out here so every later failure can give the extent back. Publishing is the point of
    // no return: after it the pages hold real bytes and both the extent and the page descriptors are
    // ours to release.
    std::optional<HostKVExtentCapability> published;
    {
        auto reservation = host_kv_extents->prepare(*text_kv_pages, pages);
        if (!reservation) {
            // A refusal here used to be one string, which cannot distinguish "the arena is full" from
            // "the policy said no". Both numbers are cheap and they are the reason this path fails.
            std::fprintf(stderr,
                         "[kvt] no Host extent: %u page(s) of %zu B, extents %u/%u occupied, "
                         "state image %llu B\n",
                         kv.page_count, host_kv_extents->page_layout(*text_kv_pages).page_stride,
                         host_kv_extents->occupied(), host_kv_extents->capacity(),
                         static_cast<unsigned long long>(want.state_image_bytes));
            std::fflush(stderr);
            // Nothing is published yet, so the pages are the only thing to give back. Releasing them
            // returns their Device leases too; without this a run of failed imports would drain the
            // page pool and the symptom would look like "allocation stopped working" much later.
            for (const LogicalKVPageHandle page : pages) { (void)text_kv_pages->release(page); }
            (void)state_store->release(*state);
            error = "no Host extent for the restored pages";
            return std::nullopt;
        }
        const auto view = host_kv_extents->writable_view(*reservation);
        const std::size_t expected = kv.page_bytes * static_cast<std::size_t>(kv.page_count);
        const std::size_t available =
            static_cast<std::size_t>(view.page_count()) * view.layout().page_stride;
        if (view.data() == nullptr || available < expected) {
            host_kv_extents->abort(*reservation);
            for (const LogicalKVPageHandle page : pages) { (void)text_kv_pages->release(page); }
            (void)state_store->release(*state);
            error = "Host extent is smaller than the restored payload";
            return std::nullopt;
        }
        std::memcpy(view.data(), kv.kv_bytes.data(), expected);
        // No diagnostics here. There was a pair of fprintf calls printing the first byte before and
        // after publish while the byte-for-byte mismatch was being chased; they belong to that
        // investigation, not to every restore, and they printed payload bytes into the server log.
        published = host_kv_extents->publish(std::move(*reservation));

        // The Device replicas reserved in step 2 exist but hold nothing: every byte went into the Host
        // extent. Two consequences, and they have to be fixed together:
        //
        //   * a request that extends this prefix forks it Device -> Device, so an empty replica keeps
        //     the copy and hands zeros forward, with nothing downstream noticing;
        //   * prepare_prefix_fork() also refuses the page, because materialize() leaves it a writer
        //     (writer_references == 1) and can_pin_source() requires 0.
        //
        // So the bytes are copied down here and the pages are then marked read-only. This is NOT the
        // eager copy that was tried before and left the Host extent reading as zeros: that one went
        // through the writable view, this one reads the published extent and only writes to Device.
        bool rollback_pending = false;
        for (const LogicalKVPageHandle page : pages) {
            const auto page_host = host_kv_extents->page_view(*text_kv_pages, page);
            if (!page_host || !page_host->valid()) {
                rollback_pending = true;
                break;
            }
            const std::array<DeviceKVPageHandle, 1> destination{text_kv_pages->physical(page)};
            (void)text_kv_pages->physical_pool().copy_from_host(*page_host, destination,
                                                                device.transfer_stream);
        }
        if (!rollback_pending &&
            cudaStreamSynchronize(device.transfer_stream) != cudaSuccess) {
            rollback_pending = true;
        }
        if (rollback_pending) {
            if (published) { (void)host_kv_extents->release(*published); }
            for (const LogicalKVPageHandle page : pages) { (void)text_kv_pages->release(page); }
            (void)state_store->release(*state);
            error = "could not fill the Device copies of the restored pages";
            return std::nullopt;
        }
    }
    // From here on, a failure has to give back the published extent and the pages behind it. The
    // pages come with their Device leases, so one release per page covers both tiers.
    const auto rollback = [&]() {
        if (published) { (void)host_kv_extents->release(*published); }
        for (const LogicalKVPageHandle page : pages) { (void)text_kv_pages->release(page); }
        // The state image is the other half of the source and is still ours at this point.
        (void)state_store->release(*state);
    };

    // 4. Map them into a fresh inactive address.
    std::optional<KVAddressSpaceHandle> address = text_kv_addresses->create_inactive();
    if (!address) {
        rollback();
        error = "no free KV address space";
        return std::nullopt;
    }
    if (!text_kv_addresses->adopt_pages(*address, pages)) {
        (void)text_kv_addresses->release(*address);
        rollback();
        error = "could not map the restored pages";
        return std::nullopt;
    }
    // Commit the frontier and protect it. commit_frontier() is for active addresses only, so the
    // inactive twin does the same work here: it also fills each page's committed_columns, which
    // materialize() left at 0 -- without that the pages look empty to anything reading coverage, and
    // the protection below would refuse a frontier above zero.
    text_kv_addresses->commit_inactive_frontier(*address, kv.frontier);
    // port (sm_89): a record restored from the third tier is NOT protected here.
    //
    // The requirement exists so a checkpoint a request is still using cannot be evicted. For an
    // import it does the opposite of what the caller needs: it pins the pages, and the adopt path
    // moves or copies exactly those pages, so pinning turns "take this over" into "copy these and
    // find room", which the capacity account then refuses. The authoritative copy is on disk, so
    // there is nothing here worth protecting - if this one is evicted, load it again.
    if (pin_pages) { text_kv_addresses->set_checkpoint_requirement(*address, kv.frontier); }

    // Only now are the pages made read-only, and the order is not cosmetic: commit_coverage() above
    // requires writer_references == 1, while can_pin_source() -- what a later fork needs -- requires
    // 0. Clearing the flag before the commit made the import itself throw "committed coverage is not
    // monotonic"; clearing it after leaves these pages in the state a captured-then-demoted record
    // has, which is what the fork path expects.

    // port (sm_89): give back the creator's reference.
    //
    // materialize() sets references = 1 so the code that made the page can hand it on, and
    // adopt_pages() retains a second for the address. A page that is meant to live in exactly one
    // address has to end at 1: at 2, movable() refuses it, and every restore is priced as a copy
    // instead of a hand-over - which the capacity account then declines, so the request fails with
    // every resource reading zero. Measured before this: addr_refs=2 where 1 was required.
    //
    // Only on the unpinned path. A record that keeps its checkpoint requirement is one a request is
    // still using, and source_pins == 1 with references == 1 makes the release refuse by design.
    if (!pin_pages) {
        for (const LogicalKVPageHandle page : pages) {
            (void)text_kv_pages->release_reference(page, false);
        }
    }

    for (const LogicalKVPageHandle page : pages) {
        if (text_kv_pages->writer_references(page) != 0) {
            text_kv_pages->set_writer(page, false);
        }
    }

    if (!pin_pages) {
        // port (sm_89): the readings the adopt path decides on, taken at the moment the source is
        // complete. Without them "the import cannot be taken over" is a guess about three counters.
        const auto tail_index = text_kv_addresses->page_count(*address) - 1U;
        const auto tail       = text_kv_addresses->logical_page(*address, tail_index);
        std::fprintf(stderr,
                     "[kvt] import refs: addr_refs=%u src_pins=%d truncate=%d state_refs=%u "
                     "state_pins=%d tail_page=%u/%u\n",
                     text_kv_pages->address_references(tail), text_kv_pages->source_pins(tail) ? 1 : 0,
                     text_kv_addresses->can_destructive_truncate_inactive(*address, kv.frontier, true)
                         ? 1
                         : 0,
                     state_store->checkpoint_references(*state),
                     state_store->source_pins(*state) ? 1 : 0, tail_index,
                     text_kv_addresses->page_count(*address));
        std::fflush(stderr);
    }

    // Both halves leave in one struct. Nothing is attached to a record, nothing is pinned, and no
    // handle is retained twice: the caller either adopts this into a record (which takes the
    // reference) or hands it to a request (which takes the ownership).
    TierSource source;
    source.text                   = *address;
    source.frontier               = kv.frontier;
    source.state                  = *state;
    source.materialised_host_copy = true;
    return source;
}

std::optional<CheckpointHandle> ProgramImpl::adopt_checkpoint_source(
    TierSource&& source, std::shared_ptr<const PreparedCaptureBacking> identity,
    const PrefixShortlistKey* key, std::string& error) {
    // 5. The record. Everything below has both halves in hand, so the only failure left is a full
    //    slot pool -- and one call gives them back rather than each branch repeating the unwind.
    const auto handle = reserve_checkpoint();
    if (!handle) {
        discard_tier_source(source);
        error = "no free checkpoint slot";
        return std::nullopt;
    }
    // A record owns its image through a checkpoint reference; the import carried none because no
    // record existed. Taken here, after the slot exists, so a refused adoption never leaves one.
    state_store->retain_checkpoint_reference(*source.state);
    auto history  = std::make_shared<KVHistory>();
    history->text = *source.text;
    // The record's address goes back when the last holder of the history does. Without this an
    // imported record releases its state image but leaves its address run allocated, which is a
    // slow leak that reads as "the page pool shrank for no reason".
    history->owner = this;
    if (source.backend) { history->backend = *source.backend; }
    // Identity and key come from the request that is restoring this prefix. Without them the record
    // is readable but checkpoint_matches() will never accept it -- the comparison is on token
    // content, not on bytes, so nothing about the payload can substitute for them.
    checkpoints[handle->index].value =
        CheckpointState{.kv       = std::move(history),
                        .state    = *source.state,
                        .identity = std::move(identity),
                        .key      = key != nullptr ? *key : PrefixShortlistKey{},
                        .frontier = source.frontier};
    checkpoints[handle->index].reserved = false;
    source.text.reset();
    source.state.reset();
    source.materialised_host_copy = false;
    return handle;
}

void ProgramImpl::discard_tier_source(TierSource& source) noexcept {
    // Give back exactly what the import took, in the reverse of the order it took it, and only what
    // is still here: each field is cleared as it is released, so a second call is a no-op and the
    // transaction abort and the pre-transaction guard cannot both release the same page run.
    try {
        if (source.text) {
            // A pinned run protects a checkpoint a request may still be using. A discard is the one
            // caller that knows nothing else will, so the pin goes first -- otherwise the release
            // below refuses and this terminates.
            text_kv_addresses->set_checkpoint_requirement(*source.text, 0);
            if (!text_kv_addresses->release_after_deactivate(*source.text)) { std::terminate(); }
            source.text.reset();
        }
        if (source.backend) {
            backend_kv_addresses->set_checkpoint_requirement(*source.backend, 0);
            if (!backend_kv_addresses->release_after_deactivate(*source.backend)) {
                std::terminate();
            }
            source.backend.reset();
        }
        if (source.materialised_host_copy && host_kv_extents) {
            // The pages are unreferenced now, and a Host extent only goes back when every page in it
            // does. A source that never became a record has no history to run this on its way out.
            (void)host_kv_extents->release_unreferenced();
            source.materialised_host_copy = false;
        }
        if (source.state) {
            if (!state_store->release(*source.state)) { std::terminate(); }
            source.state.reset();
        }
    } catch (...) { std::terminate(); }
}

std::optional<CheckpointHandle> ProgramImpl::import_checkpoint(
    const CheckpointExport& kv, const std::vector<std::string>& state_shards,
    const ninfer::core::fingerprint::CheckpointFingerprint& want, std::string& error,
    std::shared_ptr<const PreparedCaptureBacking> identity, const PrefixShortlistKey* key,
    bool pin_pages) {
    auto source = import_checkpoint_source(kv, state_shards, want, error, pin_pages);
    if (!source) { return std::nullopt; }
    return adopt_checkpoint_source(std::move(*source), std::move(identity), key, error);
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
    const auto started = std::chrono::steady_clock::now();
    const auto imported = import_checkpoint(*source, split.shards, print, why);
    const double import_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (imported) {
        std::fprintf(stderr, "[ckpt] import check: import took %.2f ms for %u pages + %zu B state\n",
                     import_ms, source->page_count, source_state->bytes.size());
    }
    if (!imported) {
        std::fprintf(stderr, "[ckpt] import check: REFUSED %s\n", why.c_str());
        return;
    }
    const auto again = export_checkpoint(*imported);
    if (!again) {
        // One of two things refuses the export; say which rather than guessing.
        const bool slot_valid = valid_checkpoint(*imported);
        std::uint32_t pages = 0;
        bool address_valid = false;
        if (slot_valid) {
            const CheckpointState& record = checkpoint(*imported);
            if (record.kv && text_kv_addresses->valid(record.kv->text)) {
                address_valid = true;
                pages = text_kv_addresses->page_count(record.kv->text);
            }
        }
        std::fprintf(stderr,
                     "[ckpt] import check: export refused (slot_valid=%d, address_valid=%d, "
                     "pages=%u)\n",
                     slot_valid ? 1 : 0, address_valid ? 1 : 0, pages);
        if (address_valid) {
            const CheckpointState& record = checkpoint(*imported);
            const std::size_t stride =
                host_kv_extents->page_layout(*text_kv_pages).page_stride;
            std::size_t dev = 0;
            std::size_t host = 0;
            std::size_t spilled = 0;
            for (std::uint32_t i = 0; i < pages; ++i) {
                const LogicalKVPageHandle page =
                    text_kv_addresses->logical_page(record.kv->text, i);
                if (text_kv_pages->device_resident(page)) { ++dev; }
                if (text_kv_pages->host_resident(page)) { ++host; }
                if (text_kv_pages->spilled(page)) { ++spilled; }
            }
            std::fprintf(stderr,
                         "[ckpt] import check: stride=%zu device=%zu host=%zu spilled=%zu of %u\n",
                         stride, dev, host, spilled, pages);
        }
    }
    // Page by page, then byte by byte: "the bytes differ" is not actionable, but "page 0 differs at
    // byte 0" points at the copy, and "page 3 differs at byte 0" points at ordering or offsets.
    bool same = again.has_value() && again->kv_bytes.size() == source->kv_bytes.size() &&
                again->frontier == source->frontier;
    if (same) {
        for (std::uint32_t page = 0; page < source->page_count; ++page) {
            const std::size_t base = static_cast<std::size_t>(page) * source->page_bytes;
            if (std::memcmp(again->kv_bytes.data() + base, source->kv_bytes.data() + base,
                            source->page_bytes) == 0) {
                continue;
            }
            std::size_t offset = 0;
            while (offset < source->page_bytes &&
                   again->kv_bytes[base + offset] == source->kv_bytes[base + offset]) {
                ++offset;
            }
            std::fprintf(stderr,
                         "[ckpt] import check: first difference page %u, byte %zu of %zu "
                         "(0x%02X vs 0x%02X)\n",
                         page, offset, source->page_bytes,
                         static_cast<unsigned>(again->kv_bytes[base + offset]),
                         static_cast<unsigned>(source->kv_bytes[base + offset]));
            same = false;
            break;
        }
    }
    std::fprintf(stderr, "[ckpt] import check: %s (%u pages, %zu B)\n",
                 same ? "KV round-trip OK" : "KV MISMATCH",
                 again ? again->page_count : 0u,
                 again ? again->kv_bytes.size() : 0);
}

// Forces the one failure that is easy to provoke and expensive to leak: the state image imports,
// then the Device reservation cannot be satisfied. A payload asking for the whole page pool is the
// cheapest way to guarantee that -- the pool always has something in use by the time a checkpoint
// exists.
void ProgramImpl::verify_import_rollback(CheckpointHandle handle) {
    const auto source = export_checkpoint(handle);
    const auto source_state = export_state(handle);
    if (!source || !source_state) {
        std::fprintf(stderr, "[ckpt] rollback check: nothing to export\n");
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
        std::fprintf(stderr, "[ckpt] rollback check: split failed: %s\n", split.error.c_str());
        return;
    }

    CheckpointExport oversize;
    oversize.frontier   = source->frontier;
    oversize.page_bytes = source->page_bytes;
    oversize.page_count = text_kv_pages->physical_pool().capacity_pages();
    oversize.kv_bytes.assign(oversize.page_bytes * oversize.page_count, std::byte{0});

    const std::uint32_t state_before = host_state_images ? host_state_images->occupied() : 0;
    const std::uint32_t pages_before = text_kv_pages->occupied();
    const std::uint32_t devices_before =
        text_kv_pages->physical_pool().capacity_pages() - pages_before;

    std::string why;
    const auto refused = import_checkpoint(oversize, split.shards, print, why);

    const std::uint32_t state_after = host_state_images ? host_state_images->occupied() : 0;
    const std::uint32_t pages_after = text_kv_pages->occupied();
    const bool clean = refused == std::nullopt && state_after == state_before &&
                       pages_after == pages_before;
    std::fprintf(stderr,
                 "[ckpt] rollback check: %s (refused=%d: %s; state %u->%u, pages %u->%u, "
                 "free device before=%u)\n",
                 clean ? "CLEAN" : "LEAKED", refused ? 0 : 1, why.c_str(), state_before, state_after,
                 pages_before, pages_after, devices_before);
}
} // namespace ninfer::models::qwen3_5::detail
