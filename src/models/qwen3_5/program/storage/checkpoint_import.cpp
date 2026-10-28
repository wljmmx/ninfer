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

std::optional<CheckpointHandle> ProgramImpl::import_checkpoint_current_layout(
    const CheckpointExport& kv, const std::vector<std::string>& state_shards,
    std::shared_ptr<const PreparedCaptureBacking> identity, const PrefixShortlistKey* key,
    std::string* error) {
    ninfer::core::fingerprint::CheckpointFingerprint want;
    host_state_fingerprint(want);

    // Coverage and pages have to describe the same amount of text. A record whose frontier says 384
    // while its payload holds 8 pages would commit an address whose pages and coverage disagree --
    // that is wrong KV, not a miss, and nothing downstream would notice.
    const auto geometry = checkpoint_geometry();
    if (geometry.page_tokens > 0 &&
        static_cast<std::uint64_t>(kv.page_count) * geometry.page_tokens != kv.frontier) {
        if (error != nullptr) {
            *error = "frontier " + std::to_string(kv.frontier) + " does not match " +
                     std::to_string(kv.page_count) + " page(s) x " +
                     std::to_string(geometry.page_tokens);
        }
        return std::nullopt;
    }

    // A cache stores the image as one blob (it has no reason to shard it), while the import path
    // joins self-describing shards and refuses anything without a header. Splitting here keeps that
    // requirement local: the caller hands over what it has, and the format it has to be in stays a
    // property of this path rather than something every producer must know.
    std::vector<std::string> shards = state_shards;
    if (shards.size() == 1 && !shards[0].empty()) {
        ninfer::core::state_shards::RegionSizes sizes;
        sizes.conv_bytes  = want.state_conv_bytes;
        sizes.layer_bytes = want.state_layer_bytes;
        sizes.layers      = want.state_layers;
        const std::uint64_t covered = sizes.conv_bytes + sizes.layer_bytes * sizes.layers;
        sizes.tail_bytes =
            want.state_image_bytes > covered ? want.state_image_bytes - covered : 0;
        const auto split =
            ninfer::core::state_shards::split(shards[0], sizes, 6ull * 1024 * 1024);
        // A shard that already carries a header also arrives as one element; split() refusing it is
        // the signal to pass it through untouched rather than a failure.
        if (split.ok) { shards = split.shards; }
    }

    std::string local;
    auto handle = import_checkpoint(kv, shards, want, local, std::move(identity), key);
    if (!handle && error != nullptr) { *error = std::move(local); }
    return handle;
}

std::optional<CheckpointHandle> ProgramImpl::import_checkpoint(
    const CheckpointExport& kv, const std::vector<std::string>& state_shards,
    const ninfer::core::fingerprint::CheckpointFingerprint& want, std::string& error,
    std::shared_ptr<const PreparedCaptureBacking> identity, const PrefixShortlistKey* key) {
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
        // The state image was adopted first, so it is the one thing to give back here. Dropping its
        // checkpoint reference and releasing it returns the Host slot; leaving it behind would make
        // the state pool shrink by one image per failed import.
        state_store->release_checkpoint_reference(*state);
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
            // Nothing is published yet, so the pages are the only thing to give back. Releasing them
            // returns their Device leases too; without this a run of failed imports would drain the
            // page pool and the symptom would look like "allocation stopped working" much later.
            for (const LogicalKVPageHandle page : pages) { (void)text_kv_pages->release(page); }
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
            error = "Host extent is smaller than the restored payload";
            return std::nullopt;
        }
        std::memcpy(view.data(), kv.kv_bytes.data(), expected);
        // No diagnostics here. There was a pair of fprintf calls printing the first byte before and
        // after publish while the byte-for-byte mismatch was being chased; they belong to that
        // investigation, not to every restore, and they printed payload bytes into the server log.
        published = host_kv_extents->publish(std::move(*reservation));

        // The Device copy is deliberately not made here. A restored checkpoint is a Host-only
        // artifact, exactly like one that was captured and demoted, and restore_addressed_host_pages()
        // promotes what it needs at use time. Copying down eagerly duplicated that work and, measured,
        // left the Host extent reading back as zeros afterwards.
    }
    // From here on, a failure has to give back the published extent and the pages behind it. The
    // pages come with their Device leases, so one release per page covers both tiers.
    const auto rollback = [&]() {
        if (published) { (void)host_kv_extents->release(*published); }
        for (const LogicalKVPageHandle page : pages) { (void)text_kv_pages->release(page); }
        // The state image is the other half of the record and is still ours at this point.
        state_store->release_checkpoint_reference(*state);
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
    text_kv_addresses->set_checkpoint_requirement(*address, kv.frontier);

    // 5. The record.
    const auto handle = reserve_checkpoint();
    if (!handle) {
        (void)text_kv_addresses->release(*address);
        rollback();
        error = "no free checkpoint slot";
        return std::nullopt;
    }
    auto history  = std::make_shared<KVHistory>();
    history->text = *address;
    // Identity and key come from the request that is restoring this prefix. Without them the record
    // is readable but checkpoint_matches() will never accept it -- the comparison is on token
    // content, not on bytes, so nothing about the payload can substitute for them.
    checkpoints[handle->index].value =
        CheckpointState{.kv       = std::move(history),
                        .state    = *state,
                        .identity = std::move(identity),
                        .key      = key != nullptr ? *key : PrefixShortlistKey{},
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
