#pragma once

// Q54: what the third tier hands to a request, and what it deliberately is not.
//
// A checkpoint is a REFERENCE: it names an address space some sequence already owns, and the whole
// capture path is built on that - identity, pins, reference counts, a record that outlives the
// request that made it. An import from disk is an OWNERSHIP TRANSFER: the bytes arrive, nothing in
// this process owns them yet, and the request that asked for them becomes the owner.
//
// Trying to express the second as the first is what produced a chain of failures: a record with no
// identity that checkpoint_matches() must refuse, pages pinned by a requirement that exists to
// protect a request still using them, a page reference count that has to be 1 for a hand-over but 2
// after materialize+adopt, and two inspect_source() passes that must agree about a source whose
// identity was established by a lookup rather than by a ledger.
//
// So the import gets its own type and its own entry into binding. Nothing here is a checkpoint.

#include "core/kv_tier.h"
#include "models/qwen3_5/program/storage/kv_address_space.h"

#include <cstdint>
#include <memory>

namespace ninfer::models::qwen3_5::detail {

// Owns everything an import brought in, and owns it alone. Dereference order matters: the address
// space is released before the page store it borrows, so the fields are declared in reverse of the
// release order a destructor would need if the store were owned here (it is not - both pointers are
// owned by the program).
struct TierSource {
    // The KV run, as a brand-new inactive address space holding exactly the imported pages.
    std::optional<KVAddressSpaceHandle> text;
    std::optional<KVAddressSpaceHandle> backend;
    std::uint32_t frontier       = 0;
    std::uint32_t backend_frontier = 0;

    // The state image, already published as a host replica. It is not attached to any checkpoint
    // record: it belongs to the request that adopts this source, and to nothing else.
    std::optional<StateImageHandle> state;

    // Who must give this back if the request is never admitted. Kept explicit rather than implied by
    // a handle, because there is no record to look it up from - that is the whole point.
    bool materialised_host_copy = false;
};

} // namespace ninfer::models::qwen3_5::detail
