#pragma once

#include "core/cpu/system.h"
#include "core/memory.h"
#include "models/qwen4_exp/ple/table.h"

#include <exception>
#include <optional>

namespace ninfer::models::qwen4_exp {

struct PleReadStats {
    std::uint64_t rows         = 0; // Output row demands, including duplicates.
    std::uint64_t cache_hits   = 0; // Unique rows fetched from the persistent raw cache.
    std::uint64_t shared_rows  = 0; // Demands reusing a retained row of a live window.
    std::uint64_t shared_pages = 0; // Fragments joining an already queued/in-flight page.
    std::uint64_t reads        = 0;
    std::uint64_t read_bytes   = 0; // Actual returned bytes, including alignment overhead.
    // Where the preparation worker's wall time goes. The stage exists because the table is on
    // disk, so a window's cost has to be attributable before it can be argued about: the hash
    // (Host, per row), the blocking wait for a read to retire, and the decode that turns an
    // arrived row into BF16 words. `pages` counts the aligned reads issued, which is what the
    // read amplification of a two-plane row shows up in.
    std::uint64_t pages     = 0;
    std::uint64_t hash_ns   = 0;
    std::uint64_t wait_ns   = 0;
    std::uint64_t decode_ns = 0;
    // The most reads this reader has ever had in flight. A window whose wall time looks like one
    // read after another shows up here as a depth the queue never reached.
    std::uint64_t peak_inflight = 0;
};

struct PlePrepareOptions {
    std::uint32_t max_positions = 0; // Maximum T*B, not max(T)*max(B).
    std::uint32_t windows       = 2;
    std::uint32_t io_depth      = 128;
    std::size_t cache_bytes     = 64ULL * 1024 * 1024;
    CpuAffinity affinity;
};

struct PleTicket {
    std::uint32_t slot;
    std::uint64_t generation;
    bool operator==(const PleTicket&) const = default;
};

struct PleSequenceIdentity {
    std::uint64_t sequence = 0, epoch = 0, frontier = 0;
    std::array<std::int32_t, 2> raw_tail{};
};

struct PleInputRow {
    PleSequenceIdentity identity;
    std::span<const std::int32_t> ids;
};

struct PleAppend {
    std::uint32_t batch_row;
    std::span<const std::int32_t> ids;
};

enum class PlePrepareStatus { Preparing, HostReady, Cancelled, Failed };

struct PleWindowSnapshot {
    PlePrepareStatus status;
    bool drained;
    std::uint32_t tokens, batch;
    std::array<std::uint32_t, 8> valid_columns;
    std::array<PleSequenceIdentity, 8> identities;
    std::exception_ptr error;
};

struct PleHostEmbedding {
    ConstMemorySpan values; // BF16 [2560,T,B], dimension zero fastest.
    PleWindowSnapshot window;
};

// Program-owned Host preparation service with one I/O worker. It borrows the immutable table
// and disjoint CPU outputs through destruction; each output holds max_positions*2560 BF16 words.
// Control calls come from one Program thread. IDs/identities are copied before returning.
// The worker alone hashes, reads, decodes and publishes output; it never calls CUDA.
class PlePrepareService {
public:
    PlePrepareService(const PleTableSource&, std::int32_t eos, PlePrepareOptions,
                      std::span<const MemorySpan> outputs);
    ~PlePrepareService();
    PlePrepareService(const PlePrepareService&)            = delete;
    PlePrepareService& operator=(const PlePrepareService&) = delete;

    // B is rows.size(), in [1,8]. No free slot returns nullopt. Optional prefetch leaves
    // room for a required window; prioritize promotes it when it becomes the current input.
    [[nodiscard]] std::optional<PleTicket>
    prepare(std::uint32_t tokens, std::span<const PleInputRow> rows, bool required = true);
    void prioritize(PleTicket);
    void append(PleTicket, std::span<const PleAppend>);
    // Final lengths do not exceed known prefixes. Sealing truncates candidates for this
    // input window; it does not commit model state. Inactive E columns become exact +0.
    void seal(PleTicket, std::span<const std::uint32_t> valid_columns);
    void cancel(PleTicket);
    [[nodiscard]] PleWindowSnapshot poll(PleTicket) const;
    [[nodiscard]] PleHostEmbedding embedding(PleTicket) const;
    // Requires a terminal, Host-drained window and no external users of its output.
    // Invalidates the handle immediately; worker reclamation makes its slot available again.
    // Preparing must first be cancelled, including when all currently known rows are ready.
    void release(PleTicket);

    [[nodiscard]] PleReadStats stats() const;
    // Owned Host array capacities, excluding borrowed outputs and allocator/system overhead.
    [[nodiscard]] std::size_t memory_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
