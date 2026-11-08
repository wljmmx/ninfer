#pragma once

#include "models/qwen4_exp/ple/prepare.h"

#include <exception>

namespace ninfer::models::qwen4_exp::ple_detail {

// Single-threaded row preparation mechanism. Each owner borrows a contiguous BF16 output;
// destination row i occupies [160*i,160*(i+1)). Each destination is requested at most once.
// Cancel suppresses output writes; release additionally requires all its reads to retire.
// Only wake() is cross-thread. Capacity includes retained rows of all unreleased owners.
class RowReader {
public:
    RowReader(const PleTableSource&, std::uint32_t owners, std::uint32_t rows_per_owner,
              std::uint32_t io_depth, std::size_t cache_bytes);
    ~RowReader();
    RowReader(const RowReader&)            = delete;
    RowReader& operator=(const RowReader&) = delete;

    void begin(std::uint32_t owner, std::span<std::uint16_t> output, bool required);
    void request(std::uint32_t owner, std::uint32_t destination, std::uint32_t row);
    void cancel(std::uint32_t owner, std::uint32_t begin, std::uint32_t end);
    void prioritize(std::uint32_t owner);
    void release(std::uint32_t owner);
    void pump();
    [[nodiscard]] bool runnable() const noexcept;
    void wait();
    void wake();

    [[nodiscard]] std::size_t pending(std::uint32_t owner) const;
    [[nodiscard]] bool drained(std::uint32_t owner) const;
    [[nodiscard]] std::exception_ptr error(std::uint32_t owner) const;
    [[nodiscard]] PleReadStats stats() const noexcept;
    [[nodiscard]] std::size_t memory_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp::ple_detail
