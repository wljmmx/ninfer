#pragma once

// Unified traffic accounting for the GDN/ReplaySSM benchmarks (upstream 551ebef0, adapted).
//
// Every measured row reports the same three layers, in the same order:
//   logical  - unique boundary operands of the complete Op, workspace excluded;
//   state    - the recurrent-state subset of the logical operands;
//   records  - the raw key/value/gate and convolution records, also a logical subset.
// `modeled_request_bytes` is opt-in and only filled where the measured implementation has a
// real request model; unique logical bytes are never substituted for it.

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <optional>
#include <sstream>
#include <span>
#include <string>

namespace ninfer::bench::gdn {

struct Geometry {
    static constexpr std::uint64_t head_dim = 128;
    int qk_heads, value_heads, tokens, batch = 1;

    [[nodiscard]] std::uint64_t qk_bytes() const {
        return std::uint64_t(head_dim) * qk_heads * tokens * batch * sizeof(std::uint16_t);
    }
    [[nodiscard]] std::uint64_t value_bytes() const {
        return std::uint64_t(head_dim) * value_heads * tokens * batch * sizeof(std::uint16_t);
    }
    [[nodiscard]] std::uint64_t gate_bytes() const {
        return std::uint64_t(value_heads) * tokens * batch * sizeof(float);
    }
    [[nodiscard]] std::uint64_t state_bytes() const {
        return std::uint64_t(head_dim) * head_dim * value_heads * batch * sizeof(float);
    }
};

struct IoBytes {
    std::uint64_t read = 0, write = 0;
    [[nodiscard]] std::uint64_t total() const { return read + write; }
};

struct Traffic {
    IoBytes logical;
    IoBytes state;
    IoBytes records;
    IoBytes workspace_requests;
    std::optional<std::uint64_t> modeled_request_bytes;
    int active_rows = 0, valid_tokens = 0;

    static constexpr const char* columns =
        "active_rows,valid_tokens,logical_read_bytes,logical_write_bytes,logical_bytes,"
        "state_read_bytes,state_write_bytes,record_read_bytes,record_write_bytes,"
        "workspace_read_request_bytes,workspace_write_request_bytes,modeled_request_bytes,"
        "logical_GBps,modeled_request_GBps,memory_roofline_GBps,streaming_memory_reference_us";

    [[nodiscard]] std::string csv(double gpu_us, double memory_gbs) const {
        const auto rate = [gpu_us](std::uint64_t bytes) {
            return gpu_us > 0.0 ? bytes / (gpu_us * 1000.0) : 0.0;
        };
        std::ostringstream out;
        out << std::fixed << std::setprecision(6) << active_rows << ',' << valid_tokens << ','
            << logical.read << ',' << logical.write << ',' << logical.total() << ',' << state.read
            << ',' << state.write << ',' << records.read << ',' << records.write << ','
            << workspace_requests.read << ',' << workspace_requests.write << ',';
        if (modeled_request_bytes) out << *modeled_request_bytes;
        out << ',' << rate(logical.total()) << ',';
        if (modeled_request_bytes) out << rate(*modeled_request_bytes);
        out << ',' << memory_gbs << ',';
        if (memory_gbs > 0.0) out << logical.total() / (memory_gbs * 1000.0);
        return out.str();
    }
};

// Complete recurrence over one physical block: every reading of q/k/g plus the read-modify-write
// of the state and the written value block.
inline Traffic running_traffic(const Geometry& p) {
    Traffic result;
    result.state   = {p.state_bytes(), p.state_bytes()};
    result.logical = {2 * p.qk_bytes() + p.value_bytes() + 2 * p.gate_bytes() + result.state.read,
                      p.value_bytes() + result.state.write};
    result.active_rows  = p.batch;
    result.valid_tokens = p.tokens * p.batch;
    return result;
}

// Verify-side record: reads the initial state and the live prefix of q/k/g, writes the value
// block plus the compact records.
inline Traffic record_traffic(int qk_heads, int value_heads, int width,
                              std::span<const int> valid, bool has_valid_tensor) {
    const int batch  = static_cast<int>(valid.size());
    const int tokens = std::accumulate(valid.begin(), valid.end(), 0);
    const Geometry live{qk_heads, value_heads, tokens};
    const Geometry physical{qk_heads, value_heads, width, batch};
    Traffic result;
    result.state.read    = physical.state_bytes();
    result.records.write = live.qk_bytes() + live.value_bytes() + 2 * live.gate_bytes();
    result.logical       = {2 * live.qk_bytes() + live.value_bytes() + 2 * live.gate_bytes() +
                                result.state.read + sizeof(std::int32_t) * batch * (1 + has_valid_tensor),
                            physical.value_bytes() + result.records.write};
    result.active_rows  = batch;
    result.valid_tokens = tokens;
    return result;
}

// Commit-side fold: each row reads its state and record prefix, replays it and writes the state
// and the three-slot convolution history window.
inline Traffic fold_traffic(int layers, int qk_heads, int value_heads, int conv_channels,
                            std::span<const int> commits) {
    Traffic result;
    for (int count : commits) {
        if (!count) continue;
        const Geometry live{qk_heads, value_heads, count};
        const auto state        = live.state_bytes() * layers;
        const auto conv_column  = 2ULL * conv_channels * layers;
        const auto conv_records = conv_column * std::min(3, count);
        const auto old_history  = conv_column * std::max(0, 3 - count);
        const auto record =
            (live.qk_bytes() + live.value_bytes() + 2 * live.gate_bytes()) * layers + conv_records;
        result.state.read += state;
        result.state.write += state;
        result.records.read += record;
        result.logical.read += state + record + old_history;
        result.logical.write += state + 3 * conv_column;
        ++result.active_rows;
        result.valid_tokens += count;
    }
    return result;
}

} // namespace ninfer::bench::gdn
