#pragma once

// The checkpoint half of the KV tier: store a whole checkpoint blob under its content digest, and
// read it back with the same validation the decoder does.
//
// Separate type from RedisKvTier on purpose. A chunk tier addresses "the KV of tokens [a,b)" by a
// block hash and speaks LMCache's key format; a checkpoint tier addresses "a resumable prefix,
// including its recurrent state" by a content digest, in its own namespace. Merging them would
// invite a chunk key to be read as a checkpoint and silently skip prefill without the state.
//
// What this does not do: create the checkpoint inside the engine. Handles, page tables and state
// slots are process-local, so a fresh process still needs an import path on the model side before a
// stored blob can become a usable source. Until that exists, this is the transport half.

#include "core/kv_checkpoint_blob.h"
#include "core/resp.h"

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::core {

class CheckpointTier {
public:
    CheckpointTier(std::string host, std::uint16_t port, int timeout_ms = 2000)
        : host_(std::move(host)), port_(port), timeout_ms_(timeout_ms) {}

    [[nodiscard]] bool connect() { return client_.connect_to(host_, port_, timeout_ms_); }
    [[nodiscard]] const std::string& error() const { return client_.error(); }

    // Encodes the spans, stores them, and returns the digest that names them. The digest is the
    // content hash, so storing the same prefix twice writes the same key with the same bytes.
    [[nodiscard]] std::optional<std::uint64_t> store(std::uint32_t frontier,
                                                     std::span<const CheckpointSpan> spans) {
        const auto blob = encode_checkpoint(frontier, spans);
        // 12 MiB takes far longer than the default; scale so a big blob is not mistaken for a dead link.
        client_.set_timeout(std::max(5000, static_cast<int>(blob.size() / 1024)));
        const auto view = decode_checkpoint(blob);
        if (!view) { return std::nullopt; }  // never store something we cannot read back
        const std::string key = checkpoint_key(view->digest);
        const auto written    = client_.set(key, std::span<const std::byte>(blob));
        if (!written.has_value() || *written != blob.size()) { return std::nullopt; }
        bytes_written_ += *written;
        return view->digest;
    }

    // Returns the decoded view into `buffer`. The buffer must be exactly the stored size; a short
    // buffer is refused by the transport rather than truncated.
    [[nodiscard]] std::optional<CheckpointBlobView> load(std::uint64_t digest,
                                                         std::vector<std::byte>& buffer) {
        const std::string key = checkpoint_key(digest);
        const auto size       = client_.length(key);
        if (!size.has_value()) { return std::nullopt; }
        buffer.assign(*size, std::byte{0});
        const auto read = client_.get(key, std::span<std::byte>(buffer));
        if (!read.has_value()) { return std::nullopt; }
        bytes_read_ += *read;
        return decode_checkpoint(std::span<const std::byte>(buffer));
    }

    [[nodiscard]] bool exists(std::uint64_t digest) {
        return client_.exists(checkpoint_key(digest)).value_or(false);
    }

    [[nodiscard]] std::uint64_t bytes_written() const { return bytes_written_; }
    [[nodiscard]] std::uint64_t bytes_read() const { return bytes_read_; }

private:
    std::string host_;
    std::uint16_t port_;
    int timeout_ms_;
    RespClient client_;
    std::uint64_t bytes_written_ = 0;
    std::uint64_t bytes_read_    = 0;
};

} // namespace ninfer::core




