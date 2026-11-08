#include "runtime/engine/context_cache/prefix_checkpoint.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

constexpr std::uint64_t kOffset = 1469598103934665603ULL; // FNV-1a 64 offset basis
constexpr std::uint64_t kPrime  = 1099511628211ULL;

std::uint64_t mix(std::uint64_t digest, std::uint64_t value) noexcept {
    digest ^= value;
    digest *= kPrime;
    return digest;
}

} // namespace

PrefixBlockHasher::PrefixBlockHasher(std::uint32_t block_tokens) : block_tokens_(block_tokens) {
    if (block_tokens == 0) {
        throw std::invalid_argument("prefix block size must be positive");
    }
    reset();
}

void PrefixBlockHasher::reset() noexcept {
    tokens_ = 0;
    chain_  = kOffset;
    blocks_.clear();
}

void PrefixBlockHasher::push(TokenId token) {
    // The first token of a new block is the successor of the block that just closed, so the
    // lookahead can be completed exactly once.
    if (tokens_ != 0 && tokens_ % block_tokens_ == 0 && !blocks_.empty() &&
        !blocks_.back().lookahead) {
        blocks_.back().lookahead = mix(blocks_.back().digest, static_cast<std::uint64_t>(token));
    }
    chain_ = mix(chain_, static_cast<std::uint64_t>(token));
    ++tokens_;
    if (tokens_ % block_tokens_ == 0) { blocks_.push_back(Block{.digest = chain_}); }
}

std::uint64_t PrefixBlockHasher::block_digest(std::uint32_t block) const {
    if (block >= blocks_.size()) {
        throw std::out_of_range("prefix block exceeds the complete blocks");
    }
    return blocks_[block].digest;
}

std::optional<std::uint64_t>
PrefixBlockHasher::lookahead_digest(std::uint32_t block) const {
    if (block >= blocks_.size()) {
        throw std::out_of_range("prefix block exceeds the complete blocks");
    }
    return blocks_[block].lookahead;
}

std::uint32_t shared_prefix_cap(std::uint32_t common_prefix_tokens,
                                std::uint32_t first_output_position) noexcept {
    return std::min(common_prefix_tokens, first_output_position);
}

PrefixCheckpointDecision plan_prefix_checkpoint(const PrefixCheckpointRequest& request,
                                                const PrefixCheckpointPolicy& policy) {
    if (policy.interval == 0) {
        return {.reason = PrefixCheckpointReason::IntervalDisabled};
    }
    // T010: ranking a point is pointless when the bytes behind it cannot be restored together.
    if (policy.speculative && !policy.draft_state_in_payload) {
        return {.reason = PrefixCheckpointReason::DraftStateAbsent};
    }
    if (request.end <= request.begin || request.last_saved > request.begin) {
        return {.reason = PrefixCheckpointReason::NoProgress};
    }

    const std::uint64_t target = static_cast<std::uint64_t>(request.last_saved) + policy.interval;
    if (request.end < target) { return {.reason = PrefixCheckpointReason::BelowInterval}; }
    // A semantic capture that will happen anyway, close to this window, already covers the prefix.
    if (request.next_semantic &&
        *request.next_semantic <= static_cast<std::uint64_t>(request.end) + policy.interval / 4U) {
        return {.reason = PrefixCheckpointReason::NearSemanticBoundary};
    }

    const std::uint32_t cap = std::min(request.end, policy.shared_prefix_limit);
    if (cap <= request.begin) { return {.reason = PrefixCheckpointReason::BeyondSharedPrefix}; }

    // The shared-prefix cap wins over the interval, but a point that saves less than half the
    // spacing is not worth a StateImage.
    if (static_cast<std::uint64_t>(cap) <=
        static_cast<std::uint64_t>(request.last_saved) + policy.interval / 2U) {
        return {.reason = PrefixCheckpointReason::BelowInterval};
    }

    std::uint32_t chosen   = cap;
    std::uint64_t distance = std::numeric_limits<std::uint64_t>::max();
    for (const auto boundary : request.semantic_boundaries) {
        if (boundary <= request.begin || boundary > cap) { continue; }
        const std::uint64_t difference =
            boundary < target ? target - boundary : boundary - target;
        if (difference <= policy.interval / 4U && difference < distance) {
            chosen   = boundary;
            distance = difference;
        }
    }

    if (chosen < policy.minimum_frontier) {
        return {.reason = PrefixCheckpointReason::BelowMinimumFrontier};
    }
    if (chosen <= request.last_saved) { return {.reason = PrefixCheckpointReason::NoProgress}; }
    return {.frontier = chosen, .reason = PrefixCheckpointReason::Selected};
}

} // namespace ninfer::runtime
