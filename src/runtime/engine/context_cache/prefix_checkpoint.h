#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::runtime {

// T042 (upstream d0ab3ba9 + 75be1384): a per-request rolling cursor over block digests.
//
// The cache used to re-hash the whole prefix for every lookup. This cursor advances one token at a
// time and keeps one digest per completed block, so a lookup at frontier F reads a value instead of
// walking F tokens. `lookahead` covers the token *after* a block (L = 1): a block hash that ends on
// its last token cannot distinguish two prompts that differ only at the next token, and that is
// exactly the boundary a prefix checkpoint is taken at.
class PrefixBlockHasher {
public:
    explicit PrefixBlockHasher(std::uint32_t block_tokens);

    void reset() noexcept;
    void push(TokenId token);

    [[nodiscard]] std::uint32_t block_tokens() const noexcept { return block_tokens_; }
    [[nodiscard]] std::uint32_t tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::uint32_t complete_blocks() const noexcept {
        return static_cast<std::uint32_t>(blocks_.size());
    }
    [[nodiscard]] std::uint64_t block_digest(std::uint32_t block) const;
    // Returns nullopt until the successor token of `block` has been pushed.
    [[nodiscard]] std::optional<std::uint64_t> lookahead_digest(std::uint32_t block) const;

private:
    struct Block {
        std::uint64_t digest = 0;
        std::optional<std::uint64_t> lookahead;
    };

    std::uint32_t block_tokens_ = 0;
    std::uint32_t tokens_       = 0;
    std::uint64_t chain_        = 0;
    std::vector<Block> blocks_;
};

// Why a point was or was not selected. The caller logs this rather than guessing from a nullopt.
enum class PrefixCheckpointReason : std::uint8_t {
    Selected,
    IntervalDisabled,
    NoProgress,
    BelowInterval,
    NearSemanticBoundary,
    BeyondSharedPrefix,
    BelowMinimumFrontier,
    // T010 conclusion 4: the tier payload carries Main KV + one state image, but no MTP KV span,
    // no mtp frontier and no tail hidden row. A checkpoint taken while MTP is on would restore a
    // draft context that never saw the prefix; fail closed until T197 adds the span.
    DraftStateAbsent,
};

struct PrefixCheckpointPolicy {
    // Target spacing. 0 disables intermediate checkpoints (semantic captures still happen).
    std::uint32_t interval         = 0;
    // The cache cannot address a prefix shorter than two chunks.
    std::uint32_t minimum_frontier = 256;
    // L10: shared-prefix points are capped at the first output position. UINT32_MAX means no cap.
    std::uint32_t shared_prefix_limit = UINT32_MAX;
    // Fail closed when the checkpoint payload cannot carry the speculative draft context.
    bool speculative             = false;
    bool draft_state_in_payload  = false;
};

struct PrefixCheckpointRequest {
    // Execution window: the cursor has already covered `begin`, the permit reaches `end`.
    std::uint32_t begin      = 0;
    std::uint32_t end        = 0;
    std::uint32_t last_saved = 0;
    // The next semantic capture the plan already scheduled, if any.
    std::optional<std::uint32_t> next_semantic;
    // Proven message boundaries inside the window (monotonic).
    std::span<const std::uint32_t> semantic_boundaries;
};

struct PrefixCheckpointDecision {
    std::optional<std::uint32_t> frontier;
    PrefixCheckpointReason reason = PrefixCheckpointReason::NoProgress;
};

// Chooses at most one intermediate checkpoint inside (last_saved, end].
//
// The point is the end of the current execution window (upstream 049d5fb4: "otherwise the end of
// the next ordinary execution window"), shortened by the shared-prefix cap and snapped to a proven
// message boundary within a quarter interval of the spacing target.
//
// Order of decisions matters:
//   1. a speculative run without draft state is refused outright (fail closed);
//   2. a point that adds less than half an interval is not worth a StateImage;
//   3. a semantic capture that is already near is preferred over a new point;
//   4. the shared-prefix cap may shorten the point, and never extends it;
//   5. a proven message boundary within a quarter interval of the target replaces it.
[[nodiscard]] PrefixCheckpointDecision plan_prefix_checkpoint(const PrefixCheckpointRequest&,
                                                              const PrefixCheckpointPolicy&);

// L10: shared-prefix checkpoints stop at the first output position. Never returns a value above
// `common_prefix_tokens`, and 0 means "there is no shared prefix to checkpoint".
[[nodiscard]] std::uint32_t
shared_prefix_cap(std::uint32_t common_prefix_tokens, std::uint32_t first_output_position) noexcept;

} // namespace ninfer::runtime
