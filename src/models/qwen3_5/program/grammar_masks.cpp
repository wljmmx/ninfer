#include "models/qwen3_5/program/program_impl.h"
#include "core/device.h"
#include "core/transfer.h"

namespace ninfer::models::qwen3_5::detail {

ops::SamplingMask ProgramImpl::bind_grammar_mask(runtime::TokenMaskProvider* provider,
                                                 std::size_t row) {
    grammar_dead_positions[row] = 0;
    if (!provider || !provider->constrained(row)) { return {}; }
    const auto words = grammar_masks_device.ne[0];
    return {static_cast<const std::uint32_t*>(grammar_masks_device.data) +
                row * (draft_window + 1) * words,
            words};
}

ops::SamplingMask ProgramImpl::fill_grammar_mask(runtime::TokenMaskProvider* provider,
                                                 std::size_t row, std::span<const TokenId> drafts) {
    grammar_dead_positions[row] = 0;
    if (!provider || !provider->constrained(row)) { return {}; }
    const auto words  = static_cast<std::size_t>(grammar_masks_device.ne[0]);
    const auto offset = row * (draft_window + 1) * words;
    std::span<std::uint32_t> host(static_cast<std::uint32_t*>(grammar_masks_host->data()) + offset,
                                  (drafts.size() + 1) * words);
    grammar_dead_positions[row] = provider->fill(row, drafts, host);
    if (std::getenv("NINFER_GRAMMAR_DEBUG") != nullptr) {
        std::fprintf(stderr, "[grammar] provider saw | row %zu | %zu draft(s):", row, drafts.size());
        for (std::size_t index = 0; index < drafts.size() && index < 8; ++index) {
            std::fprintf(stderr, " %lld", static_cast<long long>(drafts[index]));
        }
        std::fprintf(stderr, " | dead 0x%08x\n", grammar_dead_positions[row]);
        std::fflush(stderr);
    }
    auto* device_words          = static_cast<std::uint32_t*>(grammar_masks_device.data) + offset;
    core::copy_bytes(device_words, host.data(), host.size_bytes(),
                     core::TransferDirection::HostToDevice, device.stream);
    provider->uploaded(row, host.size_bytes());
    return {device_words, static_cast<std::int32_t>(words)};
}

} // namespace ninfer::models::qwen3_5::detail
