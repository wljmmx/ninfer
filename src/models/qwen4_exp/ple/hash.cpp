#include "models/qwen4_exp/ple/hash.h"

#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen4_exp {

void ple_hash(std::span<const std::int32_t> ids, std::array<std::int32_t, 2> tail, std::int32_t eos,
              const PleHashControls& controls, std::span<std::uint32_t> output) {
    if (output.size() / kPleHeads != ids.size() || output.size() % kPleHeads || eos < 0 ||
        tail[0] < 0 || tail[1] < 0)
        throw std::invalid_argument("invalid PLE hash operands");
    for (auto id : ids)
        if (id < 0) throw std::invalid_argument("negative PLE token ID");
    for (std::size_t h = 0; h < kPleHeads; ++h)
        if (controls.vocab_sizes[h] <= 0 || controls.offsets[h] < 0 ||
            std::uint64_t(controls.offsets[h]) > std::numeric_limits<std::uint32_t>::max() ||
            std::uint64_t(controls.vocab_sizes[h] - 1) >
                std::numeric_limits<std::uint32_t>::max() - std::uint64_t(controls.offsets[h]))
            throw std::invalid_argument("PLE hash head exceeds U32 rows");
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const auto older = tail[1] == eos ? eos : tail[0];
        const auto pair  = std::uint64_t(ids[t]) * std::uint64_t(controls.multipliers[0]) ^
                          std::uint64_t(tail[1]) * std::uint64_t(controls.multipliers[1]);
        const auto triple = pair ^ std::uint64_t(older) * std::uint64_t(controls.multipliers[2]);
        for (std::size_t h = 0; h < kPleHeads; ++h) {
            // The reference (llama.cpp's qwen4exp graph) reduces the mixed word modulo 2^64 and
            // takes an unsigned remainder; reinterpreting the word as signed and re-basing the
            // negative result is only equal while the top bit never sets. Keep the unsigned form.
            const std::uint64_t word = h < 8 ? pair : triple;
            const std::uint64_t index =
                word % static_cast<std::uint64_t>(controls.vocab_sizes[h]);
            output[t * kPleHeads + h] =
                static_cast<std::uint32_t>(index + std::uint64_t(controls.offsets[h]));
        }
        tail = {tail[1], ids[t]};
    }
}

} // namespace ninfer::models::qwen4_exp
