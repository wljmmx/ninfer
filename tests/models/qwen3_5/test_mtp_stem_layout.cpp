// The MTP stem's buffers, at the geometry that used to be broken.
//
// Both stem arms stage one term at a time and project it: `[h, T]` in and `[h, T]` out. The
// buffers come from workspace::mtp_stem (single-stream) and workspace::mtp_hc_roots (four-stream),
// which allocate the residual planes as one `[h*count, T]` block (each stream is a plane of it) and
// the fused pack as `[2h, T]`.
//
// The defect this pins, in both arms: the split form - Qwen4Exp's artifact keeps mtp/fc_embedding
// and mtp/fc_hidden as two parents, so the stem projects `e` and the per-stream `h` term separately
// - used the FIRST HALF of the `[2h, T]` pack as the projection's output
// (`packed_input.slice(0, 0, h)`). The slice keeps the parent's row pitch, so the view is only
// contiguous when T == 1 (one row, nothing to stride over); at T > 1 it is a strided `[h, T]` whose
// column pitch is 2h, which `ops::linear` refuses outright. That is why `--spec mtp` used to stop at
// "linear: x/out must be contiguous" on every prefill chunk while a single-token step looked fine.
//
// Both arms now project into a real contiguous `[h, T]` - the four-stream arm into `input`, the
// single-stream arm into `attention_hidden` (nothing reads it before the norm that overwrites it) -
// so this test asserts the whole per-term buffer set is contiguous at every width and states the
// property that separates the alias from them.

#include "core/tensor.h"
#include "models/qwen3_5/execution/workspace.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <new>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen3_5;
using namespace ninfer::models::qwen3_5::execution;

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "mtp stem layout: " << label << " failed\n";
        ++failures;
    }
}

// A real Tensor needs a real backing; the stem's buffers never leave the host in this test, and the
// shape/strides are what it is about, so a bumped host block is enough.
class BumpArena {
  public:
    explicit BumpArena(std::size_t bytes) : storage_(bytes) {}

    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape) {
        const std::size_t need = Tensor::metadata(dtype, shape).bytes();
        cursor_                = (cursor_ + 255) & ~std::size_t{255};
        if (cursor_ + need > storage_.size()) { throw std::bad_alloc(); }
        void* const base = storage_.data() + cursor_;
        cursor_ += need;
        return Tensor(base, Device::cpu(), dtype, shape);
    }

  private:
    std::vector<std::byte> storage_;
    std::size_t cursor_ = 0;
};

TextConfig hc_config() {
    TextConfig config;
    config.hidden_size = 8;
    config.hc          = HcConfig{4, 2}; // four streams, rank-2 pair
    return config;
}

// The pack's first half is the alias both split arms used to project into. At T == 1 its row count
// is one, so Tensor::is_contiguous skips the column-stride check and the alias looks fine; at T > 1
// the pitch is 2h and `linear` refuses it. Nothing in either stem may write an `[h, T]` result
// there, and `linear`'s own precondition is what makes that a hard rule rather than a preference.
void check_pack_alias(const Tensor& packed_input, std::int32_t hidden, std::int32_t tokens,
                      std::int64_t row_bytes) {
    const Tensor first_half = packed_input.slice(0, 0, hidden);
    check(first_half.ne[0] == hidden && first_half.ne[1] == tokens,
          "the pack's first half is [h, T]");
    check(first_half.nb[1] == 2 * row_bytes, "the pack's first half keeps the 2h pitch");
    check(first_half.is_contiguous() == (tokens == 1),
          "the pack's first half is contiguous only at T == 1");
    check(!first_half.is_contiguous() || tokens == 1,
          "a T > 1 projection output must not be the pack's first half");
}

void check_stream_tile(const Tensor& tile, std::int32_t hidden, std::int32_t tokens,
                       std::int64_t row_bytes, const char* name) {
    check(tile.ne[0] == hidden && tile.ne[1] == tokens, name);
    check(tile.is_contiguous(), name);
    check(tile.nb[1] == row_bytes, name);
}

void check_stream_width(const TextConfig& config, std::int32_t tokens) {
    const auto hidden            = static_cast<std::int32_t>(config.hidden_size);
    const std::int64_t row_bytes = static_cast<std::int64_t>(hidden) * sizeof(std::uint16_t);

    // The single-stream stem: the split arm projects `fc_embedding(e)` into `residual` and
    // `fc_hidden(h)` into `attention_hidden`, adds, and then overwrites `attention_hidden` with the
    // block's input norm.
    {
        BumpArena arena(1U << 20U);
        const auto roots = workspace::mtp_stem(arena, config, tokens, /*allocate_embedding=*/true);
        check(roots.embedding.ne[0] == hidden && roots.embedding.ne[1] == tokens &&
                  roots.embedding.is_contiguous(),
              "stem embedding is [h, T] and contiguous");
        check_stream_tile(roots.normalized_embedding, hidden, tokens, row_bytes,
                          "stem normalized embedding is [h, T], contiguous, one h between columns");
        check_stream_tile(roots.normalized_hidden, hidden, tokens, row_bytes,
                          "stem normalized hidden is [h, T], contiguous, one h between columns");
        check_stream_tile(roots.residual, hidden, tokens, row_bytes,
                          "stem residual is [h, T], contiguous, one h between columns");
        // The staging tile the split arm's `fc_hidden` term lands in: contiguous at every width and
        // its own storage, not a window of the pack.
        check_stream_tile(roots.attention_hidden, hidden, tokens, row_bytes,
                          "stem attention_hidden is [h, T], contiguous, one h between columns");
        check(roots.packed_input.ne[0] == 2 * hidden && roots.packed_input.ne[1] == tokens &&
                  roots.packed_input.is_contiguous(),
              "stem pack is [2h, T] and contiguous");
        check_pack_alias(roots.packed_input, hidden, tokens, row_bytes);
        check(roots.attention_hidden.data != roots.packed_input.data,
              "the split arm's output tile is not the pack");
    }

    // The four-stream stem: same two tiles, but the residual is one [h*count, T] block and the
    // split arm projects into `input` instead.
    {
        const auto plane = static_cast<std::int32_t>(config.hc->stream_width(config.hidden_size));
        BumpArena arena(1U << 20U);
        const auto roots = workspace::mtp_hc_roots(arena, config, tokens);

        // The residual is one [h*count, T] block: a stream occupies the rows [s*h, (s+1)*h), which
        // is exactly the [h, count, T] geometry the HC ops read it as.
        check(roots.residual.ne[0] == plane && roots.residual.ne[1] == tokens &&
                  roots.residual.is_contiguous(),
              "hc residual is [h*count, T] and contiguous");
        check(roots.residual.nb[1] == static_cast<std::int64_t>(plane) * sizeof(std::uint16_t),
              "hc residual columns are one h*count plane apart");
        check(roots.normalized.ne[0] == plane && roots.normalized.is_contiguous(),
              "hc normalized mirrors the residual");

        // `stream_input` is the staged stream slice, `input` is where the fix projects that stream
        // into (and what the fold overwrites afterwards), `stream_output` carries the embedding
        // term.
        check_stream_tile(roots.stream_input, hidden, tokens, row_bytes,
                          "hc stream_input is [h, T], contiguous, one h between columns");
        check_stream_tile(roots.input, hidden, tokens, row_bytes,
                          "hc input is [h, T], contiguous, one h between columns");
        check_stream_tile(roots.stream_output, hidden, tokens, row_bytes,
                          "hc stream_output is [h, T], contiguous, one h between columns");

        check(roots.packed_input.ne[0] == 2 * hidden && roots.packed_input.ne[1] == tokens &&
                  roots.packed_input.is_contiguous(),
              "hc pack is [2h, T] and contiguous");
        check_pack_alias(roots.packed_input, hidden, tokens, row_bytes);
        check(roots.input.data != roots.packed_input.data,
              "the hc split arm's output tile is not the pack");
        check(roots.stream_output.data != roots.packed_input.data,
              "the embedding term's tile is not the pack");
    }
}

} // namespace

int main() {
    const TextConfig config = hc_config();
    check_stream_width(config, 1); // the decode width the alias happened to survive
    check_stream_width(config, 5); // a prefill chunk: the width that refused it
    check_stream_width(config, 2); // the narrowest width that strides at all

    if (failures != 0) {
        std::cerr << "mtp stem layout: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "mtp stem layout: ok\n";
    return 0;
}
