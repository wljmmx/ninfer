// The four-stream MTP stem's buffers, at the one geometry that used to be broken.
//
// The stem stages one residual stream at a time and projects it: `[h, T]` in and `[h, T]` out. The
// buffers come from workspace::mtp_hc_roots, which allocates the residual planes as one
// `[h*count, T]` block (each stream is a plane of it) and the fused pack as `[2h, T]`.
//
// The defect this pins: the split-stem arm used the FIRST HALF of that `[2h, T]` pack as the
// projection's output (`packed_input.slice(0, 0, h)`). The slice keeps the parent's row pitch, so
// the view is only contiguous when T == 1 (one row, nothing to stride over); at T > 1 it is a
// strided `[h, T]` whose column pitch is 2h, which `ops::linear` refuses outright. That is why
// `--spec mtp` used to stop at "linear: x/out must be contiguous" on every prefill chunk while a
// single-token step looked fine.
//
// The stem now projects into `input` - a real contiguous `[h, T]` - so this test asserts the whole
// per-stream buffer set is contiguous at both widths and states the property that separates the
// alias from them.

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

void check_width(const TextConfig& config, std::int32_t tokens) {
    const auto hidden = static_cast<std::int32_t>(config.hidden_size);
    const auto count  = static_cast<std::int32_t>(config.hc->count);
    const auto plane  = static_cast<std::int32_t>(config.hc->stream_width(config.hidden_size));
    const std::int64_t row_bytes = static_cast<std::int64_t>(hidden) * sizeof(std::uint16_t);

    BumpArena arena(1U << 20U);
    const auto roots = workspace::mtp_hc_roots(arena, config, tokens);

    // The residual is one [h*count, T] block: a stream occupies the rows [s*h, (s+1)*h), which is
    // exactly the [h, count, T] geometry the HC ops read it as.
    check(roots.residual.ne[0] == plane && roots.residual.ne[1] == tokens &&
              roots.residual.is_contiguous(),
          "residual is [h*count, T] and contiguous");
    check(roots.residual.nb[1] == static_cast<std::int64_t>(plane) * sizeof(std::uint16_t),
          "residual columns are one h*count plane apart");
    check(roots.normalized.ne[0] == plane && roots.normalized.is_contiguous(),
          "normalized mirrors the residual");

    // The two linear inputs and their output: all three are [h, T] with nothing between the columns.
    // `stream_input` is the staged stream slice, `input` is where the fix projects that stream into
    // (and what the fold overwrites afterwards), `stream_output` carries the embedding term.
    for (const Tensor* tile : {&roots.stream_input, &roots.input, &roots.stream_output}) {
        check(tile->ne[0] == hidden && tile->ne[1] == tokens, "per-stream tile is [h, T]");
        check(tile->is_contiguous(), "per-stream tile is contiguous");
        check(tile->nb[1] == row_bytes, "per-stream tile columns are one h apart");
    }

    // The pack the fused form feeds to `input_projection`: [2h, T], contiguous as a whole.
    check(roots.packed_input.ne[0] == 2 * hidden && roots.packed_input.ne[1] == tokens &&
              roots.packed_input.is_contiguous(),
          "packed input is [2h, T] and contiguous");

    // Its first half is the alias the split arm used to project into. At T == 1 the row count is
    // one, so Tensor::is_contiguous skips the column-stride check and the alias looks fine; at
    // T > 1 the pitch is 2h and `linear` refuses it. Nothing in the stem may write a [h, T] result
    // there.
    const Tensor first_half = roots.packed_input.slice(0, 0, hidden);
    check(first_half.ne[0] == hidden && first_half.ne[1] == tokens, "the pack's first half is [h, T]");
    check(first_half.nb[1] == 2 * row_bytes, "the pack's first half keeps the 2h pitch");
    check(first_half.is_contiguous() == (tokens == 1),
          "the pack's first half is contiguous only at T == 1");

    // `linear`'s own precondition, stated where the stem's choice is made: the output tile has to
    // be contiguous, so a projection may only target the buffers asserted above.
    check(!first_half.is_contiguous() || tokens == 1,
          "a T > 1 projection output must not be the pack's first half");
}

} // namespace

int main() {
    const TextConfig config = hc_config();
    check_width(config, 1); // the decode width the alias happened to survive
    check_width(config, 5); // a prefill chunk: the width that refused it
    check_width(config, 2); // the narrowest width that strides at all

    if (failures != 0) {
        std::cerr << "mtp stem layout: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "mtp stem layout: ok\n";
    return 0;
}
