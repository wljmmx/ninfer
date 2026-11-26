// The GGML block decoders: eleven encodings, checked against an independent implementation.
//
// src/core/ggml_dequant.h is this port's own reading of the GGML block formats, written against
// llama.cpp's dequantisers. tests/core/ggml_dequant_fixture.h carries the first two blocks of a real
// tensor of every encoding the packed Qwen4Exp artifact uses, together with the values gguf-py
// decodes those same blocks to. Two implementations of one specification agreeing is the evidence;
// this test never compares the decoder with itself.
//
// Tolerance: both sides compute the same expression in binary32, but the C++ side may contract a
// multiply and a subtract into one FMA (`d*q - m`) while numpy rounds the product first. That is a
// one-ulp difference, so the check is a relative bound rather than bit equality - and the test also
// pins the decode's own scale, because a wrong d or a swapped plane is orders of magnitude off, not
// one ulp.

#include "core/ggml_dequant.h"

#include "core/ggml_dequant_fixture.h"

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* encoding, const char* what) {
    if (condition) { return; }
    std::fprintf(stderr, "FAIL [%s] %s\n", encoding, what);
    ++failures;
}

using DecodeFn = void (*)(const std::uint8_t*, std::span<float>);

// The container's own format name is `ggml_<type>`; the fixture names the type. Mapping here keeps
// the pairing explicit: a fixture whose block geometry disagrees with the decoder it selects would
// otherwise decode cleanly into plausible garbage.
struct Decoder {
    const char* encoding;
    DecodeFn fn;
};

constexpr Decoder kDecoders[] = {
    {"Q4_K", ninfer::core::decode_ggml_q4_k},
    {"Q5_K", ninfer::core::decode_ggml_q5_k},
    {"Q6_K", ninfer::core::decode_ggml_q6_k},
    {"Q2_0", ninfer::core::decode_ggml_q2_0},
    {"IQ4_NL", ninfer::core::decode_ggml_iq4_nl},
    {"IQ4_XS", ninfer::core::decode_ggml_iq4_xs},
    {"IQ2_XXS", ninfer::core::decode_ggml_iq2_xxs},
    {"IQ2_XS", ninfer::core::decode_ggml_iq2_xs},
    {"IQ2_S", ninfer::core::decode_ggml_iq2_s},
    {"IQ3_XXS", ninfer::core::decode_ggml_iq3_xxs},
    {"IQ3_S", ninfer::core::decode_ggml_iq3_s},
};

DecodeFn find_decoder(const char* encoding) {
    for (const auto& entry : kDecoders) {
        if (std::string_view(entry.encoding) == encoding) { return entry.fn; }
    }
    return nullptr;
}

// Relative bound for the FMA/reassociation difference, with an absolute floor so values decoded
// from a near-zero scale cannot fail on the ratio alone.
constexpr double kRelativeTolerance = 2e-6;
constexpr double kAbsoluteFloor     = 1e-7;

} // namespace

int main() {
    for (const auto& fixture : ninfer::test::kGgmlFixtures) {
        const auto decode = find_decoder(fixture.encoding);
        check(decode != nullptr, fixture.encoding, "no decoder for this encoding");
        if (decode == nullptr) { continue; }

        check(fixture.bytes % fixture.type_bytes == 0, fixture.encoding,
              "fixture bytes are not whole blocks");
        const std::size_t blocks = fixture.bytes / fixture.type_bytes;
        check(fixture.elements == blocks * fixture.block, fixture.encoding,
              "fixture element count disagrees with its block count");

        std::vector<float> decoded(fixture.elements, 0.0f);
        ninfer::core::decode_ggml(fixture.data, fixture.elements, fixture.block, fixture.type_bytes,
                                  decode, decoded);

        double worst_relative = 0;
        std::size_t worst_index = 0;
        bool finite = true;
        for (std::size_t i = 0; i < decoded.size(); ++i) {
            const double got  = decoded[i];
            const double want = fixture.expected[i];
            if (!std::isfinite(got)) { finite = false; break; }
            const double diff  = std::fabs(got - want);
            const double scale = std::fabs(want);
            const double bound = std::max(kAbsoluteFloor, kRelativeTolerance * scale);
            if (diff > bound) {
                const double rel = scale > 0 ? diff / scale : diff;
                if (rel > worst_relative) {
                    worst_relative = rel;
                    worst_index    = i;
                }
            }
        }
        check(finite, fixture.encoding, "decoder produced a non-finite value");
        check(worst_relative == 0, fixture.encoding, "decoded values disagree with gguf-py");
        if (worst_relative != 0) {
            std::fprintf(stderr, "      element %zu: got %.9g, expected %.9g (rel %.3g)\n",
                         worst_index, decoded[worst_index], fixture.expected[worst_index],
                         worst_relative);
        }

        // A decoder that silently produced zeros would pass a "no huge error" check only by being
        // compared against zeros; the fixture's own values are the guard, so state the evidence.
        double check_sum = 0;
        for (std::size_t i = 0; i < decoded.size(); ++i) check_sum += std::fabs(decoded[i]);
        std::printf("%-8s %-32s blocks=%zu elements=%zu |decoded|=%.6g\n", fixture.encoding,
                    fixture.source, blocks, fixture.elements, check_sum);
        check(check_sum > 0, fixture.encoding, "decode produced an all-zero block");
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ggml dequant: %zu encodings agree with the reference\n",
                sizeof(ninfer::test::kGgmlFixtures) / sizeof(ninfer::test::kGgmlFixtures[0]));
    return 0;
}
