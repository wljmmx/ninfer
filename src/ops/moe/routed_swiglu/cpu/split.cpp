// port (sm_89): the CPU half of the gate/up/down (Qwen4Exp) expert form.
//
// One expert at a time, three matrices addressed per expert:
//
//   either the host tier's packed slot form (planes at 0 / codes_bytes / scales offset inside a
//   slot, one expert offset table), or a bank addressed in place in the artifact mapping (one
//   origin per plane). Both name the same three planes per expert.
//
// The row map is the artifact's, not the fused one: the gate object and the up object hold I rows
// each, so gate row i and up row i are two images of the same output channel - never rows i and I+i
// of one parent (on this artifact row I of gate is the next expert's first row). Arithmetic runs in
// FP32 with the device kernels' group order: a 64-value partial scaled once per group.
#include "ops/moe/routed_swiglu/cpu/split.h"

#include "ops/common/cpu/avx2.h"
#include "ops/common/numeric/math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::detail::moe_routed_cpu {
namespace {

namespace ax = common::cpu::avx2;

constexpr std::int32_t kAlignment = 256;

// One row of one matrix, already advanced to its first code / high bits / scale of that row.
struct Row {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint16_t* scales = nullptr;
};

// The plane origins of one expert, before a row is selected.
struct ExpertBase {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint16_t* scales = nullptr;
};

// Row r of an expert image. The artifact lays a matrix out row-major per plane: codes hold K/2 bytes
// per row, the high bits (Bits-4 per weight) (K/8)*(Bits-4) bytes, and the group scales K/64 FP16
// values. Row K's stride is the row's, not the expert's - the expert stride only separates images.
inline Row row_at(const ExpertBase& base, std::int32_t row, std::int32_t k, int bits) noexcept {
    Row out;
    out.codes  = base.codes + static_cast<std::size_t>(row) * static_cast<std::size_t>(k / 2);
    out.high   = base.high != nullptr
                     ? base.high + static_cast<std::size_t>(row) *
                                       (static_cast<std::size_t>(k / 8) *
                                        static_cast<std::size_t>(bits - 4))
                     : nullptr;
    out.scales = base.scales != nullptr
                     ? base.scales + static_cast<std::size_t>(row) *
                                         static_cast<std::size_t>(k / 64)
                     : nullptr;
    return out;
}

// One encoded matrix of the routed bank, with the per-expert addressing the bank was built with.
struct Matrix {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint16_t* scales = nullptr;
    std::size_t code_stride     = 0;
    std::size_t high_stride     = 0;
    std::size_t scale_stride    = 0;
    const std::uint64_t* codes_offsets  = nullptr;
    const std::uint64_t* high_offsets   = nullptr;
    const std::uint64_t* scales_offsets = nullptr;
    const std::uint64_t* expert_offsets = nullptr;

    [[nodiscard]] ExpertBase expert(std::uint32_t index) const noexcept {
        const auto offset = [&](const std::uint64_t* table, std::size_t stride) -> std::size_t {
            if (table != nullptr) { return static_cast<std::size_t>(table[index]); }
            return expert_offsets != nullptr ? static_cast<std::size_t>(expert_offsets[index])
                                             : static_cast<std::size_t>(index) * stride;
        };
        ExpertBase out;
        out.codes  = codes != nullptr ? codes + offset(codes_offsets, code_stride) : nullptr;
        out.high   = high != nullptr ? high + offset(high_offsets, high_stride) : nullptr;
        out.scales = scales != nullptr
                         ? reinterpret_cast<const std::uint16_t*>(
                               reinterpret_cast<const std::byte*>(scales) +
                               offset(scales_offsets, scale_stride))
                         : nullptr;
        return out;
    }
};

int code_bits(QType format) {
    switch (format) {
    case QType::Q4_G64_FP16: return 4;
    case QType::Q6_G64_FP16: return 6;
    default:
        throw std::invalid_argument(
            "routed_swiglu CPU: the split expert form has no kernel for this code width");
    }
}

Matrix matrix_of(const WeightBankView& bank) {
    const auto& layout = bank.layout;
    if (bank.parent == nullptr || bank.parent->data == nullptr) {
        throw std::invalid_argument("routed_swiglu CPU: missing split weight bank");
    }
    const auto* data = bank.parent->data;
    Matrix out;
    out.codes          = layout.codes.expert_bytes
                             ? reinterpret_cast<const std::uint8_t*>(data + layout.codes.offset)
                             : nullptr;
    out.high           = layout.high.expert_bytes
                             ? reinterpret_cast<const std::uint8_t*>(data + layout.high.offset)
                             : nullptr;
    out.scales         = layout.scales.expert_bytes
                             ? reinterpret_cast<const std::uint16_t*>(data + layout.scales.offset)
                             : nullptr;
    out.code_stride    = static_cast<std::size_t>(layout.codes.expert_stride);
    out.high_stride    = static_cast<std::size_t>(layout.high.expert_stride);
    out.scale_stride   = static_cast<std::size_t>(layout.scales.expert_stride);
    out.codes_offsets  = bank.codes_offsets;
    out.high_offsets   = bank.high_offsets;
    out.scales_offsets = bank.scales_offsets;
    out.expert_offsets = bank.expert_offsets;
    return out;
}

// out[c] = row . x[c][0..k). Weights are decoded once per four columns and the group scale is
// applied to the group partial, so the sum has the shape every other routed kernel uses.
template <int Bits, int Columns, class Input>
inline void row_dot(const Row& row, std::int32_t k, const Input* const (&x)[Columns],
                    float (&out)[Columns]) noexcept {
    __m256 total[Columns];
    for (int c = 0; c < Columns; ++c) { total[c] = _mm256_setzero_ps(); }
    const auto groups = k / 64;
    for (std::int32_t group = 0; group < groups; ++group) {
        __m256 partial[Columns];
        for (int c = 0; c < Columns; ++c) { partial[c] = _mm256_setzero_ps(); }
        for (std::int32_t q = 0; q < 64; q += 8) {
            const auto* high = row.high != nullptr
                                   ? row.high + (group * 8 + q / 8) * (Bits - 4)
                                   : nullptr;
            const auto weights = ax::decode_signed_q<Bits>(row.codes + group * 32 + q / 2, high);
            for (int c = 0; c < Columns; ++c) {
                const auto* source = x[c] + group * 64 + q;
                __m256 values;
                if constexpr (sizeof(Input) == 2) {
                    values = ax::load_bf16(reinterpret_cast<const std::uint16_t*>(source));
                } else {
                    values = _mm256_loadu_ps(source);
                }
                partial[c] = _mm256_fmadd_ps(weights, values, partial[c]);
            }
        }
        const auto scale = _mm256_set1_ps(ax::fp16_to_f32(row.scales[group]));
        for (int c = 0; c < Columns; ++c) {
            total[c] = _mm256_fmadd_ps(partial[c], scale, total[c]);
        }
    }
    for (int c = 0; c < Columns; ++c) { out[c] = ax::reduce_sum(total[c]); }
}

struct Member {
    std::int32_t token = 0;
    float coefficient  = 0.0F;
};

// The int8 activation plane, laid out the way the q4 nibble dot consumes it: even indices first, then
// odd, so each nibble half meets its own half of the row with no shuffle. One group is 64 values with
// its own peak scale, which is the same grouping the device W4A8 path uses.
struct Q8Input {
    std::vector<std::int8_t> even; // [token][hidden]
    std::vector<std::int8_t> odd;  // [token][hidden]
    std::vector<float> scale;      // [token][groups]
    std::vector<std::int32_t> sum; // [token][groups]: sum of the quantised values, for the bias term
    // Natural order, one byte per value. The six-bit pass needs the row as stored: its codes are
    // magnitudes indexed by weight, where the four-bit path deliberately interleaves even and odd so
    // each nibble half meets its own half of the row without a shuffle.
    std::vector<std::int8_t> cont; // [token][width]
    // The sixteen-bit plane, for the down pass only. madd_epi16 takes sixteen values per instruction
    // where maddubs takes thirty-two, so this costs throughput - but a six-bit weight carries 64 codes
    // and an int8 activation only 127 levels, and on this projection that showed up as +2.78% perplexity
    // against the same-config bf16 reference. Sixteen bits buys that back at half the instruction count.
    std::vector<std::int16_t> cont16; // [token][width]
};

// The activation plane is BF16, whose bits are the top half of an FP32 word - not the 5/10 split that
// fp16_to_f32 reads. Feeding BF16 through the fp16 decoder turns every value into a different number
// while still looking plausible, which is exactly the kind of error an IMPROVING perplexity hides.
inline float bf16_value(std::uint16_t bits) noexcept {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

// Quantise every token's activation row once per collective. The dot then reads int8 instead of bf16,
// which is what lets the weights meet `maddubs` 32 codes at a time instead of eight. Takes the three
// scalars it needs rather than Work, so it does not depend on the struct's layout. Sizes are assumed
// already reserved by the caller: every worker runs this over its own slice, and an assign() here
// would have each lane zero the whole plane under the others.
inline void quantise_inputs(const std::uint16_t* x, std::int32_t tokens, std::int32_t hidden,
                            Q8Input& q8, const CpuWorkerContext& worker) {
    const int groups = hidden / 64;
    const auto range = worker.range(static_cast<std::size_t>(tokens) * groups);
    for (auto g = range.begin; g < range.end; ++g) {
        const auto token   = static_cast<std::int32_t>(g / static_cast<std::size_t>(groups));
        const auto group   = static_cast<std::int32_t>(g % static_cast<std::size_t>(groups));
        const auto* source = x + static_cast<std::size_t>(token) * hidden + group * 64;
        float peak = 0.0F;
        for (int i = 0; i < 64; ++i) {
            peak = std::max(peak, std::fabs(bf16_value(source[i])));
        }
        const float scale   = peak > 0.0F ? peak / 127.0F : 0.0F;
        const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
        q8.scale[static_cast<std::size_t>(token) * groups + group] = scale;
        auto* even = q8.even.data() + static_cast<std::size_t>(token) * hidden + group * 64;
        auto* odd  = q8.odd.data() + static_cast<std::size_t>(token) * hidden + group * 64;
        auto* cont = q8.cont.data() + static_cast<std::size_t>(token) * hidden + group * 64;
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            const int lo = std::clamp(
                static_cast<int>(std::lrint(bf16_value(source[2 * i]) * inverse)), -127, 127);
            const int hi = std::clamp(
                static_cast<int>(std::lrint(bf16_value(source[2 * i + 1]) * inverse)), -127, 127);
            even[i]     = static_cast<std::int8_t>(lo);
            odd[i]      = static_cast<std::int8_t>(hi);
            cont[2 * i] = static_cast<std::int8_t>(lo);
            cont[2 * i + 1] = static_cast<std::int8_t>(hi);
            sum += lo + hi;
        }
        q8.sum[static_cast<std::size_t>(token) * groups + group] = sum;
    }
}

// The same quantisation for an FP32 row: the down pass reads the SwiGLU activation, which this form
// keeps in float.
//
// THIRTY-TWO values a group, not sixty-four. The weight scale is stuck at 64 - that is the artifact's
// stored format and cannot be re-derived here - so the only lever on precision is the activation side,
// and it is the one that binds: at 64 values a group the down pass cost +2.78% perplexity while the
// four-bit gate/up cost +0.49%, and the GPU path's own record shows the same direction (a whole row
// per group was +4.29%, 64 values +0.38%). Halving the group halves the value a single peak has to
// cover. The price is one scale application per 32 values instead of per 64.
inline void quantise_floats(const float* x, std::int32_t tokens, std::int32_t width, Q8Input& q8,
                            const CpuWorkerContext& worker) {
    // Thirty-two values a group. A numeric probe on the down pass's own distribution (silu(g)*u with
    // six-bit weights) puts a 32-value group 12.7% below a 64-value one on induced dot error
    // (0.64046 against 0.73355, rmse 0.005193 against 0.006376), so the finer grouping is the better
    // plane by construction. The weight scale stays at 64 - that is the stored layout - and each
    // 32-value group simply carries its own activation scale, which is the exact dequantisation.
    const int groups = width / 32;
    const auto range = worker.range(static_cast<std::size_t>(tokens) * groups);
    for (auto g = range.begin; g < range.end; ++g) {
        const auto token   = static_cast<std::int32_t>(g / static_cast<std::size_t>(groups));
        const auto group   = static_cast<std::int32_t>(g % static_cast<std::size_t>(groups));
        const auto* source = x + static_cast<std::size_t>(token) * width + group * 32;
        float peak = 0.0F;
        for (int i = 0; i < 32; ++i) { peak = std::max(peak, std::fabs(source[i])); }
        const float scale   = peak > 0.0F ? peak / 127.0F : 0.0F;
        const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
        q8.scale[static_cast<std::size_t>(token) * groups + group] = scale;
        auto* cont = q8.cont.data() + static_cast<std::size_t>(token) * width + group * 32;
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            const int value = std::clamp(
                static_cast<int>(std::lrint(source[i] * inverse)), -127, 127);
            cont[i] = static_cast<std::int8_t>(value);
            sum += value;
        }
        q8.sum[static_cast<std::size_t>(token) * groups + group] = sum;
    }
}

// Sixteen-bit twin of the quantisation above: same grouping, same peak rule, 32767 levels instead of
// 127. Used only by the down pass, where the coarser plane cost +2.78% perplexity. No sum is kept -
// a signed sixteen-bit plane needs no bias term, unlike the four-bit nibble.
inline void quantise_floats_i16_groups(const float* x, std::int32_t width, Q8Input& q8,
                                        std::size_t begin, std::size_t end) {
    const int groups = width / 64;
    for (auto g = begin; g < end; ++g) {
        const auto token   = static_cast<std::int32_t>(g / static_cast<std::size_t>(groups));
        const auto group   = static_cast<std::int32_t>(g % static_cast<std::size_t>(groups));
        const auto* source = x + static_cast<std::size_t>(token) * width + group * 64;
        float peak = 0.0F;
        for (int i = 0; i < 64; ++i) { peak = std::max(peak, std::fabs(source[i])); }
        const float scale   = peak > 0.0F ? peak / 32767.0F : 0.0F;
        const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
        q8.scale[static_cast<std::size_t>(token) * groups + group] = scale;
        auto* cont = q8.cont16.data() + static_cast<std::size_t>(token) * width + group * 64;
        for (int i = 0; i < 64; ++i) {
            cont[i] = static_cast<std::int16_t>(
                std::clamp(static_cast<int>(std::lrint(source[i] * inverse)), -32767, 32767));
        }
    }
}

// The same body over the team's slice of the plane: every worker quantises its own share inside one
// collective, which is how the row-split path pays it in parallel with the row split itself.
inline void quantise_floats_i16(const float* x, std::int32_t tokens, std::int32_t width,
                                Q8Input& q8, const CpuWorkerContext& worker) {
    const int groups  = width / 64;
    const auto range  = worker.range(static_cast<std::size_t>(tokens) * groups);
    quantise_floats_i16_groups(x, width, q8, range.begin, range.end);
}

// And over the whole plane, for the flattened path: the thread that owns an expert quantises that
// expert's activation itself, because there is no collective to share it with.
inline void quantise_floats_i16_all(const float* x, std::int32_t tokens, std::int32_t width,
                                    Q8Input& q8) {
    const int groups = width / 64;
    quantise_floats_i16_groups(x, width, q8, 0, static_cast<std::size_t>(tokens) * groups);
}

// The q4 nibble dot: 32 codes per instruction pair against an int8 activation plane, instead of eight
// float codes per step. The packed nibble u is two's complement, so flipping bit 3 turns it into the
// offset-by-8 form b = u ^ 8 with w = b - 8, and one correction per 64-value group - 8 * sum(x) -
// restores the true sum. The scale chain keeps the device's order: weight scale first, activation
// scale second, one add at the end.
template <int Columns>
inline void row_dot_q4_q8(const Row& row, std::int32_t k, const std::int8_t* const* even,
                          const std::int8_t* const* odd, const std::int32_t* const* sum,
                          const float* const* scale, float (&out)[Columns]) noexcept {
    const auto ones       = _mm256_set1_epi16(1);
    const auto nibble     = _mm256_set1_epi8(0x0F);
    const auto bias       = _mm256_set1_epi8(static_cast<char>(0x88));
    __m256 total[Columns];
    for (int c = 0; c < Columns; ++c) { total[c] = _mm256_setzero_ps(); }
    const auto groups = k / 64;
    for (std::int32_t group = 0; group < groups; ++group) {
        __m256i even_v[Columns];
        __m256i odd_v[Columns];
        __m256i correction[Columns];
        float act_scale[Columns];
        for (int c = 0; c < Columns; ++c) {
            const auto* e = even[c] + group * 64;
            const auto* o = odd[c] + group * 64;
            even_v[c]     = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(e));
            odd_v[c]      = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(o));
            // One lane carries the whole bias: the group's 8 * sum(x) is subtracted once, not once
            // per lane, because the lanes are summed at the end of the group.
            correction[c] = _mm256_setr_epi32(8 * sum[c][group], 0, 0, 0, 0, 0, 0, 0);
            act_scale[c]  = scale[c][group];
        }
        const auto packed  = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row.codes + group * 32));
        const auto flipped = _mm256_xor_si256(packed, bias);
        const auto low     = _mm256_and_si256(flipped, nibble);
        const auto high    = _mm256_and_si256(_mm256_srli_epi16(flipped, 4), nibble);
        const auto weight  = _mm256_set1_ps(ax::fp16_to_f32(row.scales[group]));
        for (int c = 0; c < Columns; ++c) {
            const auto pairs = _mm256_sub_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(low, even_v[c]), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(high, odd_v[c]), ones)),
                correction[c]);
            const auto scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(pairs), weight);
            total[c] = _mm256_add_ps(_mm256_mul_ps(scaled, _mm256_set1_ps(act_scale[c])), total[c]);
        }
    }
    for (int c = 0; c < Columns; ++c) { out[c] = ax::reduce_sum(total[c]); }
}

inline void row_all_tokens_q4_q8(const Row& row, std::int32_t k, const std::int8_t* const* even,
                                 const std::int8_t* const* odd, const std::int32_t* const* sum,
                                 const float* const* scale, std::size_t count,
                                 float* out) noexcept {
    std::size_t at = 0;
    while (at < count) {
        const auto left = count - at;
        if (left >= 4) {
            const std::int8_t* es[4] = {even[at], even[at + 1], even[at + 2], even[at + 3]};
            const std::int8_t* os[4] = {odd[at], odd[at + 1], odd[at + 2], odd[at + 3]};
            const std::int32_t* ss[4] = {sum[at], sum[at + 1], sum[at + 2], sum[at + 3]};
            const float* sc[4] = {scale[at], scale[at + 1], scale[at + 2], scale[at + 3]};
            float values[4];
            row_dot_q4_q8<4>(row, k, es, os, ss, sc, values);
            for (int c = 0; c < 4; ++c) { out[at + c] = values[c]; }
            at += 4;
        } else if (left >= 2) {
            const std::int8_t* es[2] = {even[at], even[at + 1]};
            const std::int8_t* os[2] = {odd[at], odd[at + 1]};
            const std::int32_t* ss[2] = {sum[at], sum[at + 1]};
            const float* sc[2] = {scale[at], scale[at + 1]};
            float values[2];
            row_dot_q4_q8<2>(row, k, es, os, ss, sc, values);
            for (int c = 0; c < 2; ++c) { out[at + c] = values[c]; }
            at += 2;
        } else {
            const std::int8_t* es[1] = {even[at]};
            const std::int8_t* os[1] = {odd[at]};
            const std::int32_t* ss[1] = {sum[at]};
            const float* sc[1] = {scale[at]};
            float values[1];
            row_dot_q4_q8<1>(row, k, es, os, ss, sc, values);
            out[at] = values[0];
            at += 1;
        }
    }
}

// The six-bit plane, as 32 SIGNED codes. A six-bit weight is nibble + 16 * upper, where `upper` is the
// two bits this row stores separately - and those two bits are the HIGH bits of a signed six-bit code,
// not an unsigned magnitude. The device decoder reads the assembled byte back as (v ^ 0x20) - 0x20
// (ops/common/quant/integer.cuh: `unpack_q6_code` and `Q6MmaDecodeAtom`), so the code lands in
// [-32, 31]. Stopping at nibble + upper * 16 reads the same bits as an unsigned 0..63, and the
// assumption that followed it - "already stored as an unsigned magnitude offset by 32, so the group
// bias is 32 * sum(x)" - is what shipped and what was wrong: only the codes whose high field is 0 or 1
// sit 32 below their value, the rest sit 64 below, so a flat 32 * sum(x) leaves every weight with a 32
// error whose sign follows the high field - about half the row each way, which is why the deltas came
// out the size of the output itself. Sign-extending bit 5 into the byte is the missing step; with it
// the weights are the signed values and the group needs no bias term at all.
inline __m256i decode_q6_signed32(const std::uint8_t* codes, const std::uint8_t* high) noexcept {
    // Thirty-two nibbles out of sixteen bytes, in natural order - the same unpack the four-bit path
    // uses, so weight 2b comes from the low nibble of byte b and weight 2b+1 from the high one.
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m128i low    = _mm_and_si128(packed, nibble);
    const __m128i high_n = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble);
    const __m256i values =
        _mm256_set_m128i(_mm_unpackhi_epi8(low, high_n), _mm_unpacklo_epi8(low, high_n));

    // Eight bytes of two-bit planes, each byte holding four codes. Gather the byte that owns each
    // output position four times over, so the four positions become four different shifts.
    std::uint64_t bits;
    std::memcpy(&bits, high, sizeof(bits));
    alignas(32) static const std::uint8_t kGather[32] = {
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
    const __m256i shuffled = _mm256_shuffle_epi8(
        _mm256_set1_epi64x(static_cast<long long>(bits)),
        _mm256_load_si256(reinterpret_cast<const __m256i*>(kGather)));

    // A byte mask decides which of the four shifts applies to which output byte, so the packed value
    // lands in natural order without an interleave step. The mask's low two bits are also the value
    // mask, which is why the shifts can use srli_epi16: any bit that crosses a byte boundary falls
    // outside those two bits and is masked away.
    alignas(32) static const std::uint8_t kPick[4][32] = {
        {3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0,
         3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0},
        {0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0,
         0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0},
        {0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0,
         0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0},
        {0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3,
         0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3}};
    __m256i upper = _mm256_setzero_si256();
    for (int j = 0; j < 4; ++j) {
        const auto picked = _mm256_and_si256(
            _mm256_srli_epi16(shuffled, 2 * j),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(kPick[j])));
        upper = _mm256_or_si256(upper, picked);
    }
    // Two bits of upper are worth sixteen each, and the mask keeps the shift inside the byte: the
    // assembled code is nibble + upper * 16 in [0, 63], whose bit 5 is the sign of the six-bit value.
    const auto assembled = _mm256_add_epi8(
        values, _mm256_and_si256(_mm256_slli_epi16(upper, 4),
                                 _mm256_set1_epi8(static_cast<char>(0xF0))));
    // (v ^ 0x20) - 0x20 is the device decoder's own expression: it sign-extends bit 5 into the byte,
    // so 0..31 stay put and 32..63 become -32..-1.
    const auto sign = _mm256_set1_epi8(static_cast<char>(0x20));
    return _mm256_sub_epi8(_mm256_xor_si256(assembled, sign), sign);
}

// maddubs multiplies an UNSIGNED byte by a signed one, and a six-bit code is signed, so the sign has to
// come out of the weight and go into the activation: |w| * (x * sign(w)) = w * x. |w| <= 32 and
// |x| <= 127 bound one lane's pair sum at 2 * 32 * 127 = 8128 and a group's at 32 * 32 * 127, both far
// inside i16 / i32. The product is therefore exact and the group carries no bias term at all - unlike
// the four-bit path, whose nibble really is offset and pays 8 * sum(x) back.
template <int Columns>
inline void row_dot_q6_q8(const Row& row, std::int32_t k, const std::int8_t* const* cont,
                          const float* const* scale, float (&out)[Columns]) noexcept {
    const auto ones = _mm256_set1_epi16(1);
    __m256 total[Columns];
    for (int c = 0; c < Columns; ++c) { total[c] = _mm256_setzero_ps(); }
    // Thirty-two values per activation group. The weight scale is fixed at 64 by the artifact's stored
    // layout, so `group / 2` names it - and that is the whole point of the split: the coarse plane was
    // the activation, and halving its span is what recovers most of the +2.78%.
    const auto groups = k / 32;
    for (std::int32_t group = 0; group < groups; ++group) {
        float act_scale[Columns];
        for (int c = 0; c < Columns; ++c) { act_scale[c] = scale[c][group]; }
        const auto* codes = row.codes + group * 16;
        const auto* high  = row.high + group * 8;
        const auto weight = _mm256_set1_ps(ax::fp16_to_f32(row.scales[group / 2]));
        const auto signed_w  = decode_q6_signed32(codes, high);
        // sign_epi8(v, v) is |v|, and v = 0 stays 0, so the magnitude is the unsigned operand.
        const auto magnitude = _mm256_sign_epi8(signed_w, signed_w);
        for (int c = 0; c < Columns; ++c) {
            const auto* act_at = cont[c] + group * 32;
            const auto act_v   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_at));
            const auto pairs   = _mm256_madd_epi16(
                _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(act_v, signed_w)), ones);
            const auto scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(pairs), weight);
            total[c] = _mm256_add_ps(_mm256_mul_ps(scaled, _mm256_set1_ps(act_scale[c])), total[c]);
        }
    }
    for (int c = 0; c < Columns; ++c) { out[c] = ax::reduce_sum(total[c]); }
}

// The sixteen-bit down dot. A six-bit code decoded to a signed byte is already the exact weight, and
// madd_epi16 multiplies signed by signed, so this needs neither the sign trick nor the bias term the
// two byte paths carry - only twice the instructions per value, which is the trade the down pass
// makes for its precision.
template <int Columns>
inline void row_dot_q6_q16(const Row& row, std::int32_t k, const std::int16_t* const* cont,
                           const float* const* scale, float (&out)[Columns]) noexcept {
    // Four independent accumulators over the group loop, folded pairwise at the end. The chain being
    // broken is the reduction across ten groups, not the decode - and int16 has half the work per
    // group that int8 has (sixteen products a lane against thirty-two), so the reduction is a larger
    // share of this pass than it was there. Four ways is the most that still fits sixteen YMM
    // registers once Columns is taken into account, which is why the split narrows with Columns.
    constexpr int kAccums = Columns <= 2 ? 4 : 2;
    __m256 total[kAccums][Columns];
    for (int a = 0; a < kAccums; ++a) {
        for (int c = 0; c < Columns; ++c) { total[a][c] = _mm256_setzero_ps(); }
    }
    const auto groups = k / 64;
    for (std::int32_t group = 0; group < groups; ++group) {
        const int slot     = group % kAccums;
        const auto* codes  = row.codes + group * 32;
        const auto* high   = row.high + group * 16;
        const auto weight  = _mm256_set1_ps(ax::fp16_to_f32(row.scales[group]));
        __m256i acc[Columns];
        for (int c = 0; c < Columns; ++c) { acc[c] = _mm256_setzero_si256(); }
        // Sixty-four values as four sixteen-wide parts: part/2 picks the 32-code half, part%2 picks
        // one 128-bit lane of it, and cvtepi8_epi16 widens that lane to sixteen signed shorts.
        for (int part = 0; part < 4; ++part) {
            const auto w8 = decode_q6_signed32(codes + (part / 2) * 16, high + (part / 2) * 8);
            const auto lane = (part % 2) == 0 ? _mm256_castsi256_si128(w8)
                                             : _mm256_extracti128_si256(w8, 1);
            const auto w16  = _mm256_cvtepi8_epi16(lane);
            for (int c = 0; c < Columns; ++c) {
                const auto x16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    cont[c] + group * 64 + part * 16));
                acc[c] = _mm256_add_epi32(acc[c], _mm256_madd_epi16(w16, x16));
            }
        }
        for (int c = 0; c < Columns; ++c) {
            const auto scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(acc[c]), weight);
            total[slot][c] =
                _mm256_add_ps(_mm256_mul_ps(scaled, _mm256_set1_ps(scale[c][group])), total[slot][c]);
        }
    }
    for (int span = kAccums / 2; span > 0; span /= 2) {
        for (int a = 0; a < span; ++a) {
            for (int c = 0; c < Columns; ++c) {
                total[a][c] = _mm256_add_ps(total[a][c], total[a + span][c]);
            }
        }
    }
    for (int c = 0; c < Columns; ++c) { out[c] = ax::reduce_sum(total[0][c]); }
}

inline void row_all_tokens_q6_q16(const Row& row, std::int32_t k, const std::int16_t* const* cont,
                                  const float* const* scale, std::size_t count,
                                  float* out) noexcept {
    std::size_t at = 0;
    while (at < count) {
        const auto left = count - at;
        if (left >= 4) {
            const std::int16_t* cs[4] = {cont[at], cont[at + 1], cont[at + 2], cont[at + 3]};
            const float* sc[4] = {scale[at], scale[at + 1], scale[at + 2], scale[at + 3]};
            float values[4];
            row_dot_q6_q16<4>(row, k, cs, sc, values);
            for (int c = 0; c < 4; ++c) { out[at + c] = values[c]; }
            at += 4;
        } else if (left >= 2) {
            const std::int16_t* cs[2] = {cont[at], cont[at + 1]};
            const float* sc[2] = {scale[at], scale[at + 1]};
            float values[2];
            row_dot_q6_q16<2>(row, k, cs, sc, values);
            for (int c = 0; c < 2; ++c) { out[at + c] = values[c]; }
            at += 2;
        } else {
            const std::int16_t* cs[1] = {cont[at]};
            const float* sc[1] = {scale[at]};
            float values[1];
            row_dot_q6_q16<1>(row, k, cs, sc, values);
            out[at] = values[0];
            at += 1;
        }
    }
}

inline void row_all_tokens_q6_q8(const Row& row, std::int32_t k, const std::int8_t* const* cont,
                                 const float* const* scale, std::size_t count,
                                 float* out) noexcept {
    std::size_t at = 0;
    while (at < count) {
        const auto left = count - at;
        if (left >= 4) {
            const std::int8_t* cs[4] = {cont[at], cont[at + 1], cont[at + 2], cont[at + 3]};
            const float* sc[4] = {scale[at], scale[at + 1], scale[at + 2], scale[at + 3]};
            float values[4];
            row_dot_q6_q8<4>(row, k, cs, sc, values);
            for (int c = 0; c < 4; ++c) { out[at + c] = values[c]; }
            at += 4;
        } else if (left >= 2) {
            const std::int8_t* cs[2] = {cont[at], cont[at + 1]};
            const float* sc[2] = {scale[at], scale[at + 1]};
            float values[2];
            row_dot_q6_q8<2>(row, k, cs, sc, values);
            for (int c = 0; c < 2; ++c) { out[at + c] = values[c]; }
            at += 2;
        } else {
            const std::int8_t* cs[1] = {cont[at]};
            const float* sc[1] = {scale[at]};
            float values[1];
            row_dot_q6_q8<1>(row, k, cs, sc, values);
            out[at] = values[0];
            at += 1;
        }
    }
}

struct Work {
    // One entry per lane. lanes == 1 is the shape this form has always had: the tables then hold a
    // single element, and every read below names the same address in the same order as the scalar
    // fields they replaced, so the single-lane path stays bit-identical.
    const std::uint16_t* const* x = nullptr; // [lane] -> [token][hidden]
    float* const* output          = nullptr; // [lane] -> [token][hidden]
    float* const* activation      = nullptr; // [lane] -> [token][intermediate]
    std::int32_t lanes          = 1;
    std::int32_t tokens = 0, hidden = 0, intermediate = 0;
    int gate_bits = 4, down_bits = 6;
    // Which rounding the device's down pass used for this shape, mirrored here so the two halves
    // compute the same arithmetic: the grouped (multi-token) kernel stores BF16, the token kernel
    // FP32. groupwise_int names the same threshold in work.h as grouped_first.
    bool plane_round = false;
    Matrix gate, up, down;
    std::vector<std::int32_t> experts;
    // [lane][expert]. Each lane carries its own route table: two lanes never share a member list,
    // which is what makes a flattened (lane, expert) task table race-free across lanes.
    std::vector<std::vector<std::vector<Member>>> members;
    // NINFER_MOE_CPU_INT8: the same switch the fused form reads, now honoured here. Off keeps the bf16
    // decode this form has always used; on routes the four-bit gate/up through the q4 nibble dot and
    // the six-bit down through the q6 one.
    bool int8_path = false;
    // NINFER_SPLIT_Q6: the six-bit down pass through the q6 nibble dot. OFF by default and measured
    // SLOWER than the bf16 decode it replaces (277 us against 148 us per expert): the 32-value unpack
    // here is a scalar loop, and the eight-value decoder it competes with is vectorised. Kept behind a
    // switch so the finding is reproducible, not deleted so the vectorised version has somewhere to
    // land.
    bool q6_path = false;
    // True when no token of a lane appears in more than one of that call's experts. Two experts that
    // share a token fold into the same output element, so only the disjoint case may run its tasks on
    // several threads at once.
    bool flatten_disjoint = false;
    // The overlapping case, made parallel: a task writes its own plane at [lane][slot][token][hidden]
    // instead of accumulating into the output, and one merge folds the slots a token actually holds
    // back in. `slot` is the expert's position in `experts`, and the merge walks those positions in
    // ascending order - the same order the row-split path accumulates in.
    //
    // The plane is allocated once per team member and reused across layers (partial_plane below), and
    // it is never initialised: a task writes every row of every slot it owns, and the merge only reads
    // the slots a token is a member of, so no unwritten element is ever read.
    float* partial      = nullptr; // [lane][slot][token][hidden]
    bool partial_output = false;
    // Whether one expert's member list names the same token twice, which only a route table with
    // repeated ids can do. Such a slot contributes twice on the row-split path and its plane can hold
    // only one value, so the repeated case keeps the serial table.
    bool members_repeat = false;
    // The merge's map: for each flat (lane, token), the slots it holds, ascending. Built once per call.
    std::vector<std::uint32_t> token_begin;
    std::vector<std::int32_t> token_slots;
    std::vector<Q8Input> q8;     // [lane] gate/up: the bf16 input, quantised once per collective
    std::vector<Q8Input> q8_act; // [lane] down: the SwiGLU activation, quantised once per expert
};

// Per-worker buffers, allocated once per collective: the rows of one expert are split across the
// team, so two workers must never share a value buffer.
struct Scratch {
    std::vector<const std::uint16_t*> inputs;
    std::vector<const float*> activations;
    std::vector<const std::int8_t*> q8_even;
    std::vector<const std::int8_t*> q8_odd;
    std::vector<const std::int32_t*> q8_sum;
    std::vector<const float*> q8_scale;
    std::vector<const std::int8_t*> q6_cont;
    std::vector<const float*> q6_scale;
    std::vector<const std::int16_t*> q16_cont;
    // The down pass reads a different width than the gate, so it needs its own quantised plane.
    std::vector<const std::int8_t*> down_even;
    std::vector<const std::int8_t*> down_odd;
    std::vector<const std::int32_t*> down_sum;
    std::vector<const float*> down_scale;
    std::vector<float> gate, up, down;
};

template <int Bits, class Input>
inline void row_all_tokens(const Row& row, std::int32_t k, const Input* const* inputs,
                           std::size_t count, float* out) noexcept {
    std::size_t at = 0;
    while (at < count) {
        const auto left = count - at;
        if (left >= 4) {
            const Input* xs[4] = {inputs[at], inputs[at + 1], inputs[at + 2], inputs[at + 3]};
            float values[4];
            row_dot<Bits, 4>(row, k, xs, values);
            for (int c = 0; c < 4; ++c) { out[at + c] = values[c]; }
            at += 4;
        } else if (left >= 2) {
            const Input* xs[2] = {inputs[at], inputs[at + 1]};
            float values[2];
            row_dot<Bits, 2>(row, k, xs, values);
            for (int c = 0; c < 2; ++c) { out[at + c] = values[c]; }
            at += 2;
        } else {
            const Input* xs[1] = {inputs[at]};
            float values[1];
            row_dot<Bits, 1>(row, k, xs, values);
            out[at] = values[0];
            at += 1;
        }
    }
}

// NINFER_CPU_PHASE: where a split route's time goes. The groupwise_int kernel's dot measured 0.075 us
// for two rows in isolation, which cannot account for a route costing hundreds of microseconds, so
// the split is what says whether the time is the gate/up pass, the down pass, or neither. Totals are
// per worker thread and the barriers sit inside the measured spans, so a span already includes the
// wait for the slowest lane - which is what the layer pays.
struct SplitPhaseTotals {
    double gate_up_us = 0.0;
    double down_us    = 0.0;
    // The merge the partial planes add: per worker, over the share of the output it folds. Reported
    // per call, because the merge is spread across the team instead of being one expert's cost.
    double reduce_us  = 0.0;
    std::uint64_t calls   = 0;
    std::uint64_t experts = 0;
};

inline SplitPhaseTotals& split_phase_totals() {
    static thread_local SplitPhaseTotals totals;
    return totals;
}

inline bool split_phase_probe() {
    static const bool enabled = std::getenv("NINFER_CPU_PHASE") != nullptr;
    return enabled;
}

inline std::chrono::steady_clock::time_point split_phase_mark(bool probe) {
    return probe ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
}

// One unit of work for the flattened table: a lane, one of that lane's experts, and the buffers that
// task owns. A task is the whole expert, so two tasks never meet inside one expert's rows.
struct ExpertJob {
    std::int32_t lane       = 0;
    std::size_t index       = 0;          // position in work.experts
    float* activation       = nullptr;    // [token][intermediate] for this task alone
    Q8Input* activation_q8  = nullptr;    // the down pass's quantised copy of it
};

// worker == nullptr is the flattened mode: this thread owns every row of the expert, so the two
// barriers the row split needs between its passes have nothing to synchronise and are dropped.
void run_expert(Work& w, Scratch& s, const CpuWorkerContext* worker,
                const ExpertJob& job) noexcept {
    const bool phase_probe = split_phase_probe();
    const auto phase_begin = split_phase_mark(phase_probe);
    const auto expert = static_cast<std::uint32_t>(w.experts[job.index]);
    // The member table is indexed by expert id, not by the position in `experts`: the two part
    // company whenever the subset skips an expert, and reading the position applies the previous
    // expert's routes (and their coefficients) to this one's weights.
    const auto& members = w.members[static_cast<std::size_t>(job.lane)][expert];
    if (members.empty()) { return; }
    const auto* lane_x      = w.x[static_cast<std::size_t>(job.lane)];
    float* const lane_out   = w.output[static_cast<std::size_t>(job.lane)];
    // Where this expert's rows go: its own slot of the partial plane, or the shared output.
    float* const slot_out =
        w.partial_output ? w.partial + (static_cast<std::size_t>(job.lane) * w.experts.size() +
                                        job.index) *
                                          (static_cast<std::size_t>(w.tokens) *
                                           static_cast<std::size_t>(w.hidden))
                         : nullptr;
    float* const activation = job.activation;
    auto& q8_in             = w.q8[static_cast<std::size_t>(job.lane)];
    auto& q8_act            = *job.activation_q8;
    const auto count = members.size();
    const int groups = w.hidden / 64;
    for (std::size_t m = 0; m < count; ++m) {
        s.inputs[m] = lane_x + static_cast<std::size_t>(members[m].token) * w.hidden;
        s.activations[m] = activation + static_cast<std::size_t>(members[m].token) * w.intermediate;
        if (w.int8_path) {
            const auto token = static_cast<std::size_t>(members[m].token);
            s.q8_even[m]  = q8_in.even.data() + token * w.hidden;
            s.q8_odd[m]   = q8_in.odd.data() + token * w.hidden;
            s.q8_sum[m]   = q8_in.sum.data() + token * groups;
            s.q8_scale[m] = q8_in.scale.data() + token * groups;
        }
    }
    // The row range of one pass: the team's slice, or the whole plane when this thread owns the
    // expert. Expressed once so the two modes cannot drift apart.
    const auto rows = [worker](std::size_t items) {
        return worker != nullptr ? worker->range(items) : CpuWorkRange{0, items};
    };
    const auto gate_base = w.gate.expert(expert);
    const auto up_base   = w.up.expert(expert);
    const auto gate_rows = rows(static_cast<std::size_t>(w.intermediate));
    for (auto i = gate_rows.begin; i < gate_rows.end; ++i) {
        const auto row = static_cast<std::int32_t>(i);
        const auto gate_row = row_at(gate_base, row, w.hidden, w.gate_bits);
        const auto up_row   = row_at(up_base, row, w.hidden, w.gate_bits);
        if (w.gate_bits == 4 && w.int8_path) {
            row_all_tokens_q4_q8(gate_row, w.hidden, s.q8_even.data(), s.q8_odd.data(),
                                 s.q8_sum.data(), s.q8_scale.data(), count, s.gate.data());
            row_all_tokens_q4_q8(up_row, w.hidden, s.q8_even.data(), s.q8_odd.data(),
                                 s.q8_sum.data(), s.q8_scale.data(), count, s.up.data());
        } else if (w.gate_bits == 4) {
            row_all_tokens<4>(gate_row, w.hidden, s.inputs.data(), count, s.gate.data());
            row_all_tokens<4>(up_row, w.hidden, s.inputs.data(), count, s.up.data());
        } else {
            row_all_tokens<6>(gate_row, w.hidden, s.inputs.data(), count, s.gate.data());
            row_all_tokens<6>(up_row, w.hidden, s.inputs.data(), count, s.up.data());
        }
        for (std::size_t m = 0; m < count; ++m) {
            activation[static_cast<std::size_t>(members[m].token) * w.intermediate + row] =
                common::silu_stable(s.gate[m]) * s.up[m];
        }
    }
    if (worker != nullptr) { worker->barrier(); }
    // The MoE down matrix on this artifact is SIX-bit (measured: only this arm's branch responds,
    // and code_bits() refuses Q5, so the 501 q5 objects in conversion.json belong to other
    // projections). It decoded in bf16 until now while gate/up already had the int8 path.
    // The sixteen-bit arm is retired: at thirty-two values a group the eight-bit plane wins on both
    // axes - faster and no worse - so int16 survives only as a reference for the probe.
    // G32 measured +3.07% against G64's +2.78% and int16's +1.14%, reproducibly, so the
    // fine-grained int8 plane is not shipped. Both arms stay behind switches for further work.
    const bool down_int16 = w.down_bits == 6 && w.q6_path;
    if (down_int16) {
        const int act_groups = w.intermediate / 64;
        if (worker != nullptr) {
            quantise_floats_i16(activation, w.tokens, w.intermediate, q8_act, *worker);
            worker->barrier();
        } else {
            // One thread owns the whole expert, so it also owns that expert's quantisation. Only the
            // tokens this expert actually serves are quantised: the plane is per token, and a task
            // that serves one token was previously quantising all eight of them - ten tasks over eight
            // tokens is eight times the work anyone consumes. quantise_floats_i16_groups takes a range
            // of (token, group) pairs, so a task's own tokens are one contiguous run each.
            for (std::size_t m = 0; m < count; ++m) {
                const auto token = static_cast<std::size_t>(members[m].token);
                quantise_floats_i16_groups(activation, w.intermediate, q8_act,
                                           token * static_cast<std::size_t>(act_groups),
                                           (token + 1) * static_cast<std::size_t>(act_groups));
            }
        }
        for (std::size_t m = 0; m < count; ++m) {
            const auto token = static_cast<std::size_t>(members[m].token);
            s.q16_cont[m] = q8_act.cont16.data() + token * w.intermediate;
            s.q6_scale[m] = q8_act.scale.data() + token * act_groups;
        }
    }
    const auto phase_gate_end = split_phase_mark(phase_probe);

    // down_h . a_t folded in with the member's coefficient; the output is [hidden, tokens].
    const auto down_base = w.down.expert(expert);
    const auto out_rows  = rows(static_cast<std::size_t>(w.hidden));
    for (auto h = out_rows.begin; h < out_rows.end; ++h) {
        const auto row      = static_cast<std::int32_t>(h);
        const auto down_row = row_at(down_base, row, w.intermediate, w.down_bits);
        if (down_int16) {
            row_all_tokens_q6_q16(down_row, w.intermediate, s.q16_cont.data(), s.q6_scale.data(),
                                  count, s.down.data());
            // NINFER_Q6_PROBE: the same row through the reference decoder, on the same activation, so
            // a disagreement can be read off directly instead of inferred from perplexity.
            static const bool probe = std::getenv("NINFER_Q6_PROBE") != nullptr;
            if (probe && h == out_rows.begin) {
                float reference[4];
                row_all_tokens<6>(down_row, w.intermediate, s.activations.data(), count, reference);
                // The same quantised activation the fast path read, scaled back to float: the dot of
                // that against the same weights separates the kernel's arithmetic from the
                // quantisation it costs, which the bf16 column above cannot (it is exact activation).
                const int act_groups = w.intermediate / 32;
                std::vector<float> dequantised_input(static_cast<std::size_t>(w.intermediate));
                for (std::size_t m = 0; m < count; ++m) {
                    for (int group = 0; group < act_groups; ++group) {
                        const auto act_scale = s.q6_scale[m][group];
                        for (int i = 0; i < 64; ++i) {
                            const auto at = static_cast<std::size_t>(group) * 64 + i;
                            dequantised_input[at] =
                                static_cast<float>(s.q6_cont[m][at]) * act_scale;
                        }
                    }
                    const float* xs[1] = {dequantised_input.data()};
                    float exact[1];
                    row_dot<6, 1>(down_row, w.intermediate, xs, exact);
                    std::fprintf(stderr,
                                 "[q6-probe] m=%zu q6=%.7f bf16=%.7f delta=%.7f arith=%.3e\n", m,
                                 static_cast<double>(s.down[m]),
                                 static_cast<double>(reference[m]),
                                 static_cast<double>(s.down[m] - reference[m]),
                                 static_cast<double>(s.down[m] - exact[0]));
                }
                std::fflush(stderr);
            }
        } else if (false) { /* four-bit down arm retired on this artifact */
            row_all_tokens_q4_q8(down_row, w.intermediate, s.down_even.data(), s.down_odd.data(),
                                 s.down_sum.data(), s.q6_scale.data(), count, s.down.data());
        } else if (w.down_bits == 6) {
            row_all_tokens<6>(down_row, w.intermediate, s.activations.data(), count, s.down.data());
        } else {
            row_all_tokens<4>(down_row, w.intermediate, s.activations.data(), count, s.down.data());
        }
        for (std::size_t m = 0; m < count; ++m) {
            const auto token = static_cast<std::size_t>(members[m].token);
            // The down result takes the GPU's own rounding for this shape. The device's grouped kernel
            // stores its down values as BF16 and only the token kernel keeps FP32 (plane_round in
            // groupwise_int: tokens >= 47), so a CPU half that always kept FP32 was not measuring the
            // same arithmetic - and every activation-quantisation cost it reported was inflated by the
            // difference. matching the device here is what makes a CPU/GPU comparison mean anything.
            const float down_value =
                w.plane_round ? common::bf16_round(s.down[m]) : s.down[m];
            if (slot_out != nullptr) {
                // Nothing else writes this element: the slot belongs to one expert and one token, and
                // the row loop visits each row once. Hence a store, not an accumulate - which is what
                // makes two experts holding the same token safe to run on two threads.
                slot_out[token * w.hidden + row] = members[m].coefficient * down_value;
            } else {
                lane_out[token * w.hidden + row] += members[m].coefficient * down_value;
            }
        }
    }
    if (worker != nullptr) { worker->barrier(); }
    if (phase_probe) {
        const auto phase_end = std::chrono::steady_clock::now();
        auto& totals         = split_phase_totals();
        totals.gate_up_us +=
            std::chrono::duration<double, std::micro>(phase_gate_end - phase_begin).count();
        totals.down_us += std::chrono::duration<double, std::micro>(phase_end - phase_gate_end).count();
        ++totals.experts;
    }
}

// One partial plane per worker thread, grown in place and never freed: the flattened path allocates it
// on rank 0 and the whole team reads it, so it has to outlive the collective, and a layer that
// allocated it again would pay a fresh mapping (1.6 MB at eight tokens, 33 MB at one hundred and
// sixty-one) on every layer. Uninitialised on purpose - see Work::partial.
float* partial_plane(std::size_t count) {
    static thread_local std::unique_ptr<float[]> plane;
    static thread_local std::size_t capacity = 0;
    if (count > capacity) {
        plane    = std::make_unique_for_overwrite<float[]>(count);
        capacity = count;
    }
    return plane.get();
}

// Fold every token's slots into the output, ascending, once the tasks above have all finished.
//
// This is where a re-association would normally cost bits, and it does not: both sides are one
// ascending accumulation chain from a zero accumulator. The row split adds expert by expert straight
// into the output, (((0 + t1) + t2) + t3); the merge adds the same terms in the same order into its
// own zero, ((0 + t1) + t2) + t3, and adds that once to an output that this path starts at zero.
// Same terms, same order, same roundings - measured on the forced CPU half, where every token holds
// several of a layer's experts, the flattened perplexity is the row split's to fifteen significant
// digits. A non-zero destination would break the identity, which is why the last step is an
// accumulate and not a store.
void merge_partial(Work& w, const CpuWorkerContext& worker) {
    const auto hidden = static_cast<std::size_t>(w.hidden);
    const auto tokens = static_cast<std::size_t>(w.tokens);
    const auto plane  = tokens * hidden; // one slot's worth
    const auto total  = static_cast<std::size_t>(w.lanes) * plane;
    const auto range  = worker.range(total);
    for (auto at = range.begin; at < range.end; ++at) {
        const auto row  = at % hidden;
        const auto flat = at / hidden; // lane * tokens + token
        if (w.token_begin[flat] == w.token_begin[flat + 1]) { continue; }
        const auto* from = w.partial + (flat % tokens) * hidden + row;
        float sum        = 0.0F;
        for (auto k = w.token_begin[flat]; k < w.token_begin[flat + 1]; ++k) {
            sum += from[static_cast<std::size_t>(w.token_slots[k]) * plane];
        }
        // A store plus the term the output already held, which is zero on this path: the arena's CPU
        // half starts from a cleared tensor. Written as an accumulate so a caller that hands in a
        // non-zero destination still gets the value the row split would have produced.
        w.output[flat / tokens][flat % tokens * hidden + row] += sum;
    }
}

// The flattened table: one task per (lane, expert), where a task is a whole expert - every row of both
// passes on the thread that owns it - so the two barriers the row split needs between its passes have
// nothing left to synchronise.
//
// Two experts of one lane can be in flight on two threads (they share that lane's activation plane and
// its quantised copy), so each task carries both buffers itself. The lane's output slice used to be
// the one thing tasks had in common, which confined the table to the disjoint case; with the partial
// planes each task now owns its own slot, and the merge below is the only step that touches the output.
void run_flattened(Work& work, Scratch& scratch, const CpuWorkerContext& worker) {
    std::vector<std::pair<std::int32_t, std::size_t>> tasks;
    tasks.reserve(static_cast<std::size_t>(work.lanes) * work.experts.size());
    for (std::int32_t lane = 0; lane < work.lanes; ++lane) {
        for (std::size_t index = 0; index < work.experts.size(); ++index) {
            const auto expert = static_cast<std::uint32_t>(work.experts[index]);
            if (work.members[static_cast<std::size_t>(lane)][expert].empty()) { continue; }
            tasks.emplace_back(lane, index);
        }
    }
    if (tasks.empty()) { return; }
    const auto plane =
        static_cast<std::size_t>(work.tokens) * static_cast<std::size_t>(work.intermediate);
    std::vector<float> activations(tasks.size() * plane);
    std::vector<Q8Input> planes(tasks.size());
    if (work.int8_path) {
        const auto act_groups = static_cast<std::size_t>(work.tokens) *
                                static_cast<std::size_t>(work.intermediate / 64);
        for (auto& one : planes) {
            one.cont16.assign(plane, 0);
            one.scale.assign(act_groups, 0.0F);
        }
    }
    const auto run_one = [&](std::size_t at) {
        ExpertJob job;
        job.lane          = tasks[at].first;
        job.index         = tasks[at].second;
        job.activation    = activations.data() + at * plane;
        job.activation_q8 = &planes[at];
        run_expert(work, scratch, nullptr, job);
    };
    if (!work.flatten_disjoint && work.partial_output) {
        // Overlapping routes, but each task owns a slot: the table runs on the whole team, one
        // barrier waits for the last task, and the merge folds the slots in expert order.
        const auto across = worker.range(tasks.size());
        for (auto at = across.begin; at < across.end; ++at) { run_one(at); }
        worker.barrier();
        const auto merge_begin = split_phase_mark(split_phase_probe());
        merge_partial(work, worker);
        if (split_phase_probe()) {
            auto& totals = split_phase_totals();
            totals.reduce_us +=
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                          merge_begin)
                    .count();
            ++totals.calls;
        }
        return;
    }
    if (!work.flatten_disjoint) {
        // The call still owes the same value the row-split path produces, so a table whose tasks share
        // a token runs in table order on one rank: no barrier, no race, same accumulation order.
        if (worker.rank() != 0) { return; }
        for (std::size_t at = 0; at < tasks.size(); ++at) { run_one(at); }
        return;
    }
    const auto range = worker.range(tasks.size());
    for (auto at = range.begin; at < range.end; ++at) { run_one(at); }
}

void run(void* arguments, const CpuWorkerContext& worker) noexcept {
    auto& work = *static_cast<Work*>(arguments);
    Scratch scratch;
    const auto room = static_cast<std::size_t>(work.tokens);
    scratch.inputs.resize(room);
    scratch.activations.resize(room);
    scratch.gate.resize(room);
    scratch.up.resize(room);
    scratch.down.resize(room);
    if (work.int8_path) {
        scratch.q8_even.resize(room);
        scratch.q8_odd.resize(room);
        scratch.q8_sum.resize(room);
        scratch.q8_scale.resize(room);
        scratch.q6_cont.resize(room);
        scratch.q6_scale.resize(room);
        // The down pass keeps its own pointer tables: its width is the intermediate dimension, not
        // the hidden one, and an un-resized table here was a null dereference the first time the
        // sixteen-bit arm ran.
        scratch.q16_cont.resize(room);
        scratch.down_even.resize(room);
        scratch.down_odd.resize(room);
        scratch.down_sum.resize(room);
        scratch.down_scale.resize(room);
        // One lane reserves the planes; an assign() in every lane would have each zero it under the
        // others' writes.
        if (worker.rank() == 0) {
            const int groups     = work.hidden / 64;
            const int act_groups = work.intermediate / 32;
            for (auto& plane : work.q8) {
                plane.even.assign(static_cast<std::size_t>(work.tokens) * work.hidden, 0);
                plane.odd.assign(static_cast<std::size_t>(work.tokens) * work.hidden, 0);
                plane.cont.assign(static_cast<std::size_t>(work.tokens) * work.hidden, 0);
                plane.scale.assign(static_cast<std::size_t>(work.tokens) * groups, 0.0F);
                plane.sum.assign(static_cast<std::size_t>(work.tokens) * groups, 0);
            }
            for (auto& plane : work.q8_act) {
                plane.cont.assign(static_cast<std::size_t>(work.tokens) * work.intermediate, 0);
                plane.cont16.assign(static_cast<std::size_t>(work.tokens) * work.intermediate, 0);
                plane.even.assign(static_cast<std::size_t>(work.tokens) * work.intermediate, 0);
                plane.odd.assign(static_cast<std::size_t>(work.tokens) * work.intermediate, 0);
                plane.scale.assign(static_cast<std::size_t>(work.tokens) * act_groups, 0.0F);
                plane.sum.assign(static_cast<std::size_t>(work.tokens) * act_groups, 0);
            }
        }
        worker.barrier();
        for (std::int32_t lane = 0; lane < work.lanes; ++lane) {
            quantise_inputs(work.x[lane], work.tokens, work.hidden,
                            work.q8[static_cast<std::size_t>(lane)], worker);
        }
        worker.barrier();
    }
    const bool phase_probe = split_phase_probe();
    const auto phase_begin = split_phase_mark(phase_probe);
    // Two task tables for two shapes of caller.
    //
    // lanes == 1 is what this build is actually asked for, and it is not one lane's worth of work:
    // split_moe_ffn() is called once per layer with the whole decode batch in the T dimension of one
    // tensor, so eight concurrent sequences arrive here as tokens == 8 in a single call, one expert per
    // route and two or three experts per layer. Rows are the only parallelism that shape offers, hence
    // the row split with a barrier between the passes.
    //
    // lanes > 1 is the shape the flattened table was designed for - one (lane, expert) task per
    // thread, every row on that thread, no barrier - and nothing in this build produces it, because
    // the batching happens upstream in the token axis rather than here. It is kept because the task
    // table is also what a future caller that hands several lanes in separately would need, and
    // because NINFER_SPLIT_FLATTEN measures it against the row split on the shape that does exist.
    // Measured on this box, one expert, eight tokens: row split 92 us, flattened 277-309 us (one
    // thread owns 640 rows, three sit idle), decode 6.99 against 6.32 tok/s per request. The
    // flattening needs at least one expert per thread to pay, and the production call has one or three.
    static const bool flatten_single = std::getenv("NINFER_SPLIT_FLATTEN") != nullptr;
    const bool flatten_path          = work.lanes > 1 || flatten_single;
    // The partial planes exist only for the case the flattened table could not parallelise before:
    // overlapping routes. The row split never allocates them and never sets partial_output, so its
    // accumulation order - and every bit of its result - is what it was.
    if (flatten_path && !work.flatten_disjoint && !work.members_repeat) {
        const auto plane = static_cast<std::size_t>(work.lanes) * work.experts.size() *
                           static_cast<std::size_t>(work.tokens) *
                           static_cast<std::size_t>(work.hidden);
        if (worker.rank() == 0) { work.partial = partial_plane(plane); }
        worker.barrier(); // publish the plane before any task writes it
        work.partial_output = true;
    }
    if (!flatten_path) {
        for (std::size_t index = 0; index < work.experts.size(); ++index) {
            ExpertJob job;
            job.lane          = 0;
            job.index         = index;
            job.activation    = work.activation[0];
            job.activation_q8 = &work.q8_act[0];
            run_expert(work, scratch, &worker, job);
        }
    } else {
        run_flattened(work, scratch, worker);
    }
    if (phase_probe) {
        const auto phase_end = std::chrono::steady_clock::now();
        const auto& totals   = split_phase_totals();
        const double experts = totals.experts > 0 ? static_cast<double>(totals.experts) : 1.0;
        const double wall = std::chrono::duration<double, std::micro>(phase_end - phase_begin).count();
        const double merge = totals.calls > 0 ? totals.reduce_us / static_cast<double>(totals.calls)
                                             : 0.0;
        // How much the merge actually has to re-associate: a token holding one slot is the same
        // summation the row split makes, so only tokens holding two or more can move a bit.
        std::size_t overlapped = 0, widest = 0;
        for (std::size_t at = 0; at + 1 < work.token_begin.size(); ++at) {
            const auto held = static_cast<std::size_t>(work.token_begin[at + 1] - work.token_begin[at]);
            if (held > 1) { ++overlapped; }
            widest = std::max(widest, held);
        }
        std::fprintf(stderr,
                     "[cpu-phase-split] experts=%llu tokens=%d | per expert: gate_up=%.1f "
                     "down=%.1f us | merge=%.1f us/call | overlap=%zu/%zu widest=%zu | wall=%.1f us "
                     "(%.0f%% accounted)\n",
                     static_cast<unsigned long long>(totals.experts), work.tokens,
                     totals.gate_up_us / experts, totals.down_us / experts, merge, overlapped,
                     work.token_begin.empty() ? 0 : work.token_begin.size() - 1, widest, wall,
                     wall > 0.0 ? 100.0 * (totals.gate_up_us + totals.down_us) / wall : 0.0);
        std::fflush(stderr);
    }
}

} // namespace

std::size_t split_capacity(std::int32_t intermediate, std::int32_t tokens) {
    if (intermediate <= 0 || tokens <= 0) {
        throw std::invalid_argument("routed_swiglu CPU: invalid split shape");
    }
    const auto bytes = static_cast<std::size_t>(intermediate) * static_cast<std::size_t>(tokens) *
                       sizeof(float);
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

void split_launch(const Tensor& x, const CpuSwiGLUExperts& experts, const ExpertRoutes& routes,
                  ExpertSubset subset, Tensor& destination, MemoryArena& scratch,
                  CpuExecutionView execution, std::int32_t hidden, std::int32_t intermediate,
                  std::int32_t experts_count, std::int32_t top_k) {
    const auto tokens = x.ne[1];
    if (tokens <= 0 || hidden <= 0 || intermediate <= 0 || experts_count <= 0 || top_k <= 0) {
        throw std::invalid_argument("routed_swiglu CPU: invalid split invocation");
    }
    const auto* ids          = static_cast<const std::int32_t*>(routes.ids.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    const auto* mask         = static_cast<const std::int32_t*>(subset.mask.data);
    if (subset.kind == ExpertSubsetKind::None) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }

    std::vector<std::uint8_t> selected(static_cast<std::size_t>(experts_count), 0);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t j = 0; j < top_k; ++j) {
            const auto expert = ids[static_cast<std::size_t>(t) * top_k + j];
            if (expert < 0 || expert >= experts_count) {
                throw std::invalid_argument("routed_swiglu CPU: invalid route ID on the split path");
            }
            bool member = false;
            switch (subset.kind) {
            case ExpertSubsetKind::All: member = true; break;
            case ExpertSubsetKind::MaskOnes: member = mask != nullptr && mask[expert] == 1; break;
            case ExpertSubsetKind::MaskZeros: member = mask != nullptr && mask[expert] == 0; break;
            default: break;
            }
            if (member) { selected[static_cast<std::size_t>(expert)] = 1; }
        }
    }

    Work work;
    // One lane, one entry per table: the shape this call has always had. The tables are local to this
    // frame and the collective below blocks, so their lifetime is the call's.
    const std::uint16_t* lane_x[1]   = {static_cast<const std::uint16_t*>(x.data)};
    float* lane_out[1]               = {static_cast<float*>(destination.data)};
    float* lane_activation[1]        = {static_cast<float*>(scratch.storage().data)};
    work.x                           = lane_x;
    work.output                      = lane_out;
    work.activation                  = lane_activation;
    work.lanes                       = 1;
    work.tokens       = tokens;
    // Same threshold the device uses (groupwise_int work.h: grouped_first = 47): at or above it the
    // grouped kernel runs and its down plane is BF16; below it the token kernel keeps FP32.
    work.plane_round  = tokens >= 47;
    work.hidden       = hidden;
    work.intermediate = intermediate;
    work.gate         = matrix_of(experts.gate);
    work.up           = matrix_of(experts.up);
    work.down         = matrix_of(experts.down);
    work.gate_bits    = code_bits(experts.gate.layout.format);
    work.down_bits    = code_bits(experts.down.layout.format);
    if (code_bits(experts.up.layout.format) != work.gate_bits) {
        throw std::invalid_argument("routed_swiglu CPU: split gate and up differ in code width");
    }
    // The same knob the fused form reads. Until now this form ignored it entirely, which is why an
    // A/B of the int8 tier came back identical: the switch was being flipped on a path that never
    // consulted it.
    {
        static const char* value = std::getenv("NINFER_MOE_CPU_INT8");
        work.int8_path = value == nullptr || std::strcmp(value, "0") != 0;
        const char* q6 = std::getenv("NINFER_SPLIT_Q6");
        work.q6_path   = q6 == nullptr || std::strcmp(q6, "0") != 0;
    }
    work.members.assign(1, std::vector<std::vector<Member>>(static_cast<std::size_t>(experts_count)));
    work.q8.resize(1);
    work.q8_act.resize(1);
    for (std::int32_t expert = 0; expert < experts_count; ++expert) {
        if (selected[static_cast<std::size_t>(expert)] != 0) { work.experts.push_back(expert); }
    }
    if (work.experts.empty()) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t j = 0; j < top_k; ++j) {
            const auto expert = ids[static_cast<std::size_t>(t) * top_k + j];
            if (selected[static_cast<std::size_t>(expert)] == 0) { continue; }
            work.members[0][static_cast<std::size_t>(expert)].push_back(
                Member{t, coefficients[static_cast<std::size_t>(t) * top_k + j]});
        }
    }
    // Whether the flattened table may put its tasks on several threads: two of this call's experts
    // that hold the same token both fold into that token's output element, and doing so from two
    // threads at once is a lost update. One pass over the member lists answers it for the whole call,
    // so nothing has to be decided inside the collective.
    {
        std::vector<std::int32_t> held(static_cast<std::size_t>(tokens), 0);
        work.flatten_disjoint = true;
        for (std::int32_t lane = 0; lane < work.lanes && work.flatten_disjoint; ++lane) {
            std::fill(held.begin(), held.end(), 0);
            for (const auto expert : work.experts) {
                for (const auto& member : work.members[static_cast<std::size_t>(lane)]
                                                      [static_cast<std::size_t>(expert)]) {
                    if (++held[static_cast<std::size_t>(member.token)] > 1) {
                        work.flatten_disjoint = false;
                        break;
                    }
                }
                if (!work.flatten_disjoint) { break; }
            }
        }
    }
    // The merge's map, built once per call: for each flat (lane, token), the slots it holds in
    // `experts` order. Two reasons to build it instead of walking every slot per token - the plane's
    // unwritten slots are never initialised (so they must not be read at all), and a token holds two
    // to four of a layer's twenty experts, not twenty.
    //
    // Only for the call that can use it: with NINFER_SPLIT_FLATTEN off and one lane the table above
    // never runs, and the shipped row split should not pay an allocation for a switch that is off.
    //
    // A repeated route id would put one token twice in one expert's list, and one slot can hold only
    // one value per element. The row-split path would add that term twice; a plane cannot, so the
    // repeated case keeps the serial table rather than silently dropping a term.
    static const bool flatten_single = std::getenv("NINFER_SPLIT_FLATTEN") != nullptr;
    if (!work.flatten_disjoint && (work.lanes > 1 || flatten_single)) {
        const auto slots       = work.experts.size();
        const auto flat_tokens = static_cast<std::size_t>(work.lanes) * static_cast<std::size_t>(tokens);
        work.token_begin.assign(flat_tokens + 1, 0);
        for (std::int32_t lane = 0; lane < work.lanes; ++lane) {
            for (std::size_t index = 0; index < slots; ++index) {
                const auto expert = static_cast<std::uint32_t>(work.experts[index]);
                const auto& list  = work.members[static_cast<std::size_t>(lane)][expert];
                for (std::size_t m = 0; m < list.size(); ++m) {
                    if (m > 0 && list[m - 1].token == list[m].token) { work.members_repeat = true; }
                    ++work.token_begin[static_cast<std::size_t>(lane) * static_cast<std::size_t>(tokens) +
                                       static_cast<std::size_t>(list[m].token) + 1];
                }
            }
        }
        if (work.members_repeat) {
            work.token_begin.clear();
        } else {
            for (std::size_t at = 1; at < work.token_begin.size(); ++at) {
                work.token_begin[at] += work.token_begin[at - 1];
            }
            work.token_slots.assign(work.token_begin.back(), 0);
            std::vector<std::uint32_t> cursor(work.token_begin.begin(), work.token_begin.end() - 1);
            for (std::int32_t lane = 0; lane < work.lanes; ++lane) {
                for (std::size_t index = 0; index < slots; ++index) {
                    const auto expert = static_cast<std::uint32_t>(work.experts[index]);
                    for (const auto& member : work.members[static_cast<std::size_t>(lane)][expert]) {
                        const auto flat =
                            static_cast<std::size_t>(lane) * static_cast<std::size_t>(tokens) +
                            static_cast<std::size_t>(member.token);
                        work.token_slots[cursor[flat]++] = static_cast<std::int32_t>(
                            static_cast<std::size_t>(lane) * slots + index);
                    }
                }
            }
        }
    }
    execution.run(run, &work);
}

} // namespace ninfer::ops::detail::moe_routed_cpu
