#pragma once

// Host copies of the E8-cylinder constant tables in ops/kernel/e8_root_codec.cuh. The Op family
// reads them from device constant memory, so a host reference cannot include that header; this
// file mirrors the three tables verbatim (regenerate if the device tables change) plus the decode
// the kernels apply: out[i] = round((root[i] + axis[i]) * radius), with the value plane using the
// calibrated Type-B radius centre.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ninfer::test::qsa::e8 {

inline constexpr std::uint64_t kRoot[256] = {
    0x000000000000fcfcULL, 0x00000000000004fcULL, 0x000000000000fc04ULL, 0x0000000000000404ULL,
    0x0000000000fc00fcULL, 0x00000000000400fcULL, 0x0000000000fc0004ULL, 0x0000000000040004ULL,
    0x00000000fc0000fcULL, 0x00000000040000fcULL, 0x00000000fc000004ULL, 0x0000000004000004ULL,
    0x000000fc000000fcULL, 0x00000004000000fcULL, 0x000000fc00000004ULL, 0x0000000400000004ULL,
    0x0000fc00000000fcULL, 0x00000400000000fcULL, 0x0000fc0000000004ULL, 0x0000040000000004ULL,
    0x00fc0000000000fcULL, 0x00040000000000fcULL, 0x00fc000000000004ULL, 0x0004000000000004ULL,
    0xfc000000000000fcULL, 0x04000000000000fcULL, 0xfc00000000000004ULL, 0x0400000000000004ULL,
    0x0000000000fcfc00ULL, 0x000000000004fc00ULL, 0x0000000000fc0400ULL, 0x0000000000040400ULL,
    0x00000000fc00fc00ULL, 0x000000000400fc00ULL, 0x00000000fc000400ULL, 0x0000000004000400ULL,
    0x000000fc0000fc00ULL, 0x000000040000fc00ULL, 0x000000fc00000400ULL, 0x0000000400000400ULL,
    0x0000fc000000fc00ULL, 0x000004000000fc00ULL, 0x0000fc0000000400ULL, 0x0000040000000400ULL,
    0x00fc00000000fc00ULL, 0x000400000000fc00ULL, 0x00fc000000000400ULL, 0x0004000000000400ULL,
    0xfc0000000000fc00ULL, 0x040000000000fc00ULL, 0xfc00000000000400ULL, 0x0400000000000400ULL,
    0x00000000fcfc0000ULL, 0x0000000004fc0000ULL, 0x00000000fc040000ULL, 0x0000000004040000ULL,
    0x000000fc00fc0000ULL, 0x0000000400fc0000ULL, 0x000000fc00040000ULL, 0x0000000400040000ULL,
    0x0000fc0000fc0000ULL, 0x0000040000fc0000ULL, 0x0000fc0000040000ULL, 0x0000040000040000ULL,
    0x00fc000000fc0000ULL, 0x0004000000fc0000ULL, 0x00fc000000040000ULL, 0x0004000000040000ULL,
    0xfc00000000fc0000ULL, 0x0400000000fc0000ULL, 0xfc00000000040000ULL, 0x0400000000040000ULL,
    0x000000fcfc000000ULL, 0x00000004fc000000ULL, 0x000000fc04000000ULL, 0x0000000404000000ULL,
    0x0000fc00fc000000ULL, 0x00000400fc000000ULL, 0x0000fc0004000000ULL, 0x0000040004000000ULL,
    0x00fc0000fc000000ULL, 0x00040000fc000000ULL, 0x00fc000004000000ULL, 0x0004000004000000ULL,
    0xfc000000fc000000ULL, 0x04000000fc000000ULL, 0xfc00000004000000ULL, 0x0400000004000000ULL,
    0x0000fcfc00000000ULL, 0x000004fc00000000ULL, 0x0000fc0400000000ULL, 0x0000040400000000ULL,
    0x00fc00fc00000000ULL, 0x000400fc00000000ULL, 0x00fc000400000000ULL, 0x0004000400000000ULL,
    0xfc0000fc00000000ULL, 0x040000fc00000000ULL, 0xfc00000400000000ULL, 0x0400000400000000ULL,
    0x00fcfc0000000000ULL, 0x0004fc0000000000ULL, 0x00fc040000000000ULL, 0x0004040000000000ULL,
    0xfc00fc0000000000ULL, 0x0400fc0000000000ULL, 0xfc00040000000000ULL, 0x0400040000000000ULL,
    0xfcfc000000000000ULL, 0x04fc000000000000ULL, 0xfc04000000000000ULL, 0x0404000000000000ULL,
    0xfefefefefefefefeULL, 0x02fefefefefefe02ULL, 0x02fefefefefe02feULL, 0xfefefefefefe0202ULL,
    0x02fefefefe02fefeULL, 0xfefefefefe02fe02ULL, 0xfefefefefe0202feULL, 0x02fefefefe020202ULL,
    0x02fefefe02fefefeULL, 0xfefefefe02fefe02ULL, 0xfefefefe02fe02feULL, 0x02fefefe02fe0202ULL,
    0xfefefefe0202fefeULL, 0x02fefefe0202fe02ULL, 0x02fefefe020202feULL, 0xfefefefe02020202ULL,
    0x02fefe02fefefefeULL, 0xfefefe02fefefe02ULL, 0xfefefe02fefe02feULL, 0x02fefe02fefe0202ULL,
    0xfefefe02fe02fefeULL, 0x02fefe02fe02fe02ULL, 0x02fefe02fe0202feULL, 0xfefefe02fe020202ULL,
    0xfefefe0202fefefeULL, 0x02fefe0202fefe02ULL, 0x02fefe0202fe02feULL, 0xfefefe0202fe0202ULL,
    0x02fefe020202fefeULL, 0xfefefe020202fe02ULL, 0xfefefe02020202feULL, 0x02fefe0202020202ULL,
    0x02fe02fefefefefeULL, 0xfefe02fefefefe02ULL, 0xfefe02fefefe02feULL, 0x02fe02fefefe0202ULL,
    0xfefe02fefe02fefeULL, 0x02fe02fefe02fe02ULL, 0x02fe02fefe0202feULL, 0xfefe02fefe020202ULL,
    0xfefe02fe02fefefeULL, 0x02fe02fe02fefe02ULL, 0x02fe02fe02fe02feULL, 0xfefe02fe02fe0202ULL,
    0x02fe02fe0202fefeULL, 0xfefe02fe0202fe02ULL, 0xfefe02fe020202feULL, 0x02fe02fe02020202ULL,
    0xfefe0202fefefefeULL, 0x02fe0202fefefe02ULL, 0x02fe0202fefe02feULL, 0xfefe0202fefe0202ULL,
    0x02fe0202fe02fefeULL, 0xfefe0202fe02fe02ULL, 0xfefe0202fe0202feULL, 0x02fe0202fe020202ULL,
    0x02fe020202fefefeULL, 0xfefe020202fefe02ULL, 0xfefe020202fe02feULL, 0x02fe020202fe0202ULL,
    0xfefe02020202fefeULL, 0x02fe02020202fe02ULL, 0x02fe0202020202feULL, 0xfefe020202020202ULL,
    0x0202fefefefefefeULL, 0xfe02fefefefefe02ULL, 0xfe02fefefefe02feULL, 0x0202fefefefe0202ULL,
    0xfe02fefefe02fefeULL, 0x0202fefefe02fe02ULL, 0x0202fefefe0202feULL, 0xfe02fefefe020202ULL,
    0xfe02fefe02fefefeULL, 0x0202fefe02fefe02ULL, 0x0202fefe02fe02feULL, 0xfe02fefe02fe0202ULL,
    0x0202fefe0202fefeULL, 0xfe02fefe0202fe02ULL, 0xfe02fefe020202feULL, 0x0202fefe02020202ULL,
    0xfe02fe02fefefefeULL, 0x0202fe02fefefe02ULL, 0x0202fe02fefe02feULL, 0xfe02fe02fefe0202ULL,
    0x0202fe02fe02fefeULL, 0xfe02fe02fe02fe02ULL, 0xfe02fe02fe0202feULL, 0x0202fe02fe020202ULL,
    0x0202fe0202fefefeULL, 0xfe02fe0202fefe02ULL, 0xfe02fe0202fe02feULL, 0x0202fe0202fe0202ULL,
    0xfe02fe020202fefeULL, 0x0202fe020202fe02ULL, 0x0202fe02020202feULL, 0xfe02fe0202020202ULL,
    0xfe0202fefefefefeULL, 0x020202fefefefe02ULL, 0x020202fefefe02feULL, 0xfe0202fefefe0202ULL,
    0x020202fefe02fefeULL, 0xfe0202fefe02fe02ULL, 0xfe0202fefe0202feULL, 0x020202fefe020202ULL,
    0x020202fe02fefefeULL, 0xfe0202fe02fefe02ULL, 0xfe0202fe02fe02feULL, 0x020202fe02fe0202ULL,
    0xfe0202fe0202fefeULL, 0x020202fe0202fe02ULL, 0x020202fe020202feULL, 0xfe0202fe02020202ULL,
    0x02020202fefefefeULL, 0xfe020202fefefe02ULL, 0xfe020202fefe02feULL, 0x02020202fefe0202ULL,
    0xfe020202fe02fefeULL, 0x02020202fe02fe02ULL, 0x02020202fe0202feULL, 0xfe020202fe020202ULL,
    0xfe02020202fefefeULL, 0x0202020202fefe02ULL, 0x0202020202fe02feULL, 0xfe02020202fe0202ULL,
    0x020202020202fefeULL, 0xfe0202020202fe02ULL, 0xfe020202020202feULL, 0x0202020202020202ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
};

inline constexpr std::uint64_t kAxis[16] = {
    0x0000000000000001ULL, 0x00000000000000ffULL, 0x0000000000000100ULL, 0x000000000000ff00ULL,
    0x0000000000010000ULL, 0x0000000000ff0000ULL, 0x0000000001000000ULL, 0x00000000ff000000ULL,
    0x0000000100000000ULL, 0x000000ff00000000ULL, 0x0000010000000000ULL, 0x0000ff0000000000ULL,
    0x0001000000000000ULL, 0x00ff000000000000ULL, 0x0100000000000000ULL, 0xff00000000000000ULL,
};

inline constexpr float kRadius[16] = {
    0.0000f, 0.0992f, 0.1250f, 0.1575f,
    0.1984f, 0.2500f, 0.3150f, 0.3969f,
    0.5000f, 0.6300f, 0.7937f, 1.0000f,
    1.2599f, 1.5874f, 2.0000f, 2.5198f,
};

inline constexpr float kValueRadiusCentreB = 1.115f;

// One eight-dim subspace of the E8-cylinder carrier.
struct EncodeDiag {
    float a_score = 0.0f, b_score = 0.0f, norm = 0.0f;
    int type = 0, best_dim = 0, first = 0, second = 0, radius = 0;
    std::uint8_t root = 0, radial = 0;
    std::array<float, 8> res{};
};

inline void decode_8d(std::uint8_t root_code, std::uint8_t radial, bool value_plane,
                      std::array<int, 8>& decoded) {
    const int radius_index = radial >> 4, axis_index = radial & 0x0F;
    if (radius_index == 0) {
        decoded.fill(0);
        return;
    }
    const float scale = kRadius[radius_index] *
                        (!value_plane ? 1.0f
                                      : (root_code < 112 ? 1.0f : kValueRadiusCentreB));
    const std::uint64_t root = kRoot[root_code], axis = kAxis[axis_index];
    for (int i = 0; i < 8; ++i) {
        const int r = int(static_cast<std::int8_t>((root >> (8 * i)) & 0xFF));
        const int a = int(static_cast<std::int8_t>((axis >> (8 * i)) & 0xFF));
        decoded[i] = int(std::nearbyint(float(r + a) * scale));
    }
}

// The subspace quantizer, one lane of the device's warp form. Every reduction below reproduces the
// device's butterfly order exactly - `e8_encode_cylinder_8d_warp` reduces the squared norm, the
// absolute sum, the primary-axis projection, the residual maximum and the type-B bit mask with XOR
// masks 4,2,1, and the top-two selection with 1,2,4 - because a different order changes the last
// float bits and can select a different radius index or root at a boundary.
//
// STATUS (2026-11-11): the K side of the dyadic `exact_key_encoding` fixture is bit-identical
// (1024/1024 subspace codes), and the earlier 13/1024 residue was the residual contraction, fixed
// by the `std::fma` below. What is left is the V side: with `store_rk_value` -> `store_rk_e8<false>`
// wired through the three carriers, `norm_alignment_and_phase` still reports one radial byte per
// fixture (device 0xD7 vs host 0xD0 at row 199/head 1/group 0/subspace 0 of the 321-token case).
// The device dump (`NINFER_QSA_E8_DUMP`) shows the same `absmax` (8), the same `step` (1.14257812),
// the same radius index (13) and the same root (52), so the absmax scope and the scale landing
// order are both ruled out; only the residual-axis argmax differs, and the host's own `res` has no
// tie there (|res[0]| = 0.355 against |res[3]| = 0.211), so a near-tie re-selection is ruled out
// too. Locating it needs the device's own `res` for that subspace, which the printf buffer drops at
// the dump's volume; the carriers therefore stay out of `test_prepare`'s storage list.
inline void encode_8d(const std::array<float, 8>& value, float ks, std::uint8_t& out_root,
                      std::uint8_t& out_radial, EncodeDiag* diag = nullptr) {
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    // A lane reads its partner's pre-step value, so each butterfly stage writes a fresh array.
    const auto butterfly_sum = [](const std::array<float, 8>& start) {
        std::array<float, 8> v = start;
        for (int mask = 4; mask > 0; mask >>= 1) {
            std::array<float, 8> next{};
            for (int l = 0; l < 8; ++l) next[l] = v[l] + v[l ^ mask];
            v = next;
        }
        return v;
    };

    std::array<float, 8> squares{};
    for (int l = 0; l < 8; ++l) squares[l] = value[l] * value[l];
    const float norm = std::sqrt(butterfly_sum(squares)[0]);

    const float relative = norm / (ks * 2.82842712474619f + 1e-8f);
    int radius_index     = 0;
    if (relative >= 0.08f) {
        const float log_value = 3.0f * (std::log(relative) * 1.4426950408889634f) + 8.0f;
        const int rounded     = static_cast<int>(std::rint(log_value));
        radius_index          = rounded < 1 ? 1 : (rounded > 15 ? 15 : rounded);
    }

    const float inverse = 1.0f / (norm + 1e-8f);
    std::array<float, 8> unit{}, magnitude{}, sign{};
    for (int l = 0; l < 8; ++l) {
        unit[l]      = value[l] * inverse;
        magnitude[l] = std::fabs(unit[l]);
        sign[l]      = unit[l] >= 0.0f ? 1.0f : -1.0f;
    }

    // Type A: the two largest coordinates, selected by mask 1,2,4 with an index tie-break.
    std::array<float, 8> top1 = magnitude;
    std::array<float, 8> top2;
    top2.fill(-1.0f);
    std::array<int, 8> top1_index{}, top2_index{};
    for (int l = 0; l < 8; ++l) {
        top1_index[l] = l;
        top2_index[l] = -1;
    }
    for (int mask = 1; mask <= 4; mask <<= 1) {
        std::array<float, 8> next1 = top1, next2 = top2;
        std::array<int, 8> next1i = top1_index, next2i = top2_index;
        for (int l = 0; l < 8; ++l) {
            const int p           = l ^ mask;
            const float other1    = top1[p];
            const int other1i     = top1_index[p];
            const float other2    = top2[p];
            const int other2i     = top2_index[p];
            // The device breaks every tie by value then by index, which is a strict total order and
            // therefore converges across lanes; the second place is the best of the losing first
            // place and both second places. A plain `>` on the second place would let two lanes hold
            // different pairs when two coordinates are exactly equal, and the device's root would
            // then name one pair while its residual used another.
            const bool peer1_wins = other1 > top1[l] || (other1 == top1[l] && other1i < top1_index[l]);
            const float win1      = peer1_wins ? other1 : top1[l];
            const int win1i       = peer1_wins ? other1i : top1_index[l];
            const float lose1     = peer1_wins ? top1[l] : other1;
            const int lose1i      = peer1_wins ? top1_index[l] : other1i;
            const auto better     = [](float v, int i, float bv, int bi) {
                return i >= 0 && (v > bv || (v == bv && (bi < 0 || i < bi)));
            };
            float second     = lose1;
            int second_index_l = lose1i;
            if (better(top2[l], top2_index[l], second, second_index_l)) {
                second         = top2[l];
                second_index_l = top2_index[l];
            }
            if (better(other2, other2i, second, second_index_l)) {
                second         = other2;
                second_index_l = other2i;
            }
            next1[l]  = win1;
            next1i[l] = win1i;
            next2[l]  = second;
            next2i[l] = second_index_l;
        }
        top1 = next1;
        top1_index = next1i;
        top2 = next2;
        top2_index = next2i;
    }
    const int first_index  = std::min(top1_index[0], top2_index[0]);
    const int second_index = std::max(top1_index[0], top2_index[0]);
    const int pair_index   = (first_index * (15 - first_index)) / 2 + (second_index - first_index - 1);
    const int sign_first   = sign[first_index] > 0 ? 1 : 0;
    const int sign_second  = sign[second_index] > 0 ? 1 : 0;
    const std::uint32_t type_a_code =
        (std::uint32_t(pair_index) << 2) | (std::uint32_t(sign_first) << 1) |
        std::uint32_t(sign_second);
    const float type_a_score = top1[0] + top2[0];

    // Type B: the square-root-2 family, chosen by the summed magnitude, the parity of the negative
    // coordinates and the smallest coordinate.
    const float magnitude_sum = butterfly_sum(magnitude)[0];
    int negative_count = 0;
    for (int l = 0; l < 8; ++l) negative_count += sign[l] < 0 ? 1 : 0;
    std::array<float, 8> smallest = magnitude;
    std::array<int, 8> smallest_index{};
    for (int l = 0; l < 8; ++l) smallest_index[l] = l;
    for (int mask = 4; mask > 0; mask >>= 1) {
        std::array<float, 8> next = smallest;
        std::array<int, 8> next_index = smallest_index;
        for (int l = 0; l < 8; ++l) {
            const int p       = l ^ mask;
            const float other = smallest[p];
            const int other_i = smallest_index[p];
            if (other < smallest[l] || (other == smallest[l] && other_i < smallest_index[l])) {
                next[l]       = other;
                next_index[l] = other_i;
            }
        }
        smallest       = next;
        smallest_index = next_index;
    }
    float type_b_score = 0.5f * magnitude_sum;
    if ((negative_count & 1) != 0) type_b_score -= smallest[0];

    std::array<float, 8> flipped{};
    for (int l = 0; l < 8; ++l) {
        float chosen = sign[l];
        if (l == smallest_index[0] && (negative_count & 1) != 0) chosen = -chosen;
        flipped[l] = chosen;
    }
    std::array<float, 8> mask_bits{};
    for (int l = 0; l < 8; ++l) {
        mask_bits[l] = (l < 7 && flipped[l] > 0) ? float(1u << l) : 0.0f;
    }
    const auto type_b_code =
        static_cast<std::uint8_t>(std::uint32_t(butterfly_sum(mask_bits)[0] + 0.5f));

    std::array<float, 8> primary{};
    if (type_a_score >= type_b_score) {
        out_root = static_cast<std::uint8_t>(type_a_code);
        for (int l = 0; l < 8; ++l) {
            if (l == first_index) primary[l] = sign_first ? 1.0f : -1.0f;
            else if (l == second_index) primary[l] = sign_second ? 1.0f : -1.0f;
            else primary[l] = 0.0f;
        }
    } else {
        out_root = static_cast<std::uint8_t>(112 + type_b_code);
        for (int l = 0; l < 8; ++l) primary[l] = flipped[l] > 0 ? 0.5f : -0.5f;
    }

    std::array<float, 8> projections{};
    for (int l = 0; l < 8; ++l) projections[l] = unit[l] * (primary[l] * kInvSqrt2);
    const float projection = butterfly_sum(projections)[0];
    std::array<float, 8> residual{};
    for (int l = 0; l < 8; ++l) {
        // The device writes this as one expression, `u - dot * (v1 * kInvSqrt2)`, which nvcc
        // contracts into a fused multiply-add. Splitting it into a multiply and a subtract rounds
        // twice instead of once, and the residual axis is chosen by an argmax that flips on a
        // one-ulp tie.
        const float scale = primary[l] * kInvSqrt2;
        residual[l]       = std::fma(-projection, scale, unit[l]);
    }

    std::array<float, 8> magnitude_residual{};
    for (int l = 0; l < 8; ++l) magnitude_residual[l] = std::fabs(residual[l]);
    std::array<float, 8> largest = magnitude_residual;
    std::array<int, 8> largest_index{};
    for (int l = 0; l < 8; ++l) largest_index[l] = l;
    for (int mask = 4; mask > 0; mask >>= 1) {
        std::array<float, 8> next = largest;
        std::array<int, 8> next_index = largest_index;
        for (int l = 0; l < 8; ++l) {
            const int p       = l ^ mask;
            const float other = largest[p];
            const int other_i = largest_index[p];
            if (other > largest[l] || (other == largest[l] && other_i < largest_index[l])) {
                next[l]       = other;
                next_index[l] = other_i;
            }
        }
        largest       = next;
        largest_index = next_index;
    }
    const int best_dimension = largest_index[0];
    const int residual_sign  = residual[best_dimension] >= 0.0f ? 0 : 1;
    const int axis_index     = (best_dimension << 1) | residual_sign;

    if (radius_index == 0) {
        out_root   = 0;
        out_radial = 0;
    } else {
        out_radial = static_cast<std::uint8_t>((radius_index << 4) | (axis_index & 0x0F));
    }
    if (std::getenv("NINFER_E8_HOST_DUMP") != nullptr) {
        if (diag != nullptr) {
            diag->a_score  = type_a_score;
            diag->b_score  = type_b_score;
            diag->norm     = norm;
            diag->type     = type_a_score >= type_b_score ? 0 : 1;
            diag->best_dim = best_dimension;
            diag->first    = first_index;
            diag->second   = second_index;
            diag->radius   = radius_index;
            diag->root     = out_root;
            diag->radial   = out_radial;
            diag->res      = residual;
        }
        std::fprintf(stderr,
                     "[e8host] rel=%.9g rad=%d a=%.9g b=%.9g i=%d j=%d dim=%d root=%u radb=%u "
                     "base=%.9g\n",
                     double(relative), radius_index, double(type_a_score), double(type_b_score),
                     first_index, second_index, best_dimension, unsigned(out_root),
                     unsigned(out_radial), double(norm));
    }
}

} // namespace ninfer::test::qsa::e8
