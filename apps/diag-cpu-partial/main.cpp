// port (sm_89), engine-integration P3 Phase E: repository-level CPU-partial verification
// harness for the KV-partition seam (int8/host_partial.h + ops/softmax_attention/cpu/partial.h).
//
// Two modes:
//   partial (default) - builds random and deterministic rk4v4 / rk8v4 paged caches (the engine
//     append op writes the packed planes, exactly like the causal-cache test fixture), runs the
//     PRODUCTION grouped instance through ops::causal_softmax_attention_cached, reads the call's
//     per-split partial buffers back from its own workspace arena, and compares them with the
//     CPU partial (partial.h) split by split, head by head, column by column. It then merges
//     the GPU partials and the CPU partials with partial.h's host merge mirror and checks both
//     against the production bf16 output.
//   ab - long-workload A/B THROUGH THE SEAM: a filled prefix (868 and 2048 tokens) plus 512
//     production-shaped decode steps (the fused append+attention entry), each run twice with
//     identical inputs - once all-GPU (no host view installed) and once with
//     NINFER_KV_CPU_PARTITION forcing one split's partial through the CPU leg - and the merged
//     bf16 outputs compared element by element.
//
// Bars (stated once, enforced per element):
//   acc     |cpu-gpu| <= max(1e-5*|gpu|, 1e-6*acc_scale, 2^-10*v_bound)
//           (the relative channel, the ulp-noise channel for cancelled elements, and the
//            single-FP16-quantum channel of one key's contribution - the Phase B "1e-5 relative
//            or one 2^-11 quantum" rule with the ulp padded across the binade)
//   maximum |cpu-gpu| <= max(1e-5*|gpu|, 1e-6*scale)  (scores are integer-exact per group; the
//           expected diff is zero, so this channel also catches real drift)
//   sum     |cpu-gpu| <= max(1e-5*|gpu|, 1e-6*scale)
//   merged bf16 outputs: every differing element must be the adjacent representable bf16 value
//           (one ulp); the headline number is the L2 relative difference (< 1e-5).
//
// usage: diag-cpu-partial [partial|ab] [seam-split-for-ab, default 0]

#include "core/arena.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/cpu/partial.h"
#include "ops/softmax_attention/dense/causal_cache/int8/host_partial.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace softmax_cpu = ninfer::ops::detail::softmax_cpu;
namespace seam         = ninfer::ops::detail;
using namespace ninfer;

constexpr std::int32_t kHeadDim = 256;
constexpr float kAttentionScale = 0.0625f;  // 1/sqrt(256), the production shape
constexpr int kPageSize         = 64;
constexpr double kRelBar        = 1e-5;
constexpr double kQuantumPad    = 1.0 / 1024.0;  // 2^-10, one fp16 ulp padded

void set_env(const std::string& assignment) {
#ifdef _WIN32
    _putenv(assignment.c_str());
#else
    const auto equals = assignment.find('=');
    if (equals == std::string::npos) { throw std::runtime_error("bad env assignment"); }
    setenv(assignment.substr(0, equals).c_str(), assignment.substr(equals + 1).c_str(), 1);
#endif
}

[[nodiscard]] std::uint16_t f32_to_bf16(float value) {
    return softmax_cpu::f32_to_bf16_bits_rn(value);
}

[[nodiscard]] float bf16_to_f32(std::uint16_t bits) { return softmax_cpu::bf16_bits_to_f32(bits); }

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint64_t seed, float lo, float hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uniform(lo, hi);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) { bits[i] = f32_to_bf16(uniform(rng)); }
    return bits;
}

// Deterministic fixture rows: zero rows (the absmax-0 codec edge), saturating rows (codes at
// the rail) and a fixed LCG pattern, so both sides read back the exact stored bytes.
[[nodiscard]] std::uint16_t deterministic_bf16(int token, int head, int d, bool key, float amp) {
    if ((token * 3 + head * 7 + (key ? 0 : 5)) % 23 == 0) { return 0; }  // all-zero row
    if ((token * 5 + head * 11 + (key ? 3 : 1)) % 19 == 4) {
        return f32_to_bf16((d & 1) != 0 ? amp : -amp);  // saturating row
    }
    const std::uint32_t mixed = 1103515245u * (key ? 1u : 2u) +
                                12345u * static_cast<std::uint32_t>(d) +
                                6789u * static_cast<std::uint32_t>(token) +
                                97u * static_cast<std::uint32_t>(head);
    const float unit = static_cast<float>((mixed >> 16) & 0xffu) / 255.0f;  // [0,1)
    return f32_to_bf16((unit * 2.0f - 1.0f) * amp);
}

struct Geometry {
    const char* name;
    int q_heads;
    int kv_heads;
};

constexpr Geometry kGeometries[] = {
    {"h24kv4", 24, 4},
    {"h16kv2", 16, 2},
};

enum class FixtureKind { Random, Deterministic };

// --- the device fixture ------------------------------------------------------------------------

struct Fixture {
    Geometry geometry{};
    seam::RkHostKvDType dtype{};
    KvCacheStorage storage{};
    int window         = 0;  // populated context tokens
    int width          = 0;  // query tile
    int envelope_min   = 0;
    int envelope_max   = 0;  // envelope.max_visible_keys used by every call
    int logical_pages  = 0;
    int physical_pages = 0;
    std::vector<std::int32_t> table_host{};

    DeviceBuffer k_pages{};
    DeviceBuffer v_pages{};
    DeviceBuffer k_scales{};
    DeviceBuffer v_scales{};
    DeviceBuffer table{};
    DeviceBuffer table_row{};  // I32 [1]: the fused entry's kv_table_rows operand
    DeviceBuffer q{};
    DeviceBuffer pos{};
    DeviceBuffer out{};

    // host mirrors the CPU leg and the seam view consume
    std::vector<std::uint8_t> k_codes{};
    std::vector<std::uint8_t> v_codes{};
    std::vector<std::uint16_t> k_scales_host{};
    std::vector<std::uint16_t> v_scales_host{};
    std::vector<std::uint16_t> q_bits{};
    std::vector<std::int32_t> positions{};

    [[nodiscard]] int key_extent() const {
        return dtype == seam::RkHostKvDType::Rk4v4 ? kHeadDim / 2 : kHeadDim;
    }

    [[nodiscard]] PagedKVLayerView view() const {
        PagedKVLayerView result;
        const DType key_code = dtype == seam::RkHostKvDType::Rk4v4 ? DType::U8 : DType::I8;
        result.k_pages = Tensor(k_pages.p, key_code,
                                {key_extent(), kPageSize, geometry.kv_heads, physical_pages});
        result.v_pages = Tensor(v_pages.p, DType::U8,
                                {kHeadDim / 2, kPageSize, geometry.kv_heads, physical_pages});
        result.k_scale_pages =
            Tensor(k_scales.p, DType::FP16, {4, kPageSize, geometry.kv_heads, physical_pages});
        result.v_scale_pages =
            Tensor(v_scales.p, DType::FP16, {4, kPageSize, geometry.kv_heads, physical_pages});
        result.block_table  = Tensor(table.p, DType::I32, {logical_pages});
        result.head_dim     = kHeadDim;
        result.num_kv_heads = geometry.kv_heads;
        result.storage      = storage;
        return result;
    }

    [[nodiscard]] PagedKVBatchLayerView batch_view() const {
        const PagedKVLayerView direct = view();
        PagedKVBatchLayerView result;
        result.k_pages       = direct.k_pages;
        result.v_pages       = direct.v_pages;
        result.k_scale_pages = direct.k_scale_pages;
        result.v_scale_pages = direct.v_scale_pages;
        result.block_tables  = direct.block_table.view({logical_pages, 1});
        result.head_dim     = direct.head_dim;
        result.num_kv_heads = direct.num_kv_heads;
        result.storage      = direct.storage;
        return result;
    }

    [[nodiscard]] Tensor table_row_tensor() const { return Tensor(table_row.p, DType::I32, {1}); }

    [[nodiscard]] softmax_cpu::RkPartialKvPages host_pages() const {
        softmax_cpu::RkPartialKvPages pages;
        pages.k_codes     = k_codes.data();
        pages.v_codes     = v_codes.data();
        pages.k_scales    = k_scales_host.data();
        pages.v_scales    = v_scales_host.data();
        pages.block_table = table_host.data();
        pages.kv_heads    = geometry.kv_heads;
        return pages;
    }
};

std::size_t plane_bytes(int extent, int kv_heads, int physical_pages) {
    return static_cast<std::size_t>(extent) * kPageSize * kv_heads * physical_pages;
}

// Builds a paged cache filled by the ENGINE append op (the harness owns no codec), with a
// fragmented block table (logical page -> 2*page+1) and evicted logical pages marked absent in
// the device table, exactly like the causal-cache test fixture does.
Fixture build_fixture(Geometry geometry, seam::RkHostKvDType dtype, int window, int width,
                      int envelope_min, int envelope_max, FixtureKind kind, std::uint64_t seed,
                      std::span<const int> absent_pages) {
    Fixture f;
    f.geometry      = geometry;
    f.dtype         = dtype;
    f.storage       = dtype == seam::RkHostKvDType::Rk4v4
                          ? KvCacheStorage::RotatedInt4KeyInt4ValueGroup64
                          : KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
    f.window        = window;
    f.width         = width;
    f.envelope_min  = envelope_min;
    f.envelope_max  = envelope_max;
    f.logical_pages  = (std::max(std::max(window, envelope_max), width) + kPageSize - 1) / kPageSize;
    const int populated = (window + kPageSize - 1) / kPageSize;
    // Every logical page up to the envelope is mapped to its own physical page: the decode
    // steps of ab mode append into the tail pages, and the append op's caller contract
    // requires a materialized table entry for every appended position (an unmapped -1 entry
    // would make the append kernel write through physical_page = -1, before the planes).
    f.physical_pages = 2 * f.logical_pages + 1;

    f.table_host.assign(static_cast<std::size_t>(f.logical_pages), -1);
    for (int page = 0; page < f.logical_pages; ++page) {
        f.table_host[static_cast<std::size_t>(page)] = 2 * page + 1;
    }
    for (const int page : absent_pages) {
        // Absent pages must sit in the populated prefix: the decode tail is appended later.
        if (page < 0 || page >= populated) { throw std::runtime_error("absent page out of range"); }
    }

    const std::size_t kv_count = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * window;
    std::vector<std::uint16_t> k_bits(kv_count);
    std::vector<std::uint16_t> v_bits(kv_count);
    if (kind == FixtureKind::Random) {
        k_bits = random_bf16(kv_count, seed, -0.25f, 0.25f);
        v_bits = random_bf16(kv_count, seed + 1u, -1.0f, 1.0f);
    } else {
        for (int head = 0; head < geometry.kv_heads; ++head) {
            for (int token = 0; token < window; ++token) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const std::size_t index =
                        static_cast<std::size_t>(d) +
                        static_cast<std::size_t>(kHeadDim) *
                            (static_cast<std::size_t>(head) +
                             static_cast<std::size_t>(geometry.kv_heads) * token);
                    k_bits[index] = deterministic_bf16(token, head, d, true, 0.25f);
                    v_bits[index] = deterministic_bf16(token, head, d, false, 1.0f);
                }
            }
        }
    }

    f.k_pages  = DeviceBuffer(plane_bytes(f.key_extent(), geometry.kv_heads, f.physical_pages));
    f.v_pages  = DeviceBuffer(plane_bytes(kHeadDim / 2, geometry.kv_heads, f.physical_pages));
    // scale planes are FP16: plane_bytes counts elements, the buffer wants bytes
    f.k_scales = DeviceBuffer(plane_bytes(4, geometry.kv_heads, f.physical_pages) * 2);
    f.v_scales = DeviceBuffer(plane_bytes(4, geometry.kv_heads, f.physical_pages) * 2);
    f.table    = DeviceBuffer(f.table_host.size() * sizeof(std::int32_t));
    f.table_row = DeviceBuffer(sizeof(std::int32_t));
    const std::int32_t row_value = 0;
    f.table_row.copy_from_host(&row_value, sizeof(row_value));

    DeviceBuffer k_input(k_bits.size() * sizeof(std::uint16_t));
    DeviceBuffer v_input(v_bits.size() * sizeof(std::uint16_t));
    std::vector<std::int32_t> append_positions(static_cast<std::size_t>(window));
    for (int i = 0; i < window; ++i) { append_positions[static_cast<std::size_t>(i)] = i; }
    DeviceBuffer append_pos(append_positions.size() * sizeof(std::int32_t));
    k_input.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    v_input.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    append_pos.copy_from_host(append_positions.data(),
                              append_positions.size() * sizeof(std::int32_t));
    f.table.copy_from_host(f.table_host.data(), f.table_host.size() * sizeof(std::int32_t));

    ops::kv_cache_append(
        Tensor(k_input.p, DType::BF16, {kHeadDim, geometry.kv_heads, window}),
        Tensor(v_input.p, DType::BF16, {kHeadDim, geometry.kv_heads, window}),
        Tensor(append_pos.p, DType::I32, {window}), f.view(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Absent pages are marked AFTER the append, exactly like an eviction policy would (the
    // test fixture's mark_absent_pages): the append op requires a materialized table entry
    // for every position it writes, so marking first would make it write through page -1.
    for (const int page : absent_pages) {
        f.table_host[static_cast<std::size_t>(page)] = -1;
    }
    if (!absent_pages.empty()) {
        f.table.copy_from_host(f.table_host.data(), f.table_host.size() * sizeof(std::int32_t));
    }
    f.k_codes.resize(f.k_pages.bytes);
    f.v_codes.resize(f.v_pages.bytes);
    f.k_scales_host.resize(f.k_scales.bytes / 2);
    f.v_scales_host.resize(f.v_scales.bytes / 2);
    f.k_pages.copy_to_host(f.k_codes.data(), f.k_codes.size());
    f.v_pages.copy_to_host(f.v_codes.data(), f.v_codes.size());
    f.k_scales.copy_to_host(f.k_scales_host.data(), f.k_scales_host.size() * 2);
    f.v_scales.copy_to_host(f.v_scales_host.data(), f.v_scales_host.size() * 2);

    const std::size_t q_count = static_cast<std::size_t>(kHeadDim) * geometry.q_heads * width;
    f.q_bits    = random_bf16(q_count, seed + 7u, -0.25f, 0.25f);
    f.positions.resize(static_cast<std::size_t>(width));
    for (int t = 0; t < width; ++t) {
        f.positions[static_cast<std::size_t>(t)] = window - width + t;
    }
    f.q   = DeviceBuffer(q_count * sizeof(std::uint16_t));
    f.pos = DeviceBuffer(f.positions.size() * sizeof(std::int32_t));
    f.out = DeviceBuffer(q_count * sizeof(std::uint16_t));
    f.q.copy_from_host(f.q_bits.data(), q_count * sizeof(std::uint16_t));
    f.pos.copy_from_host(f.positions.data(), f.positions.size() * sizeof(std::int32_t));
    return f;
}

// Rebinds one query tile: new Q bits and absolute positions, uploaded to the device tensors
// and mirrored into the vectors the seam view points at.
void set_query_tile(Fixture& f, std::vector<std::uint16_t> q_bits,
                    std::vector<std::int32_t> positions) {
    f.q_bits    = std::move(q_bits);
    f.positions = std::move(positions);
    f.q.copy_from_host(f.q_bits.data(), f.q_bits.size() * sizeof(std::uint16_t));
    f.pos.copy_from_host(f.positions.data(), f.positions.size() * sizeof(std::int32_t));
}

[[nodiscard]] std::size_t output_count(const Fixture& f) {
    return static_cast<std::size_t>(kHeadDim) * f.geometry.q_heads * f.width;
}

// |v16| upper bound over the PRESENT pages only: |fp16(code*vs)| <= 7*max|vs| with one fp16
// rounding pad. Unpopulated/absent pages keep uninitialized bytes neither side ever reads.
double v_plane_bound(const Fixture& f) {
    double max_scale = 0.0;
    const int kv_heads = f.geometry.kv_heads;
    for (const std::int32_t entry : f.table_host) {
        if (entry < 0) { continue; }
        for (int head = 0; head < kv_heads; ++head) {
            for (int tip = 0; tip < kPageSize; ++tip) {
                for (int g = 0; g < 4; ++g) {
                    const std::int64_t index =
                        softmax_cpu::plane_offset(4, kv_heads, entry, head, tip, g);
                    max_scale = std::max(max_scale, std::abs(static_cast<double>(softmax_cpu::
                                                    fp16_bits_to_f32(
                                                        f.v_scales_host[static_cast<std::size_t>(
                                                            index)]))));
                }
            }
        }
    }
    return 7.0 * max_scale * (1.0 + kQuantumPad);
}

// --- comparison helpers ------------------------------------------------------------------------

struct PartialCheck {
    long elements     = 0;
    long failures     = 0;
    long quantum_hits = 0;  // passed only through the fp16-quantum channel
    double max_rel    = 0.0;  // max rel over elements with |gpu| >= 1% of the array scale
    double worst_diff = 0.0;
};

// The element rule from the header comment. `extra_floor` carries the acc quantum channel
// (2^-10 * v_bound); 0 for maximum/sum.
[[nodiscard]] bool element_passes(double diff, double gpu, double scale, double extra_floor,
                                  bool has_quantum_channel, PartialCheck& check) {
    ++check.elements;
    const double rel_channel = kRelBar * std::max(std::abs(gpu), 1e-6 * scale);
    const double pass_bar     = std::max(rel_channel, extra_floor);
    if (std::abs(gpu) >= 0.01 * scale) {
        check.max_rel = std::max(check.max_rel, diff / std::max(std::abs(gpu), 1e-300));
    }
    check.worst_diff = std::max(check.worst_diff, diff);
    if (diff <= pass_bar) {
        if (has_quantum_channel && diff > rel_channel) { ++check.quantum_hits; }
        return true;
    }
    return false;
}

PartialCheck compare_partials(const std::vector<float>& gpu, const std::vector<float>& cpu,
                              int q_heads, int width, int active, double v_bound, bool is_acc) {
    PartialCheck check;
    const double scale = [&] {
        double max_abs = 0.0;
        for (const float value : gpu) { max_abs = std::max(max_abs, std::abs((double)value)); }
        return max_abs > 0.0 ? max_abs : 1.0;
    }();
    const double extra = is_acc ? kQuantumPad * v_bound : 0.0;
    for (int split = 0; split < active; ++split) {
        for (int token = 0; token < width; ++token) {
            for (int head = 0; head < q_heads; ++head) {
                const std::int64_t stat = head + static_cast<std::int64_t>(q_heads) *
                                                       (token + static_cast<std::int64_t>(width) *
                                                                    split);
                for (std::int64_t e = 0; e < (is_acc ? 256 : 1); ++e) {
                    const std::int64_t index = is_acc ? stat * 256 + e : stat;
                    const double g = static_cast<double>(gpu[static_cast<std::size_t>(index)]);
                    const double c = static_cast<double>(cpu[static_cast<std::size_t>(index)]);
                    if (c == g) { ++check.elements; continue; }  // fast path, covers -inf slots
                    const double diff = std::abs(c - g);
                    if (!element_passes(diff, g, scale, extra, is_acc, check)) { ++check.failures; }
                }
            }
        }
    }
    return check;
}

enum class Bf16Delta { Exact, OneUlp, Other };

[[nodiscard]] Bf16Delta classify_bf16(std::uint16_t a, std::uint16_t b) {
    if (a == b) { return Bf16Delta::Exact; }
    if ((a == 0x0000u && b == 0x8000u) || (a == 0x8000u && b == 0x0000u)) {
        return Bf16Delta::OneUlp;  // +-0 are neighbouring values
    }
    if ((a & 0x8000u) == (b & 0x8000u)) {
        const unsigned diff = a > b ? a - b : b - a;
        if (diff == 1u) { return Bf16Delta::OneUlp; }
    }
    return Bf16Delta::Other;
}

// Max split merge weight per (head, token) from the reference partials: the largest share
// any single split's acc can hold in the merged value. One legal fp16 p-quantum flip inside
// that split's acc propagates to at most kQuantumPad * v_bound * w_max on the output element -
// the Phase B "1e-5 relative or one 2^-11 quantum" rule expressed at merge granularity.
[[nodiscard]] std::vector<double> max_split_weights(const std::vector<float>& maximum,
                                                   const std::vector<float>& sum, int q_heads,
                                                   int width, int active) {
    std::vector<double> weights(static_cast<std::size_t>(q_heads) * width, 0.0);
    for (int token = 0; token < width; ++token) {
        for (int head = 0; head < q_heads; ++head) {
            const auto stat = [&](int split) {
                return head + static_cast<std::int64_t>(q_heads) *
                                  (token + static_cast<std::int64_t>(width) * split);
            };
            double m_all = -std::numeric_limits<double>::infinity();
            for (int s = 0; s < active; ++s) { m_all = std::max(m_all, (double)maximum[stat(s)]); }
            double denominator = 0.0;
            for (int s = 0; s < active; ++s) {
                const double l = sum[stat(s)];
                denominator += l > 0.0 && m_all > -INFINITY ?
                                   l * std::exp((double)maximum[stat(s)] - m_all) : 0.0;
            }
            double w_max = 0.0;
            if (denominator > 0.0) {
                for (int s = 0; s < active; ++s) {
                    const double l = sum[stat(s)];
                    if (l > 0.0 && m_all > -INFINITY) {
                        w_max = std::max(w_max, l * std::exp((double)maximum[stat(s)] - m_all) /
                                                     denominator);
                    }
                }
            }
            weights[static_cast<std::size_t>(head) +
                    static_cast<std::size_t>(q_heads) * token] = w_max;
        }
    }
    return weights;
}

struct MergeCheck {
    long elements = 0;
    long exact    = 0;
    long one_ulp  = 0;
    long floor_pass = 0;  // beyond one ulp but inside the 1e-5 * scale near-zero floor
    long beyond    = 0;
    double l2_num = 0.0;
    double l2_den = 0.0;
    double l2_rel = 0.0;
    double max_rel = 0.0;
};

// Merged-output element rule: exact, or one bf16 ulp (the rounding of a sub-1e-5 fp32
// deviation), or |diff| <= max(1e-5 * output_scale, kQuantumPad * v_bound * w_max[head,token]) -
// the propagated single-quantum bound, which is the only legal way a near-zero cancelled
// element may move by more than one ulp. `weights` is max_split_weights of the reference
// partials (nullptr skips the bound leg).
MergeCheck compare_merged(const std::vector<std::uint16_t>& reference,
                          const std::vector<std::uint16_t>& actual, int q_heads,
                          const std::vector<double>* weights, double v_bound) {
    if (reference.size() != actual.size()) { throw std::runtime_error("merged size mismatch"); }
    double scale = 0.0;
    for (const std::uint16_t bits : reference) {
        scale = std::max(scale, std::abs(static_cast<double>(bf16_to_f32(bits))));
    }
    const double floor_bar = kRelBar * (scale > 0.0 ? scale : 1.0);
    MergeCheck check;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        ++check.elements;
        const Bf16Delta delta = classify_bf16(reference[i], actual[i]);
        const double a = static_cast<double>(bf16_to_f32(actual[i]));
        const double r = static_cast<double>(bf16_to_f32(reference[i]));
        const double diff = std::abs(a - r);
        double quantum_bar = 0.0;
        if (weights != nullptr) {
            const std::size_t head = (i / 256) % static_cast<std::size_t>(q_heads);
            const std::size_t token = i / 256 / static_cast<std::size_t>(q_heads);
            quantum_bar = kQuantumPad * v_bound *
                          (*weights)[head + static_cast<std::size_t>(q_heads) * token];
        }
        switch (delta) {
        case Bf16Delta::Exact: ++check.exact; break;
        case Bf16Delta::OneUlp:
            ++check.one_ulp;
            break;
        case Bf16Delta::Other:
            if (diff <= std::max(floor_bar, quantum_bar)) {
                ++check.floor_pass;
            } else {
                ++check.beyond;
                static long printed = 0;  // cap the diagnostic across the whole process
                if (printed < 40) {
                    ++printed;
                    std::printf("    beyond-bar element %zu: ref=0x%04x (%.6e) actual=0x%04x "
                                "(%.6e) diff=%.3e floor=%.3e quantum=%.3e\n",
                                i, reference[i], r, actual[i], a, diff, floor_bar, quantum_bar);
                }
            }
            break;
        }
        const double signed_diff = a - r;
        check.l2_num += signed_diff * signed_diff;
        check.l2_den += r * r;
        if (delta != Bf16Delta::Exact) {
            check.max_rel = std::max(check.max_rel, diff / std::max(std::abs(r), 1e-300));
        }
    }
    check.l2_rel = check.l2_den > 0.0 ? std::sqrt(check.l2_num / check.l2_den) : 0.0;
    return check;
}

void report_merge(const char* label, const MergeCheck& check) {
    std::printf("  %-24s exact=%ld/%ld one-ulp=%ld floor=%ld beyond=%ld l2_rel=%.3e\n", label,
                check.exact, check.elements, check.one_ulp, check.floor_pass, check.beyond,
                check.l2_rel);
}

// --- partial mode ------------------------------------------------------------------------------

struct ProductionRun {
    std::vector<float> acc, maximum, sum;
    std::vector<std::uint16_t> out_bits;
    int capacity  = 0;
    int active    = 0;
    int key_tile  = 0;
    std::size_t workspace_bytes = 0;
};

[[nodiscard]] int resolve_key_tile(int q_heads, int width) {
    using seam::CausalD256H24Kv4;
    using seam::CausalD256H16Kv2;
    if (q_heads == 24) {
        switch (width) {
        case 1: return seam::Int8KvGroupedInstance<CausalD256H24Kv4, 1>::Schedule::kKeyRows;
        case 8: return seam::Int8KvGroupedInstance<CausalD256H24Kv4, 8>::Schedule::kKeyRows;
        default: throw std::logic_error("unsupported harness width");
        }
    }
    switch (width) {
    case 1: return seam::Int8KvGroupedInstance<CausalD256H16Kv2, 1>::Schedule::kKeyRows;
    case 8: return seam::Int8KvGroupedInstance<CausalD256H16Kv2, 8>::Schedule::kKeyRows;
    default: throw std::logic_error("unsupported harness width");
    }
}

ProductionRun run_production_cached(Fixture& f, DeviceExecutionView execution) {
    ProductionRun run;
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(f.envelope_min),
                                                          static_cast<std::uint32_t>(f.envelope_max)};
    const ops::AttentionHeadGeometry geometry{kHeadDim, f.geometry.q_heads, f.geometry.kv_heads};
    const auto plan = seam::make_int8_kv_causal_plan(f.geometry.q_heads, f.width, 1, envelope,
                                                    execution.multiprocessor_count);
    if (plan.family != seam::Int8KvFamily::Grouped) {
        throw std::logic_error("harness case did not route to the grouped family");
    }
    run.capacity = plan.partition.capacity;
    run.key_tile = resolve_key_tile(f.geometry.q_heads, f.width);

    run.workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        geometry, f.storage, envelope, 1, f.width, f.width, execution);
    DeviceBuffer workspace(std::max<std::size_t>(run.workspace_bytes, 256));
    WorkspaceArena arena(DeviceSpan{workspace.p, workspace.bytes});

    Tensor tq(f.q.p, DType::BF16, {kHeadDim, f.geometry.q_heads, f.width});
    Tensor tp(f.pos.p, DType::I32, {f.width});
    Tensor tout(f.out.p, DType::BF16, {kHeadDim, f.geometry.q_heads, f.width});
    ops::causal_softmax_attention_cached(tq, tp, geometry, kAttentionScale, f.view(), envelope,
                                        arena, tout, execution);
    CUDA_CHECK(cudaDeviceSynchronize());

    // The partial buffers are the call's only workspace allocations (the test suite asserts the
    // same high-water identity); read them back from the arena this harness owns.
    const std::size_t acc_bytes = static_cast<std::size_t>(256) * f.geometry.q_heads * f.width *
                                  run.capacity * sizeof(float);
    const std::size_t stat_bytes =
        static_cast<std::size_t>(f.geometry.q_heads) * f.width * run.capacity * sizeof(float);
    const auto aligned          = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    const std::size_t max_offset = aligned(acc_bytes);
    const std::size_t sum_offset = aligned(max_offset + stat_bytes);
    const std::size_t expected_peak = sum_offset + stat_bytes;
    if (arena.peak_used() != expected_peak || arena.peak_used() != run.workspace_bytes) {
        std::fprintf(stderr,
                     "diag-cpu-partial: workspace layout drifted (peak=%llu expected=%llu "
                     "capacity-query=%llu); the partial snoop is invalid\n",
                     static_cast<unsigned long long>(arena.peak_used()),
                     static_cast<unsigned long long>(expected_peak),
                     static_cast<unsigned long long>(run.workspace_bytes));
        throw std::logic_error("partial snoop layout mismatch");
    }
    std::vector<std::uint8_t> raw(expected_peak);
    workspace.copy_to_host(raw.data(), raw.size());
    run.acc.resize(acc_bytes / sizeof(float));
    run.maximum.resize(stat_bytes / sizeof(float));
    run.sum.resize(stat_bytes / sizeof(float));
    std::memcpy(run.acc.data(), raw.data(), acc_bytes);
    std::memcpy(run.maximum.data(), raw.data() + max_offset, stat_bytes);
    std::memcpy(run.sum.data(), raw.data() + sum_offset, stat_bytes);

    run.out_bits.resize(output_count(f));
    f.out.copy_to_host(run.out_bits.data(), run.out_bits.size() * sizeof(std::uint16_t));
    run.active = plan.partition.active(f.positions.back() + 1);
    return run;
}

void cpu_partials(const Fixture& f, int capacity, int active, int key_tile,
                  std::vector<float>& acc, std::vector<float>& maximum,
                  std::vector<float>& sum) {
    const int q_heads = f.geometry.q_heads;
    const int width   = f.width;
    acc.assign(static_cast<std::size_t>(256) * q_heads * width * capacity, 0.0f);
    maximum.assign(static_cast<std::size_t>(q_heads) * width * capacity, 0.0f);
    sum.assign(static_cast<std::size_t>(q_heads) * width * capacity, 0.0f);
    const int window = f.positions.back() + 1;
    for (int split = 0; split < active; ++split) {
        softmax_cpu::RkPartialRequest request;
        request.q_bf16    = f.q_bits.data();
        request.positions = f.positions.data();
        request.scale     = kAttentionScale;
        request.width    = width;
        request.tokens   = width;
        request.split     = split;
        request.range     = softmax_cpu::rk_partial_split_range(window, split, active, key_tile);
        request.key_tile  = key_tile;
        const softmax_cpu::RkPartialOutput out{acc.data(), maximum.data(), sum.data()};
        if (q_heads == 24) {
            if (f.dtype == seam::RkHostKvDType::Rk4v4)
                softmax_cpu::rk_attention_partial<24, 4, true>(f.host_pages(), request, out);
            else
                softmax_cpu::rk_attention_partial<24, 4, false>(f.host_pages(), request, out);
        } else {
            if (f.dtype == seam::RkHostKvDType::Rk4v4)
                softmax_cpu::rk_attention_partial<16, 2, true>(f.host_pages(), request, out);
            else
                softmax_cpu::rk_attention_partial<16, 2, false>(f.host_pages(), request, out);
        }
    }
}

int run_partial_mode(DeviceExecutionView execution) {
    struct CaseResult {
        std::string label;
        bool pass;
        double acc_max_rel;
        double stat_max_rel;
        double merge_l2;
        long merge_beyond_ulp;
    };
    std::vector<CaseResult> results;
    const char* dtype_names[] = {"rk4v4", "rk8v4"};
    const int windows[]       = {64, 320, 2048};
    const int widths[]        = {1, 8};

    for (const auto& geometry : kGeometries) {
        for (int dtype_index = 0; dtype_index < 2; ++dtype_index) {
            const auto dtype =
                dtype_index == 0 ? seam::RkHostKvDType::Rk4v4 : seam::RkHostKvDType::Rk8v4;
            for (int kind_index = 0; kind_index < 2; ++kind_index) {
                const FixtureKind kind =
                    kind_index == 0 ? FixtureKind::Random : FixtureKind::Deterministic;
                for (int window : windows) {
                    for (int width : widths) {
                        std::vector<int> absent;
                        if (window >= 320) {
                            const int populated = (window + kPageSize - 1) / kPageSize;
                            absent = {std::min(window / 128, populated - 2),
                                      std::min(window / 128 + 3, populated - 1)};
                        }
                        Fixture f = build_fixture(
                            geometry, dtype, window, width, window, window, kind,
                            0x9E3779B9u + static_cast<std::uint64_t>(window) * 131u + kind_index,
                            absent);
                        ProductionRun run = run_production_cached(f, execution);
                        std::vector<float> cpu_acc, cpu_max, cpu_sum;
                        cpu_partials(f, run.capacity, run.active, run.key_tile, cpu_acc, cpu_max,
                                     cpu_sum);
                const double bound = v_plane_bound(f);
                // The bound is computed from the prefix snapshot; the decode steps append V rows
                // of the same +-1 bf16 distribution, so the prefix's absmax already saturates it.
                        const PartialCheck acc_check = compare_partials(
                            run.acc, cpu_acc, geometry.q_heads, width, run.active, bound, true);
                        const PartialCheck max_check = compare_partials(
                            run.maximum, cpu_max, geometry.q_heads, width, run.active, 0.0, false);
                        const PartialCheck sum_check = compare_partials(
                            run.sum, cpu_sum, geometry.q_heads, width, run.active, 0.0, false);

                        std::vector<std::uint16_t> merged_gpu(output_count(f));
                        std::vector<std::uint16_t> merged_cpu(merged_gpu.size());
                        if (geometry.q_heads == 24) {
                            softmax_cpu::rk_merge_partials<24>(run.acc.data(), run.maximum.data(),
                                                             run.sum.data(), width, run.active,
                                                             merged_gpu.data());
                            softmax_cpu::rk_merge_partials<24>(cpu_acc.data(), cpu_max.data(),
                                                             cpu_sum.data(), width, run.active,
                                                             merged_cpu.data());
                        } else {
                            softmax_cpu::rk_merge_partials<16>(run.acc.data(), run.maximum.data(),
                                                             run.sum.data(), width, run.active,
                                                             merged_gpu.data());
                            softmax_cpu::rk_merge_partials<16>(cpu_acc.data(), cpu_max.data(),
                                                             cpu_sum.data(), width, run.active,
                                                             merged_cpu.data());
                        }
                        const auto gpu_weights =
                            max_split_weights(run.maximum, run.sum, geometry.q_heads, width,
                                              run.active);
                        const MergeCheck gpu_merge = compare_merged(
                            run.out_bits, merged_gpu, geometry.q_heads, &gpu_weights, bound);
                        const MergeCheck cpu_merge = compare_merged(
                            run.out_bits, merged_cpu, geometry.q_heads, &gpu_weights, bound);

                        char label[96];
                        std::snprintf(label, sizeof label, "%s %s T=%-5d W=%d %s", dtype_names[dtype_index],
                                      geometry.name, window, width,
                                      kind_index == 0 ? "rand" : "det ");
                        const bool pass = acc_check.failures == 0 && max_check.failures == 0 &&
                                          sum_check.failures == 0 && gpu_merge.beyond == 0 &&
                                          cpu_merge.beyond == 0;
                        std::printf("%s active=%-3d key_tile=%d %s\n", label, run.active,
                                    run.key_tile, pass ? "PASS" : "FAIL");
                        std::printf("  acc : max_rel=%.2e worst_abs=%.2e fail=%ld quantum=%ld\n",
                                    acc_check.max_rel, acc_check.worst_diff, acc_check.failures,
                                    acc_check.quantum_hits);
                        std::printf("  stat: max_rel(max)=%.2e max_rel(sum)=%.2e fail=%ld+%ld\n",
                                    max_check.max_rel, sum_check.max_rel, max_check.failures,
                                    sum_check.failures);
                        report_merge("merge gpu-vs-device", gpu_merge);
                        report_merge("merge cpu-vs-device", cpu_merge);
                        results.push_back({label, pass,
                                           acc_check.max_rel,
                                           std::max(max_check.max_rel, sum_check.max_rel),
                                           std::max(gpu_merge.l2_rel, cpu_merge.l2_rel),
                                           gpu_merge.beyond + cpu_merge.beyond});
                    }
                }
            }
        }
    }

    std::printf("\n== per-dtype summary (partial mode; element rule: rel<=1e-5 or one fp16 quantum "
                "for acc, rel<=1e-5 for stats; merge: one ulp or 1e-5-scale floor) ==\n");
    int failing_dtypes = 0;
    for (int dtype_index = 0; dtype_index < 2; ++dtype_index) {
        double acc_rel = 0.0, stat_rel = 0.0, merge_l2 = 0.0;
        long cases = 0, passed = 0, merge_beyond = 0, element_fails = 0;
        for (const auto& r : results) {
            if (r.label.rfind(dtype_names[dtype_index], 0) == 0) {
                ++cases;
                passed += r.pass ? 1 : 0;
                acc_rel  = std::max(acc_rel, r.acc_max_rel);
                stat_rel = std::max(stat_rel, r.stat_max_rel);
                merge_l2 = std::max(merge_l2, r.merge_l2);
                merge_beyond += r.merge_beyond_ulp;
                element_fails += r.pass ? 0 : 1;
            }
        }
        const bool pass = passed == cases && stat_rel < kRelBar && merge_beyond == 0;
        failing_dtypes += pass ? 0 : 1;
        std::printf("%s: cases=%ld pass=%ld acc_max_rel=%.2e (quantum-channel deviations) "
                    "stat_max_rel=%.2e merge_l2(max case)=%.3e merge_beyond_bar=%ld -> %s\n",
                    dtype_names[dtype_index], cases, passed, acc_rel, stat_rel, merge_l2,
                    merge_beyond, pass ? "PASS" : "FAIL");
    }
    return failing_dtypes == 0 ? 0 : 1;
}

// --- ab mode ------------------------------------------------------------------------------------

// One decode step through the PRODUCTION fused entry (append + attention), width 1, with the
// call's partial buffers snooped back from the harness-owned arena (slot 0 acc + the stats
// planes; the full acc plane when `deep` is set).
struct StepSnoop {
    std::vector<float> slot0_acc{};
    std::vector<float> full_acc{};
    std::vector<float> maximum{};
    std::vector<float> sum{};
    std::vector<std::uint16_t> out_bits{};
};

StepSnoop run_decode_step(Fixture& f, DeviceExecutionView execution, WorkspaceArena& arena,
                          const DeviceBuffer& step_k, const DeviceBuffer& step_v,
                          const DeviceBuffer& workspace, int capacity, bool deep) {
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(f.envelope_min),
                                                          static_cast<std::uint32_t>(f.envelope_max)};
    const ops::AttentionHeadGeometry geometry{kHeadDim, f.geometry.q_heads, f.geometry.kv_heads};
    Tensor tq(f.q.p, DType::BF16, {kHeadDim, f.geometry.q_heads, 1});
    Tensor tk(step_k.p, DType::BF16, {kHeadDim, f.geometry.kv_heads, 1});
    Tensor tv(step_v.p, DType::BF16, {kHeadDim, f.geometry.kv_heads, 1});
    Tensor tp(f.pos.p, DType::I32, {1});
    Tensor tout(f.out.p, DType::BF16, {kHeadDim, f.geometry.q_heads, 1});
    ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, f.table_row_tensor(), geometry,
                                   kAttentionScale, f.batch_view(), envelope, arena, tout,
                                   execution);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::size_t acc_bytes = static_cast<std::size_t>(256) * f.geometry.q_heads * capacity *
                                  sizeof(float);
    const std::size_t stat_bytes =
        static_cast<std::size_t>(f.geometry.q_heads) * capacity * sizeof(float);
    const auto aligned = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    const std::size_t max_offset = aligned(acc_bytes);
    const std::size_t sum_offset = aligned(max_offset + stat_bytes);
    if (arena.peak_used() != sum_offset + stat_bytes) {
        std::fprintf(stderr,
                     "diag-cpu-partial: fused-entry workspace layout drifted (peak=%llu "
                     "expected=%llu); the partial snoop is invalid\n",
                     static_cast<unsigned long long>(arena.peak_used()),
                     static_cast<unsigned long long>(sum_offset + stat_bytes));
        throw std::logic_error("ab partial snoop layout mismatch");
    }

    StepSnoop snoop;
    snoop.slot0_acc.resize(static_cast<std::size_t>(256) * f.geometry.q_heads);
    workspace.copy_to_host(snoop.slot0_acc.data(), snoop.slot0_acc.size() * sizeof(float), 0);
    if (deep) {
        snoop.full_acc.resize(acc_bytes / sizeof(float));
        workspace.copy_to_host(snoop.full_acc.data(), acc_bytes, 0);
    }
    snoop.maximum.resize(stat_bytes / sizeof(float));
    snoop.sum.resize(stat_bytes / sizeof(float));
    workspace.copy_to_host(snoop.maximum.data(), stat_bytes, max_offset);
    workspace.copy_to_host(snoop.sum.data(), stat_bytes, sum_offset);
    snoop.out_bits.resize(output_count(f));
    f.out.copy_to_host(snoop.out_bits.data(), snoop.out_bits.size() * sizeof(std::uint16_t));
    return snoop;
}

// Accumulates the fp32-level comparison of the seam's slot (split 0) between two legs: the
// strict 1e-5 evidence, with the same element rule as partial mode (rel, ulp noise, quantum).
void accumulate_slot0(const StepSnoop& gpu, const StepSnoop& cpu, int q_heads, double v_bound,
                      PartialCheck& acc_check, PartialCheck& max_check, PartialCheck& sum_check) {
    double acc_scale = 0.0;
    for (const float value : gpu.slot0_acc) {
        acc_scale = std::max(acc_scale, std::abs(static_cast<double>(value)));
    }
    acc_scale = acc_scale > 0.0 ? acc_scale : 1.0;
    const int elements = static_cast<int>(gpu.slot0_acc.size());
    for (int i = 0; i < elements; ++i) {
        const double g = gpu.slot0_acc[static_cast<std::size_t>(i)];
        const double c = cpu.slot0_acc[static_cast<std::size_t>(i)];
        if (c == g) { ++acc_check.elements; continue; }
        if (!element_passes(std::abs(c - g), g, acc_scale, kQuantumPad * v_bound, true,
                            acc_check)) {
            ++acc_check.failures;
        }
    }
    double m_scale = 0.0, l_scale = 0.0;
    for (int head = 0; head < q_heads; ++head) {
        m_scale = std::max(m_scale, std::abs((double)gpu.maximum[head]));
        l_scale = std::max(l_scale, std::abs((double)gpu.sum[head]));
    }
    for (int head = 0; head < q_heads; ++head) {
        const double gm = gpu.maximum[head], cm = cpu.maximum[head];
        if (gm != cm) {
            if (!element_passes(std::abs(cm - gm), gm, m_scale, 0.0, false, max_check)) {
                ++max_check.failures;
            }
        } else {
            ++max_check.elements;
        }
        const double gl = gpu.sum[head], cl = cpu.sum[head];
        if (gl != cl) {
            if (!element_passes(std::abs(cl - gl), gl, l_scale, 0.0, false, sum_check)) {
                ++sum_check.failures;
            }
        } else {
            ++sum_check.elements;
        }
    }
}

int run_ab_mode(DeviceExecutionView execution, int seam_split) {
    set_env("NINFER_KV_CPU_PARTITION=" + std::to_string(seam_split));
    constexpr int kSteps = 512;
    const int prefixes[] = {868, 2048};
    const char* dtype_names[] = {"rk4v4", "rk8v4"};
    int failing_cases = 0;
    std::printf("== ab mode: seam split %d, %d decode steps per case, fused append+attention "
                "entry ==\n", seam_split, kSteps);

    for (const auto& geometry : kGeometries) {
        for (int dtype_index = 0; dtype_index < 2; ++dtype_index) {
            const auto dtype =
                dtype_index == 0 ? seam::RkHostKvDType::Rk4v4 : seam::RkHostKvDType::Rk8v4;
            const std::size_t q_count = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
            const std::size_t kv_count = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads;
            const std::uint64_t seed_base =
                0xC0FFEE00u + static_cast<std::uint64_t>(geometry.q_heads) * 7919u;

            for (const int prefix : prefixes) {
                const int envelope_max = prefix + kSteps;
                const int populated = (prefix + kPageSize - 1) / kPageSize;
                std::vector<int> absent{std::min(prefix / 128, populated - 2),
                                        std::min(prefix / 128 + 3, populated - 1)};
                Fixture f = build_fixture(geometry, dtype, prefix, 1, prefix, envelope_max,
                                          FixtureKind::Random, seed_base + prefix, absent);

                // The seam's CPU leg reads the host mirror of the prefix planes; the decode
                // steps append to later pages only, so for split 0 (the earliest 64 keys) the
                // mirror stays valid for the whole run. Verified below before the loop.
                const ops::CausalAttentionExecutionEnvelope envelope{
                    static_cast<std::uint32_t>(f.envelope_min),
                    static_cast<std::uint32_t>(f.envelope_max)};
                const auto plan = seam::make_int8_kv_causal_plan(
                    geometry.q_heads, 1, 1, envelope, execution.multiprocessor_count);
                if (plan.family != seam::Int8KvFamily::Grouped) {
                    throw std::logic_error("ab decode step did not route to the grouped family");
                }
                const int capacity = plan.partition.capacity;
                const int window_last = prefix + kSteps;
                const int active_last  = plan.partition.active(window_last);
                const auto probe = softmax_cpu::rk_partial_split_range(window_last, seam_split,
                                                                       active_last, 32);
                if (seam_split != 0 || probe.end > 64) {
                    std::fprintf(stderr, "diag-cpu-partial: ab case needs a mirror covering "
                                         "appended pages (split range [%d,%d)); out of scope\n",
                                 probe.begin, probe.end);
                    return 2;
                }

                // workspace sized once for the whole decode loop (fixed envelope, width 1)
                const ops::AttentionHeadGeometry geometry_spec{kHeadDim, geometry.q_heads,
                                                               geometry.kv_heads};
                const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
                    geometry_spec, f.storage, envelope, 1, 1, 1, execution);
                DeviceBuffer workspace(std::max<std::size_t>(workspace_bytes, 256));
                WorkspaceArena arena(DeviceSpan{workspace.p, workspace.bytes});

                seam::RkHostPartialView view{};
                view.dtype       = dtype;
                view.query_heads  = geometry.q_heads;
                view.pages        = f.host_pages();
                view.q_bf16       = f.q_bits.data();
                view.positions    = f.positions.data();
                // NOTE: set_query_tile replaces f.q_bits / f.positions each step (move-assign
                // swaps the heap buffer), so the view's mirrors are REBOUND after every upload -
                // a pointer captured once would dangle and the CPU leg would read freed memory.

                DeviceBuffer step_k(kv_count * sizeof(std::uint16_t));
                DeviceBuffer step_v(kv_count * sizeof(std::uint16_t));

                std::vector<StepSnoop> gpu_steps(kSteps), cpu_steps(kSteps);
                const auto prepare_step = [&](int step) {
                    const auto k_bits =
                        random_bf16(kv_count, seed_base + 100u + step, -0.25f, 0.25f);
                    const auto v_bits =
                        random_bf16(kv_count, seed_base + 200u + step, -1.0f, 1.0f);
                    set_query_tile(f, random_bf16(q_count, seed_base + step, -0.25f, 0.25f),
                                   {prefix + step});
                    view.q_bf16    = f.q_bits.data();
                    view.positions = f.positions.data();
                    step_k.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
                    step_v.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
                };

                // leg A: all GPU (the knob is armed but no view is installed, so the gate
                // falls back once and every split runs on the device)
                seam::rk_host_partial_install(nullptr);
                for (int step = 0; step < kSteps; ++step) {
                    prepare_step(step);
                    gpu_steps[step] =
                        run_decode_step(f, execution, arena, step_k, step_v, workspace, capacity,
                                        step == 0);
                }

                // leg B: identical inputs, the seam forces split `seam_split` through the CPU
                seam::rk_host_partial_install(&view);
                const std::uint64_t engaged_before = seam::rk_host_partial_engagements();
                for (int step = 0; step < kSteps; ++step) {
                    prepare_step(step);
                    cpu_steps[step] =
                        run_decode_step(f, execution, arena, step_k, step_v, workspace, capacity,
                                        step == 0);
                }
                const std::uint64_t engaged =
                    seam::rk_host_partial_engagements() - engaged_before;
                seam::rk_host_partial_install(nullptr);

                const double bound = v_plane_bound(f);
                // deep integrity: the GPU slots the kernel actually WROTE this step (split <
                // active) must be bit-identical between the legs; slots >= active are never
                // written and hold stale bytes from earlier calls, so they are not comparable.
                long deep_diffs = 0;
                const int active_step0 = plan.partition.active(prefix + 1);
                if (!gpu_steps[0].full_acc.empty()) {
                    const std::size_t written =
                        256 * static_cast<std::size_t>(geometry.q_heads) *
                        static_cast<std::size_t>(active_step0);
                    for (std::size_t i = 256 * static_cast<std::size_t>(geometry.q_heads);
                         i < written; ++i) {
                        const bool same = gpu_steps[0].full_acc[i] == cpu_steps[0].full_acc[i];
                        if (!same && deep_diffs < 3) {
                            std::printf("    integrity probe: full_acc[%zu] leg_a=%.6e "
                                        "leg_b=%.6e\n",
                                        i,
                                        static_cast<double>(gpu_steps[0].full_acc[i]),
                                        static_cast<double>(cpu_steps[0].full_acc[i]));
                        }
                        deep_diffs += same ? 0 : 1;
                    }
                }

                PartialCheck acc_check, max_check, sum_check;
                long elements = 0, exact = 0, one_ulp = 0, floor_pass = 0, beyond = 0;
                long stat_mismatch = 0;
                double l2_num = 0.0, l2_den = 0.0;
                for (int step = 0; step < kSteps; ++step) {
                    const int window = prefix + step + 1;
                    const int active = plan.partition.active(window);
                    accumulate_slot0(gpu_steps[step], cpu_steps[step], geometry.q_heads, bound,
                                    acc_check, max_check, sum_check);
                    // stats of the splits the kernel wrote this step (1..active); slots >=
                    // active hold stale bytes and are not comparable between the legs
                    for (int s = 1; s < active; ++s) {
                        for (int head = 0; head < geometry.q_heads; ++head) {
                            const std::size_t stat =
                                static_cast<std::size_t>(head + geometry.q_heads * s);
                            stat_mismatch += gpu_steps[step].maximum[stat] ==
                                                    cpu_steps[step].maximum[stat]
                                                ? 0
                                                : 1;
                            stat_mismatch += gpu_steps[step].sum[stat] == cpu_steps[step].sum[stat]
                                                ? 0
                                                : 1;
                        }
                    }
                    const auto weights = max_split_weights(gpu_steps[step].maximum,
                                                           gpu_steps[step].sum, geometry.q_heads,
                                                           1, active);
                    const MergeCheck check = compare_merged(gpu_steps[step].out_bits,
                                                           cpu_steps[step].out_bits, geometry.q_heads,
                                                           &weights, bound);
                    elements += check.elements;
                    exact += check.exact;
                    one_ulp += check.one_ulp;
                    floor_pass += check.floor_pass;
                    beyond += check.beyond;
                    l2_num += check.l2_num;
                    l2_den += check.l2_den;
                }
                const double l2_rel = l2_den > 0.0 ? std::sqrt(l2_num / l2_den) : 0.0;
                const bool pass = engaged == static_cast<std::uint64_t>(kSteps) &&
                                  beyond == 0 && deep_diffs == 0 && stat_mismatch == 0 &&
                                  acc_check.failures == 0 && max_check.failures == 0 &&
                                  sum_check.failures == 0;
                failing_cases += pass ? 0 : 1;
                std::printf("%s %s prefix=%-5d steps=%d window=%d..%d capacity=%d engaged=%llu/%d\n",
                            dtype_names[dtype_index], geometry.name, prefix, kSteps, prefix + 1,
                            window_last, capacity, static_cast<unsigned long long>(engaged),
                            kSteps);
                std::printf("  seam slot partial: acc_max_rel=%.2e fail=%ld quantum=%ld; "
                            "max_rel(max)=%.2e max_rel(sum)=%.2e fail=%ld+%ld; "
                            "gpu-slot integrity=%ld\n",
                            acc_check.max_rel, acc_check.failures, acc_check.quantum_hits,
                            max_check.max_rel, sum_check.max_rel, max_check.failures,
                            sum_check.failures, deep_diffs + stat_mismatch);
                std::printf("  merged output: exact=%.4f%% one-ulp=%ld floor=%ld beyond=%ld "
                            "l2_rel=%.3e -> %s\n",
                            100.0 * static_cast<double>(exact) / static_cast<double>(elements),
                            one_ulp, floor_pass, beyond, l2_rel, pass ? "PASS" : "FAIL");
            }
        }
    }
    return failing_cases == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string mode = argc > 1 ? argv[1] : "partial";
    const int seam_split    = argc > 2 ? std::atoi(argv[2]) : 0;

    try {
        DeviceContext device(0);
        device.bind_to_current_thread();
        const DeviceExecutionView execution = device.execution_view();

        if (mode == "partial") { return run_partial_mode(execution); }
        if (mode == "ab") { return run_ab_mode(execution, seam_split); }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "diag-cpu-partial: fatal: %s\n", error.what());
        return 1;
    }
    std::fprintf(stderr, "usage: diag-cpu-partial [partial|ab] [seam-split]\n");
    return 2;
}
