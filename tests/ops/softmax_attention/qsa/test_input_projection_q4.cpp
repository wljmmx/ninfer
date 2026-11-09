#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/qsa.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/linear_test_common.h"

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <vector>

// The indexed QSA layer of the real Qwen4Exp artifact (H=2560) keeps six projections in the formats
// the artifact actually stores, not the fused Q8/BF16 forms qsa_input_proj registers:
//
//   query   q4_g64_fp16 [6144,2560]   rows [0,6144) of one q4 [12288,2560] parent
//   gate    q4_g64_fp16 [6144,2560]   rows [6144,12288) of that same parent
//   key     q4_g64_fp16 [512,2560]
//   value   q5_g64_fp16 [512,2560]
//   index_query q4_g64_fp16 [512,2560]
//   index_key   q4_g64_fp16 [128,2560]
//
// The model layer therefore takes M3's "separate rows" route (weight_input.cpp: every logical
// matrix is a complete parent region) and attention_projection runs one Linear per projection.
// This case pins that route at the operator level: the four attention projections go through the
// public four-parent attn_input_proj, the two index projections through ops::linear, all against an
// FP64 oracle decoded from the parent weights, and the workspace capacity the planner queries is
// mirrored by what the arena actually hands out.

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;
namespace {

constexpr std::int32_t kHidden = 2560;
// One A16 arithmetic profile: the same criterion the fused QSA input projection suite and the
// Linear A16 suite use. Not relaxed for the decomposed route.
constexpr ReductionCriterion kA16Criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

quantized_weight::PatternedWeightOptions row_split_options() {
    quantized_weight::PatternedWeightOptions options;
    options.row_split_scale = quantized_weight::RowSplitScalePattern::Small;
    options.row_split_codes = quantized_weight::RowSplitCodePattern::Hashed;
    return options;
}

// One parent tensor as the artifact stores it: a payload uploaded once, then row regions of it
// bound as independent Weight views (what an artifact `parts` range becomes at load time).
class Parent {
public:
    Parent(QType qtype, std::int32_t rows, std::int32_t columns, std::uint32_t seed)
        : host_(quantized_weight::make_patterned_weight(qtype, rows, columns, seed,
                                                        row_split_options())),
          device_(host_.payload.size()) {
        device_.copy_from_host(host_.payload.data(), host_.payload.size());
    }

    [[nodiscard]] const quantized_weight::PackedWeight& host() const { return host_; }
    [[nodiscard]] QType qtype() const { return host_.weight.qtype; }

    [[nodiscard]] Weight view() const { return host_.device_weight(device_.p); }

    [[nodiscard]] Weight rows(std::int32_t begin, std::int32_t count) const {
        return host_.device_row_view(device_.p, begin, count);
    }

    int verify_preserved(const std::string& label) const {
        std::vector<std::uint8_t> after(host_.payload.size());
        device_.copy_to_host(after.data(), after.size());
        if (after == host_.payload) { return 0; }
        std::cerr << label << ": packed weight was modified\n";
        return 1;
    }

private:
    quantized_weight::PackedWeight host_;
    DeviceBuffer device_;
};

struct Segment {
    const char* name;
    const Parent* parent;
    std::int32_t row_begin;
    std::int32_t rows;
};

struct Layer {
    // One q4 [12288,2560] parent holds the query and the output gate, in that row order.
    explicit Layer()
        : query_gate(QType::Q4_G64_FP16, 12288, kHidden, 307U),           // query + gate
          key(QType::Q4_G64_FP16, 512, kHidden, 311U),                   // key
          value(QType::Q5_G64_FP16, 512, kHidden, 313U),                 // value (Q5 in the artifact)
          index_query(QType::Q4_G64_FP16, 512, kHidden, 317U),           // index_query
          index_key(QType::Q4_G64_FP16, 128, kHidden, 331U) {}           // index_key

    Parent query_gate, key, value, index_query, index_key;
};

// The order the model runs them in: attention_projection's four parents (query, key, gate, value)
// then the indexer's two.
std::array<Segment, 6> segments(const Layer& layer) {
    return {Segment{"query", &layer.query_gate, 0, 6144},
            Segment{"key", &layer.key, 0, 512},
            Segment{"gate", &layer.query_gate, 6144, 6144},
            Segment{"value", &layer.value, 0, 512},
            Segment{"index_query", &layer.index_query, 0, 512},
            Segment{"index_key", &layer.index_key, 0, 128}};
}

std::vector<int> rows_to_check(const Segment& segment, bool full) {
    if (full) {
        std::vector<int> rows(static_cast<std::size_t>(segment.rows));
        std::iota(rows.begin(), rows.end(), 0);
        return rows;
    }
    const auto sampled = sampled_rows(segment.rows);
    return {sampled.begin(), sampled.end()};
}

int verify_segment(const std::string& label, const Segment& segment, const std::vector<float>& input,
                   const Tensor& output) {
    const int tokens  = output.ne[1];
    const auto words  = from_device<std::uint16_t>(output.data,
                                                   std::size_t(segment.rows) * tokens);
    for (auto word : words) {
        if ((word & 0x7f80) == 0x7f80) {
            std::cerr << label << ": output is unwritten or non-finite\n";
            return 1;
        }
    }
    const auto rows = rows_to_check(segment, tokens <= 8);
    std::vector<double> actual, expected;
    actual.reserve(rows.size() * static_cast<std::size_t>(tokens));
    expected.reserve(actual.capacity());
    for (int row : rows) {
        std::vector<double> decoded(kHidden);
        for (std::int32_t column = 0; column < kHidden; ++column) {
            decoded[column] = quantized_weight::logical_weight_fp64(
                segment.parent->host(), segment.row_begin + row, column);
        }
        for (int token = 0; token < tokens; ++token) {
            const float* column = input.data() + static_cast<std::size_t>(token) * kHidden;
            double sum          = 0.0;
            for (std::int32_t k = 0; k < kHidden; ++k) {
                sum += decoded[k] * static_cast<double>(column[k]);
            }
            actual.push_back(bf16_to_f32(words[std::size_t(token) * segment.rows + row]));
            expected.push_back(sum);
        }
    }
    return verify_reduction(label, actual, expected, kA16Criterion);
}

std::size_t capacity_bytes(const Layer& layer, std::int32_t tokens) {
    // The planner queries the whole request interval before the run: both ends of it must resolve
    // to the same workspace, and that workspace is what the arena below hands out.
    std::size_t bytes = 0;
    for (const auto& segment : segments(layer)) {
        for (const std::int32_t at : {1, tokens}) {
            bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                        segment.parent->qtype(), segment.rows, kHidden,
                                        ops::LinearPolicy::A16Only, at, at));
        }
    }
    return bytes;
}

int run_case(DeviceContext& device, const Layer& layer, std::int32_t tokens) {
    const auto input = make_bf16_activation(kHidden, tokens, 401U + static_cast<std::uint32_t>(tokens));
    const auto bits  = bf16_bits(input);
    DeviceBuffer data(bits.size() * 2);
    data.copy_from_host(bits.data(), data.bytes);
    Tensor x(data.p, DType::BF16, {kHidden, tokens});

    std::vector<std::unique_ptr<GuardedBf16Tensor>> backing;
    std::array<Tensor, 6> out;
    const auto segment_table = segments(layer);
    for (std::size_t i = 0; i < segment_table.size(); ++i) {
        backing.push_back(
            std::make_unique<GuardedBf16Tensor>(segment_table[i].rows, tokens));
        out[i] = backing.back()->tensor();
    }

    const std::size_t capacity = capacity_bytes(layer, tokens);
    // DeviceArena rejects a zero capacity; a 1-byte arena is the exact mirror of "the query says
    // this route needs no scratch", which the usage/peak assertions below then confirm.
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 1));
    workspace.reset_peak();

    Weight query = layer.query_gate.rows(0, 6144);
    Weight key   = layer.key.rows(0, 512);
    Weight gate  = layer.query_gate.rows(6144, 6144);
    Weight value = layer.value.rows(0, 512);
    ops::attn_input_proj(x, query, key, gate, value, out[0], out[2], out[1], out[3],
                         ops::LinearPolicy::A16Only, workspace, device.stream);
    ops::linear(x, layer.index_query.rows(0, 512), out[4], ops::LinearPolicy::A16Only, workspace,
                device.stream);
    ops::linear(x, layer.index_key.rows(0, 128), out[5], ops::LinearPolicy::A16Only, workspace,
                device.stream);
    device.synchronize();

    const std::string base = "QSA_Q4 T=" + std::to_string(tokens);
    int failures           = 0;
    if (workspace.used() != 0 || workspace.peak_used() != 0 || capacity != 0) {
        std::cerr << base << ": workspace query " << capacity << " reported, arena used "
                  << workspace.used() << " peak " << workspace.peak_used() << '\n';
        ++failures;
    }
    for (std::size_t i = 0; i < segment_table.size(); ++i) {
        failures += verify_segment(base + " " + segment_table[i].name, segment_table[i], input,
                                   out[i]);
        failures += backing[i]->verify_guards(base + " " + segment_table[i].name);
    }
    failures += verify_preserved(base + " input", data, bits);
    failures += layer.query_gate.verify_preserved(base + " query/gate parent");
    failures += layer.key.verify_preserved(base + " key parent");
    failures += layer.value.verify_preserved(base + " value parent");
    failures += layer.index_query.verify_preserved(base + " index_query parent");
    failures += layer.index_key.verify_preserved(base + " index_key parent");
    return failures;
}

template <class F> int rejects(const char* label, F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": accepted invalid operands\n";
    return 1;
}

// The fused Op stays the Q8/BF16 form it was written for; the artifact's Q4/Q5 split is served by
// the separate-row route above. Pin the boundary so a future edit cannot silently claim the fused
// kernel is what runs here.
int fused_op_boundary(DeviceContext& device, const Layer& layer) {
    DeviceBuffer data(std::size_t(kHidden) * 2 * 2);
    Tensor x(data.p, DType::BF16, {kHidden, 2});
    GuardedBf16Tensor query(6144, 2), gate(6144, 2), key(512, 2), value(512, 2),
        index_query(512, 2), index_key(128, 2);
    Tensor query_view = query.tensor(), gate_view = gate.tensor(), key_view = key.tensor(),
           value_view = value.tensor(), index_query_view = index_query.tensor(),
           index_key_view = index_key.tensor();
    int failures = 0;
    failures += rejects("fused query/gate accepts Q4", [&] {
        ops::qsa_input_proj(x, layer.query_gate.view(), query_view, gate_view, device.stream);
    });
    failures += rejects("fused kv/index accepts Q4+Q5", [&] {
        ops::qsa_input_proj(x, layer.key.view(), key_view, value_view, index_query_view,
                            index_key_view, device.stream);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        DeviceContext device;
        Layer layer;
        int failures = fused_op_boundary(device, layer);
        for (const std::int32_t tokens : {1, 2, 8, 17, 64, 128}) {
            failures += run_case(device, layer, tokens);
        }
        if (failures) return 1;
        std::cout << "OK QSA Q4/Q5 decomposed input projections (artifact formats, capacity "
                     "mirror)\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
