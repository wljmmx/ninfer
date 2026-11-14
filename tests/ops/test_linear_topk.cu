#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_topk.h"

#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kFullRows  = 248320;
constexpr std::int32_t kValidRows = 248077;
constexpr std::int32_t kShortRows = 131072;
constexpr std::int32_t kTopK      = 16;
constexpr int kMaxColumns         = 257;

constexpr ReductionCriterion kA16ScoreCriterion{
    /*relative_l2=*/1.0 / 256.0,
    /*gross_absolute=*/1.0 / 256.0,
    /*gross_relative_to_max_reference=*/2.0 / 256.0,
};

struct FixtureWeight {
    DeviceBuffer payload;
    Weight weight;
    std::uint64_t scale_offset = 0;
};

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

FixtureWeight make_rowsplit(QType qtype, std::int32_t rows) {
    const std::int32_t group         = qtype == QType::Q8_G32_FP16 ? 32 : 64;
    const std::uint64_t groups       = static_cast<std::uint64_t>(rows) * (kHidden / group);
    const std::uint64_t code_bytes   = qtype == QType::Q8_G32_FP16
                                           ? static_cast<std::uint64_t>(rows) * kHidden
                                           : static_cast<std::uint64_t>(rows) * kHidden / 2;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = groups * sizeof(std::uint16_t);
    FixtureWeight result{
        DeviceBuffer(static_cast<std::size_t>(scale_offset + scale_bytes)), {}, scale_offset};
    result.payload.fill(0);
    result.weight.payload          = result.payload.p;
    result.weight.payload_bytes    = result.payload.bytes;
    result.weight.high_plane_bytes = 0;
    result.weight.qtype            = qtype;
    result.weight.group_size       = group;
    result.weight.shape[0]         = rows;
    result.weight.shape[1]         = kHidden;
    result.weight.padded_shape[0]  = rows;
    result.weight.padded_shape[1]  = kHidden;
    result.weight.ndim             = 2;
    result.weight.qdata            = result.payload.p;
    result.weight.qhigh            = nullptr;
    result.weight.scales           = static_cast<std::uint8_t*>(result.payload.p) + scale_offset;
    result.weight.n                = rows;
    result.weight.k                = kHidden;
    result.weight.group            = group;
    result.weight.layout           = QuantLayout::RowSplit;
    result.weight.scale_dtype      = DType::FP16;
    return result;
}

FixtureWeight make_fp8() {
    const std::uint64_t code_bytes   = static_cast<std::uint64_t>(kFullRows) * kHidden;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = static_cast<std::uint64_t>(kFullRows) * 2;
    FixtureWeight result{
        DeviceBuffer(static_cast<std::size_t>(scale_offset + scale_bytes)), {}, scale_offset};
    result.payload.fill(0);
    result.weight.payload          = result.payload.p;
    result.weight.payload_bytes    = result.payload.bytes;
    result.weight.high_plane_bytes = 0;
    result.weight.qtype            = QType::FP8_E4M3FN_ROW_BF16;
    result.weight.group_size       = kHidden;
    result.weight.shape[0]         = kFullRows;
    result.weight.shape[1]         = kHidden;
    result.weight.padded_shape[0]  = kFullRows;
    result.weight.padded_shape[1]  = kHidden;
    result.weight.ndim             = 2;
    result.weight.qdata            = result.payload.p;
    result.weight.qhigh            = nullptr;
    result.weight.scales           = static_cast<std::uint8_t*>(result.payload.p) + scale_offset;
    result.weight.n                = kFullRows;
    result.weight.k                = kHidden;
    result.weight.group            = kHidden;
    result.weight.layout           = QuantLayout::RowScale;
    result.weight.scale_dtype      = DType::BF16;
    result.weight.scale_ne[0]      = kFullRows;
    result.weight.scale_nb[0]      = 2;
    result.weight.scale_nb[1]      = static_cast<std::int64_t>(kFullRows) * 2;
    result.weight.scale_nb[2]      = result.weight.scale_nb[1];
    result.weight.scale_nb[3]      = result.weight.scale_nb[1];
    return result;
}

const std::array<std::int32_t, 17> kFullWinnerRows{
    3,     127,   511,   512,   1023,   4095,   8191,   15872,  16383,
    16384, 16895, 32767, 65535, 131071, 196607, 247808, 248076,
};

const std::array<std::int32_t, 17> kShortWinnerRows{
    3,     127,   511,   512,   1023,  4095,  8191,   15872,  16383,
    16384, 16895, 32767, 32768, 65535, 65536, 130560, 131071,
};

float factor_for(std::size_t index) { return index >= 15 ? 17.0F : static_cast<float>(index + 1); }

float group_base_scale(std::int32_t group) {
    return 0.0137F + static_cast<float>(group % 7) * 0.0071F;
}

std::int32_t rowsplit_code(QType qtype, std::int32_t k) {
    const int value = 1 + k % (qtype == QType::Q8_G32_FP16 ? 113 : 7);
    return k % 5 == 0 ? -value : value;
}

std::uint8_t fp8_code(std::int32_t k) {
    constexpr std::uint8_t codes[]{0x30, 0x32, 0x34, 0x38}; // 0.5, 0.625, 0.75, 1.0
    return codes[k & 3] | (k % 5 == 0 ? 0x80 : 0);
}

void patch_rowsplit_row(FixtureWeight& fixture, QType qtype, std::int32_t row, float factor) {
    const std::size_t code_row_bytes = qtype == QType::Q8_G32_FP16 ? kHidden : kHidden / 2;
    std::vector<std::uint8_t> codes(code_row_bytes);
    if (qtype == QType::Q8_G32_FP16) {
        for (std::int32_t k = 0; k < kHidden; ++k) {
            codes[static_cast<std::size_t>(k)] = static_cast<std::uint8_t>(rowsplit_code(qtype, k));
        }
    } else {
        for (std::int32_t pair = 0; pair < kHidden / 2; ++pair) {
            const std::uint8_t low  = static_cast<std::uint8_t>(rowsplit_code(qtype, pair * 2));
            const std::uint8_t high = static_cast<std::uint8_t>(rowsplit_code(qtype, pair * 2 + 1));
            codes[static_cast<std::size_t>(pair)] =
                static_cast<std::uint8_t>((low & 15) | (high << 4));
        }
    }
    fixture.payload.copy_from_host(codes.data(), codes.size(),
                                   static_cast<std::size_t>(row) * code_row_bytes);
    const std::int32_t groups = kHidden / (qtype == QType::Q8_G32_FP16 ? 32 : 64);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(groups));
    for (std::int32_t group = 0; group < groups; ++group) {
        scales[static_cast<std::size_t>(group)] =
            quantized_weight::detail::f32_to_f16(factor * group_base_scale(group));
    }
    fixture.payload.copy_from_host(scales.data(), scales.size() * sizeof(std::uint16_t),
                                   static_cast<std::size_t>(fixture.scale_offset) +
                                       static_cast<std::size_t>(row) * groups *
                                           sizeof(std::uint16_t));
}

void patch_fp8_row(FixtureWeight& fixture, std::int32_t row, float factor) {
    std::vector<std::uint8_t> codes(kHidden);
    for (std::int32_t k = 0; k < kHidden; ++k) { codes[static_cast<std::size_t>(k)] = fp8_code(k); }
    fixture.payload.copy_from_host(codes.data(), codes.size(),
                                   static_cast<std::size_t>(row) * kHidden);
    const std::uint16_t scale = f32_to_bf16(factor);
    fixture.payload.copy_from_host(&scale, sizeof(scale),
                                   static_cast<std::size_t>(fixture.scale_offset) +
                                       static_cast<std::size_t>(row) * sizeof(scale));
}

std::vector<std::uint16_t> make_hidden() {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(kHidden) * kMaxColumns);
    for (std::int32_t column = 0; column < kMaxColumns; ++column) {
        for (std::int32_t k = 0; k < kHidden; ++k) {
            const float source = 0.25F + static_cast<float>(k & 7) * 0.125F +
                                 static_cast<float>(column % 17) * 0.0625F;
            const std::uint16_t bits                               = f32_to_bf16(source);
            hidden[static_cast<std::size_t>(column) * kHidden + k] = bits;
        }
    }
    return hidden;
}

// Decode the actual represented signed code and stored scale, without staging BF16 casts.
std::vector<double> base_scores(QType qtype, const std::vector<std::uint16_t>& hidden) {
    std::vector<double> scores(18 * kMaxColumns);
    for (int factor = 1; factor <= 17; ++factor)
        for (int column = 0; column < kMaxColumns; ++column) {
            double sum = 0;
            for (int k = 0; k < kHidden; ++k) {
                double weight;
                if (qtype == QType::FP8_E4M3FN_ROW_BF16) {
                    weight =
                        quantized_weight::detail::decode_e4m3fn(fp8_code(k)) *
                        static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(factor))));
                } else {
                    const int group = k / (qtype == QType::Q8_G32_FP16 ? 32 : 64);
                    const auto scale =
                        quantized_weight::detail::f32_to_f16(factor * group_base_scale(group));
                    weight = static_cast<double>(rowsplit_code(qtype, k)) *
                             quantized_weight::detail::f16_to_f32(scale);
                }
                sum += weight * bf16_to_f32(hidden[static_cast<std::size_t>(column) * kHidden + k]);
            }
            scores[factor * kMaxColumns + column] = sum;
        }
    return scores;
}

template <std::size_t N>
std::vector<std::pair<float, std::int32_t>>
expected_order(const std::array<std::int32_t, N>& rows, const std::vector<std::int32_t>* id_map) {
    std::vector<std::pair<float, std::int32_t>> candidates;
    candidates.reserve(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const std::int32_t id = id_map == nullptr ? rows[index] : (*id_map)[rows[index]];
        candidates.emplace_back(factor_for(index), id);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
    });
    candidates.resize(kTopK);
    return candidates;
}

int verify_invocation(const char* profile, std::int32_t columns, const Tensor& ids_tensor,
                      const Tensor& scores_tensor, const std::vector<double>& base_score,
                      const std::vector<std::pair<float, std::int32_t>>& expected) {
    const std::size_t count = static_cast<std::size_t>(kTopK) * columns;
    const auto ids          = from_device<std::int32_t>(ids_tensor.data, count);
    const auto scores       = from_device<float>(scores_tensor.data, count);
    std::vector<double> actual_scores(count);
    std::vector<double> reference_scores(count);
    int failures = 0;
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t offset       = static_cast<std::size_t>(column) * kTopK + rank;
            const std::int32_t expected_id = expected[rank].second;
            const double expected_score =
                base_score[static_cast<int>(expected[rank].first) * kMaxColumns + column];
            if (ids[offset] != expected_id) {
                std::cerr << profile << " U=" << columns << " column=" << column << " rank=" << rank
                          << " id got=" << ids[offset] << " expected=" << expected_id << '\n';
                ++failures;
            }
            actual_scores[offset]    = static_cast<double>(scores[offset]);
            reference_scores[offset] = expected_score;
        }
    }
    failures += verify_reduction(std::string(profile) + " U=" + std::to_string(columns) + " scores",
                                 actual_scores, reference_scores, kA16ScoreCriterion);
    return failures;
}

template <class Launch>
void replay_graph_twice(Launch&& launch, cudaStream_t stream) {
    cudaGraph_t graph         = nullptr;
    cudaGraphExec_t execution = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "linear_topk begin capture");
    launch(stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "linear_topk end capture");
    cuda_check(cudaGraphInstantiate(&execution, graph, 0), "linear_topk instantiate graph");
    cuda_check(cudaGraphLaunch(execution, stream), "linear_topk first graph replay");
    cuda_check(cudaGraphLaunch(execution, stream), "linear_topk second graph replay");
    cuda_synchronize(stream);
    cuda_check(cudaGraphExecDestroy(execution), "linear_topk destroy graph execution");
    cuda_check(cudaGraphDestroy(graph), "linear_topk destroy graph");
}

std::vector<int> test_columns() {
    std::vector<int> result;
    for (int n = 1; n <= 120; ++n) result.push_back(n);
    for (int n : {121, 127, 128, 129, 257}) result.push_back(n);
    return result;
}

int verify_zero_ties(const FixtureWeight& fixture, const Tensor* id_map,
                     const std::vector<std::int32_t>* host_map) {
    constexpr int columns = 3;
    DeviceBuffer hidden(static_cast<std::size_t>(kHidden) * columns * 2);
    hidden.fill(0);
    GuardedDeviceBuffer ids(kTopK * columns * 4), scores(kTopK * columns * 4);
    Tensor x(hidden.p, DType::BF16, {kHidden, columns});
    Tensor out_ids(ids.data(), DType::I32, {kTopK, columns});
    Tensor out_scores(scores.data(), DType::FP32, {kTopK, columns});
    WorkspaceArena workspace(ops::linear_topk_workspace_capacity_bytes(
        fixture.weight.qtype, fixture.weight.n, kHidden, columns, columns));
    if (id_map)
        ops::linear_topk(x, fixture.weight, *id_map, out_ids, out_scores, workspace, nullptr);
    else
        ops::linear_topk(x, fixture.weight, kValidRows, out_ids, out_scores, workspace, nullptr);
    cuda_synchronize();
    std::vector<int> expected;
    if (host_map) {
        expected = *host_map;
        std::sort(expected.begin(), expected.end());
        expected.resize(kTopK);
    } else
        for (int rank = 0; rank < kTopK; ++rank) expected.push_back(rank);
    const auto got_ids    = from_device<int>(ids.data(), kTopK * columns);
    const auto got_scores = from_device<float>(scores.data(), kTopK * columns);
    int failures          = 0;
    for (int i = 0; i < kTopK * columns; ++i)
        if (got_ids[i] != expected[i % kTopK] || got_scores[i] != 0) ++failures;
    if (failures) std::cerr << "linear_topk full-vocabulary zero tie failure\n";
    return failures + ids.verify_guards("zero-tie ids") + scores.verify_guards("zero-tie scores");
}

int run_full(QType qtype, const char* profile, const DeviceBuffer& hidden,
             const std::vector<double>& base_score) {
    FixtureWeight fixture =
        qtype == QType::Q8_G32_FP16 ? make_rowsplit(qtype, kFullRows) : make_fp8();
    for (std::size_t index = 0; index < kFullWinnerRows.size(); ++index) {
        if (qtype == QType::Q8_G32_FP16) {
            patch_rowsplit_row(fixture, qtype, kFullWinnerRows[index], factor_for(index));
        } else {
            patch_fp8_row(fixture, kFullWinnerRows[index], factor_for(index));
        }
    }
    if (qtype == QType::Q8_G32_FP16) {
        patch_rowsplit_row(fixture, qtype, kFullRows - 1, 64.0F);
    } else {
        patch_fp8_row(fixture, kFullRows - 1, 64.0F);
    }

    const auto expected = expected_order(kFullWinnerRows, nullptr);
    const std::size_t capacity =
        ops::linear_topk_workspace_capacity_bytes(qtype, kFullRows, kHidden, 1, kMaxColumns);
    GuardedDeviceBuffer graph_scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{graph_scratch.data(), graph_scratch.bytes()});
    DeviceBuffer ids(static_cast<std::size_t>(kTopK) * kMaxColumns * sizeof(std::int32_t));
    DeviceBuffer scores(static_cast<std::size_t>(kTopK) * kMaxColumns * sizeof(float));
    int failures = 0;
    for (int columns : test_columns()) {
        GuardedDeviceBuffer x(static_cast<std::size_t>(kHidden) * columns * 2);
        GuardedDeviceBuffer out_ids(static_cast<std::size_t>(kTopK) * columns * 4);
        GuardedDeviceBuffer out_scores(static_cast<std::size_t>(kTopK) * columns * 4);
        cuda_check(cudaMemcpy(x.data(), hidden.p, x.bytes(), cudaMemcpyDeviceToDevice),
                   "copy input fixture");
        const auto before = from_device<std::uint16_t>(x.data(), x.bytes() / 2);
        const auto exact  = ops::linear_topk_workspace_capacity_bytes(
            fixture.weight.qtype, fixture.weight.n, kHidden, columns, columns);
        GuardedDeviceBuffer scratch(exact);
        WorkspaceArena point_workspace(DeviceSpan{scratch.data(), exact});
        Tensor hidden_tensor(x.data(), DType::BF16, {kHidden, columns});
        Tensor ids_tensor(out_ids.data(), DType::I32, {kTopK, columns});
        Tensor scores_tensor(out_scores.data(), DType::FP32, {kTopK, columns});
        out_ids.fill(0xcd);
        out_scores.fill(0xff);
        ops::linear_topk(hidden_tensor, fixture.weight, kValidRows, ids_tensor, scores_tensor,
                         point_workspace, nullptr);
        cuda_synchronize();
        failures +=
            verify_invocation(profile, columns, ids_tensor, scores_tensor, base_score, expected);
        failures += out_ids.verify_guards("ids tail");
        failures += out_scores.verify_guards("scores tail");
        failures += scratch.verify_guards("workspace tail");
        failures += x.verify_guards("hidden guards");
        if (from_device<std::uint16_t>(x.data(), x.bytes() / 2) != before) {
            std::cerr << "linear_topk modified hidden\n";
            ++failures;
        }
        if (point_workspace.used() != 0 || point_workspace.peak_used() > exact) {
            std::cerr << "linear_topk workspace scope/peak failure\n";
            ++failures;
        }
    }

    for (int graph_columns : {1,  8,  9,  16, 17, 24, 25, 32,  33,  48,  49, 64,
                              65, 80, 81, 88, 89, 96, 97, 112, 113, 120, 129}) {
        Tensor graph_hidden(hidden.p, DType::BF16, {kHidden, graph_columns});
        Tensor graph_ids(ids.p, DType::I32, {kTopK, graph_columns});
        Tensor graph_scores(scores.p, DType::FP32, {kTopK, graph_columns});
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreate(&stream), "linear_topk create graph stream");
        replay_graph_twice(
            [&](cudaStream_t captured_stream) {
                ops::linear_topk(graph_hidden, fixture.weight, kValidRows, graph_ids, graph_scores,
                                 workspace, captured_stream);
            },
            stream);
        cuda_check(cudaStreamDestroy(stream), "linear_topk destroy graph stream");
        failures += verify_invocation(profile, graph_columns, graph_ids, graph_scores, base_score,
                                      expected);
    }
    failures += graph_scratch.verify_guards("graph workspace tail");
    failures += verify_zero_ties(fixture, nullptr, nullptr);
    return failures;
}

int run_q4(const DeviceBuffer& hidden, const std::vector<double>& base_score) {
    FixtureWeight fixture = make_rowsplit(QType::Q4_G64_FP16, kShortRows);
    for (std::size_t index = 0; index < kShortWinnerRows.size(); ++index) {
        patch_rowsplit_row(fixture, QType::Q4_G64_FP16, kShortWinnerRows[index], factor_for(index));
    }
    std::vector<std::int32_t> host_map(kShortRows);
    for (std::int32_t row = 0; row < kShortRows; ++row) { host_map[row] = kValidRows - 1 - row; }
    DeviceBuffer map(host_map.size() * sizeof(std::int32_t));
    map.copy_from_host(host_map.data(), map.bytes);
    const auto expected = expected_order(kShortWinnerRows, &host_map);

    const std::size_t capacity = ops::linear_topk_workspace_capacity_bytes(
        QType::Q4_G64_FP16, kShortRows, kHidden, 1, kMaxColumns);
    GuardedDeviceBuffer graph_scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{graph_scratch.data(), graph_scratch.bytes()});
    DeviceBuffer ids(static_cast<std::size_t>(kTopK) * kMaxColumns * sizeof(std::int32_t));
    DeviceBuffer scores(static_cast<std::size_t>(kTopK) * kMaxColumns * sizeof(float));
    Tensor map_tensor(map.p, DType::I32, {kShortRows});
    int failures = 0;
    for (int columns : test_columns()) {
        GuardedDeviceBuffer x(static_cast<std::size_t>(kHidden) * columns * 2);
        GuardedDeviceBuffer out_ids(static_cast<std::size_t>(kTopK) * columns * 4);
        GuardedDeviceBuffer out_scores(static_cast<std::size_t>(kTopK) * columns * 4);
        cuda_check(cudaMemcpy(x.data(), hidden.p, x.bytes(), cudaMemcpyDeviceToDevice),
                   "copy input fixture");
        const auto before = from_device<std::uint16_t>(x.data(), x.bytes() / 2);
        const auto exact  = ops::linear_topk_workspace_capacity_bytes(
            fixture.weight.qtype, fixture.weight.n, kHidden, columns, columns);
        GuardedDeviceBuffer scratch(exact);
        WorkspaceArena point_workspace(DeviceSpan{scratch.data(), exact});
        Tensor hidden_tensor(x.data(), DType::BF16, {kHidden, columns});
        Tensor ids_tensor(out_ids.data(), DType::I32, {kTopK, columns});
        Tensor scores_tensor(out_scores.data(), DType::FP32, {kTopK, columns});
        out_ids.fill(0xcd);
        out_scores.fill(0xff);
        ops::linear_topk(hidden_tensor, fixture.weight, map_tensor, ids_tensor, scores_tensor,
                         point_workspace, nullptr);
        cuda_synchronize();
        failures += verify_invocation("q4-optimized", columns, ids_tensor, scores_tensor,
                                      base_score, expected);
        failures += out_ids.verify_guards("ids tail");
        failures += out_scores.verify_guards("scores tail");
        failures += scratch.verify_guards("workspace tail");
        failures += x.verify_guards("hidden guards");
        if (from_device<std::uint16_t>(x.data(), x.bytes() / 2) != before) {
            std::cerr << "linear_topk modified hidden\n";
            ++failures;
        }
        if (point_workspace.used() != 0 || point_workspace.peak_used() > exact) {
            std::cerr << "linear_topk workspace scope/peak failure\n";
            ++failures;
        }
    }

    for (int graph_columns : {1,  8,  9,  16, 17, 24, 25, 32,  33,  48,  49, 64,
                              65, 80, 81, 88, 89, 96, 97, 112, 113, 120, 129}) {
        Tensor graph_hidden(hidden.p, DType::BF16, {kHidden, graph_columns});
        Tensor graph_ids(ids.p, DType::I32, {kTopK, graph_columns});
        Tensor graph_scores(scores.p, DType::FP32, {kTopK, graph_columns});
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreate(&stream), "linear_topk create graph stream");
        replay_graph_twice(
            [&](cudaStream_t captured_stream) {
                ops::linear_topk(graph_hidden, fixture.weight, map_tensor, graph_ids, graph_scores,
                                 workspace, captured_stream);
            },
            stream);
        cuda_check(cudaStreamDestroy(stream), "linear_topk destroy graph stream");
        failures += verify_invocation("q4-optimized graph", graph_columns, graph_ids, graph_scores,
                                      base_score, expected);
    }
    failures += graph_scratch.verify_guards("graph workspace tail");
    failures += verify_zero_ties(fixture, &map_tensor, &host_map);
    if (from_device<std::int32_t>(map.p, host_map.size()) != host_map) {
        std::cerr << "linear_topk modified id map\n";
        ++failures;
    }
    return failures;
}

// Q5 draft head: the Qwen3.8-Flash-Next output head is Q5_G64_FP16 [248320, 2560], and the DFlash2
// candidate selection is the only reader of it in this geometry. Three planes (codes, one bit per
// value, one fp16 scale per 64 values) are staged exactly as the dense Q5 GEMMs stage them, and the
// oracle decodes the represented values on the host in FP64 - independent of every device decode -
// so a wrong high-plane bit or a scale landing on the wrong group shows up as a different id.
constexpr std::int32_t kDraftK        = 2560;
constexpr std::int32_t kDraftGroupK   = 64;
constexpr std::int32_t kDraftGroups   = kDraftK / kDraftGroupK;
constexpr std::int32_t kDraftCodeRow  = kDraftK / 2;
constexpr std::int32_t kDraftHighRow  = kDraftK / 8;
constexpr std::int32_t kDraftScaleRow = kDraftGroups * 2;

// Every winner stays below kValidRows: the full head masks the reserved tail, so a fixture row there
// is dropped by design and would look like a kernel failure.
const std::array<std::int32_t, 18> kQ5WinnerRows{
    5,     128,   511,   512,   1023,   4095,   8191,   15872,  196607,
    16383, 16384, 16895, 32767, 65535, 131071, 247806, 248075, 248076,
};

struct Q5Fixture {
    DeviceBuffer payload;
    Weight weight;
    std::uint64_t high_offset  = 0;
    std::uint64_t scale_offset = 0;
};

Q5Fixture make_q5_full() {
    const std::uint64_t code_bytes  = static_cast<std::uint64_t>(kFullRows) * kDraftCodeRow;
    const std::uint64_t high_bytes  = static_cast<std::uint64_t>(kFullRows) * kDraftHighRow;
    const std::uint64_t scale_bytes = static_cast<std::uint64_t>(kFullRows) * kDraftScaleRow;
    const std::uint64_t high_offset  = align_up(code_bytes, 256);
    const std::uint64_t scale_offset = align_up(high_offset + high_bytes, 256);
    Q5Fixture result{DeviceBuffer(static_cast<std::size_t>(scale_offset + scale_bytes) + 256), {},
                     0, 0};
    result.payload.fill(0);
    result.high_offset  = high_offset;
    result.scale_offset = scale_offset;
    auto* base          = static_cast<std::uint8_t*>(result.payload.p);
    result.weight.payload          = result.payload.p;
    result.weight.payload_bytes    = result.payload.bytes;
    result.weight.high_plane_bytes = high_bytes;
    result.weight.qtype            = QType::Q5_G64_FP16;
    result.weight.group_size       = kDraftGroupK;
    result.weight.shape[0]         = kFullRows;
    result.weight.shape[1]         = kDraftK;
    result.weight.padded_shape[0]  = kFullRows;
    result.weight.padded_shape[1]  = kDraftK;
    result.weight.ndim             = 2;
    result.weight.qdata            = base;
    result.weight.qhigh            = base + high_offset;
    result.weight.scales           = base + scale_offset;
    result.weight.n                = kFullRows;
    result.weight.k                = kDraftK;
    result.weight.group            = kDraftGroupK;
    result.weight.layout           = QuantLayout::RowSplit;
    result.weight.scale_dtype      = DType::FP16;
    return result;
}

// Five-bit signed value at (group, lane): the low four bits live in the code plane, the fifth in the
// high plane, and the value is sign-extended from bit 4. The pattern is row independent, so a row's
// score is its scale factor times one fixed projection - which is what makes the ranking below
// exact - while still using both signs, all five bits and every group.
std::int32_t q5_value(std::int32_t group, std::int32_t lane) {
    const std::int32_t span = 1 + (group * 3 + lane * 5) % 15;
    return (group + lane) % 7 == 0 ? -span : span;
}

void patch_q5_row(Q5Fixture& fixture, std::int32_t row, float factor) {
    auto* base = static_cast<std::uint8_t*>(fixture.payload.p);
    std::vector<std::uint8_t> codes(kDraftCodeRow, 0);
    std::vector<std::uint8_t> high(kDraftHighRow, 0);
    for (std::int32_t group = 0; group < kDraftGroups; ++group) {
        for (std::int32_t lane_pair = 0; lane_pair < kDraftGroupK / 2; ++lane_pair) {
            const std::int32_t v0 = q5_value(group, lane_pair * 2);
            const std::int32_t v1 = q5_value(group, lane_pair * 2 + 1);
            const std::uint8_t low0 = static_cast<std::uint8_t>(v0 & 0x0f);
            const std::uint8_t low1 = static_cast<std::uint8_t>(v1 & 0x0f);
            const std::uint8_t high0 = static_cast<std::uint8_t>((v0 >> 4) & 1);
            const std::uint8_t high1 = static_cast<std::uint8_t>((v1 >> 4) & 1);
            codes[static_cast<std::size_t>(group) * 32 + lane_pair] =
                static_cast<std::uint8_t>(low0 | (low1 << 4));
            const std::int32_t high_byte = group * 8 + (lane_pair * 2) / 8;
            high[static_cast<std::size_t>(high_byte)] |= static_cast<std::uint8_t>(
                (high0 << ((lane_pair * 2) % 8)) | (high1 << ((lane_pair * 2 + 1) % 8)));
        }
    }
    std::vector<std::uint16_t> scales(kDraftGroups);
    for (std::int32_t group = 0; group < kDraftGroups; ++group) {
        scales[static_cast<std::size_t>(group)] =
            quantized_weight::detail::f32_to_f16(factor * group_base_scale(group));
    }
    fixture.payload.copy_from_host(codes.data(), codes.size(),
                                   static_cast<std::size_t>(row) * kDraftCodeRow);
    fixture.payload.copy_from_host(high.data(), high.size(),
                                   static_cast<std::size_t>(fixture.high_offset) +
                                       static_cast<std::size_t>(row) * kDraftHighRow);
    fixture.payload.copy_from_host(scales.data(), scales.size() * sizeof(std::uint16_t),
                                   static_cast<std::size_t>(fixture.scale_offset) +
                                       static_cast<std::size_t>(row) * kDraftScaleRow);
    (void)base;
}

// FP64 scores of the represented weights against one hidden vector, for the rows that are not zero.
std::vector<double> q5_base_scores(const std::vector<std::uint16_t>& hidden) {
    std::vector<double> scores(kQ5WinnerRows.size());
    for (std::size_t index = 0; index < kQ5WinnerRows.size(); ++index) {
        double sum = 0;
        for (std::int32_t group = 0; group < kDraftGroups; ++group) {
            const auto scale_bits = quantized_weight::detail::f32_to_f16(
                static_cast<float>(index + 1) * group_base_scale(group));
            const double scale = quantized_weight::detail::f16_to_f32(scale_bits);
            for (std::int32_t lane = 0; lane < kDraftGroupK; ++lane) {
                const std::int32_t value = q5_value(group, lane);
                const double weight      = static_cast<double>(value) * scale;
                sum += weight * bf16_to_f32(
                                   hidden[static_cast<std::size_t>(group * kDraftGroupK + lane)]);
            }
        }
        scores[index] = sum;
    }
    return scores;
}

// Deterministic BF16 hidden of the draft width, positive like a real residual stream (a random-sign
// vector would cancel a five-bit projection down to noise); three seeds stand in for three decode
// steps.
std::vector<std::uint16_t> make_draft_hidden(std::int32_t seed, std::int32_t columns) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(kDraftK) * columns);
    std::uint32_t state = 0x9e3779b9u ^ static_cast<std::uint32_t>(seed * 2654435761u);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        const float unit = static_cast<float>(state >> 8) / static_cast<float>(1u << 24);
        hidden[index]    = f32_to_bf16(0.25F + unit * 1.5F);
    }
    return hidden;
}

int run_q5_draft_head() {
    Q5Fixture fixture = make_q5_full();
    for (std::size_t index = 0; index < kQ5WinnerRows.size(); ++index) {
        patch_q5_row(fixture, kQ5WinnerRows[index], static_cast<float>(index + 1));
    }

    int failures = 0;
    for (std::int32_t seed = 0; seed < 3; ++seed) {
        const std::vector<std::uint16_t> host_hidden = make_draft_hidden(seed, 1);
        const auto base                              = q5_base_scores(host_hidden);
        std::vector<std::pair<double, std::int32_t>> ranked;
        for (std::size_t index = 0; index < kQ5WinnerRows.size(); ++index) {
            ranked.emplace_back(base[index], kQ5WinnerRows[index]);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
        });
        ranked.resize(kTopK);

        DeviceBuffer hidden(host_hidden.size() * sizeof(std::uint16_t));
        hidden.copy_from_host(host_hidden.data(), hidden.bytes);
        GuardedDeviceBuffer out_ids(static_cast<std::size_t>(kTopK) * sizeof(std::int32_t));
        GuardedDeviceBuffer out_scores(static_cast<std::size_t>(kTopK) * sizeof(float));
        const std::size_t capacity = ops::linear_topk_workspace_capacity_bytes(
            QType::Q5_G64_FP16, kFullRows, kDraftK, 1, 1);
        GuardedDeviceBuffer scratch(capacity);
        WorkspaceArena workspace(DeviceSpan{scratch.data(), capacity});
        Tensor x(hidden.p, DType::BF16, {kDraftK, 1});
        Tensor ids_tensor(out_ids.data(), DType::I32, {kTopK, 1});
        Tensor scores_tensor(out_scores.data(), DType::FP32, {kTopK, 1});
        ops::linear_topk(x, fixture.weight, kValidRows, ids_tensor, scores_tensor, workspace, nullptr);
        cuda_synchronize();

        const auto ids    = from_device<std::int32_t>(out_ids.data(), kTopK);
        const auto scores = from_device<float>(out_scores.data(), kTopK);

        // The dense Q5 head is the path the ordinary decode samples from; its logits are the second
        // opinion, so the top-1 id and the top-8 set are compared against it as well.
        std::vector<float> logits(static_cast<std::size_t>(kFullRows));
        {
            // The public activation and output of every Linear route are BF16, so the second opinion
            // is read back in BF16 and widened here.
            DeviceBuffer dense(static_cast<std::size_t>(kFullRows) * sizeof(std::uint16_t));
            Tensor dense_tensor(dense.p, DType::BF16, {kFullRows, 1});
            ops::linear(x, fixture.weight, dense_tensor, nullptr);
            cuda_synchronize();
            const auto wide = from_device<std::uint16_t>(dense.p, kFullRows);
            for (std::size_t index = 0; index < wide.size(); ++index) {
                logits[index] = bf16_to_f32(wide[index]);
            }
        }
        std::vector<std::int32_t> order(kFullRows);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 8, order.end(),
                          [&](std::int32_t lhs, std::int32_t rhs) {
                              return logits[static_cast<std::size_t>(lhs)] >
                                         logits[static_cast<std::size_t>(rhs)] ||
                                     (logits[static_cast<std::size_t>(lhs)] ==
                                          logits[static_cast<std::size_t>(rhs)] &&
                                      lhs < rhs);
                          });

        int matched_ids = 0;
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            if (ids[rank] == ranked[static_cast<std::size_t>(rank)].second) {
                ++matched_ids;
            } else {
                std::cerr << "q5-draft seed=" << seed << " rank=" << rank << " id got=" << ids[rank]
                          << " oracle=" << ranked[static_cast<std::size_t>(rank)].second << '\n';
                ++failures;
            }
        }
        int dense_top8 = 0;
        {
            std::vector<double> actual_scores(kTopK);
            std::vector<double> reference_scores(kTopK);
            for (std::int32_t rank = 0; rank < kTopK; ++rank) {
                actual_scores[static_cast<std::size_t>(rank)] = scores[rank];
                reference_scores[static_cast<std::size_t>(rank)] =
                    ranked[static_cast<std::size_t>(rank)].first;
            }
            failures += verify_reduction("q5-draft scores seed=" + std::to_string(seed),
                                         actual_scores, reference_scores, kA16ScoreCriterion);
        }        for (int rank = 0; rank < 8; ++rank) {
            if (std::find(ids.begin(), ids.begin() + 8, order[static_cast<std::size_t>(rank)]) !=
                ids.begin() + 8) {
                ++dense_top8;
            }
        }
        const bool dense_top1 = ids[0] == order[0];
        std::cout << "q5-draft seed=" << seed << " top1-vs-dense=" << (dense_top1 ? 1 : 0)
                  << " top8-intersection=" << dense_top8 << "/8 ranks-matching-oracle="
                  << matched_ids << "/" << kTopK << '\n';
        if (!dense_top1) {
            std::cerr << "q5-draft seed=" << seed << " dense top1=" << order[0]
                      << " topk top1=" << ids[0] << '\n';
            ++failures;
        }
        if (dense_top8 < 8) {
            std::cerr << "q5-draft seed=" << seed << " dense top8 intersection " << dense_top8
                      << "/8\n";
            ++failures;
        }
        failures += out_ids.verify_guards("q5 ids tail");
        failures += out_scores.verify_guards("q5 scores tail");
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const std::vector<std::uint16_t> host_hidden = make_hidden();
        DeviceBuffer hidden(host_hidden.size() * sizeof(std::uint16_t));
        hidden.copy_from_host(host_hidden.data(), hidden.bytes);

        int failures = 0;
        failures += run_full(QType::Q8_G32_FP16, "q8-full", hidden,
                             base_scores(QType::Q8_G32_FP16, host_hidden));
        failures += run_full(QType::FP8_E4M3FN_ROW_BF16, "fp8-full", hidden,
                             base_scores(QType::FP8_E4M3FN_ROW_BF16, host_hidden));
        failures += run_q4(hidden, base_scores(QType::Q4_G64_FP16, host_hidden));
        failures += run_q5_draft_head();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " linear_topk\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "linear_topk: " << error.what() << '\n';
        return 1;
    }
}



