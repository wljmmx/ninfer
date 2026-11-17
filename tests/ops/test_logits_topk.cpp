#include "ninfer/ops/logits_topk.h"
#include "ninfer/ops/target_logprobs.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The same aggregate bound the target_logprobs oracle uses: the two ops evaluate the same
// normalizer, so their log-probabilities are graded the same way.
constexpr ReductionCriterion kTopkFp32Criterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-4,
    /*gross_relative_to_max_reference=*/0.0,
};

std::vector<std::uint16_t> make_random_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                              std::int32_t columns) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(column + 1) * 1013904223u;
            const float value = -24.0f + static_cast<float>(mixed % 6144u) * (1.0f / 128.0f);
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(96.0f);
        }
    }
    return logits;
}

std::vector<std::uint16_t> make_uniform_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                               std::int32_t columns, float value) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns,
                                      f32_to_bf16(value));
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(112.0f);
        }
    }
    return logits;
}

std::vector<std::uint16_t> make_tied_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                            std::int32_t columns) {
    // Every third row repeats its predecessor's value, so ties are dense and the id tiebreak has to
    // decide the order instead of the values.
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const std::int32_t bucket = row % 3 == 0 ? row / 3 : (row - 1) / 3;
            const float value         = static_cast<float>((bucket * 7) % 23) * 0.5f - 6.0f;
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(120.0f);
        }
    }
    return logits;
}

struct ColumnOracle {
    std::vector<std::int32_t> ids;
    std::vector<double> logprobs;
    double log_normalizer = 0.0;
};

// The independent oracle: descending value, ascending id, double precision normalizer.
ColumnOracle column_oracle(const std::vector<std::uint16_t>& logits, std::int32_t physical_rows,
                           std::int32_t valid_rows, std::int32_t column, std::int32_t k) {
    const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
    std::vector<std::pair<double, std::int32_t>> candidates;
    candidates.reserve(static_cast<std::size_t>(valid_rows));
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int32_t row = 0; row < valid_rows; ++row) {
        const double value = static_cast<double>(bf16_to_f32(logits[base + row]));
        maximum            = std::max(maximum, value);
        candidates.emplace_back(value, row);
    }
    double sum = 0.0;
    for (const auto& [value, row] : candidates) {
        (void)row;
        sum += std::exp(value - maximum);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) { return lhs.first > rhs.first; }
        return lhs.second < rhs.second;
    });
    ColumnOracle oracle;
    oracle.log_normalizer = maximum + std::log(sum);
    for (std::int32_t rank = 0; rank < k; ++rank) {
        const auto& [value, row] = candidates[static_cast<std::size_t>(rank)];
        oracle.ids.push_back(row);
        oracle.logprobs.push_back(value - maximum - std::log(sum));
    }
    return oracle;
}

std::vector<double> fp32_as_double(const void* device, std::size_t count) {
    const auto values = from_device<float>(device, count);
    return {values.begin(), values.end()};
}

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
             std::int32_t columns, std::int32_t k, const std::vector<std::uint16_t>& logits) {
    const std::size_t entries = static_cast<std::size_t>(columns) * k;

    std::vector<std::int32_t> oracle_ids(entries);
    std::vector<double> oracle_logprobs(entries);
    std::vector<double> oracle_normalizer(static_cast<std::size_t>(columns));
    std::vector<std::int32_t> argmax_ids(static_cast<std::size_t>(columns));
    for (std::int32_t column = 0; column < columns; ++column) {
        const ColumnOracle oracle = column_oracle(logits, physical_rows, valid_rows, column, k);
        for (std::int32_t rank = 0; rank < k; ++rank) {
            oracle_ids[static_cast<std::size_t>(column) * k + rank] =
                oracle.ids[static_cast<std::size_t>(rank)];
            oracle_logprobs[static_cast<std::size_t>(column) * k + rank] =
                oracle.logprobs[static_cast<std::size_t>(rank)];
        }
        oracle_normalizer[static_cast<std::size_t>(column)] = oracle.log_normalizer;
        argmax_ids[static_cast<std::size_t>(column)]        = oracle.ids[0];
    }

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(entries * sizeof(std::int32_t));
    GuardedDeviceBuffer device_logprobs(entries * sizeof(float));
    GuardedDeviceBuffer device_normalizer(static_cast<std::size_t>(columns) * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_ids.fill(0xcd);
    device_logprobs.fill(0xcd);
    device_normalizer.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, columns});
    Tensor ids_tensor(device_ids.data(), DType::I32, {k, columns});
    Tensor logprobs_tensor(device_logprobs.data(), DType::FP32, {k, columns});
    Tensor normalizer_tensor(device_normalizer.data(), DType::FP32, {columns});
    ops::logits_topk(logits_tensor, valid_rows, k, ids_tensor, logprobs_tensor, normalizer_tensor,
                     nullptr);
    cuda_synchronize();

    const auto host_ids      = from_device<std::int32_t>(device_ids.data(), entries);
    const auto host_logprobs = from_device<float>(device_logprobs.data(), entries);

    int failures = verify_exact((label + " ids").c_str(), host_ids, oracle_ids);
    failures +=
        verify_reduction(label + " logprobs", fp32_as_double(device_logprobs.data(), entries),
                         oracle_logprobs, kTopkFp32Criterion);
    failures += verify_reduction(
        label + " log_normalizer",
        fp32_as_double(device_normalizer.data(), static_cast<std::size_t>(columns)),
        oracle_normalizer, kTopkFp32Criterion);

    // A candidate list is ordered by decreasing logprob and never repeats a token.
    for (std::int32_t column = 0; column < columns; ++column) {
        std::vector<std::int32_t> seen;
        seen.reserve(static_cast<std::size_t>(k));
        float previous = std::numeric_limits<float>::infinity();
        for (std::int32_t rank = 0; rank < k; ++rank) {
            const std::size_t index = static_cast<std::size_t>(column) * k + rank;
            if (host_ids[index] < 0 || host_ids[index] >= valid_rows) {
                std::cerr << label << ": candidate id " << host_ids[index]
                          << " outside the domain\n";
                return failures + 1;
            }
            if (host_logprobs[index] > previous) {
                std::cerr << label << ": logprobs increase at rank " << rank << '\n';
                return failures + 1;
            }
            previous = host_logprobs[index];
            seen.push_back(host_ids[index]);
        }
        std::sort(seen.begin(), seen.end());
        if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
            std::cerr << label << ": candidate list repeats a token\n";
            return failures + 1;
        }
    }

    // The top-1 logprob is the target logprob target_logprobs reports for the same column's argmax.
    // Both ops use one 256-thread CTA per column, the same maximum and the same reduction order, so
    // this is asserted bit for bit rather than within a tolerance.
    GuardedDeviceBuffer device_targets(static_cast<std::size_t>(columns) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_target_logprobs(static_cast<std::size_t>(columns) * sizeof(float));
    device_targets.copy_from_host(argmax_ids.data(), device_targets.bytes());
    device_target_logprobs.fill(0xcd);
    Tensor targets_tensor(device_targets.data(), DType::I32, {columns});
    Tensor target_logprobs_tensor(device_target_logprobs.data(), DType::FP32, {columns});
    ops::target_logprobs(logits_tensor, targets_tensor, valid_rows, target_logprobs_tensor,
                         nullptr);
    cuda_synchronize();
    const auto target_logprobs_host =
        from_device<float>(device_target_logprobs.data(), static_cast<std::size_t>(columns));
    std::vector<float> top1(static_cast<std::size_t>(columns));
    for (std::int32_t column = 0; column < columns; ++column) {
        top1[static_cast<std::size_t>(column)] =
            host_logprobs[static_cast<std::size_t>(column) * k];
    }
    failures +=
        verify_exact((label + " top-1 equals target_logprobs").c_str(), top1, target_logprobs_host);

    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_ids.verify_guards(label + " ids guards");
    failures += device_logprobs.verify_guards(label + " logprobs guards");
    failures += device_normalizer.verify_guards(label + " normalizer guards");
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(8 * 3 * sizeof(std::uint16_t));
    DeviceBuffer ids_data(3 * sizeof(std::int32_t));
    DeviceBuffer logprobs_data(3 * sizeof(float));
    DeviceBuffer normalizer_data(3 * sizeof(float));
    Tensor logits(logits_data.p, DType::BF16, {8, 3});
    Tensor ids(ids_data.p, DType::I32, {1, 3});
    Tensor logprobs(logprobs_data.p, DType::FP32, {1, 3});
    Tensor normalizer(normalizer_data.p, DType::FP32, {3});

    int failures = 0;
    failures += expect_invalid("logits_topk rejects k=0", [&] {
        ops::logits_topk(logits, 8, 0, ids, logprobs, normalizer, nullptr);
    });
    failures += expect_invalid("logits_topk rejects k above the capacity", [&] {
        ops::logits_topk(logits, 8, ops::kLogitsTopkCapacity + 1, ids, logprobs, normalizer,
                         nullptr);
    });
    failures += expect_invalid("logits_topk rejects k above valid_rows", [&] {
        ops::logits_topk(logits, 8, 2, ids, logprobs, normalizer, nullptr);
    });
    failures += expect_invalid("logits_topk rejects valid_rows=0", [&] {
        ops::logits_topk(logits, 0, 1, ids, logprobs, normalizer, nullptr);
    });
    failures += expect_invalid("logits_topk rejects ids shape mismatch", [&] {
        Tensor wrong_ids(ids_data.p, DType::I32, {3, 1});
        ops::logits_topk(logits, 8, 1, wrong_ids, logprobs, normalizer, nullptr);
    });
    failures += expect_invalid("logits_topk rejects logprobs dtype", [&] {
        Tensor wrong_logprobs(logprobs_data.p, DType::BF16, {1, 3});
        ops::logits_topk(logits, 8, 1, ids, wrong_logprobs, normalizer, nullptr);
    });
    failures += expect_invalid("logits_topk rejects normalizer shape", [&] {
        Tensor wrong_normalizer(normalizer_data.p, DType::FP32, {2});
        ops::logits_topk(logits, 8, 1, ids, logprobs, wrong_normalizer, nullptr);
    });
    failures += expect_invalid("logits_topk rejects non-contiguous logits", [&] {
        Tensor strided = logits;
        strided.nb[1] += 2;
        ops::logits_topk(strided, 8, 1, ids, logprobs, normalizer, nullptr);
    });
    failures += expect_invalid("logits_topk rejects an aliasing output", [&] {
        Tensor alias(logits_data.p, DType::FP32, {1, 3});
        ops::logits_topk(logits, 8, 1, ids, alias, normalizer, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // Full vocabulary width, the shape the causal scorer actually produces, at the capacity k.
    failures += run_case("logits_topk full vocabulary C=3", 248320, 248077, 3, 64,
                         make_random_logits(248320, 248077, 3));
    // Many columns, a short column: the per-column CTA mapping.
    failures += run_case("logits_topk non-aligned rows C=1025", 523, 509, 1025, 17,
                         make_random_logits(523, 509, 1025));
    // Uniform values: the order is decided entirely by the id tiebreak.
    failures += run_case("logits_topk uniform logits", 263, 257, 1024, 5,
                         make_uniform_logits(263, 257, 1024, 3.5f));
    failures +=
        run_case("logits_topk tied logits", 301, 293, 129, 9, make_tied_logits(301, 293, 129));
    // k=1 is the argmax path the KL tool falls back to when the payload is trimmed.
    failures +=
        run_case("logits_topk one valid row", 13, 1, 7, 1, make_uniform_logits(13, 1, 7, -7.0f));
    // k equal to valid_rows returns the whole column, in order.
    failures +=
        run_case("logits_topk k equals valid_rows", 67, 64, 33, 64, make_random_logits(67, 64, 33));
    failures += run_validation_cases();

    std::cout << (failures ? "FAIL" : "OK") << " logits_topk\n";
    return failures ? 1 : 0;
}
