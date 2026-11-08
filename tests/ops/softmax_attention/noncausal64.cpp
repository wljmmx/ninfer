// T088 (upstream 6eb6056b): D64 noncausal profiles for the packed attention path.
//
// The vision heads run at head_dim 64 with scale 1/sqrt(64) = 0.125. The text heads stay at
// 72. These cases pin the D64 route: a single key must reproduce V exactly (softmax over one
// key is 1.0), and multi-token / packed / uniform segments must match a FP64 oracle.

#include "ninfer/ops/softmax_attention.h"

#include "core/arena.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim64    = 64;
constexpr int kHeads    = 16;
constexpr float kScale  = 0.125F;
constexpr ops::AttentionHeadGeometry kGeometry64{kDim64, kHeads, kHeads};

std::size_t index_of(int token, int head, int d) {
    return (static_cast<std::size_t>(token) * kHeads + static_cast<std::size_t>(head)) * kDim64 +
           static_cast<std::size_t>(d);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

void noncausal64_oracle(const std::vector<float>& q, const std::vector<float>& k,
                        const std::vector<float>& v, const std::vector<int>& cu_seqlens,
                        std::vector<double>& out) {
    const double scale = 1.0 / std::sqrt(64.0);
    out.assign(q.size(), 0.0);
    for (std::size_t segment = 0; segment + 1 < cu_seqlens.size(); ++segment) {
        const int begin = cu_seqlens[segment];
        const int end   = cu_seqlens[segment + 1];
        naive_dense_softmax_attention(
            kGeometry64, end - begin, end - begin, scale,
            [&](int d, int head, int token) {
                return static_cast<double>(q[index_of(begin + token, head, d)]);
            },
            [&](int d, int head, int token) {
                return static_cast<double>(k[index_of(begin + token, head, d)]);
            },
            [&](int d, int head, int token) {
                return static_cast<double>(v[index_of(begin + token, head, d)]);
            },
            [](int, int) { return true; },
            [&](int d, int head, int token, double value) {
                out[index_of(begin + token, head, d)] = value;
            });
    }
}

int run_case(const std::vector<int>& cu_seqlens, std::uint32_t seed, bool single_key) {
    const int tokens              = cu_seqlens.back();
    const std::size_t token_plane = static_cast<std::size_t>(kHeads) * kDim64;
    const std::size_t value_count = static_cast<std::size_t>(tokens) * token_plane;
    std::vector<float> q(value_count);
    std::vector<float> k(value_count);
    std::vector<float> v(value_count);
    fill_uniform(q, seed, -1.0f, 1.0f);
    fill_uniform(k, seed + 1, -1.0f, 1.0f);
    fill_uniform(v, seed + 2, -2.0f, 2.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    std::vector<double> reference;
    noncausal64_oracle(q, k, v, cu_seqlens, reference);

    const auto q_expected = bf16_bits(q);
    const auto k_expected = bf16_bits(k);
    const auto v_expected = bf16_bits(v);

    DeviceBuffer q_storage = to_device(q_expected);
    DeviceBuffer k_storage = to_device(k_expected);
    DeviceBuffer v_storage = to_device(v_expected);
    Tensor q_tensor(q_storage.p, DType::BF16, {kDim64, kHeads, tokens});
    Tensor k_tensor(k_storage.p, DType::BF16, {kDim64, kHeads, tokens});
    Tensor v_tensor(v_storage.p, DType::BF16, {kDim64, kHeads, tokens});

    DeviceBuffer d_cu_seqlens = to_device_i32(cu_seqlens);
    Tensor cu_tensor(d_cu_seqlens.p, DType::I32, {static_cast<std::int32_t>(cu_seqlens.size())});
    GuardedDeviceBuffer d_out(value_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    Tensor out_tensor(d_out.data(), DType::BF16, {kDim64, kHeads, tokens});

    const std::int32_t segments       = static_cast<std::int32_t>(cu_seqlens.size()) - 1;
    const std::size_t workspace_bytes = ops::packed_softmax_attention_workspace_capacity_bytes(
        kGeometry64, tokens, tokens, segments, segments);
    DeviceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    if (single_key) {
        ops::softmax_attention(q_tensor, k_tensor, v_tensor, kGeometry64, kScale, workspace,
                               out_tensor, nullptr);
    } else {
        ops::packed_softmax_attention(q_tensor, k_tensor, v_tensor, kGeometry64, kScale, cu_tensor,
                                      workspace, out_tensor, nullptr);
    }
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> out_bits(value_count);
    cudaMemcpy(out_bits.data(), out_tensor.data, value_count * sizeof(std::uint16_t),
               cudaMemcpyDeviceToHost);

    // Softmax over a single key is exactly 1, so the D64 route must reproduce V bit-exactly
    // at the element level (V is already rounded to BF16).
    if (single_key) {
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < value_count; ++i) {
            if (out_bits[i] != v_expected[i]) { ++mismatches; }
        }
        if (mismatches != 0) {
            std::cerr << "noncausal64 single-key: " << mismatches << " of " << value_count
                      << " elements differ from V\n";
            return 1;
        }
        return 0;
    }

    double worst_relative = 0.0;
    double worst_absolute = 0.0;
    for (std::size_t i = 0; i < value_count; ++i) {
        const double expected = reference[i];
        const double actual   = static_cast<double>(bf16_to_f32(out_bits[i]));
        const double error    = std::abs(actual - expected);
        const double relative = error / (std::abs(expected) + 1.0);
        worst_absolute        = std::max(worst_absolute, error);
        worst_relative        = std::max(worst_relative, relative);
    }
    if (worst_relative > 2.5e-3 || worst_absolute > 3.0e-3) {
        std::cerr << "noncausal64 oracle: worst relative " << worst_relative << " absolute "
                  << worst_absolute << "\n";
        return 1;
    }
    return 0;
}

} // namespace

int run_softmax_attention_noncausal64_tests() {
    int failures = 0;
    // Single key: output must equal V.
    failures += run_case({0, 1}, 11, true);
    // Multi-token and packed segments against the FP64 oracle.
    failures += run_case({0, 129}, 21, false);
    failures += run_case({0, 17, 64, 129}, 31, false);
    failures += run_case({0, 32, 64, 96}, 41, false);
    if (failures == 0) { std::cout << "noncausal64 ok\n"; }
    return failures;
}
