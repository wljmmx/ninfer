// port (sm_89): proves the expert-cache data path against the real artifact before any engine
// wiring can depend on it. One layer's 256 experts are (a) streamed from the artifact file into a
// cache pool and addressed by per-expert offsets, and (b) kept resident and addressed by stride.
// Both banks run the same routed SwiGLU on the same routes; the outputs must be bit-identical.

#include "artifact/expert_files.h"
#include "artifact/expert_reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/expert_cache.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include <cuda_runtime.h>

#include <chrono>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

constexpr std::int32_t kHidden = 2048;
constexpr std::int32_t kTopK   = 8;
constexpr std::int32_t kTokens = 1023;

int fail(const std::string& message) {
    std::cerr << "expert stream: " << message << '\n';
    return 1;
}

std::uint16_t bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact_path == nullptr || *artifact_path == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    try {
        DeviceContext device(0);
        artifact::Reader reader{artifact_path};
        const auto& bindings = reader.directory().bindings;

        // Layer 0 of this model is MoE; the fused parents are shared by all 256 experts.
        const auto& gate = bindings.at("text/layers/0/moe/experts/0/gate");
        const auto& up   = bindings.at("text/layers/0/moe/experts/0/up");
        const auto& down = bindings.at("text/layers/0/moe/experts/0/down");
        const auto& up_part = up.parts.front();
        const auto& gate_part = gate.parts.front();
        const auto& down_part = down.parts.front();
        if (up_part.object != gate_part.object || down_part.object == gate_part.object) {
            return fail("unexpected expert object grouping");
        }
        const auto& gate_geometry = reader.geometry(gate_part.object);
        const auto& down_geometry = reader.geometry(down_part.object);
        const auto columns = gate_geometry.shape.back();
        const auto count   = static_cast<std::uint32_t>(256);
        // gate and up are the two halves of one fused [2I,H] expert matrix.
        const auto gate_up_rows = static_cast<std::uint32_t>(
            (up_part.end - gate_part.begin) / columns);
        const auto down_rows =
            static_cast<std::uint32_t>(down_part.end / down_geometry.shape.back());

        const std::array<std::uint64_t, 2> gate_up_shape{gate_up_rows, columns};
        const std::array<std::uint64_t, 2> down_shape{down_rows, down_geometry.shape.back()};
        const auto gate_up_geometry =
            weight_geometry(gate_geometry.format, gate_geometry.layout, gate_up_shape);
        const auto down_expert_geometry =
            weight_geometry(down_geometry.format, down_geometry.layout, down_shape);

        std::cerr << "geom: gu fmt=" << int(gate_up_geometry.format) << " lay=" << int(gate_up_geometry.layout)
                  << " bytes=" << gate_up_geometry.bytes << " codes=" << gate_up_geometry.code_bytes
                  << " | dn fmt=" << int(down_expert_geometry.format) << " lay=" << int(down_expert_geometry.layout)
                  << " bytes=" << down_expert_geometry.bytes << " codes=" << down_expert_geometry.code_bytes
                  << " rows=" << gate_up_rows << "/" << down_rows << "\n";
        const std::array spec{ExpertBankSpec{{gate_up_geometry}, {down_expert_geometry}, count,
                                             false}};
        ExpertCacheConfig config;
        const auto description = describe_expert_cache(spec, {.capacity_bytes = 256});
        config.capacity_bytes  = description.banks.front().entry_bytes * count * 2;
        const auto requirements = describe_expert_cache(spec, config);

        const auto files = artifact::describe_fused_expert_files(
            reader, gate_part.object, down_part.object, gate_geometry, down_geometry, count,
            gate_up_rows, down_rows, requirements.banks.front());

        artifact::ArtifactExpertReader source{artifact_path};
        ExpertCache cache(spec, requirements, device, {}, std::span<const ExpertFileBank>(&files, 1),
                          &source);

        std::vector<ExpertCacheKey> all(count);
        for (std::uint32_t expert = 0; expert < count; ++expert) {
            all[expert] = ExpertCacheKey{0, expert};
        }
        cache.prepare(all);
        const auto upload_begin = std::chrono::steady_clock::now();
        cache.wait_for_progress();
        const auto upload_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - upload_begin).count();
        if (cache.used_bytes() != config.capacity_bytes) {
            return fail("cache did not fill its pool");
        }

        DeviceBuffer offsets(count * sizeof(std::uint64_t));
        std::vector<std::uint64_t> host_offsets(count);
        for (std::uint32_t expert = 0; expert < count; ++expert) {
            const auto offset = cache.offset(ExpertCacheKey{0, expert});
            if (!offset) { return fail("streamed expert was not published"); }
            host_offsets[expert] = *offset;
        }
        offsets.copy_from_host(host_offsets.data(), host_offsets.size() * sizeof(std::uint64_t));
        const auto pool_span = cache.pool();

        // Resident reference: the same two parents uploaded whole and addressed by stride.
        DeviceBuffer gate_up_device(gate_geometry.bytes);
        DeviceBuffer down_device(down_geometry.bytes);
        {
            const auto bytes = reader.read_object(gate_part.object);
            gate_up_device.copy_from_host(bytes.data(), bytes.size());
        }
        {
            const auto bytes = reader.read_object(down_part.object);
            down_device.copy_from_host(bytes.data(), bytes.size());
        }
        WeightParent gate_up_parent{gate_geometry, static_cast<const std::byte*>(gate_up_device.p),
                                    0.0F, Device::cuda(device.device), {}};
        WeightParent down_parent{down_geometry, static_cast<const std::byte*>(down_device.p), 0.0F,
                                 Device::cuda(device.device), {}};
        const auto gate_up_layout =
            describe_weight_bank(gate_geometry, {0, count, gate_up_rows, gate_up_rows});
        const auto down_layout =
            describe_weight_bank(down_geometry, {0, count, down_rows, down_rows});
        const auto gate_up_bank = bind_weight_bank(gate_up_parent, gate_up_layout);
        const auto down_bank    = bind_weight_bank(down_parent, down_layout);
        // Shared expert of the same layer: the engine always passes it, so the streamed path must
        // be exercised with it too.
        const auto& sh_gate = bindings.at("text/layers/0/moe/shared/gate");
        const auto& sh_up   = bindings.at("text/layers/0/moe/shared/up");
        const auto& sh_down = bindings.at("text/layers/0/moe/shared/down");
        const auto& sh_gu_geometry = reader.geometry(sh_gate.parts.at(0).object);
        const auto& sh_dn_geometry = reader.geometry(sh_down.parts.at(0).object);
        DeviceBuffer sh_gu_device(sh_gu_geometry.bytes);
        DeviceBuffer sh_dn_device(sh_dn_geometry.bytes);
        {
            const auto bytes = reader.read_object(sh_gate.parts.at(0).object);
            sh_gu_device.copy_from_host(bytes.data(), bytes.size());
        }
        {
            const auto bytes = reader.read_object(sh_down.parts.at(0).object);
            sh_dn_device.copy_from_host(bytes.data(), bytes.size());
        }
        WeightParent sh_gu_parent{sh_gu_geometry,
                                  static_cast<const std::byte*>(sh_gu_device.p), 0.0F,
                                  Device::cuda(device.device), {}};
        WeightParent sh_dn_parent{sh_dn_geometry,
                                  static_cast<const std::byte*>(sh_dn_device.p), 0.0F,
                                  Device::cuda(device.device), {}};
        const auto sh_columns = sh_gu_geometry.shape.back();
        const auto sh_rows    = (sh_up.parts.at(0).end - sh_gate.parts.at(0).begin) / sh_columns;
        const Weight sh_fused =
            native_weight(WeightView{{sh_rows, sh_columns},
                                     {{&sh_gu_parent, sh_gate.parts.at(0).begin,
                                       sh_up.parts.at(0).end}}});
        const Weight sh_down_weight =
            native_weight(WeightView{{sh_dn_geometry.shape.at(0), sh_dn_geometry.shape.at(1)},
                                     {{&sh_dn_parent, sh_down.parts.at(0).begin,
                                       sh_down.parts.at(0).end}}});
        const ops::SwiGLUWeights shared_weights{
            ops::SingleProjectionWeight{sh_fused, ops::LinearPolicy::A16Only},
            ops::SingleProjectionWeight{sh_down_weight, ops::LinearPolicy::A16Only}};
        const auto shared_profile = ops::swiglu_profile(shared_weights);
        DeviceBuffer shared_coefficients_device(kTokens * sizeof(float));
        {
            std::vector<float> values(kTokens, 0.25F);
            shared_coefficients_device.copy_from_host(values.data(), values.size() * sizeof(float));
        }
        const ops::ScaledSwiGLUTerm shared_term{
            shared_weights, Tensor(shared_coefficients_device.p, DType::FP32, {kTokens})};
        const auto profile      = ops::routed_swiglu_profile(gate_up_bank, down_bank, kTopK,
                                                             ops::LinearPolicy::A16Only,
                                                             ops::LinearPolicy::A16Only);

        // Routes: top-k over a deterministic logit table, coefficients that sum to one per token.
        std::mt19937 random(20261021);
        std::normal_distribution<float> gaussian;
        std::vector<std::int32_t> ids(std::size_t(kTopK) * kTokens);
        std::vector<float> coefficients(std::size_t(kTopK) * kTokens);
        for (std::int32_t token = 0; token < kTokens; ++token) {
            std::vector<std::int32_t> order(count);
            std::iota(order.begin(), order.end(), 0);
            std::vector<float> logits(count);
            for (auto& logit : logits) { logit = gaussian(random); }
            std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                              [&](auto a, auto b) { return logits[a] > logits[b]; });
            float total = 0.0F;
            for (std::int32_t j = 0; j < kTopK; ++j) {
                const auto weight = std::exp(logits[order[j]]);
                ids[std::size_t(j) * kTokens + token]          = order[j];
                coefficients[std::size_t(j) * kTokens + token] = weight;
                total += weight;
            }
            for (std::int32_t j = 0; j < kTopK; ++j) {
                coefficients[std::size_t(j) * kTokens + token] /= total;
            }
        }
        DeviceBuffer ids_device(ids.size() * sizeof(std::int32_t));
        DeviceBuffer coefficients_device(coefficients.size() * sizeof(float));
        ids_device.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
        coefficients_device.copy_from_host(coefficients.data(), coefficients.size() * sizeof(float));

        std::vector<std::uint16_t> input(std::size_t(kHidden) * kTokens);
        for (auto& value : input) { value = bf16(gaussian(random)); }
        DeviceBuffer input_device(input.size() * sizeof(std::uint16_t));
        input_device.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
        Tensor x(input_device.p, DType::BF16, {kHidden, kTokens});
        ops::ExpertRoutes routes{Tensor(ids_device.p, DType::I32, {kTopK, kTokens}),
                            Tensor(coefficients_device.p, DType::FP32, {kTopK, kTokens})};

        const auto workspace_bytes = ops::routed_swiglu_workspace_capacity_bytes(
            profile, shared_profile, ops::ExpertSubsetKind::All, DType::BF16,
            ops::MoeEpilogue::AddResidual, kTokens, kTokens);
        DeviceBuffer scratch(workspace_bytes);
        MemoryArena arena(MemorySpan{scratch.p, scratch.bytes, Device::cuda(device.device)});

        DeviceBuffer streamed_output(std::size_t(kHidden) * kTokens * sizeof(std::uint16_t));
        DeviceBuffer resident_output(std::size_t(kHidden) * kTokens * sizeof(std::uint16_t));

        ops::CudaSwiGLUExperts streamed{device_expert_bank(gate_up_bank, down_bank), {}};
        auto& bank = streamed.bank;
        const auto pool_base = static_cast<const std::byte*>(pool_span.data);
        const auto& entry = description.banks.front();
        bank.gate_up.codes   = pool_base + entry.codes.offset;
        bank.gate_up.high    = entry.high.bytes ? pool_base + entry.high.offset : nullptr;
        bank.gate_up.scales  = pool_base + entry.scales.offset;
        bank.gate_up.code_stride = bank.gate_up.high_stride = bank.gate_up.scale_stride = 0;
        bank.gate_up.offsets = static_cast<const std::uint64_t*>(offsets.p);
        bank.gate_up.indexed_storage = pool_span;
        bank.down            = bank.gate_up;
        bank.down.codes      = pool_base + entry.down_codes.offset;
        bank.down.high       = entry.down_high.bytes ? pool_base + entry.down_high.offset : nullptr;
        bank.down.scales     = pool_base + entry.down_scales.offset;

        std::cerr << "planes: pool=" << pool_span.bytes << " entry=" << entry.entry_bytes
                  << " gu=" << entry.codes.bytes << "/" << entry.high.bytes << "/"
                  << entry.scales.bytes << " dn=" << entry.down_codes.bytes << "/"
                  << entry.down_high.bytes << "/" << entry.down_scales.bytes << "\n";
        const auto dump = [](const char* name, const ExpertMatrixDeviceView& v) {
            std::cerr << name << " codes=" << v.codes << " high=" << v.high
                      << " scales=" << v.scales << " offsets=" << v.offsets
                      << " storage=" << v.indexed_storage.bytes << "\n";
        };
        dump("gu", bank.gate_up);
        dump("dn", bank.down);

        streamed_output.fill(0x01);
        resident_output.fill(0x01);
        Tensor streamed_destination(streamed_output.p, DType::BF16, {kHidden, kTokens});
        {
            auto scope = arena.scope();
            ops::routed_swiglu(x, profile, streamed, routes, {}, ops::MoeEpilogue::Write,
                               streamed_destination, arena, device.execution_view());
            ops::routed_swiglu(x, profile, streamed, routes, ops::ExpertSubset{ops::ExpertSubsetKind::None},
                               shared_term, ops::MoeEpilogue::AddResidual, streamed_destination,
                               arena, device.execution_view());
        }

        Tensor resident_destination(resident_output.p, DType::BF16, {kHidden, kTokens});
        {
            auto scope = arena.scope();
            ops::routed_swiglu(x, profile,
                               ops::CudaSwiGLUExperts{device_expert_bank(gate_up_bank, down_bank),
                                                      {}},
                               routes, {}, ops::MoeEpilogue::Write, resident_destination, arena,
                               device.execution_view());
            ops::routed_swiglu(x, profile,
                               ops::CudaSwiGLUExperts{device_expert_bank(gate_up_bank, down_bank),
                                                      {}},
                               routes, ops::ExpertSubset{ops::ExpertSubsetKind::None}, shared_term,
                               ops::MoeEpilogue::AddResidual, resident_destination, arena,
                               device.execution_view());
        }
        device.synchronize();

        std::vector<std::uint16_t> streamed_values(streamed_output.bytes / sizeof(std::uint16_t));
        std::vector<std::uint16_t> resident_values(resident_output.bytes / sizeof(std::uint16_t));
        streamed_output.copy_to_host(streamed_values.data(), streamed_output.bytes);
        resident_output.copy_to_host(resident_values.data(), resident_output.bytes);

        // NaN pairs are compared by payload class: NaN != NaN would otherwise report a value
        // that both sides computed identically.
        std::size_t mismatches = 0, nan_pairs = 0, printed = 0;
        for (std::size_t i = 0; i < streamed_values.size(); ++i) {
            const auto a = streamed_values[i];
            const auto b = resident_values[i];
            const auto nan = [](std::uint16_t v) {
                return (v & 0x7F80U) == 0x7F80U && (v & 0x007FU) != 0;
            };
            if (nan(a) && nan(b)) {
                ++nan_pairs;
                continue;
            }
            if (a != b) {
                if (printed < 3) {
                    std::cerr << "diff[" << i << "] streamed=0x" << std::hex << a
                              << " resident=0x" << b << std::dec << "\n";
                    ++printed;
                }
                ++mismatches;
            }
        }
        std::cerr << "nan_pairs=" << nan_pairs << " value_mismatches=" << mismatches << "\n";
        if (mismatches != 0) {
            return fail("streamed bank differs from the resident bank at " +
                        std::to_string(mismatches) + " values");
        }
        std::cout << "expert stream: ok experts=" << count
                  << " entry_bytes=" << description.banks.front().entry_bytes
                  << " uploads=" << cache.uploads() << " evictions=" << cache.evictions()
                  << " tokens=" << kTokens << " upload_s=" << upload_seconds << " pool_mib="
                  << (config.capacity_bytes >> 20) << '\n';
        return 0;
    } catch (const std::exception& error) {
        return fail(error.what());
    }
}
