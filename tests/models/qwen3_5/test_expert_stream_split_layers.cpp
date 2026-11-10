// port (sm_89): the gate/up/down (Qwen4Exp) expert form's LAYERED execution.
//
// The artifact keeps an expert's gate, up and down in three objects, so a layer owns three pool runs.
// This test proves the three things that make that form executable on a pool that cannot hold it:
//
//   (b) the CPU half - the route the pool and the tier both cannot hold - computes the value the
//       Device kernels would have, measured against an FP64 decode of the same bytes. The Device half
//       writes the FP32 term of the experts it can address, the CPU half the rest, and one merge folds
//       residual + host + device with a single rounding.
//   (a) the tier's second leg: with a pool that keeps only part of the layer and a pinned tier that
//       keeps part of the rest, the layer still completes and every route resolves to an address
//       (pool slot or tier slot), which is what the kernels need to run it rather than skip experts.
//   (c) the per-run placement: three runs, three slot tables, three strides - each run's offset table
//       entry is that run's own slot address, its mask says "addressable" and its host flag says
//       which leg serves it.
//
// The fixture is numerically valid (q4 gate/up, q6 down, group 64, FP16 scales) and uses the
// artifact's own (H=2560, I=640, E=512, K=10) profile, because the operator validates the routing
// geometry. Only the ten experts the routes name carry real bytes; the rest of the bank points at the
// first image and is never routed, which keeps the fixture at 30 MiB instead of 1.5 GiB.
#include "core/arena.h"
#include "core/cpu/executor.h"
#include "core/cpu/system.h"
#include "core/device.h"
#include "core/expert_cache.h"
#include "core/weight_bank.h"
#include "models/qwen3_5/execution/expert_stream.h"
#include "models/qwen3_5/execution/host_expert_residency.h"
#include "ninfer/ops/moe.h"
#include "ops/common/cpu/avx2.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace {

using namespace ninfer;
using models::qwen3_5::execution::ExpertMatrixBankDesc;
using models::qwen3_5::execution::ExpertMatrixRole;
using models::qwen3_5::execution::ExpertStream;
using models::qwen3_5::execution::HostExpertBankDesc;
using models::qwen3_5::execution::HostExpertResidency;
using models::qwen3_5::execution::HostWarmBudget;

// The routed geometry the operator admits (Qwen3.8-Flash-Next: H=2560, I=640, E=512, K=10).
constexpr std::int32_t kExperts   = 512;
constexpr std::int32_t kHidden    = 2560;
constexpr std::int32_t kInter     = 640;
constexpr std::int32_t kTopK      = 10;
constexpr std::int32_t kTokens    = 1;
constexpr std::int32_t kRealised  = 10; // experts whose bytes are real (the routes never leave them)

int failures = 0;

void fail(const char* what) {
    std::fprintf(stderr, "expert stream split layers: %s\n", what);
    ++failures;
}

float half_to_float(std::uint16_t half) {
    return ops::common::cpu::avx2::fp16_to_f32(half);
}

// The routed input and the merged residual are BF16 (the artifact's x dtype and the epilogue's
// output), while the group scales are FP16. Decoding a BF16 payload with the FP16 rule reads a
// different number - a different exponent width - so the two rules keep separate names here.
float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float out                = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Round to nearest even, the rule the quantizer and the kernels share.
std::uint16_t float_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t raw  = bits & 0x7fffffffu;
    if (raw > 0x7f800000u) { return static_cast<std::uint16_t>(sign | 0x7e00u); }
    if (raw >= 0x47800000u) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    std::int32_t exponent  = static_cast<std::int32_t>((raw >> 23) & 0xff) - 127;
    std::uint32_t mantissa = raw & 0x7fffffu;
    if (exponent < -14) {
        const std::int32_t shift = -14 - exponent;
        if (shift > 24) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x800000u;
        const std::uint32_t rounded = mantissa >> (shift + 1);
        const std::uint32_t tie     = (mantissa >> shift) & 1u;
        const std::uint32_t sticky  = (mantissa & ((1u << shift) - 1u)) != 0 ? 1u : 0u;
        return static_cast<std::uint16_t>(sign | (rounded + (tie & (sticky | (rounded & 1u)))));
    }
    std::uint32_t out = mantissa + 0x00000fffu + ((mantissa >> 13) & 1u);
    if ((out & 0x00800000u) != 0) {
        out = 0;
        if (++exponent >= 16) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent + 15) << 10) |
                                      (out >> 13));
}

// One encoded expert matrix, addressed exactly as the kernels address it.
struct Planes {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint16_t* scales = nullptr;
};

Planes planes_of(const std::vector<std::byte>& image, std::size_t offset,
                 const ExpertBankLayout& entry) {
    Planes out;
    out.codes = reinterpret_cast<const std::uint8_t*>(image.data() + offset + entry.codes.offset);
    out.high  = entry.high.bytes != 0
                    ? reinterpret_cast<const std::uint8_t*>(image.data() + offset +
                                                            entry.high.offset)
                    : nullptr;
    out.scales = reinterpret_cast<const std::uint16_t*>(image.data() + offset + entry.scales.offset);
    return out;
}

// values is row-major [rows, columns]. The written image is the same address arithmetic the kernels
// use: K/2 code bytes and K/64 FP16 scales per row, plus (K/8)*(bits-4) high bytes for q6.
void encode(std::vector<std::byte>& image, std::size_t offset, const ExpertBankLayout& entry,
            std::int32_t rows, std::int32_t columns, std::int32_t bits,
            const std::vector<float>& values) {
    auto* codes = reinterpret_cast<std::uint8_t*>(image.data() + offset + entry.codes.offset);
    auto* high  = entry.high.bytes != 0
                      ? reinterpret_cast<std::uint8_t*>(image.data() + offset + entry.high.offset)
                      : nullptr;
    auto* scales = reinterpret_cast<std::uint16_t*>(image.data() + offset + entry.scales.offset);
    const auto groups = columns / 64;
    const auto qmax   = bits == 4 ? 7.0F : 31.0F;
    const auto qmin   = bits == 4 ? -8 : -32;
    const auto hi_max = bits == 4 ? 7 : 31;
    for (std::int32_t row = 0; row < rows; ++row) {
        auto* row_codes  = codes + static_cast<std::size_t>(row) * (columns / 2);
        auto* row_high   = high != nullptr
                               ? high + static_cast<std::size_t>(row) *
                                            (static_cast<std::size_t>(columns / 8) * (bits - 4))
                               : nullptr;
        auto* row_scales = scales + static_cast<std::size_t>(row) * groups;
        for (std::int32_t group = 0; group < groups; ++group) {
            const float* source = values.data() + static_cast<std::size_t>(row) * columns +
                                  static_cast<std::size_t>(group) * 64;
            float widest = 0.0F;
            for (std::int32_t i = 0; i < 64; ++i) { widest = std::max(widest, std::fabs(source[i])); }
            const float scale        = widest > 0.0F ? widest / qmax : 1.0F;
            const std::uint16_t half = float_to_half(scale);
            row_scales[group]        = half;
            const float decoded_scale = half_to_float(half);
            for (std::int32_t i = 0; i < 64; ++i) {
                std::int32_t q = static_cast<std::int32_t>(std::lround(source[i] / decoded_scale));
                q              = std::min(std::max(q, qmin), hi_max);
                const auto code = static_cast<std::uint32_t>(q) & (bits == 4 ? 0xFu : 0x3Fu);
                auto& byte      = row_codes[group * 32 + i / 2];
                if (i % 2 == 0) {
                    byte = static_cast<std::uint8_t>((byte & 0xF0u) | (code & 0xFu));
                } else {
                    byte = static_cast<std::uint8_t>((byte & 0x0Fu) | ((code & 0xFu) << 4));
                }
                if (bits > 4) {
                    auto& window     = row_high[group * 8 * (bits - 4) + i / 4];
                    const auto shift = static_cast<int>(2 * (i % 4));
                    window           = static_cast<std::uint8_t>(
                        (window & ~(0x3u << shift)) | (((code >> 4) & 0x3u) << shift));
                }
            }
        }
    }
}

// The three plane addresses of row `row`: codes and the high bits both step by the row, and the
// scales by the row's groups. Missing the high plane's row step was a real bug in the first draft of
// this oracle (q6 stores two bits per weight there), so it is one place now.
struct RowView {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint16_t* scales;
};

RowView row_view(const Planes& planes, std::int32_t row, std::int32_t columns, std::int32_t bits) {
    return {planes.codes + static_cast<std::size_t>(row) * (columns / 2),
            planes.high != nullptr
                ? planes.high + static_cast<std::size_t>(row) *
                                    (static_cast<std::size_t>(columns / 8) * (bits - 4))
                : nullptr,
            planes.scales + static_cast<std::size_t>(row) * (columns / 64)};
}

// The FP64 decode of one row of one expert: the same bytes, the same scale, no SIMD.
double dot64(const RowView& row, std::int32_t columns, std::int32_t bits,
             const std::vector<double>& x) {
    double total = 0.0;
    for (std::int32_t group = 0; group < columns / 64; ++group) {
        double partial     = 0.0;
        const double scale = half_to_float(row.scales[group]);
        for (std::int32_t i = 0; i < 64; ++i) {
            const auto byte   = row.codes[group * 32 + i / 2];
            std::int32_t code = static_cast<std::int32_t>((i % 2 == 0) ? (byte & 0xFu)
                                                                       : ((byte >> 4) & 0xFu));
            if (bits == 6) {
                const auto window = row.high[group * 16 + i / 4];
                code |= static_cast<std::int32_t>((window >> (2 * (i % 4))) & 0x3u) << 4;
                if ((code & 0x20) != 0) { code -= 64; }
            } else if ((code & 0x8) != 0) {
                code -= 16;
            }
            partial += static_cast<double>(code) * x[static_cast<std::size_t>(group) * 64 + i];
        }
        total += partial * scale;
    }
    return total;
}

// The artifact tier, in memory: one image per (bank, expert), read back verbatim by the loader.
struct MemoryReader final : ExpertBlobReader {
    const std::vector<std::vector<std::byte>>* images = nullptr;

    void read_entry(const ExpertFileEntry& entry, std::span<std::byte> destination) const override {
        const auto& image = (*images)[entry.chunks.front().file_index];
        std::memcpy(destination.data(), image.data() + entry.chunks.front().file_offset,
                    static_cast<std::size_t>(entry.image_bytes));
    }
};

struct Fixture {
    std::vector<ExpertBankSpec> specs;
    ExpertCacheRequirements requirements;
    std::vector<std::vector<std::byte>> images; // one per bank: kRealised entry images
    std::vector<ExpertFileBank> files;
    std::vector<std::int32_t> ids;              // [top_k, tokens], token-major
    std::vector<float> coefficients;
    std::vector<std::uint16_t> input;           // [hidden, tokens] BF16
    std::vector<std::vector<float>> source;     // per bank: the values the images were encoded from
    WeightBankLayout gate_layout, down_layout;
    WeightGeometry gate_geometry, down_geometry;
};

} // namespace

int main() {
    try {
        DeviceContext device(0);
        const auto cuda = Device::cuda(device.device);

        Fixture fixture;
        const std::uint64_t gate_shape[2] = {kInter, kHidden};
        const std::uint64_t down_shape[2] = {kHidden, kInter};
        fixture.gate_geometry = weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit, gate_shape);
        fixture.down_geometry = weight_geometry(QType::Q6_G64_FP16, QuantLayout::RowSplit, down_shape);
        const std::uint64_t gate_placeholder_shape[2] = {1, kHidden};
        const std::uint64_t down_placeholder_shape[2] = {1, kInter};
        const auto gate_placeholder =
            weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit, gate_placeholder_shape);
        const auto down_placeholder =
            weight_geometry(QType::Q6_G64_FP16, QuantLayout::RowSplit, down_placeholder_shape);

        fixture.specs = {
            {ExpertMatrixSpec{fixture.gate_geometry}, ExpertMatrixSpec{gate_placeholder}, kExperts,
             false},
            {ExpertMatrixSpec{fixture.gate_geometry}, ExpertMatrixSpec{gate_placeholder}, kExperts,
             false},
            {ExpertMatrixSpec{fixture.down_geometry}, ExpertMatrixSpec{down_placeholder}, kExperts,
             false}};
        ExpertCacheConfig config;
        // Four seats' worth of bytes: the pool cannot hold a token's ten experts, which is the
        // arrangement the layering exists for.
        config.capacity_bytes = 4u * 1408u * 1024u;
        fixture.requirements =
            describe_expert_cache(std::span<const ExpertBankSpec>(fixture.specs), config);

        std::mt19937 random(20261110);
        std::normal_distribution<float> gaussian;
        fixture.images.resize(fixture.specs.size());
        fixture.files.resize(fixture.specs.size());
        for (std::uint32_t bank = 0; bank < fixture.specs.size(); ++bank) {
            const auto entry_bytes =
                static_cast<std::size_t>(fixture.requirements.banks[bank].entry_bytes);
            fixture.images[bank].assign(entry_bytes * kRealised, std::byte{0});
            const auto rows    = bank == 2 ? kHidden : kInter;
            const auto columns = bank == 2 ? kInter : kHidden;
            const auto bits    = bank == 2 ? 6 : 4;
            std::vector<float> values(static_cast<std::size_t>(rows) * columns);
            for (auto& value : values) { value = gaussian(random) * 0.05F; }
            fixture.source.push_back(values);
            for (std::int32_t expert = 0; expert < kRealised; ++expert) {
                encode(fixture.images[bank], static_cast<std::size_t>(expert) * entry_bytes,
                       fixture.requirements.banks[bank], rows, columns, bits, values);
            }
            // Every expert of the bank needs a file plan; the ones the routes never name point at the
            // first image, so the fixture stays 30 MiB instead of 1.5 GiB.
            for (std::int32_t expert = 0; expert < kExperts; ++expert) {
                ExpertFileEntry entry;
                entry.image_bytes = entry_bytes;
                entry.chunks.push_back({bank,
                                        static_cast<std::uint64_t>(expert % kRealised) * entry_bytes,
                                        entry_bytes, 0});
                fixture.files[bank].experts.push_back(std::move(entry));
            }
        }
        MemoryReader reader;
        reader.images = &fixture.images;
        ExpertCache cache(std::span<const ExpertBankSpec>(fixture.specs), fixture.requirements,
                          device, {}, std::span<const ExpertFileBank>(fixture.files), &reader);

        // Routes: one token, ten distinct experts, coefficients that do not sum to one on purpose (the
        // routed term must apply them as given).
        fixture.ids.resize(static_cast<std::size_t>(kTopK) * kTokens);
        fixture.coefficients.resize(static_cast<std::size_t>(kTopK) * kTokens);
        for (std::int32_t j = 0; j < kTopK; ++j) {
            fixture.ids[static_cast<std::size_t>(j)] = j;
            fixture.coefficients[static_cast<std::size_t>(j)] = 0.05F * static_cast<float>(j + 1);
        }
        fixture.input.resize(static_cast<std::size_t>(kHidden) * kTokens);
        for (auto& value : fixture.input) {
            float f = gaussian(random);
            std::uint32_t bits = 0;
            std::memcpy(&bits, &f, sizeof(bits));
            bits += 0x7fffu + ((bits >> 16) & 1u);
            value = static_cast<std::uint16_t>(bits >> 16);
        }

        // --- the artifact descriptions and the stream --------------------------------------------
        const std::uint64_t gate_stack_shape[2] = {kExperts * kInter, kHidden};
        const std::uint64_t down_stack_shape[2] = {kExperts * kHidden, kInter};
        const auto gate_stack =
            weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit, gate_stack_shape);
        const auto down_stack =
            weight_geometry(QType::Q6_G64_FP16, QuantLayout::RowSplit, down_stack_shape);
        fixture.gate_layout = describe_weight_bank(gate_stack, {0, kExperts, kInter, kInter});
        fixture.down_layout = describe_weight_bank(down_stack, {0, kExperts, kHidden, kHidden});
        const auto desc = [](const WeightBankLayout& layout, const WeightGeometry& parent,
                             std::uint64_t first_object) {
            ExpertMatrixBankDesc out;
            out.layout = layout;
            out.parent = parent;
            out.objects.resize(kExperts);
            for (std::int32_t expert = 0; expert < kExperts; ++expert) {
                out.objects[expert] = first_object + expert;
            }
            return out;
        };
        ExpertStream stream(kExperts, desc(fixture.gate_layout, fixture.gate_geometry, 1000),
                            desc(fixture.gate_layout, fixture.gate_geometry, 2000),
                            desc(fixture.down_layout, fixture.down_geometry, 3000));
        stream.attach_split(cache, 0, fixture.requirements.banks[0], 1,
                            fixture.requirements.banks[1], 2, fixture.requirements.banks[2]);
        if (!stream.host_tier_supported()) { fail("the three-run form reports no host tier"); }

        const auto profile = ops::routed_swiglu_profile(
            fixture.gate_layout, fixture.gate_layout, fixture.down_layout, kTopK,
            ops::LinearPolicy::A16Only, ops::LinearPolicy::A16Only);

        DeviceBuffer ids_device(fixture.ids.size() * sizeof(std::int32_t));
        ids_device.copy_from_host(fixture.ids.data(), fixture.ids.size() * sizeof(std::int32_t));
        DeviceBuffer coefficients_device(fixture.coefficients.size() * sizeof(float));
        coefficients_device.copy_from_host(fixture.coefficients.data(),
                                           fixture.coefficients.size() * sizeof(float));
        DeviceBuffer input_device(fixture.input.size() * sizeof(std::uint16_t));
        input_device.copy_from_host(fixture.input.data(), fixture.input.size() * sizeof(std::uint16_t));
        Tensor x(input_device.p, DType::BF16, {kHidden, kTokens});
        ops::ExpertRoutes routes{Tensor(ids_device.p, DType::I32, {kTopK, kTokens}),
                                 Tensor(coefficients_device.p, DType::FP32, {kTopK, kTokens})};

        // --- (c) + (a): the pool keeps part of the layer, the placement is per run -----------------
        const bool placed = stream.ensure(ids_device.p, kTopK, kTokens, nullptr, nullptr);
        if (!placed) { fail("ensure() refused a layer the pool can partly serve"); }
        device.synchronize();
        if (const auto latched = cudaGetLastError(); latched != cudaSuccess) {
            std::fprintf(stderr, "expert stream split layers: after ensure: %s\n",
                         cudaGetErrorString(latched));
            fail("ensure() left a CUDA error");
        }
        std::vector<std::int32_t> addressable(static_cast<std::size_t>(kExperts), 0);
        for (std::int32_t expert = 0; expert < kExperts; ++expert) { addressable[expert] = 1; }
        for (const auto role :
             {ExpertMatrixRole::Gate, ExpertMatrixRole::Up, ExpertMatrixRole::Down}) {
            const auto mask = stream.split_mask_host(role);
            const auto offsets = stream.split_offsets_host(role);
            const auto bank = stream.split_bank(role);
            std::uint32_t resident = 0;
            for (std::int32_t expert = 0; expert < kExperts; ++expert) {
                addressable[static_cast<std::size_t>(expert)] &= mask[expert] != 0 ? 1 : 0;
                const auto slot = cache.expert_slot(bank, expert);
                if (mask[expert] != 0) { ++resident; }
                if (slot && mask[expert] != 0) {
                    if (offsets[expert] != cache.slot_offset(static_cast<std::uint32_t>(*slot))) {
                        fail("a run's offset entry is not its own slot address");
                    }
                } else if (offsets[expert] != 0) {
                    fail("an unmapped expert keeps an offset");
                }
            }
            if (resident == 0) { fail("a run placed nothing"); }
            if (resident >= static_cast<std::uint32_t>(kTopK)) {
                fail("the fixture's pool is not short of the layer");
            }
            std::fprintf(stderr,
                         "expert stream split layers: run=%zu bank=%u seats=%u placed=%u\n",
                         static_cast<std::size_t>(role), bank,
                         stream.split_resident_capacity(role), resident);
        }
        std::uint32_t covered = 0;
        for (const auto value : addressable) { covered += value != 0 ? 1u : 0u; }
        std::fprintf(stderr,
                     "expert stream split layers: pool+none addresses %u of %d routed experts\n",
                     covered, kTopK);
        if (covered == 0) { fail("nothing in the layer is addressable"); }

        // --- the all-Device reference, independent of the cache -----------------------------------
        // The same bytes uploaded whole and addressed by an offset table, so the reference cannot
        // inherit a placement bug from the pool.
        std::array<DeviceBuffer, 3> image_buffers{
            DeviceBuffer(fixture.images[0].size()), DeviceBuffer(fixture.images[1].size()),
            DeviceBuffer(fixture.images[2].size())};
        std::array<std::vector<std::uint64_t>, 3> reference_offsets;
        std::array<DeviceBuffer, 3> reference_offset_buffers{
            DeviceBuffer(kExperts * sizeof(std::uint64_t)), DeviceBuffer(kExperts * sizeof(std::uint64_t)),
            DeviceBuffer(kExperts * sizeof(std::uint64_t))};
        for (std::size_t bank = 0; bank < 3; ++bank) {
            image_buffers[bank].copy_from_host(fixture.images[bank].data(),
                                               fixture.images[bank].size());
            reference_offsets[bank].resize(kExperts);
            for (std::int32_t expert = 0; expert < kExperts; ++expert) {
                reference_offsets[bank][expert] = static_cast<std::uint64_t>(expert % kRealised) *
                                                  fixture.requirements.banks[bank].entry_bytes;
            }
            reference_offset_buffers[bank].copy_from_host(
                reference_offsets[bank].data(), reference_offsets[bank].size() * sizeof(std::uint64_t));
        }
        const auto reference_view = [&](std::size_t bank) {
            ExpertMatrixDeviceView view;
            const auto& entry = fixture.requirements.banks[bank];
            const auto* base  = static_cast<const std::byte*>(image_buffers[bank].p);
            view.codes   = base + entry.codes.offset;
            view.high    = entry.high.bytes ? base + entry.high.offset : nullptr;
            view.scales  = entry.scales.bytes ? base + entry.scales.offset : nullptr;
            view.offsets = static_cast<const std::uint64_t*>(reference_offset_buffers[bank].p);
            view.indexed_storage = MemorySpan{image_buffers[bank].p, image_buffers[bank].bytes, cuda};
            return view;
        };
        ops::CudaSwiGLUExperts reference_experts;
        reference_experts.gate      = reference_view(0);
        reference_experts.up        = reference_view(1);
        reference_experts.bank.down = reference_view(2);
        reference_experts.bank.count  = kExperts;
        reference_experts.bank.device = cuda;

        std::vector<std::int32_t> all(static_cast<std::size_t>(kExperts), 1);
        DeviceBuffer all_mask_device(all.size() * sizeof(std::int32_t));
        all_mask_device.copy_from_host(all.data(), all.size() * sizeof(std::int32_t));
        const auto workspace_bytes = ops::routed_swiglu_workspace_capacity_bytes(
            profile, ops::ExpertSubsetKind::MaskOnes, DType::FP32, ops::MoeEpilogue::Write,
            kTokens, kTokens);
        DeviceBuffer scratch(workspace_bytes);
        MemoryArena arena(MemorySpan{scratch.p, scratch.bytes, cuda});
        DeviceBuffer reference_device(static_cast<std::size_t>(kHidden) * kTokens * sizeof(float));
        Tensor reference(reference_device.p, DType::FP32, {kHidden, kTokens});
        {
            Tensor mask(all_mask_device.p, DType::I32, {kExperts});
            auto scope = arena.scope();
            ops::routed_swiglu(x, profile, reference_experts, routes,
                               ops::ExpertSubset{ops::ExpertSubsetKind::MaskOnes, mask},
                               ops::MoeEpilogue::Write, reference, arena, device.execution_view());
        }
        device.synchronize();
        if (const auto latched = cudaGetLastError(); latched != cudaSuccess) {
            std::fprintf(stderr, "expert stream split layers: after the all-Device reference: %s\n",
                         cudaGetErrorString(latched));
            fail("the all-Device reference left a CUDA error");
        }

        // --- (b) the layer as the model runs it: Device half + CPU half + one merge --------------
        std::vector<std::int32_t> split_mask(addressable);
        DeviceBuffer split_mask_device(split_mask.size() * sizeof(std::int32_t));
        split_mask_device.copy_from_host(split_mask.data(),
                                         split_mask.size() * sizeof(std::int32_t));
        DeviceBuffer mixed_device(static_cast<std::size_t>(kHidden) * kTokens * sizeof(float));
        Tensor mixed(mixed_device.p, DType::FP32, {kHidden, kTokens});
        {
            Tensor mask(split_mask_device.p, DType::I32, {kExperts});
            auto scope = arena.scope();
            ops::routed_swiglu(x, profile, stream.experts(), routes,
                               ops::ExpertSubset{ops::ExpertSubsetKind::MaskOnes, mask},
                               ops::MoeEpilogue::Write, mixed, arena, device.execution_view());
        }
        // The CPU views: the same bytes in the packed form a host-tier bank presents.
        std::array<WeightParent, 3> parents;
        std::array<std::vector<std::uint64_t>, 3> cpu_offsets;
        std::array<WeightBankView, 3> views;
        for (std::size_t bank = 0; bank < 3; ++bank) {
            const auto& entry  = fixture.requirements.banks[bank];
            const auto rows    = bank == 2 ? kHidden : kInter;
            const auto columns = bank == 2 ? kInter : kHidden;
            const auto kind    = bank == 2 ? QType::Q6_G64_FP16 : QType::Q4_G64_FP16;
            const std::uint64_t shape[2] = {
                static_cast<std::uint64_t>(kExperts) * static_cast<std::uint64_t>(rows),
                static_cast<std::uint64_t>(columns)};
            auto geometry  = weight_geometry(kind, QuantLayout::RowSplit, shape);
            geometry.bytes = fixture.images[bank].size();
            parents[bank] =
                WeightParent{geometry, fixture.images[bank].data(), 0.0F, Device::cpu(), {}};
            auto layout       = bank == 2 ? fixture.down_layout : fixture.gate_layout;
            layout.codes.offset  = 0;
            layout.high.offset   = layout.codes.expert_bytes;
            layout.scales.offset = layout.codes.expert_bytes + layout.high.expert_bytes;
            cpu_offsets[bank].resize(kExperts);
            for (std::int32_t expert = 0; expert < kExperts; ++expert) {
                cpu_offsets[bank][expert] = static_cast<std::uint64_t>(expert % kRealised) *
                                            entry.entry_bytes;
            }
            views[bank].parent         = &parents[bank];
            views[bank].layout         = layout;
            views[bank].expert_offsets = cpu_offsets[bank].data();
        }
        ops::CpuSwiGLUExperts cpu_views;
        cpu_views.gate = views[0];
        cpu_views.up   = views[1];
        cpu_views.down = views[2];

        const auto cpu_capacity = ops::split_routed_swiglu_workspace_capacity_bytes(
            profile, ops::ExpertSubsetKind::MaskZeros, kTokens, kTokens);
        const auto cpu_capacity_all = ops::split_routed_swiglu_workspace_capacity_bytes(
            profile, ops::ExpertSubsetKind::All, kTokens, kTokens);
        std::vector<std::byte> cpu_scratch(cpu_capacity + 256);
        const auto cpu_base    = reinterpret_cast<std::uintptr_t>(cpu_scratch.data());
        const auto cpu_aligned = (256 - (cpu_base & 255)) & 255;
        MemoryArena cpu_arena(
            MemorySpan{cpu_scratch.data() + cpu_aligned, cpu_capacity, Device::cpu()});
        std::vector<float> cpu_term(static_cast<std::size_t>(kHidden) * kTokens, 0.0F);
        Tensor cpu_out(cpu_term.data(), Device::cpu(), DType::FP32, {kHidden, kTokens});
        {
            Tensor mask_cpu(split_mask.data(), Device::cpu(), DType::I32, {kExperts});
            ops::ExpertRoutes routes_cpu{
                Tensor(const_cast<std::int32_t*>(fixture.ids.data()), Device::cpu(), DType::I32,
                       {kTopK, kTokens}),
                Tensor(const_cast<float*>(fixture.coefficients.data()), Device::cpu(), DType::FP32,
                       {kTopK, kTokens})};
            Tensor x_cpu(const_cast<std::uint16_t*>(fixture.input.data()), Device::cpu(),
                         DType::BF16, {kHidden, kTokens});
            static CpuTopology topology;
            static CpuExecutor executor(topology, plan_cpu_team(topology.info(), 8));
            ops::routed_swiglu(x_cpu, profile, cpu_views, routes_cpu,
                               ops::ExpertSubset{ops::ExpertSubsetKind::MaskZeros, mask_cpu},
                               cpu_out, cpu_arena, executor.execution_view());
        }
        std::vector<std::uint16_t> residual(static_cast<std::size_t>(kHidden) * kTokens, 0);
        {
            DeviceBuffer host_term(cpu_term.size() * sizeof(float));
            host_term.copy_from_host(cpu_term.data(), cpu_term.size() * sizeof(float));
            DeviceBuffer residual_device(residual.size() * sizeof(std::uint16_t));
            residual_device.copy_from_host(residual.data(), residual.size() * sizeof(std::uint16_t));
            ops::moe_merge_host_term(host_term.p, mixed_device.p, residual_device.p,
                                     static_cast<int>(cpu_term.size()), nullptr);
            device.synchronize();
            residual_device.copy_to_host(residual.data(), residual.size() * sizeof(std::uint16_t));
        }

        std::vector<float> reference_values(static_cast<std::size_t>(kHidden) * kTokens);
        reference_device.copy_to_host(reference_values.data(),
                                      reference_values.size() * sizeof(float));
        std::vector<float> mixed_values(static_cast<std::size_t>(kHidden) * kTokens);
        mixed_device.copy_to_host(mixed_values.data(), mixed_values.size() * sizeof(float));

        // The oracle: FP64, decoded from the fixture's own bytes.
        std::vector<double> oracle(static_cast<std::size_t>(kHidden) * kTokens, 0.0);
        std::vector<double> xv(static_cast<std::size_t>(kHidden));
        for (std::int32_t t = 0; t < kTokens; ++t) {
            for (std::int32_t k = 0; k < kHidden; ++k) {
                xv[static_cast<std::size_t>(k)] =
                    bf16_to_float(fixture.input[static_cast<std::size_t>(t) * kHidden + k]);
            }
            for (std::int32_t j = 0; j < kTopK; ++j) {
                const auto expert = fixture.ids[static_cast<std::size_t>(t) * kTopK + j];
                const auto coefficient =
                    fixture.coefficients[static_cast<std::size_t>(t) * kTopK + j];
                const auto gate_planes = planes_of(
                    fixture.images[0],
                    static_cast<std::size_t>(expert % kRealised) *
                        fixture.requirements.banks[0].entry_bytes,
                    fixture.requirements.banks[0]);
                const auto up_planes = planes_of(
                    fixture.images[1],
                    static_cast<std::size_t>(expert % kRealised) *
                        fixture.requirements.banks[1].entry_bytes,
                    fixture.requirements.banks[1]);
                const auto down_planes = planes_of(
                    fixture.images[2],
                    static_cast<std::size_t>(expert % kRealised) *
                        fixture.requirements.banks[2].entry_bytes,
                    fixture.requirements.banks[2]);
                std::vector<double> activation(static_cast<std::size_t>(kInter));
                for (std::int32_t i = 0; i < kInter; ++i) {
                    const auto gate = dot64(row_view(gate_planes, i, kHidden, 4), kHidden, 4, xv);
                    const auto up   = dot64(row_view(up_planes, i, kHidden, 4), kHidden, 4, xv);
                    activation[static_cast<std::size_t>(i)] =
                        (gate / (1.0 + std::exp(-gate))) * up;
                }
                for (std::int32_t h = 0; h < kHidden; ++h) {
                    const auto value = dot64(row_view(down_planes, h, kInter, 6), kInter, 6,
                                             activation);
                    oracle[static_cast<std::size_t>(t) * kHidden + h] += coefficient * value;
                }
            }
        }

        // Probe: does the oracle read the bytes that were written? Row 0 of the gate of expert 0,
        // against the values the row was encoded from (a 4-bit quantisation of a smooth row is a
        // fraction of a percent, so the two must agree to ~1e-2 relative).
        {
            const auto gate_planes = planes_of(fixture.images[0], 0, fixture.requirements.banks[0]);
            std::vector<double> one_hot(static_cast<std::size_t>(kHidden), 0.0);
            for (std::int32_t k = 0; k < kHidden; ++k) { one_hot[static_cast<std::size_t>(k)] = 1.0; }
            std::vector<double> xv(static_cast<std::size_t>(kHidden));
            for (std::int32_t k = 0; k < kHidden; ++k) {
                xv[static_cast<std::size_t>(k)] =
                    bf16_to_float(fixture.input[static_cast<std::size_t>(k)]);
            }
            const auto decoded = dot64(row_view(gate_planes, 0, kHidden, 4), kHidden, 4, xv);
            double source = 0.0;
            for (std::int32_t k = 0; k < kHidden; ++k) {
                source += static_cast<double>(fixture.source[0][static_cast<std::size_t>(k)]) *
                          xv[static_cast<std::size_t>(k)];
            }
            std::fprintf(stderr,
                         "expert stream split layers: gate row 0 decoded=%.6g source=%.6g "
                         "(entry codes=%llu high=%llu scales=%llu bytes=%llu)\n",
                         decoded, source,
                         static_cast<unsigned long long>(fixture.requirements.banks[0].codes.offset),
                         static_cast<unsigned long long>(fixture.requirements.banks[0].high.offset),
                         static_cast<unsigned long long>(fixture.requirements.banks[0].scales.offset),
                         static_cast<unsigned long long>(fixture.requirements.banks[0].entry_bytes));
            // The other two matrices of the same expert, the same question.
            for (std::size_t bank = 1; bank < 3; ++bank) {
                const auto planes  = planes_of(fixture.images[bank], 0, fixture.requirements.banks[bank]);
                const auto columns = bank == 2 ? kInter : kHidden;
                const auto bits    = bank == 2 ? 6 : 4;
                const auto value   = dot64(row_view(planes, 0, columns, bits), columns, bits, xv);
                double expected    = 0.0;
                for (std::int32_t k = 0; k < columns; ++k) {
                    expected += static_cast<double>(fixture.source[bank][static_cast<std::size_t>(k)]) *
                                xv[static_cast<std::size_t>(k)];
                }
                std::fprintf(stderr,
                             "expert stream split layers: bank %zu row 0 decoded=%.6g source=%.6g "
                             "(codes=%llu high=%llu scales=%llu)\n",
                             bank, value, expected,
                             static_cast<unsigned long long>(fixture.requirements.banks[bank].codes.offset),
                             static_cast<unsigned long long>(fixture.requirements.banks[bank].high.offset),
                             static_cast<unsigned long long>(fixture.requirements.banks[bank].scales.offset));
            }
        }

        // Decisive comparison: the CPU kernels and the Device kernels on the SAME routes and the same
        // subset. If these two disagree, the oracle is not the thing to blame.
        std::vector<float> cpu_all(static_cast<std::size_t>(kHidden) * kTokens, 0.0F);
        {
            Tensor cpu_all_tensor(cpu_all.data(), Device::cpu(), DType::FP32, {kHidden, kTokens});
            ops::ExpertRoutes routes_cpu{
                Tensor(const_cast<std::int32_t*>(fixture.ids.data()), Device::cpu(), DType::I32,
                       {kTopK, kTokens}),
                Tensor(const_cast<float*>(fixture.coefficients.data()), Device::cpu(), DType::FP32,
                       {kTopK, kTokens})};
            Tensor x_cpu(const_cast<std::uint16_t*>(fixture.input.data()), Device::cpu(),
                         DType::BF16, {kHidden, kTokens});
            std::vector<std::byte> extra_scratch(cpu_capacity_all + 256);
            const auto extra_base = reinterpret_cast<std::uintptr_t>(extra_scratch.data());
            const auto extra_align = (256 - (extra_base & 255)) & 255;
            MemoryArena extra_arena(MemorySpan{extra_scratch.data() + extra_align, cpu_capacity_all,
                                               Device::cpu()});
            static CpuTopology topology2;
            static CpuExecutor executor2(topology2, plan_cpu_team(topology2.info(), 8));
            ops::routed_swiglu(x_cpu, profile, cpu_views, routes_cpu, ops::ExpertSubset{},
                               cpu_all_tensor, extra_arena, executor2.execution_view());
        }

        const auto rel_l2 = [](const std::vector<double>& a, const std::vector<double>& b) {
            double num = 0.0, den = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                const double d = a[i] - b[i];
                num += d * d;
                den += b[i] * b[i];
            }
            return den > 0.0 ? std::sqrt(num / den) : 0.0;
        };

        // Probe: one expert alone (coefficient 1 on route 0, zero on the rest), so a sign, scale or
        // plane-order mistake cannot hide behind a sum.
        std::vector<float> one_coefficients(static_cast<std::size_t>(kTopK) * kTokens, 0.0F);
        one_coefficients[0] = 1.0F;
        std::vector<float> one_device(static_cast<std::size_t>(kHidden) * kTokens, 0.0F);
        {
            DeviceBuffer coefficients_one(one_coefficients.size() * sizeof(float));
            coefficients_one.copy_from_host(one_coefficients.data(),
                                            one_coefficients.size() * sizeof(float));
            ops::ExpertRoutes one_routes{Tensor(ids_device.p, DType::I32, {kTopK, kTokens}),
                                         Tensor(coefficients_one.p, DType::FP32, {kTopK, kTokens})};
            DeviceBuffer one_buffer(static_cast<std::size_t>(kHidden) * kTokens * sizeof(float));
            Tensor one(one_buffer.p, DType::FP32, {kHidden, kTokens});
            Tensor mask(all_mask_device.p, DType::I32, {kExperts});
            auto scope = arena.scope();
            ops::routed_swiglu(x, profile, reference_experts, one_routes,
                               ops::ExpertSubset{ops::ExpertSubsetKind::MaskOnes, mask},
                               ops::MoeEpilogue::Write, one, arena, device.execution_view());
            device.synchronize();
            one_buffer.copy_to_host(one_device.data(), one_device.size() * sizeof(float));
        }
        {
            // The oracle for the same thing: expert 0, coefficient 1.
            std::vector<double> xv(static_cast<std::size_t>(kHidden));
            for (std::int32_t k = 0; k < kHidden; ++k) {
                xv[static_cast<std::size_t>(k)] = bf16_to_float(fixture.input[static_cast<std::size_t>(k)]);
            }
            const auto gate_planes = planes_of(fixture.images[0], 0, fixture.requirements.banks[0]);
            const auto up_planes   = planes_of(fixture.images[1], 0, fixture.requirements.banks[1]);
            const auto down_planes = planes_of(fixture.images[2], 0, fixture.requirements.banks[2]);
            std::vector<double> activation(static_cast<std::size_t>(kInter));
            std::vector<double> gate_values(static_cast<std::size_t>(kInter));
            std::vector<double> up_values(static_cast<std::size_t>(kInter));
            for (std::int32_t i = 0; i < kInter; ++i) {
                gate_values[static_cast<std::size_t>(i)] =
                    dot64(row_view(gate_planes, i, kHidden, 4), kHidden, 4, xv);
                up_values[static_cast<std::size_t>(i)] =
                    dot64(row_view(up_planes, i, kHidden, 4), kHidden, 4, xv);
                const double g = gate_values[static_cast<std::size_t>(i)];
                activation[static_cast<std::size_t>(i)] =
                    (g / (1.0 + std::exp(-g))) * up_values[static_cast<std::size_t>(i)];
            }
            std::vector<double> one_oracle(static_cast<std::size_t>(kHidden));
            for (std::int32_t h = 0; h < kHidden; ++h) {
                one_oracle[static_cast<std::size_t>(h)] =
                    dot64(row_view(down_planes, h, kInter, 6), kInter, 6, activation);
            }
            std::vector<double> one_double(one_device.size());
            for (std::size_t i = 0; i < one_device.size(); ++i) { one_double[i] = one_device[i]; }
            // A third opinion: the same expert evaluated from the UNQUANTISED source values. It has to
            // land between the kernels and the decoded oracle if both are right.
            double source_value = 0.0;
            {
                std::vector<double> activation(static_cast<std::size_t>(kInter));
                for (std::int32_t i = 0; i < kInter; ++i) {
                    double g = 0.0, u = 0.0;
                    for (std::int32_t k = 0; k < kHidden; ++k) {
                        const auto index = static_cast<std::size_t>(i) * kHidden + k;
                        g += static_cast<double>(fixture.source[0][index]) * xv[static_cast<std::size_t>(k)];
                        u += static_cast<double>(fixture.source[1][index]) * xv[static_cast<std::size_t>(k)];
                    }
                    activation[static_cast<std::size_t>(i)] = (g / (1.0 + std::exp(-g))) * u;
                }
                for (std::int32_t k = 0; k < kInter; ++k) {
                    source_value += static_cast<double>(fixture.source[2][static_cast<std::size_t>(k)]) *
                                    activation[static_cast<std::size_t>(k)];
                }
            }
            std::fprintf(stderr,
                         "expert stream split layers: one expert rel_l2=%.3e | device[0..3]=%.6g %.6g "
                         "%.6g %.6g oracle[0..3]=%.6g %.6g %.6g %.6g | source=%.6g | "
                         "gate[0..2]=%.6g %.6g %.6g up[0..2]=%.6g %.6g %.6g\n",
                         rel_l2(one_double, one_oracle), one_device[0], one_device[1], one_device[2],
                         one_device[3], one_oracle[0], one_oracle[1], one_oracle[2], one_oracle[3],
                         source_value, gate_values[0], gate_values[1], gate_values[2],
                         up_values[0], up_values[1], up_values[2]);
        }
        std::vector<double> reference_double(reference_values.size());
        for (std::size_t i = 0; i < reference_values.size(); ++i) {
            reference_double[i] = reference_values[i];
        }
        std::vector<double> split_double(residual.size());
        for (std::size_t i = 0; i < residual.size(); ++i) {
            split_double[i] = bf16_to_float(residual[i]);
        }
        const auto reference_error = rel_l2(reference_double, oracle);
        const auto split_error     = rel_l2(split_double, oracle);
        {
            std::vector<double> cpu_all_double(cpu_all.size());
            for (std::size_t i = 0; i < cpu_all.size(); ++i) { cpu_all_double[i] = cpu_all[i]; }
            std::fprintf(stderr,
                         "expert stream split layers: cpu-vs-device(all)=%.3e | cpu_all[0..2]=%.6g "
                         "%.6g %.6g device_all[0..2]=%.6g %.6g %.6g oracle[0..2]=%.6g %.6g %.6g\n",
                         rel_l2(cpu_all_double, reference_double), cpu_all[0], cpu_all[1], cpu_all[2],
                         reference_values[0], reference_values[1], reference_values[2], oracle[0],
                         oracle[1], oracle[2]);
        }
        // Half-by-half: which leg disagrees with the oracle, if either does.
        {
            std::vector<double> oracle_device(oracle.size(), 0.0), oracle_cpu(oracle.size(), 0.0);
            for (std::int32_t j = 0; j < kTopK; ++j) {
                auto& target = addressable[static_cast<std::size_t>(fixture.ids[j])] != 0
                                   ? oracle_device
                                   : oracle_cpu;
                const auto coefficient = fixture.coefficients[static_cast<std::size_t>(j)];
                const auto expert      = fixture.ids[static_cast<std::size_t>(j)];
                const auto gate_planes = planes_of(
                    fixture.images[0], static_cast<std::size_t>(expert) * fixture.requirements.banks[0].entry_bytes,
                    fixture.requirements.banks[0]);
                const auto up_planes = planes_of(
                    fixture.images[1], static_cast<std::size_t>(expert) * fixture.requirements.banks[1].entry_bytes,
                    fixture.requirements.banks[1]);
                const auto down_planes = planes_of(
                    fixture.images[2], static_cast<std::size_t>(expert) * fixture.requirements.banks[2].entry_bytes,
                    fixture.requirements.banks[2]);
                std::vector<double> activation(static_cast<std::size_t>(kInter));
                for (std::int32_t i = 0; i < kInter; ++i) {
                    const auto gate = dot64(row_view(gate_planes, i, kHidden, 4), kHidden, 4, xv);
                    const auto up   = dot64(row_view(up_planes, i, kHidden, 4), kHidden, 4, xv);
                    activation[static_cast<std::size_t>(i)] = (gate / (1.0 + std::exp(-gate))) * up;
                }
                for (std::int32_t h = 0; h < kHidden; ++h) {
                    target[static_cast<std::size_t>(h)] +=
                        coefficient * dot64(row_view(down_planes, h, kInter, 6), kInter, 6,
                                            activation);
                }
            }
            std::vector<double> device_double(mixed_values.size());
            for (std::size_t i = 0; i < mixed_values.size(); ++i) { device_double[i] = mixed_values[i]; }
            std::vector<double> cpu_double(cpu_term.size());
            for (std::size_t i = 0; i < cpu_term.size(); ++i) { cpu_double[i] = cpu_term[i]; }
            std::fprintf(stderr,
                         "expert stream split layers: half rel_l2 device=%.3e cpu=%.3e | "
                         "oracle[0..2]=%.6g %.6g %.6g device[0..2]=%.6g %.6g %.6g "
                         "cpu[0..2]=%.6g %.6g %.6g\n",
                         rel_l2(device_double, oracle_device), rel_l2(cpu_double, oracle_cpu),
                         oracle_device[0], oracle_device[1], oracle_device[2], mixed_values[0],
                         mixed_values[1], mixed_values[2], cpu_term[0], cpu_term[1], cpu_term[2]);
        }
        std::fprintf(stderr,
                     "expert stream split layers: rel_l2 device-only=%.3e split(cpu+device)=%.3e "
                     "(device experts=%u, cpu experts=%u)\n",
                     reference_error, split_error, covered, kTopK - covered);
        if (!(reference_error < 2e-3)) { fail("the all-Device reference does not match the FP64 oracle"); }
        // The two legs in FP32, before the residual's BF16 rounding: that sum is the arithmetic the
        // layer performs, so it is what "as close as the all-Device answer" is asked of. The published
        // residual carries one BF16 rounding of the same value (~2^-9 relative), which is judged
        // separately instead of being folded into the arithmetic's threshold.
        std::vector<double> merged(oracle.size());
        for (std::size_t i = 0; i < merged.size(); ++i) {
            merged[i] = static_cast<double>(cpu_term[i]) + static_cast<double>(mixed_values[i]);
        }
        const auto merged_error = rel_l2(merged, oracle);
        std::fprintf(stderr,
                     "expert stream split layers: rel_l2 cpu+device(fp32)=%.3e residual(bf16)=%.3e\n",
                     merged_error, split_error);
        if (!(merged_error < 2e-3)) { fail("the CPU half + merge does not match the FP64 oracle"); }
        if (!(merged_error < reference_error * 8.0 + 1e-5)) {
            fail("the layered answer is not as close to the oracle as the all-Device one");
        }
        if (!(split_error < 8e-3)) {
            fail("the layered answer does not survive the residual's BF16 rounding");
        }

        // --- (a) the tier's second leg, on a pinned arena ---------------------------------------
        {
            std::vector<HostExpertBankDesc> descs;
            for (std::size_t bank = 0; bank < 3; ++bank) {
                const auto geometry = bank == 2 ? fixture.down_geometry : fixture.gate_geometry;
                const auto layout   = bank == 2 ? fixture.down_layout : fixture.gate_layout;
                descs.push_back({fixture.requirements.banks[bank], layout, layout, geometry, geometry,
                                 false});
            }
            const auto slot_bytes = HostExpertResidency::slot_bytes_for(
                fixture.requirements.banks[2].entry_bytes);
            // Room for the three runs' complements at once: 8 routes a run are not in the pool and
            // each run's fill is its own, so a range sized for one run would let the next fill evict
            // it (a legitimate arrangement in which the evicted routes fall to the CPU half, but not
            // the one this block measures).
            auto residency = std::make_shared<HostExpertResidency>(
                descs, fixture.files, &reader,
                HostWarmBudget{.routed_bytes = 36 * slot_bytes, .prefetch_bytes = 0,
                               .second_chance_bytes = 0});
            residency->materialize_arena();
            stream.attach_host(residency);
            const bool tier_placed = stream.ensure(ids_device.p, kTopK, kTokens, nullptr, nullptr);
            if (!tier_placed) { fail("ensure() refused a layer with a host tier behind it"); }
            device.synchronize();
            std::uint32_t addressable_all = 0;
            for (std::int32_t expert = 0; expert < kExperts; ++expert) {
                if (stream.split_mask_host(ExpertMatrixRole::Gate)[expert] != 0) {
                    ++addressable_all;
                }
            }
            std::fprintf(stderr,
                         "expert stream split layers: tier mask covers %u experts, %u of %d routes "
                         "on the host leg (mapped=%d)\n",
                         addressable_all, stream.split_host_routes(), kTopK,
                         residency->arena_device_base() != nullptr ? 1 : 0);
            if (addressable_all == 0) { fail("no expert is addressable after a tier fill"); }
            // Every route must have an address on one leg or the other: that is what makes the layer
            // complete instead of silently skipping experts.
            for (std::int32_t j = 0; j < kTopK; ++j) {
                const auto expert = fixture.ids[static_cast<std::size_t>(j)];
                if (stream.split_mask_host(ExpertMatrixRole::Gate)[expert] == 0) {
                    std::fprintf(stderr,
                                 "expert stream split layers: route expert %d has no address\n",
                                 expert);
                    ++failures;
                }
            }
            for (const auto role :
                 {ExpertMatrixRole::Gate, ExpertMatrixRole::Up, ExpertMatrixRole::Down}) {
                for (std::int32_t expert = 0; expert < kTopK; ++expert) {
                    const auto slot   = residency->peek_slot(stream.split_bank(role), expert);
                    const auto offset = stream.split_offsets_host(role)[expert];
                    const auto pool   = cache.expert_slot(stream.split_bank(role), expert);
                    if (slot < 0) {
                        if (offset != 0 && !pool) { fail("a tier-absent expert keeps an offset"); }
                        continue;
                    }
                    if (pool && offset != cache.slot_offset(static_cast<std::uint32_t>(*pool))) {
                        fail("a tier-served expert's offset is not its pool address");
                    }
                }
            }
        }

        if (failures != 0) { return 1; }
        std::printf("expert stream split layers: ok experts=%d top_k=%d device=%u cpu=%u "
                    "rel_l2 device=%.3e split=%.3e\n",
                    kExperts, kTopK, covered, kTopK - covered, reference_error, split_error);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "expert stream split layers: %s\n", error.what());
        return 1;
    }
}

