#include "sparse_gqa_fixture.h"
#include "core/decode_graph.h"
#include "ops/softmax_attention/qsa/storage.h"

#include <cstdlib>
#include <string>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::qsa::sparse;

namespace {
namespace cache_reference = ninfer::test::qsa::prepare;
// Same A16 attention budget across all shapes and routes, including the fused gate and final
// BF16 store. Inputs cover near-uniform, unit-RMS and sharper QK distributions.
constexpr ReductionCriterion kSparseA16Criterion{2.8e-3, 1.0e-3, 2.7e-3};

// Every local geometry keeps twelve query heads per KV head.
std::size_t capacity(KvCacheStorage storage, int query_heads, int lo, int hi, int batch,
                     const ops::QsaSparseGqaExecutionEnvelope& envelope,
                     DeviceExecutionView execution) {
    return ops::qsa_sparse_gqa_workspace_capacity_bytes(storage, query_heads, query_heads / 12, lo,
                                                        hi, batch, envelope, execution);
}

int verify(Fixture& f, const std::string& label, bool preserved = true) {
    const auto reference = f.reference();
    const auto got       = from_device_bf16(f.out.data(), f.qhost.size());
    int failures         = verify_reduction(label, got, reference, kSparseA16Criterion);
    const auto words     = from_device<std::uint16_t>(f.out.data(), f.qhost.size());
    for (int m = 0; m < f.T * f.B; ++m)
        if (f.visible[m].empty())
            for (int i = 0; i < 256 * f.q_heads; ++i)
                if (words[std::size_t(m) * (256 * f.q_heads) + i] != 0) {
                    ++failures;
                    break;
                }
    if (preserved) {
        failures += verify_exact("query preserved",
                                 from_device<std::uint16_t>(f.q.data(), f.qhost.size()), f.qhost);
        failures += verify_exact(
            "gate preserved", from_device<std::uint16_t>(f.gate.data(), f.ghost.size()), f.ghost);
        failures += verify_exact(
            "K preserved", from_device<std::uint16_t>(f.keys.device.data(), f.keys.host.size()),
            f.keys.host);
        failures += verify_exact(
            "V preserved", from_device<std::uint16_t>(f.values.device.data(), f.values.host.size()),
            f.values.host);
        failures += verify_exact("selection preserved",
                                 from_device<int>(f.selections.data(), f.ids.size()), f.ids);
        failures += verify_exact("counts preserved",
                                 from_device<int>(f.sizes.data(), f.counts.size()), f.counts);
        failures += verify_exact("positions preserved",
                                 from_device<int>(f.pos.data(), f.positions.size()), f.positions);
        failures += verify_exact("validity preserved",
                                 from_device<int>(f.lengths.data(), f.valid.size()), f.valid);
        failures += verify_exact("selectors preserved",
                                 from_device<int>(f.rows.data(), f.selectors.size()), f.selectors);
        failures += verify_exact("tables preserved",
                                 from_device<int>(f.table.data(), f.tables.size()), f.tables);
        if (f.encoded_key)
            for (const auto* plane : {&f.encoded_key->data, &f.encoded_key->scales,
                                      &f.encoded_value->data, &f.encoded_value->scales})
                failures += plane->preserved(label);
    }
    for (const auto* buffer :
         {&f.out, &f.q, &f.gate, &f.keys.device, &f.values.device, &f.selections, &f.sizes})
        failures += buffer->verify_guards(label);
    return failures;
}

int run(DeviceContext& device, int t, int b, int history, bool padded = false, int cap = 512,
        float magnitude = 1, bool graph = false, KvCacheStorage storage = KvCacheStorage::BFloat16,
        int live_limit = std::numeric_limits<int>::max(), int query_heads = 24) {
    Fixture f(t, b, history, padded, storage, false, query_heads);
    f.live_limit     = live_limit;
    const auto bytes = capacity(storage, query_heads, t, t, b, f.envelope,
                                device.execution_view());
    GuardedDeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
    WorkspaceArena arena(DeviceSpan{scratch.data(), std::max<std::size_t>(bytes, 256)});
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int phase = 0; phase < (graph ? 3 : 1); ++phase) {
        f.reset(phase, cap, magnitude);
        if (graph && phase == 0) {
            DecodeGraphDefinition definition;
            definition.capture(device.stream, [&] { f.launch(arena, device.execution_view()); });
            executable.instantiate(definition);
        }
        if (graph)
            executable.launch(device.stream);
        else
            f.launch(arena, device.execution_view());
        device.synchronize();
        const auto label = "sparse storage=" + std::to_string(int(storage)) +
                           " T=" + std::to_string(t) + " B=" + std::to_string(b) +
                           " L=" + std::to_string(history) + " cap=" + std::to_string(cap) +
                           " phase=" + std::to_string(phase);
        failures += verify(f, label);
        failures += scratch.verify_guards(label);
        failures += arena.used() != 0 || arena.peak_used() > bytes;
        if (failures) return failures;
    }
    return failures;
}

int gate_rounding(DeviceContext& device, KvCacheStorage storage = KvCacheStorage::BFloat16,
                  int tokens = 1, std::uint32_t bound = 1, int query_heads = 24) {
    Fixture f(tokens, 1, 1, false, storage, false, query_heads);
    f.envelope.max_visible_tokens = bound;
    f.reset();
    std::fill(f.ghost.begin(), f.ghost.end(), f32_to_bf16(1.0f));
    const int page = f.tables[f.selectors[0] * f.table_stride];
    for (int h = 0; h < f.kv_heads; ++h)
        for (int d = 0; d < 256; ++d)
            f.values.host[f.values.offset(d, h, 0, page)] = half_word(1.00390625f);
    f.upload();
    if (f.encoded_value) {
        auto& v = *f.encoded_value;
        for (int h = 0; h < f.kv_heads; ++h) {
            const auto db = v.data.offset(0, h, page), sb = v.scales.offset(0, h, page);
            std::fill_n(v.data.host.begin() + db, v.format.data_extent, 0);
            std::fill_n(v.scales.host.begin() + sb,
                        v.format.scale_extent * dtype_size(v.format.scale_dtype), 0);
            if (v.format.encoding == cache_reference::Encoding::Nvfp4) {
                // R^T of basis values 16 at d=0 and 1/16 at d=16 is 1 +/- 1/256.
                // The positive half lies exactly on a BF16 midpoint before the gate.
                v.data.host[db]       = 6;
                v.data.host[db + 8]   = 6;
                v.scales.host[sb]     = cache_reference::encode_minifloat(4.0f, false);
                v.scales.host[sb + 1] = cache_reference::encode_minifloat(0.0625f / 4, false);
            } else {
                const auto one = v.format.encoding == cache_reference::Encoding::Int8
                                     ? 1
                                     : cache_reference::encode_minifloat(1.0f, false);
                std::fill_n(v.data.host.begin() + db, 256, one);
                for (int g = 0; g < v.format.scale_extent; ++g)
                    cache_reference::write_u16(v.scales.host, sb + 2 * g, half_word(1.00390625f));
            }
        }
        v.data.upload();
        v.scales.upload();
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    WorkspaceArena workspace(
        std::max<std::size_t>(256, capacity(storage, query_heads, tokens, tokens, 1, f.envelope,
                                            device.execution_view())));
    f.launch(workspace, device.execution_view());
    device.synchronize();
    return verify(f, "gate without intermediate BF16 storage=" + std::to_string(int(storage)) +
                         " T=" + std::to_string(tokens) + " bound=" + std::to_string(bound));
}

int read_footprint(DeviceContext& device, KvCacheStorage storage = KvCacheStorage::BFloat16,
                   bool padded = true, int query_heads = 24) {
    Fixture f(3, 4, 2053, padded, storage, false, query_heads);
    f.reset(0, 1);
    DeviceBuffer q(f.q.bytes()), gate(f.gate.bytes()), pos(f.pos.bytes()),
        ids(f.selections.bytes()), counts(f.sizes.bytes()), table(f.table.bytes());
    DeviceBuffer payloads[4];
    const std::uint8_t* host[4]{};
    std::size_t bytes[4]{}, leading[4]{};
    Tensor* targets[]{&f.cache.k_pages, &f.cache.v_pages, &f.cache.k_scale_pages,
                      &f.cache.v_scale_pages};
    if (f.encoded_key) {
        const EncodedPlane* planes[]{&f.encoded_key->data, &f.encoded_value->data,
                                     &f.encoded_key->scales, &f.encoded_value->scales};
        for (int i = 0; i < 4; ++i) {
            host[i]    = planes[i]->host.data();
            bytes[i]   = planes[i]->host.size();
            leading[i] = planes[i]->leading;
        }
    } else {
        host[0]    = reinterpret_cast<const std::uint8_t*>(f.keys.host.data());
        host[1]    = reinterpret_cast<const std::uint8_t*>(f.values.host.data());
        bytes[0]   = f.keys.device.bytes();
        bytes[1]   = f.values.device.bytes();
        leading[0] = f.keys.leading * 2;
        leading[1] = f.values.leading * 2;
    }
    for (int i = 0; i < 4; ++i) {
        if (!bytes[i]) continue;
        payloads[i]      = DeviceBuffer(bytes[i]);
        targets[i]->data = static_cast<std::uint8_t*>(payloads[i].p) + leading[i];
    }
    f.query.data                 = q.p;
    f.output_gate.data           = gate.p;
    f.cache.block_tables.data    = table.p;
    f.batch.positions.data       = pos.p;
    f.selected.block_ids.data    = ids.p;
    f.selected.block_counts.data = counts.p;
    const auto copy              = [&](void* dst, const void* src, std::size_t n) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, device.stream));
    };
    for (int b = 0; b < f.B; ++b)
        for (int t = 0; t < f.valid[b]; ++t) {
            const int m = b * f.T + t;
            copy(static_cast<int*>(pos.p) + m, f.positions.data() + m, 4);
            copy(static_cast<int*>(counts.p) + m, f.counts.data() + m, 4);
            if (f.counts[m])
                copy(static_cast<int*>(ids.p) + std::size_t(m) * 512,
                     f.ids.data() + std::size_t(m) * 512, f.counts[m] * 4);
            if (f.visible[m].empty()) continue;
            const auto off = std::size_t(m) * (256 * f.q_heads);
            copy(static_cast<std::uint16_t*>(q.p) + off, f.qhost.data() + off, 512 * f.q_heads);
            copy(static_cast<std::uint16_t*>(gate.p) + off, f.ghost.data() + off,
                 512 * f.q_heads);
            for (int token : f.visible[m]) {
                const int idx      = f.selectors[b] * f.table_stride + token / 64,
                          physical = f.tables[idx];
                copy(static_cast<int*>(table.p) + idx, f.tables.data() + idx, 4);
                for (int h = 0; h < f.kv_heads; ++h)
                    for (int i = 0; i < 4; ++i) {
                        if (!bytes[i]) continue;
                        const auto& plane = *targets[i];
                        const auto off    = leading[i] + physical * plane.nb[3] + h * plane.nb[2] +
                                         token % 64 * plane.nb[1];
                        copy(static_cast<std::uint8_t*>(payloads[i].p) + off, host[i] + off,
                             plane.ne[0] * dtype_size(plane.dtype));
                    }
            }
        }
    WorkspaceArena workspace(
        capacity(storage, query_heads, 3, 3, 4, f.envelope, device.execution_view()));
    f.launch(workspace, device.execution_view());
    device.synchronize();
    return verify(f,
                  "selected reads only storage=" + std::to_string(int(storage)) +
                      " padded=" + std::to_string(padded),
                  false);
}

int graph_update(DeviceContext& device, int tokens, int batch,
                 KvCacheStorage storage = KvCacheStorage::BFloat16, int query_heads = 24) {
    Fixture f(tokens, batch, 8192, true, storage, false, query_heads);
    const auto bytes = capacity(storage, query_heads, tokens, tokens, batch, {262144},
                                device.execution_view());
    WorkspaceArena workspace(std::max<std::size_t>(bytes, 256));
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int history : {64, 2051, 8192, 64, 0}) {
        f.history                     = std::max(1, history);
        f.envelope.max_visible_tokens = history;
        f.reset(history ? 0 : 2);
        DecodeGraphDefinition definition;
        definition.capture(device.stream, [&] { f.launch(workspace, device.execution_view()); });
        if (graph.ready())
            graph.update(definition);
        else
            graph.instantiate(definition);
        graph.launch(device.stream);
        device.synchronize();
        failures += verify(f, "graph update storage=" + std::to_string(int(storage)) +
                                  " T=" + std::to_string(tokens) + " B=" + std::to_string(batch) +
                                  " L=" + std::to_string(history));
        if (failures) return failures;
    }
    return failures;
}

template <class F>
int rejects(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "sparse accepted invalid contract input\n";
    return 1;
}

int validation(DeviceContext& device) {
    Fixture f(1, 1, 4, false);
    f.reset();
    WorkspaceArena workspace(4 * 1024 * 1024);
    Fixture padded(1, 1, 8192, true);
    // This aligned output intersects real rows in the padded plane's tail, beyond the
    // logical element-byte count. Checking Tensor::bytes() would miss the overlap.
    const auto logical_end =
        reinterpret_cast<std::uintptr_t>(padded.cache.k_pages.data) + padded.cache.k_pages.bytes();
    padded.output.data     = reinterpret_cast<void*>((logical_end + 15) / 16 * 16);
    const int padded_alias = rejects([&] { padded.launch(workspace, device.execution_view()); });
    int failures           = rejects([&] {
        (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(static_cast<KvCacheStorage>(-1), 24, 2, 1,
                                                           1, 1, f.envelope,
                                                           device.execution_view());
    });
    failures += padded_alias;
    // The low-bit cache profiles sit behind NINFER_QSA_SPARSE_GQA_QUANT_KV and stay rejected
    // while the switch is absent, so the default build cannot reach them by accident.
    if (std::getenv("NINFER_QSA_SPARSE_GQA_QUANT_KV") == nullptr)
        failures += rejects([&] {
            (void)capacity(KvCacheStorage::Fp8E4M3Row256, 24, 1, 1, 1, f.envelope,
                           device.execution_view());
        });
    // Every enumerator is accounted for: the five implemented profiles reach the switch's own
    // gate, and the five rk profiles the dense path owns are refused by name. A new enumerator
    // lands in whichever arm it belongs to instead of falling through a generic default.
    for (int raw = 0; raw <= static_cast<int>(KvCacheStorage::RK4V2E8); ++raw) {
        const auto storage = static_cast<KvCacheStorage>(raw);
        const bool implemented = raw <= static_cast<int>(KvCacheStorage::Fp8KeyNvfp4Value);
        if (!implemented) {
            bool named = false;
            try {
                (void)capacity(storage, 24, 2, 1, 1, f.envelope, device.execution_view());
            } catch (const std::exception& e) {
                const std::string message = e.what();
                named = message.find(ops::detail::qsa::qsa_storage_name(storage)) != std::string::npos;
            }
            failures += named ? 0 : 1;
        }
    }
    // Only the two local geometries are admitted; a mismatched pair is a contract error.
    failures += rejects([&] {
        (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(KvCacheStorage::BFloat16, 24, 1, 1, 1, 1,
                                                           f.envelope, device.execution_view());
    });
    failures += rejects([&] {
        (void)capacity(KvCacheStorage::BFloat16, 2, 1, 1, 1, f.envelope, device.execution_view());
    });
    const auto out = f.output;
    f.output.data  = f.query.data;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.output              = out;
    f.cache.v_pages.dtype = DType::BF16;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.cache.v_pages.dtype = DType::FP16;
    const auto k          = f.cache.k_pages;
    f.cache.k_pages.nb[1] = 511;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.cache.k_pages      = k;
    const auto execution = device.execution_view();
    std::size_t maximum  = 0;
    for (int t = 1; t <= 512; ++t)
        maximum = std::max(maximum, capacity(KvCacheStorage::BFloat16, 24, t, t, 1, f.envelope,
                                             execution));
    failures += maximum != capacity(KvCacheStorage::BFloat16, 24, 1, 512, 1, f.envelope, execution);
    return failures;
}

int run_bf16(DeviceContext& device) {
    int failures = 0;
    failures += gate_rounding(device);
    failures += graph_update(device, 8, 3);
    failures += graph_update(device, 33, 8);
    failures += read_footprint(device);
    for (int l : {1, 3, 4, 5, 63, 64, 65, 2047, 2048, 2049, 2051, 2052, 2053, 262144})
        failures += run(device, 1, 1, l);
    for (int cap : {0, 1, 7, 8, 9, 15, 16, 17, 511})
        failures += run(device, 3, 1, 32772, true, cap);
    for (int t : {2, 4, 8, 9, 16, 17, 32, 63, 64, 65, 128, 169, 170, 171})
        failures += run(device, t, 1, 8195, t % 2, 17);
    for (int b = 2; b <= 8; ++b) failures += run(device, 9, b, 32771, b % 2, 512, 1, true);
    for (int t : {255, 256, 257})
        for (int history : {1024, 8192}) failures += run(device, t, 1, history, false, 512);
    failures += run(device, 257, 1, 2051, true, 17);
    failures += run(device, 1024, 1, 2048, false, 512);
    failures += run(device, 1024, 1, 8192, false, 3);
    failures += run(device, 2048, 1, 8192, false, 1);
    failures += run(device, 4096, 1, 8192, false, 0);
    for (float magnitude : {0.1f, 1.8f})
        failures += run(device, 8, 1, 32772, false, 512, magnitude);
    // Local single-group geometry: twelve query heads share one KV head.
    failures += gate_rounding(device, KvCacheStorage::BFloat16, 1, 1, 12);
    failures += graph_update(device, 8, 3, KvCacheStorage::BFloat16, 12);
    failures += read_footprint(device, KvCacheStorage::BFloat16, false, 12);
    failures += read_footprint(device, KvCacheStorage::BFloat16, true, 12);
    constexpr int unbounded = std::numeric_limits<int>::max();
    for (int l : {1, 4, 64, 2048, 2051, 262144})
        failures += run(device, 1, 1, l, false, 512, 1, false, KvCacheStorage::BFloat16, unbounded, 12);
    for (int t : {9, 64, 169, 256})
        failures += run(device, t, 1, 8192, t % 2, 17, 1, false, KvCacheStorage::BFloat16, unbounded,
                        12);
    failures += run(device, 257, 1, 2051, true, 17, 1, false, KvCacheStorage::BFloat16, unbounded,
                    12);
    failures += run(device, 9, 8, 32771, true, 512, 1, true, KvCacheStorage::BFloat16, unbounded,
                    12);
    return failures;
}

int scale_layout(DeviceContext& device, KvCacheStorage storage, int tokens, int history) {
    // Scale planes have their own strides/alignment, independently of vector-aligned data.
    Fixture f(tokens, 1, history, false, storage, true);
    f.live_limit = 4;
    f.reset();
    WorkspaceArena arena(std::max<std::size_t>(
        256, capacity(storage, 24, tokens, tokens, 1, f.envelope, device.execution_view())));
    f.launch(arena, device.execution_view());
    device.synchronize();
    return verify(f, "independent scale layout storage=" + std::to_string(int(storage)) +
                         " T=" + std::to_string(tokens));
}

int quantized_validation(DeviceContext& device, KvCacheStorage storage) {
    Fixture invalid(1, 1, 4, false, storage);
    WorkspaceArena workspace(4 * 1024 * 1024);
    const auto key_scales       = invalid.cache.k_scale_pages;
    invalid.cache.k_scale_pages = {};
    int failures = rejects([&] { invalid.launch(workspace, device.execution_view()); });
    invalid.cache.k_scale_pages       = key_scales;
    invalid.cache.k_scale_pages.dtype = DType::BF16;
    failures += rejects([&] { invalid.launch(workspace, device.execution_view()); });
    invalid.cache.k_scale_pages       = key_scales;
    invalid.cache.k_scale_pages.nb[1] = 0;
    failures += rejects([&] { invalid.launch(workspace, device.execution_view()); });
    invalid.cache.k_scale_pages = key_scales;
    invalid.output.data         = invalid.cache.v_scale_pages.data;
    failures += rejects([&] { invalid.launch(workspace, device.execution_view()); });
    return failures;
}

int run_quantized(DeviceContext& device, KvCacheStorage storage) {
    int failures = quantized_validation(device, storage);
    failures += read_footprint(device, storage, false) + read_footprint(device, storage, true);
    failures += scale_layout(device, storage, 1, 2051);
    failures += scale_layout(device, storage, 192, 8195);
    failures += scale_layout(device, storage, 256, 65);
    failures += gate_rounding(device, storage);
    failures += gate_rounding(device, storage, 256, 1);
    failures += gate_rounding(device, storage, 256, 8192);
    failures += graph_update(device, 8, 3, storage);
    failures += graph_update(device, 33, 8, storage);
    for (int l : {1, 3, 4, 64, 65, 2048, 2051, 2052, 262144})
        failures += run(device, 1, 1, l, false, 512, 1, false, storage);
    for (int t : {63, 64, 65, 169, 170, 171, 255, 256, 257, 339, 340, 341})
        failures += run(device, t, 1, 8192, false, 17, 1, false, storage);
    failures += run(device, 257, 1, 2051, true, 17, 1, false, storage);
    failures += run(device, 9, 8, 32771, true, 512, 1, true, storage);
    failures += run(device, 1024, 1, 2048, false, 512, 1, false, storage);
    failures += run(device, 192, 1, 8192, false, 512, 1, false, storage);
    failures += run(device, 1024, 1, 8192, false, 512, 1, false, storage, 8);
    for (float magnitude : {0.1f, 1.8f})
        failures += run(device, 8, 1, 32772, false, 512, magnitude, false, storage);
    failures += run(device, 3, 2, 2052, true, 0, 1, true, storage);
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) return 77;
    try {
        // The quantized-KV routes are opt-in at the operator's switch layer, so the default
        // run exercises the BF16 profile only. Naming a low-bit format below enables them.
        std::vector<KvCacheStorage> formats{KvCacheStorage::BFloat16};
        if (argc != 1) {
            if (argc != 3 || std::string_view(argv[1]) != "--kv-cache")
                throw std::invalid_argument(
                    "usage: ninfer_qsa_sparse_gqa_test [--kv-cache bf16|fp8|int8|nvfp4|k8v4]");
            const std::string_view name = argv[2];
            if (name == "bf16")
                formats = {KvCacheStorage::BFloat16};
            else if (name == "fp8")
                formats = {KvCacheStorage::Fp8E4M3Row256};
            else if (name == "int8")
                formats = {KvCacheStorage::Int8Group64};
            else if (name == "nvfp4")
                formats = {KvCacheStorage::Nvfp4Group16};
            else if (name == "k8v4")
                formats = {KvCacheStorage::Fp8KeyNvfp4Value};
            else
                throw std::invalid_argument("unsupported test cache format");
#if defined(_WIN32)
            _putenv_s("NINFER_QSA_SPARSE_GQA_QUANT_KV", "1");
#else
            ::setenv("NINFER_QSA_SPARSE_GQA_QUANT_KV", "1", 1);
#endif
        }
        DeviceContext device;
        int failures = validation(device);
        for (auto format : formats)
            failures += format == KvCacheStorage::BFloat16 ? run_bf16(device)
                                                           : run_quantized(device, format);
        std::cout << "QSA Sparse GQA: " << (failures ? "FAIL" : "PASS") << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
