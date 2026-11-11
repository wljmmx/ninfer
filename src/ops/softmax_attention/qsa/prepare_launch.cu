#include "ops/softmax_attention/qsa/launch.h"
#include "ops/softmax_attention/qsa/prepare_kernel.cuh"
#include "ops/softmax_attention/qsa/storage.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::detail::qsa {
namespace {

template <class T>
PrepareColumns<T> prepare_columns(const Tensor& t) {
    return {static_cast<T*>(t.data), t.nb[1] / std::int64_t(sizeof(T)),
            t.nb[2] / std::int64_t(sizeof(T))};
}

template <class T>
PreparePage<T> prepare_page(const Tensor& t) {
    return {static_cast<T*>(t.data), t.nb[1] / std::int64_t(sizeof(T)),
            t.nb[2] / std::int64_t(sizeof(T)), t.nb[3] / std::int64_t(sizeof(T))};
}

PrepareArguments prepare_arguments(const QsaProjectionView& p, const QsaPrepareParameters& params,
                                   const QsaBatchView& batch, const Tensor& rope,
                                   const QsaTailStateView& before, QsaCacheView cache,
                                   QsaRawRecordWriteView records, Tensor& q, Tensor& iq) {
    const auto bf  = [](const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); };
    const auto i32 = [](const Tensor& t) { return static_cast<const std::int32_t*>(t.data); };
    if (const char* shape = std::getenv("NINFER_QSA_PROJ_SHAPE"); shape != nullptr) {
        const auto show = [](const char* name, const Tensor& t) {
            std::fprintf(stderr, "[proj] %s ne=%d,%d,%d,%d nb=%lld,%lld,%lld,%lld\n", name,
                         int(t.ne[0]), int(t.ne[1]), int(t.ne[2]), int(t.ne[3]),
                         (long long)t.nb[0], (long long)t.nb[1], (long long)t.nb[2],
                         (long long)t.nb[3]);
        };
        show("query", p.query);
        show("key", p.key);
        show("value", p.value);
    }
    PrepareArguments a{bf(p.query),
                       bf(p.key),
                       bf(p.value),
                       bf(p.index_query),
                       bf(p.index_key),
                       bf(params.query_norm),
                       bf(params.key_norm),
                       bf(params.index_query_norm),
                       bf(params.index_key_norm),
                       static_cast<__nv_bfloat16*>(q.data),
                       static_cast<__nv_bfloat16*>(iq.data),
                       i32(batch.positions),
                       i32(rope),
                       i32(batch.valid_columns),
                       i32(batch.table_rows),
                       i32(before.slots),
                       i32(records.offsets),
                       i32(cache.main.block_tables),
                       cache.main.block_tables.nb[1] / 4,
                       prepare_columns<const __nv_bfloat16>(before.raw_keys),
                       prepare_columns<const std::int32_t>(before.first_rope),
                       prepare_columns<__nv_bfloat16>(records.raw_keys),
                       prepare_columns<std::int32_t>(records.rope_positions),
                       prepare_page<__nv_bfloat16>(cache.index.keys),
                       prepare_page<std::uint8_t>(cache.main.k_pages),
                       prepare_page<std::uint8_t>(cache.main.v_pages),
                       prepare_page<std::uint8_t>(cache.main.k_scale_pages),
                       prepare_page<std::uint8_t>(cache.main.v_scale_pages),
                       p.query.ne[2],
                       p.query.ne[3],
                       rope.ne[1],
                       p.key.ne[1],
                       params.rms_epsilon,
                       {}};
    // Default model frequencies are constructed once on Host, with no Device allocation.
    static const auto default_frequency = [] {
        std::array<double, 32> result{};
        for (int i = 0; i < 32; ++i) result[i] = std::pow(1e7, -double(i) / 32);
        return result;
    }();
    for (int i = 0; i < 32; ++i)
        a.frequency[i] = params.rope_theta == 1e7F
                             ? default_frequency[i]
                             : std::pow(double(params.rope_theta), -double(i) / 32);
    // The filters apply to both exports: the device printf form (NINFER_QSA_E8_DUMP) and the
    // record-buffer form (NINFER_QSA_E8_TRACE), and the latter enables the trace from the launcher.
    if (const char* row = std::getenv("NINFER_QSA_E8_ROW"); row != nullptr) {
        a.e8_dump_row = std::atoi(row);
    }
    if (const char* plane = std::getenv("NINFER_QSA_E8_PLANE"); plane != nullptr) {
        a.e8_dump_plane = std::atoi(plane);
    }
    if (const char* head = std::getenv("NINFER_QSA_E8_HEAD"); head != nullptr) {
        a.e8_dump_head = std::atoi(head);
    }
    if (const char* dump = std::getenv("NINFER_QSA_E8_DUMP"); dump != nullptr && *dump != '\0') {
        a.e8_dump = 1;
    }
    return a;
}

template <KvCacheStorage Storage, bool BoundedPhase, bool VectorWeights, int KvHeads>
void submit(const PrepareArguments& args, cudaStream_t stream) {
    constexpr int groups = PrepareGeometry<KvHeads>::groups;
    // Short calls distribute a token's heads across CTAs; larger calls group heads across
    // tokens for memory throughput. The short-call bound also keeps tokens*groups in x.
    if (std::int64_t(args.tokens) * args.batch <= 320)
        prepare_kernel<Storage, BoundedPhase, VectorWeights, true, KvHeads>
            <<<dim3(args.tokens * groups, args.batch), 128, 0, stream>>>(args);
    else
        prepare_kernel<Storage, BoundedPhase, VectorWeights, false, KvHeads>
            <<<dim3(args.tokens, args.batch, groups), 128, 0, stream>>>(args);
}

template <KvCacheStorage Storage, int KvHeads>
void dispatch_geometry(const PrepareArguments& args, float theta, cudaStream_t stream) {
    const bool vector_weights = reinterpret_cast<std::uintptr_t>(args.wq) % 16 == 0 &&
                                reinterpret_cast<std::uintptr_t>(args.wiq) % 8 == 0;
    // For theta>=1 every phase is bounded by the nonnegative-I32 coordinate domain.
    if (theta >= 1) {
        if (vector_weights)
            submit<Storage, true, true, KvHeads>(args, stream);
        else
            submit<Storage, true, false, KvHeads>(args, stream);
    } else {
        if (vector_weights)
            submit<Storage, false, true, KvHeads>(args, stream);
        else
            submit<Storage, false, false, KvHeads>(args, stream);
    }
}

template <KvCacheStorage Storage>
void dispatch(const PrepareArguments& args, float theta, cudaStream_t stream) {
    if (args.kv_heads == 1)
        dispatch_geometry<Storage, 1>(args, theta, stream);
    else
        dispatch_geometry<Storage, 2>(args, theta, stream);
}

} // namespace

// NINFER_QSA_E8_TRACE=<path>: a bounded, default-off export of the E8 subspace decisions. A
// kernel-wide printf drops lines at the fixture's volume, so the records go through a device buffer
// that the launch copies back once and appends to the file.
namespace {

constexpr int kE8TraceCapacity = 64;
__device__ E8TraceRecord g_e8_trace[kE8TraceCapacity];
__device__ int g_e8_trace_count;

// NINFER_QSA_V_SRC_TRACE: four header floats (t, head, m, p) then the 256 values prepare_kv read.
constexpr int kVTraceRecords = 8, kVTraceStride = 260;
__device__ float g_v_trace[kVTraceRecords * kVTraceStride];
__device__ int g_v_trace_count;

// NINFER_QSA_PREP_ENTRY: an unfiltered counter for the prepare_kv entry log. Not reset per launch,
// so the cap bounds the whole run rather than one pass.
__device__ int g_prep_entry_count;

struct E8TraceSink {
    E8TraceRecord* out = nullptr;
    int* count         = nullptr;
    const char* path   = nullptr;
};
E8TraceSink e8_trace_sink(cudaStream_t stream) {
    E8TraceSink sink{};
    const char* path = std::getenv("NINFER_QSA_E8_TRACE");
    if (path == nullptr || *path == '\0') { return sink; }
    void* out   = nullptr;
    void* count = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress(&out, g_e8_trace));
    CUDA_CHECK(cudaGetSymbolAddress(&count, g_e8_trace_count));
    CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(int), stream));
    sink.out   = static_cast<E8TraceRecord*>(out);
    sink.count = static_cast<int*>(count);
    sink.path  = path;
    return sink;
}

void e8_trace_flush(const E8TraceSink& sink, cudaStream_t stream) {
    if (sink.out == nullptr) { return; }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, sink.count, sizeof(int), cudaMemcpyDeviceToHost));
    if (count > kE8TraceCapacity) { count = kE8TraceCapacity; }
    if (count <= 0) { return; }
    std::vector<E8TraceRecord> records(static_cast<std::size_t>(count));
    CUDA_CHECK(cudaMemcpy(records.data(), sink.out,
                          static_cast<std::size_t>(count) * sizeof(E8TraceRecord),
                          cudaMemcpyDeviceToHost));
    if (std::FILE* file = std::fopen(sink.path, "a")) {
        for (const E8TraceRecord& r : records) {
            std::fprintf(file, "%d %d %d %d %d %d %d %d %u %u %.9g %.9g %.9g %.9g %.9g %.9g",
                         r.plane, r.row, r.head, r.group, r.subspace, r.type_flag, r.best_dim,
                         r.axis, r.root, r.radial, double(r.absmax), double(r.step), double(r.rel),
                         double(r.a_score), double(r.b_score), double(r.dot));
            for (int i = 0; i < 8; ++i) std::fprintf(file, " %.9g", double(r.unit[i]));
            for (int i = 0; i < 8; ++i) std::fprintf(file, " %.9g", double(r.projection[i]));
            for (int i = 0; i < 8; ++i) std::fprintf(file, " %.9g", double(r.res[i]));
            for (int i = 0; i < 8; ++i) std::fprintf(file, " %.9g", double(r.v1[i]));
            std::fprintf(file, "\n");
        }
        std::fclose(file);
    }
}

// NINFER_QSA_V_SRC_TRACE: the input side of one (t, head). Where the E8 trace answers what the
// encoder decided, this answers what prepare_kv was handed - the row index it derived, the position
// it wrote to, and the 256 projection values - which is the only way to tell a wrong source column
// apart from a wrong store.
struct VTraceSink {
    float* out       = nullptr;
    int* count       = nullptr;
    const char* path = nullptr;
};

VTraceSink v_trace_sink(cudaStream_t stream) {
    VTraceSink sink{};
    const char* path = std::getenv("NINFER_QSA_V_SRC_TRACE");
    if (path == nullptr || *path == '\0') { return sink; }
    void* out   = nullptr;
    void* count = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress(&out, g_v_trace));
    CUDA_CHECK(cudaGetSymbolAddress(&count, g_v_trace_count));
    CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(int), stream));
    sink.out   = static_cast<float*>(out);
    sink.count = static_cast<int*>(count);
    sink.path  = path;
    return sink;
}

void v_trace_flush(const VTraceSink& sink, cudaStream_t stream) {
    if (sink.out == nullptr) { return; }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, sink.count, sizeof(int), cudaMemcpyDeviceToHost));
    if (count > kVTraceRecords) { count = kVTraceRecords; }
    if (count <= 0) { return; }
    std::vector<float> records(std::size_t(count) * kVTraceStride);
    CUDA_CHECK(cudaMemcpy(records.data(), sink.out, records.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (std::FILE* file = std::fopen(sink.path, "a")) {
        for (int i = 0; i < count; ++i) {
            const float* record = records.data() + std::size_t(i) * kVTraceStride;
            std::fprintf(file, "t=%d head=%d m=%d p=%d", int(record[0]), int(record[1]),
                         int(record[2]), int(record[3]));
            for (int j = 0; j < 256; ++j) std::fprintf(file, " %.9g", double(record[4 + j]));
            std::fprintf(file, "\n");
        }
        std::fclose(file);
    }
}

} // namespace

void launch_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const QsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                    DeviceExecutionView execution) {
    auto args = prepare_arguments(projected, parameters, batch, rope_positions, before, cache,
                                  records, query, index_query);
    const E8TraceSink sink = e8_trace_sink(execution.stream);
    if (sink.out != nullptr) {
        args.trace_out      = sink.out;
        args.trace_count    = sink.count;
        args.trace_capacity = kE8TraceCapacity;
        args.e8_dump        = 1;
    }
    const VTraceSink v_sink = v_trace_sink(execution.stream);
    if (v_sink.out != nullptr) {
        args.v_trace       = v_sink.out;
        args.v_trace_count = v_sink.count;
        args.v_trace_capacity = kVTraceRecords;
        if (const char* t = std::getenv("NINFER_QSA_TRACE_T"); t != nullptr) {
            args.v_trace_t = std::atoi(t);
        }
        if (const char* h = std::getenv("NINFER_QSA_TRACE_HEAD"); h != nullptr) {
            args.v_trace_head = std::atoi(h);
        }
    }
    if (const char* entry = std::getenv("NINFER_QSA_PREP_ENTRY"); entry != nullptr && *entry != '0') {
        void* counter = nullptr;
        CUDA_CHECK(cudaGetSymbolAddress(&counter, g_prep_entry_count));
        args.prep_entry_count = static_cast<int*>(counter);
        args.prep_entry_n     = 8;
        if (const char* n = std::getenv("NINFER_QSA_PREP_ENTRY_N"); n != nullptr) {
            args.prep_entry_n = std::atoi(n);
        }
    }
    switch (cache.main.storage) {
    case KvCacheStorage::BFloat16:
        dispatch<KvCacheStorage::BFloat16>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::Int8Group64:
        dispatch<KvCacheStorage::Int8Group64>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        dispatch<KvCacheStorage::Fp8E4M3Row256>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::Nvfp4Group16:
        dispatch<KvCacheStorage::Nvfp4Group16>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        dispatch<KvCacheStorage::Fp8KeyNvfp4Value>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        dispatch<KvCacheStorage::RotatedInt8KeyInt4ValueGroup64>(args, parameters.rope_theta,
                                                                 execution.stream);
        break;
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        dispatch<KvCacheStorage::RotatedInt4KeyInt4ValueGroup64>(args, parameters.rope_theta,
                                                                 execution.stream);
        break;
    case KvCacheStorage::RK4V4E8:
        dispatch<KvCacheStorage::RK4V4E8>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::RK2V4E8:
        dispatch<KvCacheStorage::RK2V4E8>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::RK4V2E8:
        dispatch<KvCacheStorage::RK4V2E8>(args, parameters.rope_theta, execution.stream);
        break;
    default:
        // Unreachable through the public entry, which refuses an unimplemented storage by name
        // before it gets here; kept so a new enumerator cannot be dispatched silently.
        throw_unsupported_storage("qsa_prepare", cache.main.storage);
    }
    CUDA_CHECK(cudaGetLastError());
    e8_trace_flush(sink, execution.stream);
    v_trace_flush(v_sink, execution.stream);
}

} // namespace ninfer::ops::detail::qsa
