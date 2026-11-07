// Measures the five public HC residual Ops and the two low-rank mix forms.
//
// The upstream bench of this family is written against bench/ops/op_measurement.cuh, which the
// local tree does not carry; this file reproduces the same Op coverage on ninfer_bench_common.h.
#include "ninfer/ops/gated_residual.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kWidth = 2560, kStreams = 4, kRank = 320;
constexpr float kEpsilon = 1.e-6F;

enum class Op {
    ExpandNorm,
    GroupedNorm,
    Combine,
    CombineNorm,
    AddNorm,
    BlockMix,
    FinalMix
};

const char* name(Op op) {
    switch (op) {
    case Op::ExpandNorm:
        return "expand-norm";
    case Op::GroupedNorm:
        return "grouped-rmsnorm";
    case Op::Combine:
        return "combine";
    case Op::CombineNorm:
        return "combine-norm";
    case Op::AddNorm:
        return "add-norm";
    case Op::BlockMix:
        return "block-mix";
    case Op::FinalMix:
        return "final-mix";
    }
    return "?";
}

struct Options {
    Op op = Op::CombineNorm;
    std::vector<int> tokens{1};
    int batch    = 1;
    bool in_place = false;
    int warmup   = 10;
    int repeat   = 100;
    std::string execution = "graph";
    std::string cache     = "cold";
};

std::vector<int> list(const std::string& text) {
    std::vector<int> result;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(',', start);
        result.push_back(std::stoi(text.substr(start, end - start)));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (result.empty()) throw std::invalid_argument("empty token list");
    return result;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::puts("usage: ninfer_hc_gated_residual_bench "
                      "[--op expand-norm|grouped-rmsnorm|combine|combine-norm|add-norm|"
                      "block-mix|final-mix] [--tokens T[,T...]] [--batch B] "
                      "[--alias disjoint|inplace] [--warmup N] [--repeat N] "
                      "[--execution eager|graph] [--cache cold|warm]");
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--op") {
            if (value == "expand-norm")
                o.op = Op::ExpandNorm;
            else if (value == "grouped-rmsnorm")
                o.op = Op::GroupedNorm;
            else if (value == "combine")
                o.op = Op::Combine;
            else if (value == "combine-norm")
                o.op = Op::CombineNorm;
            else if (value == "add-norm")
                o.op = Op::AddNorm;
            else if (value == "block-mix")
                o.op = Op::BlockMix;
            else if (value == "final-mix")
                o.op = Op::FinalMix;
            else
                throw std::invalid_argument("unknown Op: " + value);
        } else if (arg == "--tokens")
            o.tokens = list(value);
        else if (arg == "--batch")
            o.batch = std::stoi(value);
        else if (arg == "--alias")
            o.in_place = value == "inplace";
        else if (arg == "--warmup")
            o.warmup = std::stoi(value);
        else if (arg == "--repeat")
            o.repeat = std::stoi(value);
        else if (arg == "--execution")
            o.execution = value;
        else if (arg == "--cache")
            o.cache = value;
        else
            throw std::invalid_argument("unknown option: " + arg);
    }
    return o;
}

Weight bf16_weight(const DeviceBuffer& buffer, int n, int k) {
    Weight w;
    w.payload = w.qdata = buffer.p;
    w.payload_bytes     = buffer.bytes;
    w.qtype             = QType::BF16;
    w.layout            = QuantLayout::Contiguous;
    w.ndim              = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    return w;
}

struct Fixture {
    Fixture(Op op, int m, bool in_place)
        : op(op), in_place(in_place),
          input(bench::make_bf16(std::size_t(kWidth) * m * (op == Op::ExpandNorm ? 1 : kStreams),
                                 6101, -1.F, 1.F)),
          block(combines(op) ? bench::make_bf16(std::size_t(kWidth) * m, 6103, -0.75F, 0.75F)
                             : DeviceBuffer{}),
          weight(op != Op::Combine ? bench::make_bf16(std::size_t(kWidth) * kStreams, 6107,
                                                      -0.25F, 0.25F)
                                   : DeviceBuffer{}),
          injection(combines(op) && op != Op::AddNorm
                        ? bench::make_f32(std::size_t(kStreams) * m, 6109,
                                          op == Op::AddNorm ? 1.F : 0.5F,
                                          op == Op::AddNorm ? 1.F : 2.F)
                        : DeviceBuffer{}),
          backup(in_place ? bench::make_zeros(input.bytes) : DeviceBuffer{}),
          residual(op != Op::GroupedNorm && !in_place
                       ? bench::make_zeros(std::size_t(kWidth) * kStreams * m * 2)
                       : DeviceBuffer{}),
          normalized(op != Op::Combine ? bench::make_zeros(std::size_t(kWidth) * kStreams * m * 2)
                                       : DeviceBuffer{}),
          h(input.p, DType::BF16, {kWidth, kStreams, m}),
          b(block.p, DType::BF16, {kWidth, m}),
          w(weight.p, DType::BF16, {kWidth, kStreams}),
          j(injection.p, DType::FP32, {kStreams, m}),
          r(in_place ? input.p : residual.p, DType::BF16, {kWidth, kStreams, m}),
          z(normalized.p, DType::BF16, {kWidth, kStreams, m}) {
        if (op == Op::ExpandNorm) h = Tensor(input.p, DType::BF16, {kWidth, m});
        if (in_place)
            CUDA_CHECK(cudaMemcpy(backup.p, input.p, input.bytes, cudaMemcpyDeviceToDevice));
    }

    static bool combines(Op op) {
        return op == Op::Combine || op == Op::CombineNorm || op == Op::AddNorm;
    }

    void restore(cudaStream_t stream) {
        if (in_place)
            CUDA_CHECK(
                cudaMemcpyAsync(input.p, backup.p, input.bytes, cudaMemcpyDeviceToDevice, stream));
    }

    void launch(DeviceExecutionView e) {
        switch (op) {
        case Op::ExpandNorm:
            ops::hc_expand_norm(h, w, kEpsilon, r, z, e);
            break;
        case Op::GroupedNorm:
            ops::hc_grouped_rmsnorm(h, w, kEpsilon, z, e);
            break;
        case Op::Combine:
            ops::hc_combine(h, b, j, r, e);
            break;
        case Op::CombineNorm:
            ops::hc_combine_norm(h, b, j, w, kEpsilon, r, z, e);
            break;
        case Op::AddNorm:
            ops::hc_combine_norm(h, b, w, kEpsilon, r, z, e);
            break;
        default:
            throw std::logic_error("mix forms use MixFixture");
        }
    }

    Op op;
    bool in_place;
    DeviceBuffer input, block, weight, injection, backup, residual, normalized;
    Tensor h, b, w, j, r, z;
};

struct MixFixture {
    MixFixture(Op op, int m, DeviceExecutionView execution)
        : op(op),
          input(bench::make_bf16(std::size_t(kWidth) * kStreams * m, 6201, -1.F, 1.F)),
          down(bench::make_bf16(
              std::size_t(kRank + (op == Op::BlockMix ? kStreams : 0)) * kWidth * kStreams, 6203,
              -std::sqrt(3.F / (kWidth * kStreams)), std::sqrt(3.F / (kWidth * kStreams)))),
          up(bench::make_bf16(std::size_t(kWidth) * kStreams * kRank, 6207,
                              -4.F * std::sqrt(3.F / kRank), 4.F * std::sqrt(3.F / kRank))),
          output(bench::make_zeros(std::size_t(kWidth) * m * 2)),
          injection(op == Op::BlockMix ? bench::make_zeros(std::size_t(kStreams) * m * 4)
                                       : DeviceBuffer{}),
          wd(bf16_weight(down, kRank + (op == Op::BlockMix ? kStreams : 0), kWidth * kStreams)),
          wu(bf16_weight(up, kWidth * kStreams, kRank)),
          z(input.p, DType::BF16, {kWidth, kStreams, m}), x(output.p, DType::BF16, {kWidth, m}),
          j(injection.p, DType::FP32, {kStreams, m}),
          capacity(op == Op::BlockMix
                       ? ops::hc_mix_workspace_capacity_bytes(ops::HcMixWeights{wd, wu}, m, m,
                                                              execution)
                       : ops::hc_mix_workspace_capacity_bytes(ops::HcFinalMixWeights{wd, wu}, m, m,
                                                              execution)),
          arena(capacity) {}

    void launch(DeviceExecutionView e) {
        if (op == Op::BlockMix)
            ops::hc_mix(z, ops::HcMixWeights{wd, wu}, arena, x, j, e);
        else
            ops::hc_mix(z, ops::HcFinalMixWeights{wd, wu}, arena, x, e);
    }

    Op op;
    DeviceBuffer input, down, up, output, injection;
    Weight wd, wu;
    Tensor z, x, j;
    std::size_t capacity;
    WorkspaceArena arena;
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        int devices     = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) return 77;
        DeviceContext device;
        bench::L2FlushBuffer flush(std::size_t{96} << 20);
        std::printf("GPU=%s SMs=%d CUDART=%d op=%s alias=%s\n", device.props.name,
                    device.multiprocessor_count(), CUDART_VERSION, name(o.op),
                    o.in_place ? "inplace" : "disjoint");
        std::printf("op,tokens,batch,M,execution,cache,median_us,min_us,p95_us,workspace_bytes\n");
        const bool mix = o.op == Op::BlockMix || o.op == Op::FinalMix;
        for (const int tokens : o.tokens) {
            const int m = tokens * o.batch;
            if (mix) {
                MixFixture f(o.op, m, device.execution_view());
                bench::TimedGraph graph;
                if (o.execution == "graph")
                    graph.capture(device.stream, [&](cudaStream_t stream) {
                        f.launch(device.execution_view().on_stream(stream));
                    });
                const auto launch = [&](cudaStream_t stream) {
                    if (o.execution == "graph")
                        graph.launch(stream);
                    else
                        f.launch(device.execution_view().on_stream(stream));
                };
                if (o.cache == "cold") bench::flush_l2(flush, device.stream);
                const auto time = bench::measure_launch(launch, device.stream, o.warmup, o.repeat);
                std::printf("%s,%d,%d,%d,%s,%s,%.3f,%.3f,%.3f,%zu\n", name(o.op), tokens, o.batch,
                            m, o.execution.c_str(), o.cache.c_str(), time.median_us, time.min_us,
                            time.p95_us, f.capacity);
            } else {
                Fixture f(o.op, m, o.in_place);
                bench::TimedGraph graph;
                if (o.execution == "graph")
                    graph.capture(device.stream, [&](cudaStream_t stream) {
                        f.launch(device.execution_view().on_stream(stream));
                    });
                const auto launch = [&](cudaStream_t stream) {
                    f.restore(stream);
                    if (o.execution == "graph")
                        graph.launch(stream);
                    else
                        f.launch(device.execution_view().on_stream(stream));
                };
                if (o.cache == "cold" && !o.in_place) bench::flush_l2(flush, device.stream);
                const auto time = bench::measure_launch(launch, device.stream, o.warmup, o.repeat);
                std::printf("%s,%d,%d,%d,%s,%s,%.3f,%.3f,%.3f,0\n", name(o.op), tokens, o.batch, m,
                            o.execution.c_str(), o.cache.c_str(), time.median_us, time.min_us,
                            time.p95_us);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
