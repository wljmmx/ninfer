// Measures the public paired RMSNorm+concat and residual-add RMSNorm Ops.
#include "ninfer/ops/paired_rmsnorm_concat.h"
#include "ninfer/ops/residual_add_rmsnorm.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

struct Options {
    std::string op = "paired";
    int dim        = 4096;
    int calls      = 1;
    int warmup     = 10;
    int repeat     = 200;
    std::string execution = "graph";
    std::string cache     = "cold";
    std::vector<int> columns{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
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
    if (result.empty()) throw std::invalid_argument("empty column list");
    return result;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::puts("usage: ninfer_fused_rmsnorm_bench [--op paired|residual] [--dim D] "
                      "[--columns M,M,...] [--calls N] [--warmup N] [--repeat N] "
                      "[--execution eager|graph] [--cache cold|warm]");
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--op")
            o.op = value;
        else if (arg == "--dim")
            o.dim = std::stoi(value);
        else if (arg == "--columns")
            o.columns = list(value);
        else if (arg == "--calls")
            o.calls = std::stoi(value);
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
    if (o.op != "paired" && o.op != "residual") throw std::invalid_argument("unknown Op");
    if (o.calls > 1 && o.cache != "warm")
        throw std::invalid_argument("--calls > 1 requires --cache warm");
    return o;
}

struct Case {
    int d = 0, m = 0;
    bool paired = true;
    DeviceBuffer x, y, gx, gy, mask, out, residual;
    Tensor tx, ty, wx, wy, enabled, result, r;

    Case(bool paired_op, int dim, int columns)
        : d(dim), m(columns), paired(paired_op),
          x(bench::make_bf16(std::size_t(dim) * columns, 3901, -2.F, 2.F)),
          y(bench::make_bf16(std::size_t(dim) * columns, 3903, -1.F, 1.F)),
          gx(bench::make_bf16(std::size_t(dim), 3907, -1.F, 1.F)),
          gy(bench::make_bf16(std::size_t(dim), 3909, -1.F, 1.F)),
          mask(bench::make_zeros(std::size_t(columns) * sizeof(int))),
          out(bench::make_zeros(std::size_t(dim) * columns * (paired_op ? 4 : 2))),
          residual(bench::make_zeros(std::size_t(dim) * columns * 2)), tx(x.p, DType::BF16, {d, m}),
          ty(y.p, DType::BF16, {d, m}), wx(gx.p, DType::BF16, {d}), wy(gy.p, DType::BF16, {d}),
          enabled(mask.p, DType::I32, {m}),
          result(out.p, DType::BF16, {paired_op ? 2 * d : d, m}),
          r(residual.p, DType::BF16, {d, m}) {
        std::vector<int> host(m, 1);
        CUDA_CHECK(cudaMemcpy(mask.p, host.data(), host.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    void launch(DeviceExecutionView execution) {
        if (paired)
            ops::paired_rmsnorm_concat(tx, ty, wx, wy, enabled, 1.e-5F, result, execution);
        else
            ops::residual_add_rmsnorm(tx, ty, wx, 1.e-5F, r, result, execution);
    }
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
        std::printf("GPU=%s SMs=%d CUDART=%d op=%s\n", device.props.name,
                    device.multiprocessor_count(), CUDART_VERSION, o.op.c_str());
        std::printf("dim,columns,execution,cache,median_us,min_us,p95_us,logical_GB/s\n");
        for (const int m : o.columns) {
            Case data(o.op == "paired", o.dim, m);
            for (int call = 0; call < o.calls; ++call) data.launch(device.execution_view());
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            bench::TimedGraph graph;
            if (o.execution == "graph")
                graph.capture(device.stream, [&](cudaStream_t stream) {
                    for (int call = 0; call < o.calls; ++call)
                        data.launch(device.execution_view().on_stream(stream));
                });
            const auto launch = [&](cudaStream_t stream) {
                if (o.execution == "graph") {
                    graph.launch(stream);
                    return;
                }
                for (int call = 0; call < o.calls; ++call)
                    data.launch(device.execution_view().on_stream(stream));
            };
            if (o.cache == "cold") bench::flush_l2(flush, device.stream);
            const auto time = bench::measure_launch(launch, device.stream, o.warmup, o.repeat);
            const double bytes = double(std::size_t(o.dim) * m * 8 + std::size_t(o.dim) * 2 +
                                        (o.op == "paired" ? m * 4 : 0));
            const double per_call_us = time.median_us / o.calls;
            std::printf("%d,%d,%s,%s,%.3f,%.3f,%.3f,%.1f\n", o.dim, m, o.execution.c_str(),
                        o.cache.c_str(), per_call_us, time.min_us / o.calls,
                        time.p95_us / o.calls, bytes / (per_call_us * 1000.0));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
