#include "fixture.cuh"
using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        const auto o = rmsnorm::parse(argc, argv);
        measurement::hardware();
        DeviceContext device;
        L2FlushBuffer flush(o.timing.flush_bytes);
        measurement::Csv csv(o.timing.csv_out);
        for (int columns : o.tokens) {
            rmsnorm::Fixture f(o, columns);
            for (const auto& execution :
                 measurement::selections(o.timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](auto stream) { f.launch({stream, device.multiprocessor_count()}); },
                        [](auto) {}, flush, device.stream, o.timing, execution, cache);
                    measurement::Row row;
                    row.op          = rmsnorm::name(o.op);
                    row.format      = "bf16";
                    row.n           = o.dimension;
                    row.value_heads = o.heads;
                    row.tokens      = columns;
                    row.batch       = o.batch;
                    csv.write(row, result, execution, cache);
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
