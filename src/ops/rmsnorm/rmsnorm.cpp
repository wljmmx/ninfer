#include "ninfer/ops/rmsnorm.h"
#include "ops/rmsnorm/launch.h"

namespace ninfer::ops {
namespace norm = detail::rmsnorm;

namespace {
void execute(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps,
             norm::Epilogue kind, Tensor& out, DeviceExecutionView execution, const char* op) {
    const auto operands =
        norm::validate(x, weight, gate, eps, out, kind, execution.multiprocessor_count, op);
    if (operands.rows)
        norm::launch(x, weight, gate, eps, kind, out, operands, execution.multiprocessor_count,
                     execution.stream);
}
} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream) {
    execute(x, weight, nullptr, eps, unit_offset ? norm::Epilogue::Offset : norm::Epilogue::Plain,
            out, {stream, 0}, "rmsnorm");
}

void silu_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate, float eps,
                        Tensor& out, DeviceExecutionView execution) {
    execute(x, weight, &gate, eps, norm::Epilogue::SiluGated, out, execution, "silu_gated_rmsnorm");
}

void sigmoid_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate, float eps,
                           Tensor& out, DeviceExecutionView execution) {
    execute(x, weight, &gate, eps, norm::Epilogue::SigmoidGated, out, execution,
            "sigmoid_gated_rmsnorm");
}
} // namespace ninfer::ops
