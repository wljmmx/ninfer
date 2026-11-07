#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/launch.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* label) {
    if (!condition) throw std::invalid_argument(std::string("qsa_input_proj: invalid ") + label);
}

bool aligned(const void* p) { return p && reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

void matrix(const Tensor& t, int rows, int columns) {
    require(t.dtype == DType::BF16 && t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.is_contiguous() && aligned(t.data),
            "activation matrix");
}

struct Range {
    std::uintptr_t begin, end;
};

Range range(const void* data, std::uint64_t bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    require(bytes <= std::numeric_limits<std::uintptr_t>::max() - begin, "address range");
    return {begin, begin + bytes};
}

bool overlap(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }

template <QType Format, int... Rows>
void validate(const Tensor& x, const Weight& w,
              const std::array<const Tensor*, sizeof...(Rows)>& outputs) {
    constexpr int parent_rows = (Rows + ...);
    constexpr std::array rows{Rows...};
    constexpr std::uint64_t elements = std::uint64_t(parent_rows) * 2560;
    constexpr bool quantized         = Format == QType::Q8_G32_FP16;
    const int columns                = x.ne[1];
    require(columns > 0, "column extent");
    matrix(x, 2560, columns);
    require(w.qtype == Format && w.ndim == 2 && w.n == parent_rows && w.k == 2560 &&
                w.shape[0] == parent_rows && w.shape[1] == 2560 &&
                w.padded_shape[0] == parent_rows && w.padded_shape[1] == 2560 && aligned(w.qdata) &&
                w.qhigh == nullptr && w.high_plane_bytes == 0,
            "weight geometry/data");
    if constexpr (quantized)
        require(w.layout == QuantLayout::RowSplit && w.group == 32 && w.group_size == 32 &&
                    w.scale_dtype == DType::FP16 && aligned(w.scales),
                "Q8 G32 FP16 weight");
    else
        require(w.layout == QuantLayout::Contiguous && w.group == 0 && w.group_size == 0 &&
                    w.scales == nullptr && w.payload_bytes >= elements * 2,
                "BF16 weight");

    const auto input  = range(x.data, x.bytes());
    const auto data   = range(w.qdata, elements * (quantized ? 1 : 2));
    const auto scales = quantized ? range(w.scales, elements / 32 * 2) : Range{};
    std::array<Range, sizeof...(Rows)> written;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        matrix(*outputs[i], rows[i], columns);
        const auto out = range(outputs[i]->data, outputs[i]->bytes());
        require(!overlap(out, input) && !overlap(out, data) && !overlap(out, scales),
                "output/input overlap");
        for (std::size_t j = 0; j < i; ++j) require(!overlap(out, written[j]), "output overlap");
        written[i] = out;
    }
}

} // namespace

void qsa_input_proj(const Tensor& x, const Weight& weight, Tensor& query, Tensor& gate,
                    cudaStream_t stream) {
    validate<QType::Q8_G32_FP16, 6144, 6144>(x, weight, {&query, &gate});
    detail::qsa::launch_query_gate(x, weight, query, gate, stream);
}

void qsa_input_proj(const Tensor& x, const Weight& weight, Tensor& key, Tensor& value,
                    Tensor& index_query, Tensor& index_key, cudaStream_t stream) {
    validate<QType::BF16, 512, 512, 512, 128>(x, weight, {&key, &value, &index_query, &index_key});
    detail::qsa::launch_kv_index(x, weight, key, value, index_query, index_key, stream);
}

void qsa_input_proj(const Tensor& x, const Weight& weight, Tensor& query, Tensor& key, Tensor& gate,
                    Tensor& value, Tensor& index_query, Tensor& index_key, cudaStream_t stream) {
    validate<QType::BF16, 6144, 512, 6144, 512, 512, 128>(
        x, weight, {&query, &key, &gate, &value, &index_query, &index_key});
    detail::qsa::launch_joint(x, weight, query, key, gate, value, index_query, index_key, stream);
}

} // namespace ninfer::ops
