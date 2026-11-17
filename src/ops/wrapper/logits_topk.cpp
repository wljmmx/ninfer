// ninfer::ops - logits_topk wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/logits_topk.h"

#include "ops/launcher/logits_topk.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_rank_two(const Tensor& tensor, const char* label) {
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("logits_topk: ") + label +
                                    " must be rank-2 with positive dimensions");
    }
}

void require_candidates(std::int32_t columns, std::int32_t k, const Tensor& tensor,
                        const char* label) {
    if (tensor.ne[0] != k || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("logits_topk: ") + label +
                                    " must have shape [k, columns]");
    }
}

void require_vector(const Tensor& tensor, std::int32_t columns, const char* label) {
    if (tensor.ne[0] != columns || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("logits_topk: ") + label +
                                    " must have shape [columns]");
    }
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("logits_topk: ") + label + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("logits_topk: ") + label +
                                    " data must be non-null");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("logits_topk: ") + label +
                                    " data is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void logits_topk(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& ids,
                 Tensor& logprobs, Tensor& log_normalizer, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("logits_topk: logits must be BF16");
    }
    if (ids.dtype != DType::I32) { throw std::invalid_argument("logits_topk: ids must be I32"); }
    if (logprobs.dtype != DType::FP32) {
        throw std::invalid_argument("logits_topk: logprobs must be FP32");
    }
    if (log_normalizer.dtype != DType::FP32) {
        throw std::invalid_argument("logits_topk: log_normalizer must be FP32");
    }

    require_rank_two(logits, "logits");
    const std::int32_t columns = logits.ne[1];
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("logits_topk: valid_rows must be in [1, physical_rows]");
    }
    if (k <= 0 || k > kLogitsTopkCapacity || k > valid_rows) {
        throw std::invalid_argument(
            "logits_topk: k must be in [1, min(valid_rows, kLogitsTopkCapacity)]");
    }
    require_candidates(columns, k, ids, "ids");
    require_candidates(columns, k, logprobs, "logprobs");
    require_vector(log_normalizer, columns, "log_normalizer");

    (void)logits.bytes();
    (void)ids.bytes();
    (void)logprobs.bytes();
    (void)log_normalizer.bytes();
    require_accessible(logits, alignof(std::uint16_t), "logits");
    require_accessible(ids, alignof(std::int32_t), "ids");
    require_accessible(logprobs, alignof(float), "logprobs");
    require_accessible(log_normalizer, alignof(float), "log_normalizer");
    if (overlaps(ids, logits) || overlaps(logprobs, logits) || overlaps(log_normalizer, logits) ||
        overlaps(ids, logprobs) || overlaps(ids, log_normalizer) ||
        overlaps(logprobs, log_normalizer)) {
        throw std::invalid_argument(
            "logits_topk: outputs must not overlap the input or each other");
    }

    detail::logits_topk_launch(logits, valid_rows, k, ids, logprobs, log_normalizer, stream);
}

} // namespace ninfer::ops
