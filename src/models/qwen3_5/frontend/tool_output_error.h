#pragma once

#include "ninfer/types.h"

#include <exception>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::frontend {

enum class ToolOutputErrorKind { InvalidModelOutput, ContractViolation };

// Request-local failure of committed model output. The declaration is legal, so this is a server
// (output) error rather than an invalid request, and it never retracts already published bytes.
struct ToolOutputError {
    ToolOutputErrorKind kind = ToolOutputErrorKind::InvalidModelOutput;
    std::string message;

    [[nodiscard]] std::exception_ptr exception() const {
        if (kind == ToolOutputErrorKind::ContractViolation) {
            return std::make_exception_ptr(
                std::logic_error("tool constraint/decoder disagreement: " + message));
        }
        return std::make_exception_ptr(RequestError(RequestErrorKind::InvalidToolOutput, message,
                                                    {}, RequestErrorSource::Tools));
    }
};

} // namespace ninfer::models::qwen3_5::frontend
