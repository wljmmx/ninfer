// port (sm_89): read-only observation hook for the routed MoE ops.
//
// Deliberately a standalone header: no existing header includes it, so adding it does not
// change any translation unit's layout. A process-wide pointer is set only by the offline
// diagnostic; when it is null the ops behave exactly as before.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::probe {

struct MoeRouteProbe {
    const void* ids                 = nullptr; // int [top_k, tokens], token-major
    const void* coefficients        = nullptr; // float [top_k, tokens]
    const void* shared_coefficients = nullptr; // float [tokens]
    std::int32_t tokens             = 0;
    std::int32_t top_k              = 0;
};

inline MoeRouteProbe* route_probe = nullptr;

inline void publish_routes(const void* ids, const void* coefficients,
                           const void* shared_coefficients, std::int32_t tokens,
                           std::int32_t top_k) {
    if (route_probe == nullptr) { return; }
    *route_probe = {ids, coefficients, shared_coefficients, tokens, top_k};
}

} // namespace ninfer::ops::probe
