#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cstdlib>

namespace ninfer {

// port (sm_89): diagnostic only. Prints the faulting address, access kind and a raw return
// address trace, then lets the process die as before. Enable with NINFER_FAULT_TRACE=1.
inline LONG CALLBACK fault_trace_handler(EXCEPTION_POINTERS* info) {
    const auto* record = info->ExceptionRecord;
    if (!record || record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    std::fprintf(stderr, "FAULT: access=%llu addr=%p code=0x%08lX\n",
                 static_cast<unsigned long long>(record->ExceptionInformation[0]),
                 reinterpret_cast<void*>(record->ExceptionInformation[1]),
                 static_cast<unsigned long>(record->ExceptionCode));
    void* frames[32];
    const auto count = CaptureStackBackTrace(0, 32, frames, nullptr);
    for (WORD i = 0; i < count; ++i) {
        std::fprintf(stderr, "  #%02u %p\n", static_cast<unsigned>(i), frames[i]);
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void install_fault_trace() {
    const char* value = std::getenv("NINFER_FAULT_TRACE");
    if (value && *value && *value != '0') {
        AddVectoredExceptionHandler(1, fault_trace_handler);
    }
}

} // namespace ninfer
#endif
