#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>

#include <cstdio>
#include <cstdlib>

#pragma comment(lib, "dbghelp.lib")

namespace ninfer {

// port (sm_89): diagnostic only. Prints the faulting address, access kind and a symbolized
// return trace, then lets the process die as before. Enable with NINFER_FAULT_TRACE=1.
inline void fault_trace_print_frame(DWORD index, DWORD64 address) {
    constexpr auto max_name = 512;
    alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + max_name]{};
    auto* symbol            = reinterpret_cast<SYMBOL_INFO*>(storage);
    symbol->SizeOfStruct    = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen      = max_name;

    IMAGEHLP_MODULE64 module{};
    module.SizeOfStruct = sizeof(module);
    const char* module_name = SymGetModuleInfo64(GetCurrentProcess(), address, &module) ? module.ModuleName : "?";

    DWORD64 displacement = 0;
    if (!SymFromAddr(GetCurrentProcess(), address, &displacement, symbol)) {
        std::fprintf(stderr, "  #%02lu %p %s+0x%llx\n", static_cast<unsigned long>(index),
                     reinterpret_cast<void*>(address), module_name,
                     static_cast<unsigned long long>(address));
        return;
    }
    IMAGEHLP_LINE64 line{};
    line.SizeOfStruct = sizeof(line);
    DWORD line_displacement = 0;
    const bool has_line =
        SymGetLineFromAddr64(GetCurrentProcess(), address, &line_displacement, &line) != FALSE;
    std::fprintf(stderr, "  #%02lu %s!%s+0x%llx %s:%lu\n", static_cast<unsigned long>(index),
                 module_name, symbol->Name, static_cast<unsigned long long>(displacement),
                 has_line ? line.FileName : "?", has_line ? line.LineNumber : 0);
}

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
        fault_trace_print_frame(i, reinterpret_cast<DWORD64>(frames[i]));
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void install_fault_trace() {
    const char* value = std::getenv("NINFER_FAULT_TRACE");
    if (!value || !*value || *value == '0') { return; }
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    (void)SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    AddVectoredExceptionHandler(1, fault_trace_handler);
}

} // namespace ninfer
#endif
