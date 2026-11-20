#include "serve/crash_dump.h"

#include <cstdio>
#include <exception>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#include <dbghelp.h>

#pragma comment(lib, "dbghelp.lib")

namespace {

std::wstring g_directory;

std::wstring dump_path(const wchar_t* kind) {
    SYSTEMTIME now{};
    GetLocalTime(&now);
    wchar_t name[160];
    _snwprintf_s(name, sizeof(name) / sizeof(name[0]), _TRUNCATE,
                 L"%s\\ninfer-%s-%lu-%04u%02u%02u-%02u%02u%02u.dmp", g_directory.c_str(), kind,
                 GetCurrentProcessId(), now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute,
                 now.wSecond);
    return name;
}

// The dump is written while the process is already dying, so nothing here may allocate its way
// into the same failure: plain Win32 calls only, and a fixed-size path buffer.
void write_dump(const wchar_t* kind, EXCEPTION_POINTERS* info) {
    const std::wstring path = dump_path(kind);
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { return; }
    MINIDUMP_EXCEPTION_INFORMATION exception_information{};
    exception_information.ThreadId       = GetCurrentThreadId();
    exception_information.ExceptionPointers = info;
    exception_information.ClientPointers = FALSE;
    const auto type = static_cast<MINIDUMP_TYPE>(MiniDumpWithThreadInfo | MiniDumpWithIndirectlyReferencedMemory);
    MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file, type,
                      info != nullptr ? &exception_information : nullptr, nullptr, nullptr);
    CloseHandle(file);
    std::fwprintf(stderr, L"[crash] minidump written: %ls\n", path.c_str());
    std::fflush(stderr);
}

LONG WINAPI unhandled_exception_filter(EXCEPTION_POINTERS* info) {
    write_dump(L"crash", info);
    return EXCEPTION_EXECUTE_HANDLER;
}

void on_terminate() {
    write_dump(L"terminate", nullptr);
    std::_Exit(3);
}

} // namespace

namespace ninfer::serve {

void install_crash_handler(const std::filesystem::path& dir) {
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    g_directory = dir.wstring();
    SetUnhandledExceptionFilter(unhandled_exception_filter);
    std::set_terminate(on_terminate);
}

} // namespace ninfer::serve

#else

namespace ninfer::serve {
void install_crash_handler(const std::filesystem::path&) {}
} // namespace ninfer::serve

#endif
