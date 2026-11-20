#pragma once

#include <filesystem>

namespace ninfer::serve {

// Install the process-wide crash handlers: an unhandled SEH exception or a std::terminate writes a
// minidump into `dir` before the process dies. The engine exited once with 0xC0000409 and a log
// that simply stopped, which is not enough to name a frame; a dump is. Windows-only in practice,
// a no-op elsewhere.
void install_crash_handler(const std::filesystem::path& dir);

} // namespace ninfer::serve
