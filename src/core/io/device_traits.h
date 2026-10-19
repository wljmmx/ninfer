#pragma once

#include "core/io/file.h"

#include <filesystem>

namespace ninfer::io {

// port (sm_89): true when the volume behind `path` has a seek penalty (a rotating disk).
//
// The artifact is plane-major: one expert's codes and scales are hundreds of MiB apart, so a
// per-expert read is three seeks on a disk. Buffered I/O lets the OS read-ahead hide that;
// unbuffered I/O only wins where seeks are cheap. Callers pick their ReadMode from this.
[[nodiscard]] bool volume_incurs_seek_penalty(const std::filesystem::path& path);

// port (sm_89): expert weight reads default to buffered. Unbuffered is opt-in
// (NINFER_EXPERT_IO=direct) and only honoured on media without a seek penalty.
[[nodiscard]] ReadMode expert_read_mode(const std::filesystem::path& path);

} // namespace ninfer::io

