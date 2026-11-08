"""port (sm_89): Windows shims for the POSIX file calls the artifact tools use.

The tools were written for Linux. Only the primitives without a Windows equivalent are
provided here; every other call already exists in the standard library on both platforms.
Importing this module patches ``os`` in place so the callers stay unchanged.
"""

from __future__ import annotations

import os
import threading

# Descriptor caches shared by the shims below. They must be dropped when the caller closes the
# descriptor: on Windows an open handle blocks both the hard-link publish (the writer links the
# temporary file into place) and its removal, so a leaked duplicate turns a finished artifact
# into "another program is using this file" at exactly the last step.
_needs_shims = not hasattr(os, "pread") or not hasattr(os, "pwrite")
_sources: dict[int, object] = {}
_targets: dict[int, object] = {}
_lock = threading.Lock()
_write_lock = threading.Lock()

if not hasattr(os, "pread"):
    def _source(fd: int):
        source = _sources.get(fd)
        if source is None:
            # A separate file object keeps its own offset, so it cannot disturb readers that
            # used the descriptor directly.
            source = os.fdopen(os.dup(fd), "rb", buffering=0)
            _sources[fd] = source
        return source

    def _pread(fd: int, count: int, offset: int) -> bytes:
        with _lock:
            source = _source(fd)
            source.seek(offset)
            return source.read(count)

    os.pread = _pread  # type: ignore[attr-defined]

if not hasattr(os, "pwrite"):
    def _target(fd: int):
        target = _targets.get(fd)
        if target is None:
            target = os.fdopen(os.dup(fd), "r+b", buffering=0)
            _targets[fd] = target
        return target

    def _pwrite(fd: int, data: bytes, offset: int) -> int:
        with _write_lock:
            target = _target(fd)
            target.seek(offset)
            return target.write(data)

    os.pwrite = _pwrite  # type: ignore[attr-defined]

if _needs_shims:
    _original_close = os.close

    def _close(fd: int) -> None:
        for cache in (_sources, _targets):
            handle = cache.pop(fd, None)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        _original_close(fd)

    os.close = _close  # type: ignore[assignment]

if not hasattr(os, "link"):

    def _link(source: str, target: str) -> None:
        import shutil

        shutil.copyfile(source, target)

    os.link = _link  # type: ignore[attr-defined]
