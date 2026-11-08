"""Keep large offline transfers from retaining whole artifacts in Linux's page cache."""

from __future__ import annotations

import os

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
# port (sm_89): sysconf/posix_fadvise/fdatasync are Unix-only. Windows keeps the same API
# surface as a no-op so the offline tools still run there.
_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def pread(fd: int, count: int, offset: int) -> bytes:
    """Positioned read that also exists on Windows.

    os.pread is POSIX-only; the Windows port of the offline tools hit exactly that. The fallback
    seeks and reads on the same descriptor - these readers are single-threaded by construction
    (one source per conversion stage), so the seek costs nothing and changes nothing else.
    """
    if hasattr(os, "pread"):
        return os.pread(fd, count, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            if hasattr(os, "fdatasync"):
                os.fdatasync(fd)
            else:
                os.fsync(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
