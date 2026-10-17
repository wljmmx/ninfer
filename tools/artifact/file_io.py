"""Keep large offline transfers from retaining whole artifacts in Linux's page cache."""

from __future__ import annotations

import os

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
# port (sm_89): os.sysconf is POSIX-only. The page size only affects the alignment of the
# cache-discard range below, which is a Linux transfer optimisation.
_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    # port (sm_89): Windows has no page-cache advice; the caller only wants the pages gone from
    # the read cache, which the OS does on its own there.
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
            # port (sm_89): fdatasync is POSIX; fsync is the Windows equivalent (it also
            # flushes metadata, which is fine for an artifact about to be closed).
            (getattr(os, "fdatasync", None) or os.fsync)(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
