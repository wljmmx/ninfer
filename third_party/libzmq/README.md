# libzmq 4.3.5 (vendored)

The transport under the LMCache checkpoint cache: `core/lmcache_client.h` speaks DEALER/ROUTER over
ZeroMQ, so the serve binary needs the headers, an import library and the runtime DLL.

## Provenance

Copied from the Conda installation on the development box (`Library/{include,lib,bin}`), which is
where the Windows build had it before this was vendored. Upstream: <https://github.com/zeromq/libzmq>,
tag `v4.3.5`.

| File | Purpose |
|---|---|
| `include/zmq.h`, `include/zmq_utils.h` | Public headers |
| `lib/libzmq-mt-4_3_5.lib` | Import library for the static-CRT (`/MT`) build |
| `bin/libzmq-mt-4_3_5.dll` | Runtime, deployed beside each executable |

Only the `/MT` pair is carried, because the project builds with `/MT` throughout; mixing it with the
`/MD` variant would put two CRTs in one process.

## Licence

MPL-2.0. **The full licence text still has to be added here** -- it was not present in the Conda
prefix this was copied from, and inventing a licence file is worse than recording the gap. Until it
is added, this directory should not be redistributed outside the repository.

## Replacing it

Drop a matching set of files in the same three directories; `cmake/Dependencies.cmake` looks here
first and only falls back to a Conda prefix (or `NINFER_ZMQ_ROOT`) when this directory is incomplete.
