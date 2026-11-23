#!/usr/bin/env bash
# Start the L3 tier: the LMCache server that holds KV checkpoints off-process, inside WSL user space.
#
# This is the process the engine's LMCache client talks to. It is NOT a container: lmcache is
# installed in a venv from source (see ~/lmcache-src), and the server runs straight out of that venv.
# Nothing here uses docker, and nothing needs root.
#
#   tools/l3_lmcache.sh          start (idempotent)
#   tools/l3_lmcache.sh status   ask whether it answers
#   tools/l3_lmcache.sh stop     stop the one this script started
#
# The port and chunk size are part of the contract with the engine, not preferences:
#   - the engine defaults its endpoint to tcp://127.0.0.1:6555 (lmcache_client.h);
#   - --chunk-size must be 128, because a checkpoint key is cut at whole chunks of 128 tokens and the
#     server refuses a chunk index it was not told about (measured: "chunk index out of range").
# Both are overridable here, but changing either means changing the engine side with it.
#
# L1 defaults to 12 GB (2026-11-23). One 163k checkpoint is 2687 MiB, so a 4 GB tier held a single
# entry at 66% and a second 163k prefix collided with it: measured, the second store died at the
# transport (`receive failed: Resource temporarily unavailable` after 30 s at 84 MiB/s) with the
# first entry still resident. The engine sizes its own per-entry limit from half of this number
# (NINFER_LMCACHE_L1_MB, default 12288 now), so the two move together - change one, change both.
#
# The earlier 4 GB default existed for a different reason, kept here so it is not lost: on 2026-11-20
# a 10 GB tier plus the engine's own pinned host context exhausted cudaMallocHost, and the engine fell
# back to pageable memory and died at module init (both the release and a fresh build). 12 GB is ABOVE
# that figure, so it is the value to watch: if an engine start fails in cudaMallocHost, drop L3_L1_GB
# one step (8, then 6) rather than removing the cache - and re-record this comment with the reading.
set -u

ROOT="${L3_ROOT:-$HOME}"
VENV="${L3_VENV:-$ROOT/lmc-venv}"
PORT="${L3_PORT:-6555}"
HTTP_PORT="${L3_HTTP_PORT:-18080}"
CHUNK="${L3_CHUNK:-128}"
WORKERS="${L3_WORKERS:-4}"
L1_GB="${L3_L1_GB:-12}"
LOG="${L3_LOG:-$ROOT/l3-lmcache.log}"
PATTERN="lmcache server --transport zmq --port $PORT"
# Bind every interface, not loopback. Measured 2026-11-21 with WSL's `networkingMode=Mirrored`: a
# Windows client that connects to 127.0.0.1 reaches a user-space loopback proxy (`127.0.0.1 via
# 169.254.x dev loopback0`), and a 183 MB checkpoint moves at 64 MB/s over it against 129-130 MB/s
# when the same client connects to the distro's own address. The store direction gains less (97 ->
# 132 MB/s) but moves the same way. Set L3_HOST to override (e.g. 127.0.0.1 for a same-distro client).
HOST="${L3_HOST:-0.0.0.0}"

up() { pgrep -f "$PATTERN" >/dev/null 2>&1; }

case "${1:-start}" in
  status)
    if up; then echo "up on $PORT"; else echo "not running"; fi
    exit 0
    ;;
  stop)
    pkill -f "$PATTERN" 2>/dev/null && echo "stopped" || echo "was not running"
    exit 0
    ;;
esac

if up; then
  echo "already up on $PORT"
  exit 0
fi

if [ ! -x "$VENV/bin/lmcache" ]; then
  echo "missing $VENV/bin/lmcache"
  echo "install it with: uv pip install --python $VENV/bin/python lmcache   (see ~/lmc_setup3.sh)"
  exit 1
fi

nohup "$VENV/bin/lmcache" server \
  --transport zmq --host "$HOST" --port "$PORT" --http-port "$HTTP_PORT" \
  --supported-transfer-mode engine_driven \
  --chunk-size "$CHUNK" --max-workers "$WORKERS" \
  --l1-size-gb "$L1_GB" --eviction-policy LRU >"$LOG" 2>&1 &

# A server that is still importing torch is not ready to answer; wait for the socket, not for the pid.
# Measured on this box: the ports open ~17 s after launch, so the old 30 s ceiling was one import
# away from calling a healthy server a failure.
for _ in $(seq 1 240); do
  sleep 0.5
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "lmcache up on $PORT (http $HTTP_PORT, chunk $CHUNK, log $LOG)"
    exit 0
  fi
done

echo "start failed; see $LOG"
tail -n 20 "$LOG" 2>/dev/null
exit 1
