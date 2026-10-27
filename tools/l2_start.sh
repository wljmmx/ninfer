#!/usr/bin/env bash
# Start the L2 (KV cache remote tier) stack in WSL user space: Redis 8.10.3 on :6390, which is what
# LMCache's RESP backend speaks. No sudo, no systemd -- the binaries and the config live in ~/l2.
#
#   tools/l2_start.sh          start (idempotent)
#   tools/l2_stop.sh           stop
#
# The binaries come from a source build: make -C ~/redis-src/redis-8.10.3 -j8
set -u

ROOT="${L2_ROOT:-$HOME/l2}"
PORT="${L2_PORT:-6390}"
BIN="$ROOT/redis-server"
CLI="$ROOT/redis-cli"
CONF="$ROOT/redis.conf"

if [ ! -x "$BIN" ] || [ ! -x "$CLI" ]; then
  echo "missing binaries in $ROOT"
  echo "build them with: make -C ~/redis-src/redis-8.10.3 -j8 && cp ~/redis-src/redis-8.10.3/src/redis-{server,cli} ~/l2/"
  exit 1
fi
if [ ! -f "$CONF" ]; then
  echo "missing $CONF"
  exit 1
fi

if "$CLI" -p "$PORT" ping 2>/dev/null | grep -q PONG; then
  echo "already up on $PORT"
  exit 0
fi

"$BIN" "$CONF" --port "$PORT" --daemonize yes || exit 1
for _ in $(seq 1 20); do
  sleep 0.2
  if "$CLI" -p "$PORT" ping 2>/dev/null | grep -q PONG; then
    echo "redis up on $PORT"
    "$CLI" -p "$PORT" info server 2>/dev/null | grep -E 'redis_version|io_threads_active'
    exit 0
  fi
done

echo "start failed; see $ROOT/redis.log"
exit 1
