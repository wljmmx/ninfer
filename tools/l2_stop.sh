#!/usr/bin/env bash
# Stop the L2 stack started by tools/l2_start.sh.
set -u

ROOT="${L2_ROOT:-$HOME/l2}"
PORT="${L2_PORT:-6390}"
CLI="$ROOT/redis-cli"

if [ ! -x "$CLI" ]; then
  echo "missing $CLI"
  exit 1
fi
if ! "$CLI" -p "$PORT" ping 2>/dev/null | grep -q PONG; then
  echo "not running on $PORT"
  exit 0
fi
"$CLI" -p "$PORT" shutdown nosave 2>/dev/null
sleep 0.5
if "$CLI" -p "$PORT" ping 2>/dev/null | grep -q PONG; then
  echo "still running"
  exit 1
fi
echo "stopped"
