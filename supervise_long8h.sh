#!/usr/bin/env bash
# Lean background supervisor for the long8h run: logs a tick every INTERVAL s.
# Detached and nohup'd so it survives this agent turn. Polls the training
# process directly (never pgrep'ing its own command line), writes ticks, and
# records a final line when training stops so a future session knows.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

DIR="${1:-runs/long8h}"
TICKS="${2:-runs/long8h_ticks.log}"
INTERVAL="${3:-900}"          # 15 min
TRAIN_PID="${4:-}"

if [ -n "$TRAIN_PID" ]; then
  echo "=== supervisor armed $(date +%F' '%H:%M:%S), interval ${INTERVAL}s, train_pid=${TRAIN_PID} ===" >> "$TICKS"
else
  echo "=== supervisor armed $(date +%F' '%H:%M:%S), interval ${INTERVAL}s ===" >> "$TICKS"
fi

# Use a stable marker file rather than pgrep (which can self-match / linger).
MARK="/tmp/long8h_train.start"
: > "$MARK"

while :; do
  sleep "$INTERVAL"
  if [ -n "${TRAIN_PID:-}" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "=== training pid ${TRAIN_PID} exited $(date +%F' '%H:%M:%S) ===" >> "$TICKS"
    break
  fi
  ./monitor_long8h.sh "$DIR" >> "$TICKS" 2>&1
done
echo "=== supervisor done $(date +%F' '%H:%M:%S) ===" >> "$TICKS"
