#!/usr/bin/env bash
# Safety net for the long_v2 run.
#
# The opencode agent is supposed to monitor training, then judge it on the ladder
# and append a report to docs/NEXT_SESSION.md. If that agent dies partway through
# a 12-hour vigil (context exhaustion, provider hiccup, crash), nobody would ever
# produce the result. This watchdog guarantees the result exists.
#
# It does NOTHING unless the agent has failed to deliver: it waits for training to
# end, gives the agent a grace period to run the ladder itself, and only steps in
# if runs/wave_ladder3.json still does not exist.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

DIR=runs/long_v2
OUT=runs/wave_ladder3.json
GRACE="${1:-3600}"   # seconds to let the agent do the judging itself

log() { echo "[watchdog $(date +%F' '%H:%M:%S)] $*" >> runs/watchdog.log; }

log "armed; waiting for training to finish"
# Wait for training to exit. Poll the process list via a script-file pgrep (the
# pattern is inside this file, not on this shell's command line, so it cannot
# match itself).
while pgrep -f "roster_dir=$DIR" > /dev/null; do sleep 120; done
log "training finished"

# Give the opencode agent its chance to run the ladder.
waited=0
while [ "$waited" -lt "$GRACE" ]; do
  if [ -f "$OUT" ]; then
    log "agent produced $OUT; nothing to do"
    exit 0
  fi
  # If the agent is already running the ladder, wait for it rather than racing.
  if pgrep -f "roster_ladder" > /dev/null; then
    log "agent is running the ladder; waiting"
    while pgrep -f "roster_ladder" > /dev/null; do sleep 60; done
    [ -f "$OUT" ] && { log "agent finished the ladder"; exit 0; }
  fi
  sleep 120
  waited=$((waited + 120))
done

log "grace expired and $OUT is missing -- running the ladder myself"
./run_judge.sh 64 "$OUT" > runs/judge3.log 2>&1
log "ladder done (exit $?)"

{
  echo
  echo "## long_v2 result — WATCHDOG FALLBACK ($(date +%F' '%H:%M))"
  echo
  echo "The opencode monitoring agent did not produce this report, so the watchdog"
  echo "ran the ladder instead. Tick-by-tick monitoring history is NOT available;"
  echo "only the final judgement below."
  echo
  echo '```'
  echo "steps reached: $(tr '\r' '\n' < "$DIR/train.log" 2>/dev/null | grep -oE '[0-9]+/1000000000' | tail -1)"
  echo "snapshots:     $(ls "$DIR"/snap_u*.pt 2>/dev/null | wc -l)"
  echo "oom lines:     $(grep -c 'OutOfMemoryError' "$DIR/train.log" 2>/dev/null | head -1)"
  echo '```'
  echo
  echo 'Ratings (compare ONLY within this table; a policy wins only if its CI lower'
  echo 'bound clears the other CI upper bound):'
  echo
  echo '```'
  .venv/bin/python -c "
import json
d=json.load(open('$OUT'))
for p in sorted(d['policies'], key=lambda p:-p['rating']):
    print(f\"{p['id']:30s} {p['rating']:+.4f} [{p['rating_ci'][0]:+.4f},{p['rating_ci'][1]:+.4f}] games={p['games']}\")
" 2>&1
  echo '```'
} >> docs/NEXT_SESSION.md

log "fallback report appended to docs/NEXT_SESSION.md"
