#!/usr/bin/env bash
# Unattended supervisor for the long_v2 run.
#
# Design note (learned the hard way): `opencode run` is SINGLE-SHOT. The first
# attempt at this used one long-lived agent session to both monitor and report;
# it exited after ~4 minutes because the model ended its turn waiting on a
# background sleep, and an ended turn means the process exits. There is no loop
# to come back to.
#
# So: the recurring part is a plain shell loop (deterministic, cannot die from
# context limits or provider hiccups), and the model is invoked ONCE at the end,
# where it actually adds value -- turning the tick log plus the ladder JSON into
# a written report.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

DIR=runs/long_v2
TICKS=runs/long_v2_ticks.log
OUT=runs/wave_ladder3.json
INTERVAL="${1:-1800}"
MODEL="${2:-vllm/deepseek-v4-flash}"

echo "=== supervisor armed $(date +%F' '%H:%M:%S), interval ${INTERVAL}s ===" >> "$TICKS"

# 1) Tick until training exits.
while pgrep -f "roster_dir=$DIR" > /dev/null; do
  ./monitor_long_v2.sh >> "$TICKS" 2>&1
  sleep "$INTERVAL"
done
echo "=== training stopped $(date +%F' '%H:%M:%S) ===" >> "$TICKS"
./monitor_long_v2.sh >> "$TICKS" 2>&1

# 2) Judge. Skip if something else already did it (e.g. the watchdog).
if [ ! -f "$OUT" ]; then
  echo "=== judging $(date +%F' '%H:%M:%S) ===" >> "$TICKS"
  ./run_judge.sh 64 "$OUT" > runs/judge3.log 2>&1
  echo "=== judge exit $? ===" >> "$TICKS"
fi

# 3) One model call to write the report.
if [ -f "$OUT" ]; then
  PROMPT="You are writing a results report. Do NOT run any training or GPU work.

Read these two files in /home/mihai/personal/open_spiel_eclipse:
  - runs/long_v2_ticks.log   (periodic monitoring ticks from a 12h training run)
  - runs/wave_ladder3.json   (the head-to-head ladder that judges the result)

Append ONE section to docs/NEXT_SESSION.md titled '## long_v2 result (\$(date +%F))'
containing:
  1. Total steps reached, wall clock, and average sps, taken from the tick log.
  2. A compact table of the ticks (time, steps, sps, entropy, ev, oob).
  3. The full ratings table from runs/wave_ladder3.json, sorted by rating
     descending, showing id, rating, rating_ci and games.
  4. A short verdict.

RULES, do not deviate:
  - Judge ONLY with rating / rating_ci. A policy beats another ONLY if its
    rating_ci LOWER bound exceeds the other's rating_ci UPPER bound. If the CIs
    overlap, write exactly: 'no significant difference'.
  - NEVER use vp_all, mean_episode_return, return_trend or any 'beats Greedy'
    number as evidence of strength. They are decoupled from strength here and
    have caused two wrong conclusions already.
  - Never compare ratings across different ladder JSON files; the fit is
    relative to each tournament's pool.
  - Use the 'rating' field, never 'elo'.
  - Your ONLY write is appending to docs/NEXT_SESSION.md. Do not touch source,
    do not git commit, do not delete anything.
  - Report real numbers. If something is missing, say so plainly."

  timeout 3600 opencode run --dir "$REPO_ROOT" -m "$MODEL" --auto \
    --title "long_v2 report" "$PROMPT" >> runs/opencode_report.log 2>&1
  echo "=== report attempt exit $? $(date +%F' '%H:%M:%S) ===" >> "$TICKS"
fi

echo "=== supervisor done $(date +%F' '%H:%M:%S) ===" >> "$TICKS"
