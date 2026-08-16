#!/usr/bin/env bash
# =============================================================================
# int main() of the auto-research loop.  IMMUTABLE (checksummed).
#
# Run ONE throughput experiment: 12 parallel envs for a fixed 1-minute
# wall-clock budget, then report how many training steps the envs reached in
# that budget. Emits one machine-parseable line the agent reads to decide
# keep/discard.
#
#   bench.sh <run_dir> [envs] [seconds]
#     run_dir  - directory where this experiment's outputs land (created here)
#     envs     - number of parallel envs (default 12)
#     seconds  - wall-clock training budget (default 60)
#
# SCORE = throughput, NOT quality.  The goal of the loop is to make the
# training/environment run FASTER (more effective env-steps per wall-second).
# Under a fixed budget, the number of env-steps reached (`steps=`) is exactly
# that: rolling the env out faster -> more steps land in the same 60 s.  With a
# fixed env/step config (envs x num_steps = 12 x 128 = 1536) `steps` and
# `updates` are the same signal (steps = 1536 * updates), so a single scalar
# carries both rollout speed AND update/compute throughput.
#
# Anti-cheat (enforced here, not just instructed):
#   1. IMMUTABILITY GATE: bench.sh and immutables.sha carry checksums recorded
#      by `autoresearch/make_manifest.sh`. If either changed since recorded, the
#      run is VOID (scored INVALID, no credit). An agent that edits the runner
#      gets no reward for it.
#   2. WALL-CLOCK BUDGET: measured by the outer `timeout`, from process start to
#      finish -- includes startup/compile/load. A run that exceeds ~2x the
#      budget is killed and marked CRASH, exactly Karpathy's rule. steps/sec is
#      computed from THIS wall time, so cutting non-rollout overhead also pays.
#   3. CROSS-CHECK: the reported `steps` must reconcile with `updates x batch`
#      (batch = envs x num_steps) and with a physically plausible steps/sec vs
#      the un-tuned baseline. A mismatch is flagged (score still uses `steps`,
#      the robust scalar). A fake counter is NOT caught here -- that is the job
#      of gate_audit.sh, the independent fresh-context reviewer, which the loop
#      must run on every would-be keep.
#   4. THE HONESTY GATE: score is parsed from the training log, which the
#      editable trainer prints. bench.sh CANNOT make that counter unforgeable on
#      its own (there is no immutable step counter inside the env). The defense
#      is `autoresearch/gate_audit.sh`: for any variant that beats the current
#      best, spawn a FRESH-context subagent (no memory of prior runs) with the
#      SAME question each time and require a `pass` before the variant is kept.
#      This is the gate for "did the change actually speed things up, or did it
#      cheat / get faster-but-broken".
#
# The experiment (the *variant*) is whatever the current git working tree has
# been edited to be. bench.sh does NOT edit code. The agent edits, commits to a
# branch, then calls bench.sh. Use git worktrees for parallel variants.
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AR_DIR="$ROOT/autoresearch"
BENCH_BASH="$AR_DIR/bench.sh"
MANIFEST="$AR_DIR/immutables.sha"

RUN_DIR="${1:?usage: bench.sh <run_dir> [envs] [seconds]}"
ENVS="${2:-12}"
SECONDS="${3:-60}"
# steps-per-update = envs x num_steps (must match the training invocation below).
NUM_STEPS=128
BATCH=$((ENVS * NUM_STEPS))
mkdir -p "$RUN_DIR"

say() { echo "[bench] $*"; }

# ---- 1. Immutability gate ---------------------------------------------------
if [ ! -f "$MANIFEST" ]; then
  echo "RESULT run=$RUN_DIR status=INVALID reason=no_manifest (run make_manifest.sh first)"
  exit 1
fi
bad=0
while read -r expect path; do
  [ -z "$expect" ] && continue
  got=$(sha256sum "$path" 2>/dev/null | awk '{print $1}')
  if [ "$got" != "$expect" ]; then
    say "IMMUTABILITY VIOLATION: $path changed (manifest=$expect got=$got)"
    bad=1
  fi
done < "$MANIFEST"
if [ "$bad" = "1" ]; then
  echo "RESULT run=$RUN_DIR status=INVALID reason=immutable_changed"
  exit 1
fi
say "immutability gate passed"

# ---- 2. Baseline record (first run always establishes the number to beat) ---
RESULTS="$ROOT/results_${ENVS}env_${SECONDS}s.tsv"
if [ ! -f "$RESULTS" ]; then
  printf 'ts\tcommit\tbranch\tscore_steps\tupdates\tsteps_per_sec\tstatus\tdesc\n' > "$RESULTS"
fi

# ---- 3. Run the experiment: 12 envs, fixed budget --------------------------
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
BRANCH="$(git -C "$ROOT" branch --show-current)"
say "variant: commit=$COMMIT branch=$BRANCH envs=$ENVS budget=${SECONDS}s batch=${BATCH}/update"
say "launching ppo_eclipse (roster -> $RUN_DIR)"

export PYTHONPATH="build/open_spiel/python:$ROOT"
START=$(date +%s)
# 2x budget hard kill; the training loop itself also enforces --max_seconds.
timeout $((SECONDS * 2 + 120)) \
  "$ROOT/.venv/bin/python" -m open_spiel.python.examples.ppo_eclipse \
  --game="eclipse(players=4)" \
  --num_envs="$ENVS" \
  --num_steps="$NUM_STEPS" \
  --num_workers="$ENVS" \
  --num_minibatches=4 \
  --update_epochs=4 \
  --total_timesteps=100000000000 \
  --max_seconds="$SECONDS" \
  --snapshot_every="${SNAPSHOT_EVERY:-25}" \
  --roster_dir="$RUN_DIR" \
  --run_dir="$RUN_DIR" \
  --track="bench" \
  --noeval_greedy --noeval_random \
  --noprogress \
  > "$RUN_DIR/train.log" 2>&1
RC=$?
END=$(date +%s)
WALL=$((END - START))
say "training exited rc=$RC in ${WALL}s"

# ---- 4. If the run was killed / crashed, record and stop -------------------
if [ $RC -ne 0 ]; then
  # Still try to salvage a score from the last flushed [update] line: a run killed
  # by the outer timeout (rc=124) may have printed real updates before dying.
  last_steps=$(awk '/\[update [0-9]+\] steps=/{s=$0} END{print s}' "$RUN_DIR/train.log" 2>/dev/null \
    | sed -n 's/.*\[update \([0-9]*\)\] steps=\([0-9]*\).*/\2 \1/p')
  if [ -n "$last_steps" ]; then
    steps=$(echo "$last_steps" | awk '{print $1}')
    update=$(echo "$last_steps" | awk '{print $2}')
    sps=$((steps / WALL))
    echo "RESULT run=$RUN_DIR commit=$COMMIT status=crash rc=$RC score_steps=$steps updates=$update steps_per_sec=$sps desc=partial_${RC}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%Y%m%d-%H%M%S)" "$COMMIT" "$BRANCH" "$steps" "$update" "$sps" "crash" "partial_rc_${RC}" \
      >> "$RESULTS"
  else
    echo "RESULT run=$RUN_DIR commit=$COMMIT status=crash rc=$RC"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%Y%m%d-%H%M%S)" "$COMMIT" "$BRANCH" "0" "0" "0" "crash" "exit_rc_${RC}" \
      >> "$RESULTS"
  fi
  exit 1
fi

# ---- 5. Throughput score from train.log -------------------------------------
# Primary source: the gate line "[gate] 60s hard deadline reached (update N, steps=M)".
# Fallback (no gate line): the last flushed "[update N] steps=M" line.
gate=$(grep -E '^\[gate\].*\(update [0-9]+, steps=[0-9]+\)' "$RUN_DIR/train.log" 2>/dev/null | tail -1)
if [ -n "$gate" ]; then
  steps=$(echo "$gate" | sed -n 's/.*(update \([0-9]*\), steps=\([0-9]*\)).*/\2/p')
  update=$(echo "$gate" | sed -n 's/.*(update \([0-9]*\), steps=\([0-9]*\)).*/\1/p')
else
  last=$(grep -E '^\[update [0-9]+\] steps=' "$RUN_DIR/train.log" 2>/dev/null | tail -1)
  if [ -z "$last" ]; then
    echo "RESULT run=$RUN_DIR commit=$COMMIT status=crash reason=no_steps_in_log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%Y%m%d-%H%M%S)" "$COMMIT" "$BRANCH" "0" "0" "0" "crash" "no_steps_in_log" \
      >> "$RESULTS"
    exit 1
  fi
  steps=$(echo "$last" | sed -n 's/.*\[update \([0-9]*\)\] steps=\([0-9]*\).*/\2/p')
  update=$(echo "$last" | sed -n 's/.*\[update \([0-9]*\)\] steps=\([0-9]*\).*/\1/p')
fi
[ -z "$steps" ] && steps=0
[ -z "$update" ] && update=0
if [ "$WALL" -gt 0 ]; then
  sps=$((steps / WALL))
else
  sps=0
fi

# Cross-check: steps must reconcile with updates x batch, and steps/sec must be
# physically plausible. This catches internally-inconsistent counters, not a
# deliberate double-fake (that is gate_audit.sh's job).
expected=$((update * BATCH))
mismatch="no"
if [ "$expected" -ne "$steps" ]; then
  mismatch="yes"
  say "sanity: steps($steps) != updates($update)xbatch($BATCH)=$expected"
fi

say "throughput: steps=$steps updates=$update wall=${WALL}s steps_per_sec=$sps crosscheck=$mismatch"
echo "RESULT run=$RUN_DIR commit=$COMMIT score_steps=$steps updates=$update steps_per_sec=$sps"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date +%Y%m%d-%H%M%S)" "$COMMIT" "$BRANCH" "$steps" "$update" "$sps" "run" \
  "crosscheck=${mismatch} budget=${SECONDS}s" >> "$RESULTS"

echo "  recorded. Compare to the best score_steps in $RESULTS; the agent keeps this"
echo "  commit only if score_steps > previous best (Karpathy keep/discard), and"
echo "  only after gate_audit.sh (the fresh-context honesty review) passes it."
exit 0
