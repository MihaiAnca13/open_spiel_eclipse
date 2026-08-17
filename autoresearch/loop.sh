#!/usr/bin/env bash
# =============================================================================
# Fresh-session driver for the auto-research loop.  IMMUTABLE (checksummed).
#
# WHY: the model powering the agent (deepseek-v4-flash) starts looping its
# thoughts after opencode compacts a long session. The fix is to never let a
# session live long enough to compact: each experiment runs in a SHORT-LIVED,
# FRESH session that does exactly one thing and exits. Durable state lives on
# disk (results TSV + NOTES.md + git), not in a long transcript.
#
# Per-experiment flow. The driver owns ALL measurement and gating; the agent
# session owns only experiment selection + the edit:
#
#   1. Wait for GPU idle.
#   2. Spawn a FRESH `opencode run` session (no -c/-s => brand new session).
#      Given the results TSV + NOTES.md + the same objective each time, the
#      agent: picks ONE idea, edits, commits ("experiment: <idea>"), appends
#      its guess to NOTES.md, and EXITS. No loop inside the session.
#   3. Driver runs the immutable `bench.sh` on the commit. (The agent never
#      invokes bench.sh or gate_audit.sh, so the gates are structural.)
#   4. If score_steps beats best, driver runs `gate_audit.sh` (fresh-context
#      honesty review). pass -> keep; fail -> reset to last kept.
#   5. bench.sh records the TSV row; the driver records the keep/discard
#      decision in NOTES.md.
#
#   loop.sh [--experiments N] [--model M] [--dry-run]
#     --experiments N  run N experiments then exit (default: until interrupted)
#     --model          model for each fresh session (default
#                      vllm/deepseek-v4-flash -- provider-qualified, required)
#     --dry-run        spawn the agent + commit but skip bench/gate (mechanism
#                      test only -- leaves a candidate commit, no measurement)
#
# Run inside the git worktree where variants live. Results land in
# results_<envs>env_<secs>s.tsv beside the worktree root.
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AR_DIR="$ROOT/autoresearch"
ENVS="${ENVS:-12}"
# NB: NOT "SECONDS" -- bash reserves SECONDS and auto-increments it each second,
# so the wall-clock budget and the results filename drifted upward run-to-run.
# BUDGET_SECS is a fixed, user-settable value shared with bench.sh.
BUDGET_SECS="${BUDGET_SECS:-60}"
RUN_PREFIX="${RUN_PREFIX:-/tmp/ar_wt/run}"
MODEL="${MODEL:-vllm/deepseek-v4-flash}"
STATE="$AR_DIR/.loop_state"
EXPERIMENTS=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --model)       MODEL="$2"; shift 2 ;;
    --run-prefix)  RUN_PREFIX="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)
      echo "usage: loop.sh [--experiments N] [--model M] [--run-prefix DIR] [--dry-run]" >&2
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

RESULTS="$ROOT/results_${ENVS}env_${BUDGET_SECS}s.tsv"
NOTES="$AR_DIR/NOTES.md"
BRANCH="$(git -C "$ROOT" branch --show-current)"
OPENCODE="$(command -v opencode 2>/dev/null || echo "$HOME/.opencode/bin/opencode")"

# Results TSV is the ONLY record of measurement. The driver is the sole writer.
# Kept fixes are scored normally; DISCARDED experiments are recorded too, but
# with score_steps=0 and status discard-low / discard-audit / crash so they are
# traceable (an agent may gain real speed and still be denied by the audit gate,
# or simply not beat the best) yet NEVER surface as a best: best_score() takes
# the max, and a 0 never wins. The desc column explains the discard; the idea
# description itself lives in the agent's NOTES.md entry.
if [ ! -f "$RESULTS" ]; then
  printf 'ts\tcommit\tbranch\tscore_steps\tupdates\tsteps_per_sec\tstatus\tdesc\n' > "$RESULTS"
fi

say() { echo "[loop] $*"; }

last_kept() { cat "$STATE" 2>/dev/null || git -C "$ROOT" rev-parse --short HEAD; }

gpu_idle() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    [ -z "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | grep -v '^$')" ]
  else
    return 0
  fi
}

wait_gpu() {
  say "waiting for GPU idle..."
  while ! gpu_idle; do sleep 10; done
  say "GPU idle"
}

# Same objective text every submission, so every fresh session answers the same
# question (mirrors the audit gate's "same question each time" principle).
objective() {
cat <<EOF
You are an autonomous research agent in a throughput-optimization loop. Your
goal: make the Eclipse RL training/environment roll out MORE training env-steps
within a fixed ${BUDGET_SECS}-second budget, without cheating or breaking real
learning. The score is score_steps (training steps reached in the budget); the
number to beat is the best score_steps in results_${ENVS}env_${BUDGET_SECS}s.tsv.

Do EXACTLY ONE experiment this session, then exit:
1. Read the results TSV (best so far) and NOTES.md (what was tried, where the
   search is going). Pick ONE coherent, non-repeated idea.
2. Edit the code to implement it.
3. git add && git commit -m "experiment: <idea>".
4. Append 2-3 lines to NOTES.md: the idea, why, and your next-guess direction
   for the following session.
5. Exit 0. Do NOT run bench.sh or gate_audit.sh (the driver does that). Do NOT
   start a long loop. Make the change, commit, write NOTES, exit.

Edit anything except autoresearch/{bench.sh,gate_audit.sh,loop.sh,
immutables.sha}. A C++ change needs a build; note it in NOTES.md. Never forge a
step counter or trade away real learning for apparent speed -- a fresh-context
audit gate rejects that.
EOF
}

best_score() {
  # TSV schema: ts commit branch score_steps updates steps_per_sec status desc
  awk -F'\t' 'NR==1{next} $7!="crash"{if($4+0>m)m=$4+0} END{print m+0}' "$RESULTS"
}

exp=0
while [ "$EXPERIMENTS" -eq 0 ] || [ "$exp" -lt "$EXPERIMENTS" ]; do
  exp=$((exp + 1))
  RUN_DIR="${RUN_PREFIX}_${exp}"
  KEPT="$(last_kept)"
  say "=== experiment ${exp}: fresh session -> ${RUN_DIR} (last kept ${KEPT}) ==="

  wait_gpu

  # ---- 1. Agent session: pick idea, edit, commit, write NOTES, exit ---------
  mkdir -p "$(dirname "$RUN_DIR")"
  say "spawning fresh agent session (model=${MODEL})..."
  AGENT_OUT="$("$OPENCODE" run --format json --dir "$ROOT" \
    --model "$MODEL" --title "ar-exp-${exp}" "$(objective)" 2>"$RUN_DIR.agent.err")"
  ARC=$?
  say "agent session exited rc=${ARC}"
  COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
  # The agent commits as "experiment: <idea>", so the subject is the one-liner
  # of what was attempted. Captured before any rollback so every TSV row (keep
  # or discard) can name the attempt.
  MSG="$(git -C "$ROOT" log -1 --format=%s)"
  say "current HEAD: ${COMMIT} (${MSG})"

  if [ $ARC -ne 0 ]; then
    say "agent session failed (rc=${ARC}); leaving working tree as-is"
    continue
  fi

  if [ "$COMMIT" = "$KEPT" ]; then
    say "agent made no new commit; skipping"
    continue
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    say "[dry-run] leaving commit ${COMMIT} as candidate; NOT measuring"
    continue
  fi

  # ---- 2. Driver measures the commit with the immutable bench.sh ------------
  say "running bench.sh..."
  BENCH_OUT="$($AR_DIR/bench.sh "$RUN_DIR" "$ENVS" "$BUDGET_SECS")"
  printf '%s\n' "$BENCH_OUT"
  THIS="$(echo "$BENCH_OUT" | sed -n 's/.*score_steps=\([0-9]*\).*/\1/p' | tail -1)"
  UPDATES="$(echo "$BENCH_OUT" | sed -n 's/.*updates=\([0-9]*\).*/\1/p' | tail -1)"
  SPS="$(echo "$BENCH_OUT" | sed -n 's/.*steps_per_sec=\([0-9]*\).*/\1/p' | tail -1)"
  BEST="$(best_score)"
  say "this=${THIS:-0} best=${BEST:-0}"

  if [ -z "$THIS" ]; then
    say "bench produced no score (crash?) -> discard ${COMMIT}"
    printf '%s\t\t%s\t0\t0\t0\t%s\t%s\n' \
      "$(date +%Y%m%d-%H%M%S)" "$BRANCH" "crash" "$MSG" >> "$RESULTS"
    git -C "$ROOT" reset --hard "$KEPT" 2>/dev/null
    continue
  fi

  if [ "$THIS" -le "$BEST" ]; then
    say "no improvement (${THIS} <= ${BEST}) -> discard ${COMMIT}"
    printf '%s\t\t%s\t0\t0\t0\t%s\t%s\n' \
      "$(date +%Y%m%d-%H%M%S)" "$BRANCH" "discard-low" \
      "${MSG} (steps=${THIS} <= best=${BEST}, not kept)" >> "$RESULTS"
    printf -- '- [%s] DISCARD steps=%s (no improvement, <= best %s): %s\n' \
      "$(date +%H:%M)" "$THIS" "$BEST" "$MSG" >> "$NOTES"
    git -C "$ROOT" reset --hard "$KEPT" 2>/dev/null
    continue
  fi

  # ---- 3. Would-be keep: fresh-context honesty gate --------------------------
  say "new best (${THIS} > ${BEST}); running fresh-context audit gate (base=${KEPT})..."
  "$AR_DIR/gate_audit.sh" --run-dir "$RUN_DIR" --base "$KEPT" --head "$COMMIT" 2>&1
  if [ $? -eq 0 ]; then
    say "AUDIT pass -> KEEP ${COMMIT} as new best"
    printf '%s\n' "$COMMIT" > "$STATE"
    printf -- '- [%s] KEEP steps=%s (audit pass)\n' "$(date +%H:%M)" "$THIS" >> "$NOTES"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%Y%m%d-%H%M%S)" "$COMMIT" "$BRANCH" "${THIS:-0}" "${UPDATES:-0}" "${SPS:-0}" "run" \
      "$MSG" >> "$RESULTS"
  else
    say "AUDIT fail -> discard ${COMMIT} despite score"
    # A genuine speedup rejected by the freshness gate is recorded at score 0 so
    # it is never lost: the rollback drops the code, but the attempt + why stay
    # traceable, and a later session can see (and re-test) what was tried.
    printf '%s\t\t%s\t0\t0\t0\t%s\t%s\n' \
      "$(date +%Y%m%d-%H%M%S)" "$BRANCH" "discard-audit" \
      "${MSG} (steps=${THIS} > best=${BEST} but fresh-context audit failed)" >> "$RESULTS"
    printf -- '- [%s] AUDIT-DENIED steps=%s (was new best %s): %s\n' \
      "$(date +%H:%M)" "$THIS" "$BEST" "$MSG" >> "$NOTES"
    git -C "$ROOT" reset --hard "$KEPT" 2>/dev/null
  fi
done

say "loop finished (experiments=${EXPERIMENTS})"
exit 0
