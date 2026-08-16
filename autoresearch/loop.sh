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
#     --model          model for each fresh session (default deepseek-v4-flash)
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
SECONDS="${SECONDS:-60}"
RUN_PREFIX="${RUN_PREFIX:-/tmp/ar_wt/run}"
MODEL="${MODEL:-deepseek-v4-flash}"
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

RESULTS="$ROOT/results_${ENVS}env_${SECONDS}s.tsv"
NOTES="$AR_DIR/NOTES.md"
OPENCODE="$(command -v opencode 2>/dev/null || echo "$HOME/.opencode/bin/opencode")"

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
within a fixed ${SECONDS}-second budget, without cheating or breaking real
learning. The score is score_steps (training steps reached in the budget); the
number to beat is the best score_steps in results_${ENVS}env_${SECONDS}s.tsv.

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
  say "current HEAD: ${COMMIT}"

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
  BENCH_OUT="$($AR_DIR/bench.sh "$RUN_DIR" "$ENVS" "$SECONDS")"
  printf '%s\n' "$BENCH_OUT"
  THIS="$(echo "$BENCH_OUT" | sed -n 's/.*score_steps=\([0-9]*\).*/\1/p' | tail -1)"
  BEST="$(best_score)"
  say "this=${THIS:-0} best=${BEST:-0}"

  if [ -z "$THIS" ] || [ "$THIS" -le "$BEST" ]; then
    say "no improvement (${THIS:-0} <= ${BEST:-0}) -> discard ${COMMIT}"
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
  else
    say "AUDIT fail -> discard ${COMMIT} despite score"
    git -C "$ROOT" reset --hard "$KEPT" 2>/dev/null
    printf -- '- [%s] DISCARD steps=%s (audit fail)\n' "$(date +%H:%M)" "$THIS" >> "$NOTES"
  fi
done

say "loop finished (experiments=${EXPERIMENTS})"
exit 0
