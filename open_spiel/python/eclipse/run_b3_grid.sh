#!/usr/bin/env bash
# Sprint B3: re-adjudicate the shaping/aux grid on the fixed action space.
#
# The original Sprint-1 grid ranked these cells through a metric that was
# saturated (four cells all reported 8/8 vs both fixed baselines) while the async
# env was silently dropping ~2/3 of every legal action set. Both are fixed, so
# the ranking has to be redone before it can be trusted.
#
# Each cell is gated on wall clock (--max_seconds) rather than steps, so every
# cell in a wave gets the same compute regardless of its throughput, and each
# emits a final batched verdict at the deadline. Logs land under runs/ because
# the previous long run's stdout was lost from /tmp.
set -u

SECS=${SECS:-1080}                 # 18 min per cell
ROOT=runs/sprint_b3
mkdir -p "$ROOT"

common=(
  --num_envs=64 --num_steps=128 --total_timesteps=100000000
  --num_workers=6 --nn_width=64 --nn_depth=2
  --eval_every=20 --verdict_every_sec=300
  --eval_envs=32 --eval_games=64 --snapshot_every=200
  --noprogress --max_seconds="$SECS"
)

launch() {  # launch <cell> <seed> <phi> <aux>
  local cell=$1 seed=$2 phi=$3 aux=$4
  PYTHONPATH=build/open_spiel/python:. .venv/bin/python \
    open_spiel/python/examples/ppo_eclipse.py \
    "${common[@]}" --seed="$seed" --phi="$phi" --aux_target_mode="$aux" \
    --track="sprint_b3/$cell" --roster_dir="$ROOT/${cell}_roster" \
    > "$ROOT/$cell.log" 2>&1 &
  echo "  launched $cell (phi=$phi aux=$aux seed=$seed) pid=$!"
}

summarize() {  # summarize <cells...>
  local spec cell
  for spec in "$@"; do
    cell=${spec%%:*}
    local health verdicts
    health=$(grep -oE "wipeout=[0-9.]+ +survivors=[0-9./]+ +elim_round=[0-9./]+ +vp_all=[0-9.]+ +vp_best=[0-9.]+" \
             "$ROOT/$cell.log" 2>/dev/null | tail -1)
    verdicts=$(grep -E "\[verdict\] vs " "$ROOT/$cell.log" 2>/dev/null | tail -2 \
               | sed -E 's/.*vs +([A-Za-z]+) +utility=([-+0-9.]+) (\[[^]]*\]).*/\1=\2\3/' | tr '\n' ' ')
    if grep -qE "Traceback|Error" "$ROOT/$cell.log" 2>/dev/null; then
      echo "  RESULT $cell : FAILED -- $(grep -mE1 'Traceback|Error' "$ROOT/$cell.log" | head -1)"
    else
      echo "  RESULT $cell : ${verdicts:-no-verdict} | ${health:-no-health}"
    fi
  done
}

wave() {  # wave <label> <cells...>
  local label=$1; shift
  echo "=== wave $label starting: $# cells ==="
  for spec in "$@"; do
    IFS=: read -r cell seed phi aux <<< "$spec"
    launch "$cell" "$seed" "$phi" "$aux"
  done
  wait
  echo "=== wave $label done ==="
  summarize "$@"
}

wave A \
  a_none_rank:1:none:rank \
  a_banked_rank:1:banked:rank \
  a_soft_rank:1:soft:rank \
  a_learned_rank:1:learned:rank \
  a_telescope_rank:1:telescope:rank

wave B \
  b_none_rank:2:none:rank \
  b_banked_rank:2:banked:rank \
  b_soft_rank:2:soft:rank \
  b_learned_rank:2:learned:rank \
  b_telescope_rank:2:telescope:rank

wave C \
  c_none_noaux:1:none:none \
  c_learned_noaux:1:learned:none \
  c_telescope_noaux:1:telescope:none

echo "=== B3 grid complete ==="
