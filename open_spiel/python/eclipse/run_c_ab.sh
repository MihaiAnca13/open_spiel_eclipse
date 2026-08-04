#!/usr/bin/env bash
# Sprint C A/B: each architectural change measured against the A+B baseline on
# the same held-out eval boards, with equal wall-clock per cell.
#
# Wall-clock gating (not step gating) is the right comparison here: C1 and C2
# change the cost per step, and the question is which spends a fixed compute
# budget better, not which is better per gradient step.
#
# PHI is chosen from the B3 result; override via the environment.
set -u

SECS=${SECS:-1080}
PHI=${PHI:-telescope}
AUX=${AUX:-rank}
SEEDS=${SEEDS:-"1 2"}
ROOT=runs/sprint_c
mkdir -p "$ROOT"

common=(
  --num_envs=64 --num_steps=128 --total_timesteps=100000000
  --num_workers=6 --eval_every=20 --verdict_every_sec=300
  --eval_envs=32 --eval_games=64 --snapshot_every=200
  --noprogress --max_seconds="$SECS" --phi="$PHI" --aux_target_mode="$AUX"
)

# cell -> extra flags
declare -A CELLS=(
  [base]="--nn_width=64 --nn_depth=2"
  [c1_capacity]="--nn_width=512 --nn_depth=3 --nn_norm --nn_activation=gelu --separate_critic"
  [c2_factored]="--nn_width=64 --nn_depth=2 --factored_actions"
  [c3_distcritic]="--nn_width=64 --nn_depth=2 --rank_ce_coef=0.5"
  [c_all]="--nn_width=512 --nn_depth=3 --nn_norm --nn_activation=gelu --separate_critic --factored_actions --rank_ce_coef=0.5"
)
ORDER=(base c1_capacity c2_factored c3_distcritic c_all)

summarize() {
  local seed=$1 cell f health rnd grd
  for cell in "${ORDER[@]}"; do
    f="$ROOT/${cell}_s${seed}.log"
    if grep -qE "Traceback|Error" "$f" 2>/dev/null; then
      echo "  RESULT ${cell}_s${seed} : FAILED -- $(grep -mE1 'Traceback|Error' "$f" | head -1)"
      continue
    fi
    health=$(grep -oE "wipeout=[0-9.]+ +survivors=[0-9./]+ +elim_round=[0-9./]+ +vp_all=[0-9.]+" "$f" 2>/dev/null | tail -1)
    rnd=$(grep -E "\[verdict\] vs Random" "$f" 2>/dev/null | tail -1 | grep -oE "utility=[-+0-9.]+ \[[^]]*\]")
    grd=$(grep -E "\[verdict\] vs Greedy" "$f" 2>/dev/null | tail -1 | grep -oE "utility=[-+0-9.]+ \[[^]]*\]")
    echo "  RESULT ${cell}_s${seed} : Random ${rnd:-n/a} | Greedy ${grd:-n/a} | ${health:-no-health}"
  done
}

for seed in $SEEDS; do
  echo "=== seed $seed starting (phi=$PHI aux=$AUX, ${SECS}s per cell) ==="
  for cell in "${ORDER[@]}"; do
    # shellcheck disable=SC2086
    PYTHONPATH=build/open_spiel/python:. .venv/bin/python \
      open_spiel/python/examples/ppo_eclipse.py \
      "${common[@]}" ${CELLS[$cell]} --seed="$seed" \
      --track="sprint_c/${cell}_s${seed}" \
      --roster_dir="$ROOT/${cell}_s${seed}_roster" \
      > "$ROOT/${cell}_s${seed}.log" 2>&1 &
    echo "  launched ${cell}_s${seed}"
  done
  wait
  echo "=== seed $seed done ==="
  summarize "$seed"
done

echo "=== C A/B complete ==="
