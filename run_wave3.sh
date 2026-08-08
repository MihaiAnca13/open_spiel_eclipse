#!/usr/bin/env bash
# Wave 3: SINGLE-VARIABLE attribution against the baseline config.
#
# Why: wave 1/2 bundled separate_critic + ent_coef=0.05 + width 256 +
# factored_actions into one arm, and the bundle REGRESSED (1.22 -> 0.99 on the
# ladder). With everything changed at once the cause is unattributable. Every
# arm here changes exactly ONE thing from the baseline that scored 1.2204.
#
# Control = runs/roster (baseline, NOT relaunched). Its config, recovered from
# /tmp/opencode/real_run.sh + arch.json:
#   --encoder=spatial --nn_width=64 --nn_depth=2 --nn_activation=tanh
#   --ent_coef=0.01 --aux_target_mode=rank --aux_coef=0.1
#   --num_envs=128 --num_steps=128 --num_minibatches=4  (default 4, NOT 8)
# Baseline's LR decayed only 6% over its run, so --lr_schedule=fixed matches it.
#
# Arms run SEQUENTIALLY -- one arm saturates the 11.6 GiB card (see run_wave1.sh).
# No pgrep wait-loops here: a loop gated on a string that appears in its own
# command line matches itself and spins forever (cost ~1h of idle GPU earlier).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

SECS="${1:-5400}"

COMMON=(
  --game='eclipse(players=4)' --seed=1 --cuda
  --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2
  --ent_coef=0.01 --aux_target_mode=rank --aux_coef=0.1
  --num_envs=128 --num_steps=128 --num_workers=16 --num_minibatches=4
  --snapshot_every=25 --lr_schedule=fixed
  --total_timesteps=1000000000 --max_seconds="$SECS"
)

run_arm() {  # run_arm <name> <extra flags...>
  local name="$1"; shift
  mkdir -p "runs/w3_${name}"
  echo "=== arm ${name} start $(date +%H:%M:%S) ($*) ==="
  .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
    "${COMMON[@]}" --roster_dir="runs/w3_${name}" --track="w3_${name}" "$@" \
    > "runs/w3_${name}/train.log" 2>&1 || echo "  arm ${name} EXITED NONZERO"
  echo "=== arm ${name} done  $(date +%H:%M:%S) ==="
}

# Order: most-informative first, so an interruption still yields signal.
run_arm sep    --separate_critic                        # the aux-owns-trunk fix alone
run_arm ent    --ent_coef=0.05                          # prime suspect for the regression
run_arm fact   --factored_actions                       # control for the pointer test
run_arm ptr    --factored_actions --spatial_pointer     # the clean single-variable pointer test
run_arm wide   --nn_width=256 --nn_depth=3              # capacity alone
echo "wave 3 complete"
