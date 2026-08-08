#!/usr/bin/env bash
# Wave 2: the board-blindness fix itself.
#
# Arm P = arm C's exact flag set PLUS --spatial_pointer, so C vs P isolates the
# pointer head and nothing else. Judge both against baseline `runs/roster` on
# the merged ladder (--ladder_roster_dir takes a comma-separated list).
#
# Runs ONLY after any in-flight arm exits -- one arm at a time, see run_wave1.sh
# for the measured VRAM ceiling.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

SECS="${1:-5400}"

while pgrep -f "roster_dir=runs/exp1_combined" > /dev/null; do sleep 30; done
echo "=== arm pointer starting ($(date +%H:%M:%S), ${SECS}s) ==="
mkdir -p runs/exp1_pointer
.venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
  --game='eclipse(players=4)' --seed=1 --cuda \
  --encoder=spatial --nn_activation=tanh \
  --num_envs=128 --num_steps=128 --num_workers=16 --num_minibatches=8 \
  --snapshot_every=25 --aux_target_mode=rank --aux_coef=0.1 \
  --lr_schedule=fixed \
  --total_timesteps=1000000000 --max_seconds="$SECS" \
  --nn_width=256 --nn_depth=3 --separate_critic --ent_coef=0.05 \
  --factored_actions --spatial_pointer \
  --roster_dir=runs/exp1_pointer --track=exp1_pointer \
  > runs/exp1_pointer/train.log 2>&1
echo "=== arm pointer done ($(date +%H:%M:%S)) ==="
