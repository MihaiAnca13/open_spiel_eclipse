#!/usr/bin/env bash
# Wave 1: attribute the flag-level fixes behind the elo-1080 plateau.
#
# CTRL is runs/roster (the existing plateau baseline) -- NOT relaunched.
# Judge afterwards with the multi-roster ladder against baseline:snap_u573.
#
# ARMS RUN SEQUENTIALLY, ON PURPOSE. Measured on this box (RTX 4080 Laptop,
# 11.6 GiB usable): a width-64 arm holds ~4.7 GiB and the width-256 arm more.
# Three arms OOM'd; two arms pinned the GPU at 11.8/12.3 GiB and thrashed to
# ~45 envstep/s each. One arm solo runs at ~2950 envstep/s -- so running them
# back to back is ~30x more productive than contending. Do not "parallelize"
# this without first shrinking per-arm memory (fp16 rollout storage or fewer
# envs); wall-clock parallelism is not the constraint, VRAM is.
#
# Guardrails encoded here:
#  --lr_schedule=fixed : an unset value now resolves to the never-tested 'kl'
#                        controller, which drives LR to lr_max in ~5 updates.
#  --num_minibatches=8 : trims the learn-phase activation peak.
#
# Arm B (--ent_coef=0.05 alone) is deferred: arm C already contains that
# change, so B is only needed if C underperforms A.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

SECS="${1:-5400}"   # per arm; 90min x 2 arms = 3h total

COMMON=(
  --game='eclipse(players=4)' --seed=1 --cuda
  --encoder=spatial --nn_activation=tanh
  --num_envs=128 --num_steps=128 --num_workers=16 --num_minibatches=8
  --snapshot_every=25 --aux_target_mode=rank --aux_coef=0.1
  --lr_schedule=fixed
  --total_timesteps=1000000000 --max_seconds="$SECS"
)

run_arm() {  # run_arm <name> <extra flags...>
  local name="$1"; shift
  mkdir -p "runs/exp1_${name}"
  echo "=== arm ${name} starting ($(date +%H:%M:%S), ${SECS}s) ==="
  .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
    "${COMMON[@]}" --roster_dir="runs/exp1_${name}" --track="exp1_${name}" "$@" \
    > "runs/exp1_${name}/train.log" 2>&1
  echo "=== arm ${name} done ($(date +%H:%M:%S)) ==="
}

run_arm combined  --nn_width=256 --nn_depth=3 --separate_critic --ent_coef=0.05 --factored_actions
run_arm sepcritic --nn_width=64  --nn_depth=2 --separate_critic
echo "wave 1 complete"
