#!/usr/bin/env bash
# Long run: the 1.2204 baseline config, at 1.57x throughput, for 8-12h.
#
# Purpose: every experiment so far reached only ~16M steps, while the historical
# runs went to 400M. This asks the one question those short arms could not:
# where does the plateau actually sit when the agent gets ~150M steps?
#
# Config = the baseline that produced runs/roster/snap_u573 (rating 1.2204),
# plus the two throughput flags that measured a real gain (--amp 3337->4328,
# +--compile_encoder ->5251 envstep/s; --channels_last was a no-op so it is NOT
# used), plus the new --gamma default of 0.998.
#
# --ent_coef=0.05 (NOT the 0.01 default): wave 3's single-variable ladder ranked
# it the best policy tested -- w3_ent:main 1.1071 [1.0817,1.1375] vs
# baseline:u573 1.0553 [1.0328,1.0810], i.e. its CI lower bound clears the
# baseline's upper bound. Margin is only 0.0007, so treat it as "best available"
# rather than proven. Note it is the ONLY single change that beat baseline:
# --separate_critic and --factored_actions each rated BELOW it, --nn_width=256
# was neutral, and the bundle of all of them was the worst net on the board.
#
# CHANGED vs the 1.2204 baseline, so attribute carefully if this underperforms:
#   --ent_coef 0.01 -> 0.05 (wave 3 winner, marginal)
#   --gamma 0.99 -> 0.998   (0.99^80 ~ 0.45 discarded half the terminal signal)
#   --amp, --compile_encoder (throughput only; bf16 verified not to perturb the
#                             PPO ratio -- first-epoch clipfrac 0.0 vs 0.0)
#
# 128 envs is the hard ceiling on this 12GB card -- 192 already OOMs. Do not
# raise it here; big-batch work belongs on the cluster.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

SECS="${1:-43200}"          # 12h default
DIR="${2:-runs/long_v2}"

mkdir -p "$DIR"
echo "=== long_v2 start $(date +%F' '%H:%M:%S)  max_seconds=${SECS}  dir=${DIR} ==="
exec .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
  --game='eclipse(players=4)' --seed=1 --cuda \
  --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2 \
  --ent_coef=0.05 --aux_target_mode=rank --aux_coef=0.1 \
  --num_envs=128 --num_steps=128 --num_workers=16 --num_minibatches=4 \
  --lr_schedule=fixed \
  --amp --compile_encoder \
  --snapshot_every=100 \
  --total_timesteps=1000000000 --max_seconds="$SECS" \
  --roster_dir="$DIR" --track=long_v2 \
  >> "$DIR/train.log" 2>&1
