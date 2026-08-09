#!/usr/bin/env bash
# 8h training run after the stability/throughput sprint (2026-08-08).
#
# What changed vs the 47.4M-step long_v2 that crashed:
#   - _gumbel_sample no longer leaks the num_actions sentinel (OOB gather fixed).
#   - The obs rollout buffer moved off GPU, so --num_envs can exceed 128.
#   - --verdict_every_sec default is now 7200 (evals ate 71% of wall-clock).
#   - The act path no longer re-runs the encoder for value when the critic
#     shares the trunk.
#
# Config rationale:
#   - 256 envs: the CPU-obs offload lifts the 128-env cap (measured no-OOM to
#     384); 256 doubles the batch (16,384 -> 32,768 timesteps) at ~3920 SPS,
#     testing hide-and-seek's batch-size hypothesis while keeping throughput.
#   - --ent_coef=0.05: wave-3 single-variable winner (marginal, CI-clear).
#   - --gamma default 0.998 (was 0.99): retains the terminal signal over the
#     ~80-decision per-seat subsequence; untested on the ladder before now.
#   - --amp --compile_encoder: 1.57x wall-clock throughput win.
#   - --lr_schedule=fixed: never use the unvalidated kl controller.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

SECS="${1:-28800}"          # 8h default
DIR="${2:-runs/long8h}"
mkdir -p "$DIR"

echo "=== long8h start $(date +%F' '%H:%M:%S)  max_seconds=${SECS}  dir=${DIR} ===" > "runs/long8h_launch.log"
exec .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
  --game='eclipse(players=4)' --seed=1 --cuda \
  --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2 \
  --ent_coef=0.05 --aux_target_mode=rank --aux_coef=0.1 \
  --num_envs=256 --num_steps=128 --num_workers=16 --num_minibatches=4 \
  --lr_schedule=fixed \
  --amp --compile_encoder \
  --snapshot_every=100 \
  --total_timesteps=1000000000 --max_seconds="$SECS" \
  --roster_dir="$DIR" --track=long8h \
  >> "$DIR/train.log" 2>&1
