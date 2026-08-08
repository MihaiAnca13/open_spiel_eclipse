#!/usr/bin/env bash
# Item 5: continued long run on the fixed 24714/spatial stack, aux=rank.
# Resumes runs/roster main (past the u573 plateau) and keeps training,
# writing later snapshots + a final force-snapshot/verdict at the deadline.
# Judge afterwards on the ladder (snap_u573 = plateau baseline vs continued main).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
exec .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
  --resume=main \
  --roster_dir=runs/roster \
  --num_envs=128 \
  --num_workers=16 \
  --num_steps=128 \
  --total_timesteps=1000000000 \
  --snapshot_every=100 \
  --max_seconds="${1:-3600}" \
  --track=item5_long \
  "${@:2}"
