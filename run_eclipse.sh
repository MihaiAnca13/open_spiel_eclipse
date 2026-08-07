#!/usr/bin/env bash
# Launch the 24714-stack spatial PPO run writing to runs/roster (overwrite).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="build/open_spiel/python:$PWD"
exec .venv/bin/python -m open_spiel.python.examples.ppo_eclipse "$@"
