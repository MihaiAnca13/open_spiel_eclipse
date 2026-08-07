#!/usr/bin/env bash
# Run the roster ladder on runs/roster (32 games/pair), with the heuristic bot.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="build/open_spiel/python:$PWD"
exec .venv/bin/python -m open_spiel.python.eclipse.roster_ladder "$@"
