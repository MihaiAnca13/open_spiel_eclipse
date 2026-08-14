#!/usr/bin/env bash
# T1's judgement: ONE tournament rating BOTH arms' mid and final snapshots.
#
# One tournament, not two: ratings are NOT comparable across ladder runs. Rating
# each arm separately and comparing the numbers is the single easiest way to get
# this experiment wrong. roster_ladder tags every policy with its roster dir's
# basename ("t1_ue1:snap_0300") and pins Random at 0, so both arms land on one
# scale in one fit.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
export CUDA_VISIBLE_DEVICES=3

TAG="${1:-}"
GAMES="${2:-128}"
OUT="runs/t1_ladder${TAG}.json"

A="runs/t1_ue4${TAG}"
B="runs/t1_ue1${TAG}"
for d in "$A" "$B"; do
  [ -d "$d" ] || { echo "missing roster dir $d" >&2; exit 2; }
done

.venv/bin/python -m open_spiel.python.eclipse.roster_ladder \
  --game='eclipse(players=4)' --cuda --seed=1 \
  --ladder_roster_dir="$A,$B" \
  --ladder_games_per_dir="$GAMES" \
  --ladder_include_bots --ladder_include_heuristic \
  --eval_envs=64 --num_workers=16 \
  --ladder_out="$OUT" \
  2>&1 | tee "runs/t1_ladder${TAG}.log"

echo
.venv/bin/python tools/t1_verdict.py "$OUT"
