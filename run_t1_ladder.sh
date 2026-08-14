#!/usr/bin/env bash
# T1's judgement: ONE tournament rating BOTH arms' mid and final snapshots.
#
# One tournament, not two: ratings are NOT comparable across ladder runs. Rating
# each arm separately and comparing the numbers is the single easiest way to get
# this experiment wrong. roster_ladder tags every policy with its roster dir's
# basename ("t1_ue1:snap_u1200") and pins Random at 0, so both arms land on one
# scale in one fit.
#
# WHY IT PRUNES FIRST
#   roster_ladder is a full round-robin -- p*(p-1)/2 pairs x 2 directions x
#   games_per_dir games -- so cost is QUADRATIC in policy count, and a game costs
#   a measured 1.387 s. Both arms' full rosters (8 snapshots + main each) is
#   p=21, 210 pairs, ~10.4h at games_per_dir=64. Pruned to 4 snapshots + main it
#   is p=13, 78 pairs, ~3.9h.
#
#   Cut snapshots first and games/dir last: fewer games widens every rating CI,
#   and this test needs one arm's LOWER bound to clear the other's UPPER bound, so
#   games buy exactly the resolution the verdict depends on. Dropping
#   near-adjacent snapshots costs almost nothing by comparison -- adjacent
#   snapshots inside one run are nearly indistinguishable, which is why
#   --ladder_min_sep exists at all.
#
#   The pruned copies live in runs/t1_rated/<basename> so the BASENAME is
#   unchanged: roster_ladder tags policies by it and tools/t1_verdict.py matches
#   arms on that tag, so renaming the directory would silently break the verdict.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
export CUDA_VISIBLE_DEVICES=3

TAG="${1:-}"
GAMES="${2:-64}"
KEEP="${3:-4}"
OUT="runs/t1_ladder${TAG}.json"
RATED="runs/t1_rated${TAG}"

A_SRC="runs/t1_ue4${TAG}"
B_SRC="runs/t1_ue1${TAG}"
for d in "$A_SRC" "$B_SRC"; do
  [ -f "$d/roster.json" ] || { echo "missing roster $d/roster.json" >&2; exit 2; }
done

rm -rf "$RATED"; mkdir -p "$RATED"
A="$RATED/$(basename "$A_SRC")"
B="$RATED/$(basename "$B_SRC")"
.venv/bin/python tools/prune_roster.py "$A_SRC" "$A" --keep="$KEEP"
.venv/bin/python tools/prune_roster.py "$B_SRC" "$B" --keep="$KEEP"

echo
echo "=== ladder start $(date +%F' '%H:%M:%S)  games_per_dir=$GAMES keep=$KEEP ==="
.venv/bin/python -m open_spiel.python.eclipse.roster_ladder \
  --game='eclipse(players=4)' --cuda --seed=1 \
  --ladder_roster_dir="$A,$B" \
  --ladder_games_per_dir="$GAMES" \
  --ladder_include_bots --ladder_include_heuristic \
  --eval_envs=64 --num_workers=16 \
  --ladder_out="$OUT" \
  2>&1 | tee "runs/t1_ladder${TAG}.log"
echo "=== ladder done $(date +%F' '%H:%M:%S) ==="

echo
.venv/bin/python tools/t1_verdict.py "$OUT"
