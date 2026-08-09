#!/usr/bin/env bash
# Judge wave 1/2 arms against the plateau baseline on ONE merged ladder.
#
# A full merge of every snapshot in every roster is ~30 policies = 435 pairs,
# which is an order of magnitude more games than needed. So this builds trimmed
# single-purpose roster dirs holding only the policies that matter, then runs
# one tournament over all of them plus the shared Random/Greedy/Heuristic
# anchors.
#
# Read policies[].rating / rating_ci from the output -- NOT the elo field.
# An arm beats the plateau only if its rating_ci LOWER bound clears
# roster:snap_u573's rating_ci UPPER bound.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

GAMES="${1:-128}"
OUT="${2:-runs/wave_ladder.json}"
TRIM="runs/_judge"

# Snapshot numbering is NOT predictable -- PolicyRoster.prune keeps 4 recent +
# 4 spaced, so e.g. exp1_combined kept u25,u50,u725..u833 and has no u400.
# Auto-discover instead of hardcoding: earliest snapshot, a middle one, + main.
pick() {  # pick <destdir> <srcdir>
  local dest="$1" src="$2"
  rm -rf "$dest"; mkdir -p "$dest"
  cp "$src/arch.json" "$dest/arch.json"
  local snaps=() entries="" chosen=()
  while IFS= read -r f; do snaps+=("$f"); done < <(
    ls "$src"/snap_u*.pt 2>/dev/null \
      | sed 's/.*snap_u\([0-9]*\)\.pt/\1 &/' | sort -n | cut -d' ' -f2)
  # Middle snapshot + main only. With 5+ rosters, taking 4 policies each blows
  # the tournament up to ~253 pairs; 2 each keeps it near 78. For baseline this
  # lands on snap_u573 -- exactly the plateau reference the go/no-go uses.
  local n=${#snaps[@]}
  if [ "$n" -gt 0 ]; then chosen+=("${snaps[$((n/2))]}"); fi
  for f in "${chosen[@]}"; do
    local upd alias
    upd="$(basename "$f" | grep -oE '[0-9]+')"
    alias="u${upd}"
    cp "$f" "$dest/$alias.pt"
    entries="${entries}{\"policy_id\":\"$alias\",\"role\":\"snapshot\",\"birth_update\":$upd,\"path\":\"$REPO_ROOT/$dest/$alias.pt\",\"win_rate\":null},"
  done
  if [ -f "$src/main.pt" ]; then
    # Use main's REAL birth_update from the source roster. main is normally the
    # same weights as the newest snapshot, and matching births lets the ladder's
    # identical-policy mirror assertion actually run as a correctness check
    # instead of being silently skipped by a made-up number.
    local mainupd
    mainupd="$(python3 -c "
import json,sys
try:
    r=json.load(open('$src/roster.json'))
    print(next(int(e['birth_update']) for e in r if e.get('role')=='main'))
except Exception:
    print(99999)")"
    cp "$src/main.pt" "$dest/main.pt"
    entries="${entries}{\"policy_id\":\"main\",\"role\":\"main\",\"birth_update\":$mainupd,\"path\":\"$REPO_ROOT/$dest/main.pt\",\"win_rate\":null},"
  fi
  printf '[%s]\n' "${entries%,}" > "$dest/roster.json"
  echo "  $(basename "$dest"): $(ls "$dest"/*.pt | wc -l) policies"
}

echo "building trimmed rosters..."
pick "$TRIM/baseline" runs/roster
DIRS="$TRIM/baseline"
# Every arm dir that has a finished main.pt joins the same tournament.
for src in runs/exp1_combined runs/exp1_pointer \
           runs/w3_sep runs/w3_ent runs/w3_fact runs/w3_ptr runs/w3_wide \
           runs/long8h; do
  [ -f "$src/main.pt" ] || continue
  tag="$(basename "$src")"
  pick "$TRIM/$tag" "$src"
  DIRS="$DIRS,$TRIM/$tag"
done

echo "rosters: $DIRS"
echo "games/dir=$GAMES  ->  out=$OUT"
.venv/bin/python -m open_spiel.python.eclipse.roster_ladder \
  --ladder_roster_dir="$DIRS" \
  --ladder_games_per_dir="$GAMES" \
  --eval_envs=64 --num_workers=16 --seed=1 --cuda \
  --ladder_out="$OUT" 2>&1 | tail -40
