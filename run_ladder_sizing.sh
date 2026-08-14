#!/usr/bin/env bash
# Measure ladder game throughput, so T1's judgement can be sized instead of guessed.
#
# WHY THIS RUNS BEFORE T1
#   T1 spends ~10 GPU-hours on two arms and is then judged by ONE tournament.
#   roster_ladder is a FULL round-robin: p*(p-1)/2 pairs x 2 directions x
#   games_per_dir games, so the game count grows QUADRATICALLY in snapshots. Two
#   arms at --snapshot_every=100 over 5h reach ~9 and ~15 snapshots; with mains
#   and three bots that is p=29, i.e. 406 pairs and >100k full Eclipse games.
#   Discovering that after the arms have run is the failure mode this prevents.
#   No ladder.json exists anywhere in runs/, so seconds-per-game has never been
#   measured on this box.
#
# WHY IT TRAINS ITS OWN ROSTER FIRST
#   It cannot reuse runs/roster. Those checkpoints are from 2026-08-13 and the
#   encoder layout has moved since (actor.0.tail_mlp 1486 -> 2146,
#   entity_fc 256 -> 128), so load_state_dict raises on SIZE MISMATCH -- which it
#   does even with strict=False, so roster_ladder's "tolerant" loader cannot
#   absorb it either. A few fresh snapshots from the current tree cost ~1 minute
#   and are the only architecturally valid input.
#
#   The ratings this produces are meaningless (the nets are minutes old and
#   nearly identical). Only the WALL CLOCK is being measured.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
export CUDA_VISIBLE_DEVICES=3

KEEP="${1:-3}"          # snapshots kept for the probe tournament
GAMES="${2:-2}"         # games per pair per direction
SEED_DIR="runs/ladder_sizing_seed"
PROBE="runs/ladder_sizing_probe"

# --- 1. a tiny, architecturally-current roster -------------------------------
# Same arch flags as T1 so per-game cost is representative: the net size is what
# drives it. Small env/step counts because only the snapshots are wanted.
if [ ! -f "$SEED_DIR/roster.json" ]; then
  echo "=== building a fresh roster (runs/roster is architecturally stale) ==="
  rm -rf "$SEED_DIR"; mkdir -p "$SEED_DIR"
  .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
    --game='eclipse(players=4)' --seed=1 --cuda \
    --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2 \
    --factored_actions --noleague \
    --ent_coef=0.05 --aux_target_mode=rank --aux_coef=0.1 \
    --num_envs=64 --num_steps=32 --num_workers=8 --num_minibatches=1 \
    --lr_schedule=fixed --amp --nocompile_encoder \
    --snapshot_every=2 --verdict_every_sec=0 \
    --total_timesteps=$((12 * 64 * 32)) --max_seconds=0 \
    --roster_dir="$SEED_DIR" --run_dir="$SEED_DIR" --track=ladder_seed \
    > "$SEED_DIR/train.log" 2>&1
  echo "  rc=$?  snapshots: $(ls "$SEED_DIR" | grep -c '^snap')"
fi

rm -rf "$PROBE"
.venv/bin/python tools/prune_roster.py "$SEED_DIR" "$PROBE" --keep="$KEEP"

# --- 2. time one small tournament -------------------------------------------
t0=$(date +%s.%N)
.venv/bin/python -m open_spiel.python.eclipse.roster_ladder \
  --game='eclipse(players=4)' --cuda --seed=1 \
  --ladder_roster_dir="$PROBE" \
  --ladder_games_per_dir="$GAMES" \
  --ladder_include_bots --ladder_include_heuristic \
  --eval_envs=64 --num_workers=16 \
  --ladder_out="$PROBE/ladder.json" \
  > "$PROBE/ladder.log" 2>&1
rc=$?
t1=$(date +%s.%N)

.venv/bin/python - "$PROBE/ladder.log" "$t0" "$t1" "$rc" "$GAMES" <<'PY'
import re, sys
log, t0, t1, rc, games = sys.argv[1:6]
elapsed = float(t1) - float(t0)
text = open(log, errors="replace").read()
m = re.search(r"round-robin pairs\s*:\s*(\d+)", text)
pol = re.search(r"policies\s*:\s*(\d+)", text)
print(f"rc={rc}  elapsed={elapsed:.1f}s")
if rc != "0" or not m:
    print("!! ladder did not complete -- tail of log:")
    print("\n".join(text.strip().splitlines()[-25:]))
    raise SystemExit(1)
pairs, p = int(m.group(1)), int(pol.group(1))
total_games = pairs * 2 * int(games)
per_game = elapsed / total_games
print(f"policies={p}  pairs={pairs}  games={total_games}  "
      f"=> {per_game:.3f} s/game (includes net loading + the rating fit)")
print()
print("Extrapolated T1 tournament cost (2 arms + Random/Greedy/Heuristic):")
print(f"{'snaps/arm':>10} {'p':>4} {'pairs':>6} "
      f"{'g/dir=32':>11} {'g/dir=64':>11} {'g/dir=128':>11}")
for keep in (2, 3, 4, 5, 8, 11):
    p2 = 2 * (keep + 1) + 3          # keep snapshots + main per arm, + 3 bots
    pr = p2 * (p2 - 1) // 2
    row = "".join(f"{pr * 2 * g * per_game / 3600:>10.1f}h" for g in (32, 64, 128))
    print(f"{keep:>10} {p2:>4} {pr:>6} {row}")
print()
print("Read this as a budget, not a menu: fewer games/dir widens every rating CI,")
print("and T1's pass test needs one arm's LOWER bound to clear the other's UPPER")
print("bound. Cut snapshots per arm first, games/dir last.")
PY
