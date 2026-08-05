#!/usr/bin/env bash
# Rate a policy roster's snapshots against each other on the Random=0 scale.
#
# One roster per invocation. Arch flags must match the roster's training run
# (or a roster arch.json is preferred by the ladder when present). Random is
# pinned at rating 0, so ratings across rosters are comparable.
set -u
export PYTHONPATH=build/open_spiel/python:.
VENV=.venv/bin/python
LADDER=open_spiel/python/eclipse/roster_ladder.py

GAMES=${GAMES:-128}
ENVS=${ENVS:-32}
WORKERS=${WORKERS:-6}
SEED=${SEED:-1}

# c_all_s1 : Sprint C "all three" cell (the headline C combo).
"$VENV" "$LADDER" \
  --ladder_roster_dir=runs/sprint_c/c_all_s1_roster \
  --ladder_games_per_dir="$GAMES" --eval_envs="$ENVS" --num_workers="$WORKERS" \
  --seed="$SEED" \
  --nn_width=512 --nn_depth=3 --nn_norm --nn_activation=gelu \
  --separate_critic --factored_actions --aux_target_mode=rank

# a_soft_rank : Sprint B3's chosen default (phi=soft), seed 1.
"$VENV" "$LADDER" \
  --ladder_roster_dir=runs/sprint_b3/a_soft_rank_roster \
  --ladder_games_per_dir="$GAMES" --eval_envs="$ENVS" --num_workers="$WORKERS" \
  --seed="$SEED" \
  --nn_width=64 --nn_depth=2 --aux_target_mode=rank

echo "=== ladder complete ==="
