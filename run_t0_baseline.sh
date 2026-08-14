#!/usr/bin/env bash
# T0 -- re-baseline the throughput ladder on BIG (behemoth), 2026-08-14.
#
# WHY THIS EXISTS
#   Every pre-2026-08-13 SPS number in every doc is phase-sum-derived: `sps` and
#   `learn_share` came from a SUM of phase durations, which equals elapsed time
#   only while phases are strictly serialized. Nothing downstream of this file is
#   interpretable until BIG has honest wall-clock numbers on the merged tree
#   (5f5743be). See next_work.md T0 and docs/eclipse_rl_todo.md.
#
# WHAT EACH RUNG ISOLATES
#   The ladder in next_work.md reads "committed config -> device buffer ->
#   --compile_encoder -> 1,024 envs". Those rungs are only informative if the
#   base does NOT already have the two levers, so the base forces them off:
#
#     base      256e  obs_buffer=cpu   nocompile    <- reference
#     devbuf    256e  obs_buffer=cuda  nocompile    <- isolates the device buffer
#     compile   256e  obs_buffer=cuda  compile      <- isolates torch.compile;
#                                                      this rung IS the committed
#                                                      run_v2.sh config on BIG,
#                                                      whose obs_buffer=auto
#                                                      resolves to cuda here
#     envs1024  1024e obs_buffer=cuda  compile      <- scale
#     league    256e  obs_buffer=cuda  compile      <- league BOOKKEEPING only
#
#   Read the `league` rung narrowly. It runs --snapshot_every=0, so the roster
#   never grows past `main` and every lineup draws the same policy: K=1 distinct
#   policies in the batch. It therefore prices _refresh_lineups and the trainable
#   -mask bookkeeping, and NOT the per-policy act cost, which is the expensive
#   half of league play. That half is K-dependent and superlinear, and it is
#   measured properly by tools/t3_opponent_curve.py. Do not quote this rung as
#   "the cost of --league".
#
#   ROWS PER MINIBATCH IS HELD AT 8,192 across the env-count change (mb=4 at 256
#   envs, mb=16 at 1,024), because that is the most efficient point measured on
#   BIG and because letting it drift would confound the env-count rung with a
#   learn-cost change. num_workers is held at 16 for the same reason -- worker
#   count gets its own sweep in T5, not a free ride inside this ladder.
#
#   Rungs 1-4 are --noleague, so act runs at K=1 distinct policies. That is the
#   right reference for T3, which measures act cost as a ratio against K=1. The
#   `league` rung then prices the production cap against rung 3.
#
# GPU: 3 only. GPUs 1-2 hold two resident vLLM workers at 96.8 GB each; 0 drives
# the display. Arms run SEQUENTIALLY -- two arms once thrashed the card to ~45
# SPS and three OOM'd.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
export CUDA_VISIBLE_DEVICES=3

OUT="${OUT_DIR:-runs/t0_baseline}"
# Updates per rung, not seconds: the ladder compares per-update costs, so every
# rung must do the same amount of work. 24 at timing_every=4 gives six prints
# (u0,4,8,12,16,20); u0 -- which carries process warmup and the torch.compile
# trace -- is discarded outright, leaving five hot samples. Two samples was too
# thin to see whether a compile rung had finished warming up.
UPDATES="${UPDATES:-24}"
TIMING_EVERY=4
mkdir -p "$OUT"

rung() {  # rung <label> <num_envs> <num_minibatches> <extra flags...>
  local label="$1" envs="$2" mb="$3"; shift 3
  local log="$OUT/$label.log"
  local dir="$OUT/$label"
  rm -rf "$dir"; mkdir -p "$dir"

  # total_timesteps, not max_seconds: a fixed update count keeps the rungs
  # comparable. The timeout is a safety net sized off the slowest plausible
  # rung, never the stop condition.
  local total=$((UPDATES * envs * 128))

  echo "### rung=$label envs=$envs mb=$mb extra='$*' updates=$UPDATES" | tee -a "$OUT/summary.txt"
  local t0 t1
  t0=$(date +%s.%N)
  timeout 5400 .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
    --game='eclipse(players=4)' --seed=1 --cuda \
    --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2 \
    --factored_actions \
    --ent_coef=0.05 --aux_target_mode=rank --aux_coef=0.1 \
    --num_envs="$envs" --num_steps=128 --num_workers=16 --num_minibatches="$mb" \
    --lr_schedule=fixed --amp \
    --snapshot_every=0 --timing --timing_every="$TIMING_EVERY" \
    --verdict_every_sec=0 \
    --total_timesteps="$total" --max_seconds=0 \
    --roster_dir="$dir" --run_dir="$dir" --track="t0_$label" \
    "$@" > "$log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)

  # Real throughput is final_steps/elapsed, never the logged sps -- see the
  # Operational section of next_work.md.
  .venv/bin/python tools/parse_t0.py "$label" "$log" "$t0" "$t1" "$rc" "$total" \
    | tee -a "$OUT/summary.txt"
  echo | tee -a "$OUT/summary.txt"
}

: > "$OUT/summary.txt"
echo "=== T0 throughput ladder, BIG/behemoth, GPU 3, $(date +%F' '%H:%M:%S) ===" \
  | tee -a "$OUT/summary.txt"
echo "commit $(git rev-parse --short HEAD)  updates/rung=$UPDATES" \
  | tee -a "$OUT/summary.txt"
echo | tee -a "$OUT/summary.txt"

rung base     256  4  --noleague --obs_buffer_device=cpu  --nocompile_encoder
rung devbuf   256  4  --noleague --obs_buffer_device=cuda --nocompile_encoder
rung compile  256  4  --noleague --obs_buffer_device=cuda --compile_encoder
rung envs1024 1024 16 --noleague --obs_buffer_device=cuda --compile_encoder
rung league   256  4  --league --max_live_opponents=4 \
                      --obs_buffer_device=cuda --compile_encoder

echo "=== done $(date +%F' '%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
