#!/usr/bin/env bash
# T4 -- price network capacity on BIG (behemoth), GPU 3.
#
# WHY THIS EXISTS
#   The trained agent has 545,587 TRAINABLE parameters at --nn_width=64
#   --nn_depth=2. main.pt appears to hold 1,897,461, but two inflations stack:
#   `shared`, `critic_trunk`, `critic.0` and `actor.0` are four state_dict
#   aliases of ONE shared encoder (verified bit-identical, 2.89x), and 111,170
#   of what remains are int64 action-factorization lookup tables -- buffers, not
#   weights. tools/count_params.py prints all three readings.
#
#   Every quality lever this project has tested was tested inside that box, and
#   every one came back null: update_epochs (T1, both arms flat/regressing), the
#   spatial pointer head (null on two ladders), --aux_coef (A/B at 0.01, no
#   change to approx_kl/clipfrac), encoder choice (null, and corroborated by
#   hide-and-seek's ~9% encoder effect). Meanwhile BOTH T1 arms stop improving
#   by ~update 100-300 and every training metric -- vp_all, approx_kl, clipfrac,
#   explained_var, entropy -- goes flat and stays flat for 200M+ steps.
#
#   Capacity has never been swept on this box. This script prices it before a
#   multi-day run commits to a width, because the cost is the only part of the
#   bet that can be measured cheaply.
#
# WHAT IS AND IS NOT MEASURED
#   THIS MEASURES COST ONLY -- seconds per update at each width. It says nothing
#   about whether more capacity trains a stronger agent; that is what the long
#   run answers, on ladder rating. Do not read a throughput table as a quality
#   result. (This file exists because reading one number as another is how three
#   "the doc is wrong on BIG" findings turned out to be instrumentation bugs.)
#
#   --noleague on every rung, matching T0's rungs 1-4: league act cost scales
#   with the number of DISTINCT policies in the batch, which drifts with roster
#   growth and would confound a width comparison. Price the league cap
#   separately from T3's table (~6.5% of update at K=5).
#
#   Rows per minibatch is held at 8,192 (1,024 envs, mb=16) and num_workers at
#   16, both for the same reason T0 holds them: a rung must vary one thing.
#
# GPU: 3 only. GPUs 1-2 hold two resident vLLM workers at 96.8 GB each; 0 drives
# the display. Rungs run SEQUENTIALLY -- two concurrent arms once thrashed the
# card to ~45 SPS and three OOM'd.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
export CUDA_VISIBLE_DEVICES=3

OUT="${OUT_DIR:-runs/t4_capacity}"
# Same shape as T0: 24 updates at timing_every=4 gives six prints, of which u0
# (process warmup + the ~29 s torch.compile trace) is discarded, leaving five
# hot samples. Compile warms up for ~10 updates past u0, so STEADY -- the last
# three -- is the number to compare rungs on.
UPDATES="${UPDATES:-24}"
TIMING_EVERY=4
mkdir -p "$OUT"

rung() {  # rung <label> <width> <depth>
  local label="$1" width="$2" depth="$3"
  local log="$OUT/$label.log"
  local dir="$OUT/$label"
  rm -rf "$dir"; mkdir -p "$dir"

  local envs=1024
  local total=$((UPDATES * envs * 128))

  echo "### rung=$label width=$width depth=$depth updates=$UPDATES" \
    | tee -a "$OUT/summary.txt"
  local t0 t1
  t0=$(date +%s.%N)
  timeout 5400 .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
    --game='eclipse(players=4)' --seed=1 --cuda \
    --encoder=spatial --nn_activation=tanh \
    --nn_width="$width" --nn_depth="$depth" \
    --factored_actions --noleague \
    --ent_coef=0.05 --aux_target_mode=rank --aux_coef=0.1 \
    --num_envs="$envs" --num_steps=128 --num_workers=16 --num_minibatches=16 \
    --update_epochs=4 \
    --lr_schedule=fixed --amp --compile_encoder --overlap_record \
    --obs_buffer_device=cuda \
    --snapshot_every=0 --timing --timing_every="$TIMING_EVERY" \
    --verdict_every_sec=0 \
    --total_timesteps="$total" --max_seconds=0 \
    --roster_dir="$dir" --run_dir="$dir" --track="t4_$label" \
    > "$log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)

  # Real throughput is final_steps/elapsed, never the logged sps.
  .venv/bin/python tools/parse_t0.py "$label" "$log" "$t0" "$t1" "$rc" "$total" \
    | tee -a "$OUT/summary.txt"

  # Distinct parameter count, straight from the trainer's startup telemetry
  # (nn.Module.parameters() de-duplicates the four encoder aliases; summing a
  # state_dict does not, and overstates by ~2.9x).
  grep -o "params=[0-9,]* trainable.*" "$log" | head -1 \
    | sed 's/^/    /' | tee -a "$OUT/summary.txt"

  # A crashed run holds the GPU indefinitely: async workers block forever on
  # their semaphores and nothing releases VRAM. Refuse to start the next rung
  # into a zombie -- a 91 GB one once made a clean config look like a fresh OOM.
  local used
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    [ "$used" -lt 2000 ] && break
    sleep 10
  done
  echo "    gpu3 memory.used after rung: ${used} MiB" | tee -a "$OUT/summary.txt"
  echo | tee -a "$OUT/summary.txt"
}

: > "$OUT/summary.txt"
echo "=== T4 capacity cost ladder, BIG/behemoth, GPU 3, $(date +%F' '%H:%M:%S) ===" \
  | tee -a "$OUT/summary.txt"
echo "commit $(git rev-parse --short HEAD)  updates/rung=$UPDATES" \
  | tee -a "$OUT/summary.txt"
echo | tee -a "$OUT/summary.txt"

# w64 is the CONTROL and it is not optional: it re-derives the recorded 12.44
# s/update on this tree. If the control does not land near it, the harness
# changed and no other rung in the table is interpretable.
rung w64d2   64  2
rung w128d2  128 2
rung w256d2  256 2
rung w256d3  256 3

echo "=== done $(date +%F' '%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
