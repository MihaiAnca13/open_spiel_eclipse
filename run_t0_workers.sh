#!/usr/bin/env bash
# T0 addendum -- pick --num_workers at the long run's env count, on BIG.
#
# WHY THIS IS NOT JUST T5 CURIOSITY
#   T1 commits ~10 GPU-hours to two sequential arms. Both arms share whatever
#   num_workers is chosen, so a bad choice does not invalidate the A/B -- but it
#   does buy less policy improvement per hour, in the one experiment that is
#   least affordable to repeat. next_work.md recommends 20 for the long run; that
#   number came off the 12 GB box's 32 cores, not off BIG.
#
#   BIG has 12 physical / 24 logical cores and is SHARING them with two resident
#   vLLM workers (measured ~14% of a core each, plus an engine process). More
#   env workers than free cores does not just stop helping -- it starts stealing
#   from the main process's act/learn dispatch. This measures where that turns.
#
#   Held fixed at the long run's shape: 1,024 envs, 128 steps, 16 minibatches
#   (8,192 rows each), device obs buffer, compiled encoder.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
export CUDA_VISIBLE_DEVICES=3

OUT="${OUT_DIR:-runs/t0_workers}"
UPDATES="${UPDATES:-12}"
WORKER_LIST="${WORKER_LIST:-12 16 20 24}"
mkdir -p "$OUT"
: > "$OUT/summary.txt"

echo "=== T0 worker probe @ 1024 envs, BIG/behemoth, GPU 3, $(date +%F' '%H:%M:%S) ===" \
  | tee -a "$OUT/summary.txt"
echo "commit $(git rev-parse --short HEAD)  updates/rung=$UPDATES" \
  | tee -a "$OUT/summary.txt"
echo "host load at start: $(uptime | sed 's/.*load average/load average/')" \
  | tee -a "$OUT/summary.txt"

for W in $WORKER_LIST; do
  label="w$W"
  log="$OUT/$label.log"
  dir="$OUT/$label"
  rm -rf "$dir"; mkdir -p "$dir"
  total=$((UPDATES * 1024 * 128))
  echo "### workers=$W updates=$UPDATES" | tee -a "$OUT/summary.txt"
  t0=$(date +%s.%N)
  timeout 5400 .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
    --game='eclipse(players=4)' --seed=1 --cuda \
    --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2 \
    --factored_actions --noleague \
    --ent_coef=0.05 --aux_target_mode=rank --aux_coef=0.1 \
    --num_envs=1024 --num_steps=128 --num_workers="$W" --num_minibatches=16 \
    --lr_schedule=fixed --amp --compile_encoder --obs_buffer_device=cuda \
    --snapshot_every=0 --timing --timing_every=4 --verdict_every_sec=0 \
    --total_timesteps="$total" --max_seconds=0 \
    --roster_dir="$dir" --run_dir="$dir" --track="t0_$label" \
    > "$log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  .venv/bin/python tools/parse_t0.py "$label" "$log" "$t0" "$t1" "$rc" "$total" \
    | tee -a "$OUT/summary.txt"
  echo | tee -a "$OUT/summary.txt"
done

echo "=== done $(date +%F' '%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
echo "Pick the largest workers value before REAL_SPS stops rising; if 12 and 24" \
  | tee -a "$OUT/summary.txt"
echo "tie, prefer the SMALLER one -- it leaves the vLLM workers their cores." \
  | tee -a "$OUT/summary.txt"
