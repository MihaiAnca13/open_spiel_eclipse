#!/usr/bin/env bash
# T1 -- update_epochs 1 vs 4, the real experiment. BIG/behemoth, GPU 3.
#
# WHAT IS BEING ASKED
#   `learn` runs update_epochs * num_minibatches forward+backward passes, so
#   update_epochs=1 is the biggest throughput lever left (BIG measured 1.55x:
#   5,735 -> 8,863 at 1,024 envs). But samples/hour only converts to strength if
#   one epoch learns as much per sample. This is the largest open QUALITY question
#   and the only item that can make the long run worse rather than merely slower.
#
# WHY EQUAL WALL-CLOCK AND NOT EQUAL STEPS
#   The whole point is that one arm gets more samples per hour. Equal steps would
#   throw away the entire effect being measured. Both arms therefore get the same
#   --max_seconds and are compared on strength at that budget.
#
# HOW IT IS JUDGED -- and how it is NOT
#   ONE ladder tournament rating both arms' final AND mid snapshots together,
#   on ladder `rating` / `rating_ci` only. Never `elo`, `vp_all`,
#   `mean_episode_return`, or vs-Greedy; and never by diffing a loss curve --
#   at a fixed seed three master runs agreed exactly for updates 0-2 and then
#   diverged chaotically from GPU reduction order alone.
#
#   PASS: update_epochs=1's rating lower bound clears update_epochs=4's upper
#   bound. Anything less and keep 4 -- it is what produced `long8h`, the
#   strongest model measured.
#
# EXPECTED AND NOT DISQUALIFYING
#   approx_kl and clipfrac WILL fall (a quarter as many gradient steps per batch
#   is less policy movement, not more stability). explained_variance may fall
#   because the value head also gets a quarter of the updates. If ue=1 loses,
#   re-run ONCE with a raised LR before discarding it -- a null result at an LR
#   tuned for 4x the gradient steps is not a result. That re-run is:
#       ./run_t1_update_epochs.sh 18000 <K> 1 <raised-lr>
#
# SEQUENCING: arms run SEQUENTIALLY. Two arms once thrashed the card to ~45 SPS
# and three OOM'd. GPU 3 only -- GPUs 1-2 hold resident vLLM workers.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
export CUDA_VISIBLE_DEVICES=3

SECS="${1:-18000}"          # per arm; 5h. next_work.md recommends 4-6h.
K="${2:-4}"                 # --max_live_opponents, from T3
WORKERS="${3:-16}"          # from the T0 worker probe
LR="${4:-2.5e-4}"           # raise this for the ue=1 re-run if T1 fails
ARMS="${ARMS:-4 1}"         # incumbent first
TAG="${TAG:-}"              # set for the raised-LR re-run so it gets its own dirs

# 1,024 envs x 128 steps = 131,072 rows; /16 minibatches = 8,192 rows each, the
# most efficient minibatch measured on BIG. Raise num_envs and num_minibatches
# together or that ratio drifts and the arms stop being comparable.
ENVS=1024
STEPS=128
MB=16

echo "=== T1 update_epochs A/B ===" | tee runs/t1_launch.log
echo "secs/arm=$SECS  K=$K  workers=$WORKERS  lr=$LR  arms='$ARMS'  tag='$TAG'" \
  | tee -a runs/t1_launch.log
echo "commit $(git rev-parse --short HEAD)" | tee -a runs/t1_launch.log

for UE in $ARMS; do
  DIR="runs/t1_ue${UE}${TAG}"
  rm -rf "$DIR"; mkdir -p "$DIR"
  echo "--- arm update_epochs=$UE -> $DIR  start $(date +%F' '%H:%M:%S)" \
    | tee -a runs/t1_launch.log

  # --seed=1 on both arms: the comparison must not also be a seed lottery.
  # --snapshot_every=100 so each arm yields 8-12 rateable snapshots inside the
  # budget, which is what makes "final AND mid snapshots" possible.
  .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
    --game='eclipse(players=4)' --seed=1 --cuda \
    --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2 \
    --factored_actions --league --max_live_opponents="$K" \
    --ent_coef=0.05 --aux_target_mode=rank --aux_coef=0.1 \
    --num_envs="$ENVS" --num_steps="$STEPS" --num_workers="$WORKERS" \
    --num_minibatches="$MB" --update_epochs="$UE" \
    --learning_rate="$LR" --lr_schedule=fixed \
    --amp --compile_encoder --overlap_record \
    --obs_buffer_device=cuda \
    --snapshot_every=100 \
    --timing --timing_every=50 \
    --total_timesteps=1000000000 --max_seconds="$SECS" \
    --roster_dir="$DIR" --run_dir="$DIR" --track="t1_ue${UE}${TAG}" \
    >> "$DIR/train.log" 2>&1
  rc=$?
  echo "--- arm update_epochs=$UE done rc=$rc $(date +%F' '%H:%M:%S)" \
    | tee -a runs/t1_launch.log

  # A crashed run holds the GPU indefinitely: async workers block forever on
  # their semaphores and nothing releases VRAM. A 91 GB zombie once made a clean
  # config look like a fresh OOM. Refuse to start the next arm into that.
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    [ "$used" -lt 2000 ] && break
    sleep 10
  done
  echo "    gpu3 memory.used after arm: ${used} MiB" | tee -a runs/t1_launch.log
done

echo "=== both arms done $(date +%F' '%H:%M:%S) ===" | tee -a runs/t1_launch.log
echo "Next: rate BOTH arms in ONE tournament, mid + final snapshots together:" \
  | tee -a runs/t1_launch.log
echo "  ./run_t1_ladder.sh '$TAG'" | tee -a runs/t1_launch.log
