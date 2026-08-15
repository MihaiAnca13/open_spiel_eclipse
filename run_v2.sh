#!/usr/bin/env bash
# V2 typed-pointer training run, plus its control arm. 2026-08-11.
#
# WHY THERE ARE TWO ARMS
#   The V2 observation changed the tensor from 24,714 to 37,596 floats, so every
#   checkpoint and every Elo rating from the runs/ history is unloadable. There
#   is no comparable baseline left. "pointer" cannot be shown to beat anything
#   unless "control" is run on the SAME observation and rated on the same ladder.
#
#   control : --nofactored_actions -> plain Linear(width, 11117) actor head.
#   pointer : the typed V2 pointer head. Each action's logit reads the entity
#             that action names (cell / unit / route destination / planet slot /
#             seat), gathered per pair.
#
#   Be precise about what this isolates. Both arms share the SAME observation and
#   the SAME encoder (V2 entity branches, unit attention, embedded categoricals),
#   so the comparison attributes exactly one thing: the typed pointer head. It is
#   NOT a reproduction of the pre-V2 runs in runs/meta/hourly_log.md -- those had
#   a 24,714-float observation and no entity branches at all, and their
#   checkpoints cannot be loaded any more. If you want the head's contribution,
#   this is the right control. If you want "V2 vs pre-V2" end to end, that needs
#   a third arm on the old tensor, which no longer exists in the tree.
#
# MEASURED RESOURCE REQUIREMENTS (256 envs / 128 steps / 16 workers)
#   HOST RAM: the old "peak 17.5 GB, OOMs at a 16 GB cap" note is STALE. That
#     figure included a 4.9 GB fp32 rollout obs buffer on the host; the buffer is
#     now device-resident fp16 by default (2.46 GB of VRAM at these settings), so
#     it is off the host entirely. A league run with snapshots active completes
#     comfortably under a 14 GB cap. Re-measure before raising --num_envs.
#   VRAM: 9.4 GB allocated / 11.1 GB reserved at --num_minibatches=4, ~7-8 GB
#     at 8 (measured on a 12 GB card, league adds ~1 GB for opponent nets).
#
# 2026-08-13 -- READ BEFORE CHANGING --snapshot_every OR --max_live_opponents
#   The act path runs one encoder forward per DISTINCT policy in the batch, and
#   that is steeply superlinear: 256 rows cost 1.12x at 4 distinct policies,
#   2.05x at 8, 8.32x at 32. Before --max_live_opponents existed, lineups drew
#   from the whole roster, so act decayed for as long as the run lasted (22.8 ->
#   41.7 ms/step over 45 updates, still climbing) with nothing in the loss series
#   or the ratings to show for it. The default cap of 4 makes act plateau. Do not
#   set --max_live_opponents=0 on a long run without re-reading docs/eclipse_rl_todo.md.
#
# MEASURED CONFIG NOTES
#   --num_minibatches=8, not 4: at 4 the minibatch is 8,192 states x ~21 legal
#     actions = ~171k pairs and peak VRAM measured 9.43 GB allocated / 11.11 GB
#     reserved on a 12 GB card -- runnable but no headroom for eval spikes or
#     league opponents. At 8 it measured ~7 GB. On a 40 GB+ card, 4 is fine and
#     gives the larger batch.
#   --amp --compile_encoder: the compiled graph is now actually reachable.
#     Before, every hot call went through forward_with_context while
#     torch.compile wrapped a body only forward() could reach, so the measured
#     1.57x was silently inert.
#   --lr_schedule=fixed: never use the unvalidated kl controller.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

ARM="${1:-pointer}"             # pointer | control
SECS="${2:-28800}"              # 8h default
MB="${3:-8}"                    # 8 for a 12 GB card, 4 for 40 GB+
LEAGUE="${4:-1}"                # 1 = population self-play, 0 = plain self-play
DIR="runs/v2_${ARM}"

case "$ARM" in
  pointer) HEAD_FLAG="--factored_actions" ;;
  control) HEAD_FLAG="--nofactored_actions" ;;
  *) echo "usage: $0 [pointer|control] [seconds] [num_minibatches] [league 0|1]" >&2
     exit 2 ;;
esac
# --league was broken outright from the commit that moved the obs rollout buffer
# off the GPU (it indexed that CPU tensor with a CUDA mask) until 2026-08-12, so
# no league run exists on this observation. It is verified working now.
[ "$LEAGUE" = "1" ] && LEAGUE_FLAG="--league" || LEAGUE_FLAG="--noleague"

mkdir -p "$DIR"
echo "=== v2_${ARM} start $(date +%F' '%H:%M:%S) max_seconds=${SECS} mb=${MB} dir=${DIR} ===" \
  > "runs/v2_${ARM}_launch.log"

exec .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
  --game='eclipse(players=4)' --seed=1 --cuda \
  --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2 \
  $HEAD_FLAG $LEAGUE_FLAG \
  --ent_coef=0.05 --aux_target_mode=rank --aux_coef=0.1 \
  --num_envs=256 --num_steps=128 --num_workers=16 --num_minibatches="$MB" \
  --lr_schedule=fixed \
  --amp --compile_encoder \
  --snapshot_every=100 \
  --timing --timing_every=50 \
  --total_timesteps=1000000000 --max_seconds="$SECS" \
  --roster_dir="$DIR" --track="v2_${ARM}" \
  >> "$DIR/train.log" 2>&1
