#!/usr/bin/env bash
# The production Eclipse 4P run. BIG/behemoth, GPU 3. Trains in chunks and
# rates itself between them, so a run that has stopped learning says so in
# hours instead of at the end.
#
# WHY THIS SHAPE
#   T1 established that the failure mode here is silent. Both of its arms
#   stopped improving by ~update 100-300 and then spent 200M+ steps going
#   nowhere, and one of them *regressed* -- while vp_all ROSE (14.62 -> 15.81)
#   and mean_episode_return sat at 17. Every training-side metric said the run
#   was fine. Only a ladder saw it. So this script does not run for days and
#   then look: it stops on a cadence, rates recent snapshots against older ones
#   in ONE tournament, and prints IMPROVING / FLAT / REGRESSING.
#
#   Chunking is also why the run is safe to leave alone. --resume restores the
#   optimizer moments, the step/update counters and the roster, so a chunk
#   boundary is not a training discontinuity (before _save_train_state existed,
#   resuming restarted Adam and the LR schedule -- the Sprint-1 grid ran on a
#   sawtooth LR because of it).
#
# WHAT CHANGED FROM THE T1 CONFIG, AND WHY
#   1. CAPACITY. The T1 net had 545,587 TRAINABLE parameters at --nn_width=64.
#      Its checkpoint appears to hold 1,897,461 because two inflations stack:
#      `shared`, `critic_trunk`, `critic.0` and `actor.0` are four state_dict
#      aliases of ONE shared encoder (verified bit-identical, 2.89x), and
#      111,170 of the remainder are int64 action-factorization lookup tables,
#      which are buffers and not capacity. See tools/count_params.py.
#      EVERY quality lever this project has tested was tested inside that box
#      and every one came back null: update_epochs, the spatial pointer head,
#      --aux_coef at 0.01, encoder choice. Capacity itself was never swept on
#      this machine. run_t4_capacity.sh prices it; WIDTH comes from that table.
#      This is a bet, not a measured win -- the gates are what will judge it.
#   2. THE ROSTER KEEPS ITS WHOLE HISTORY (--roster_keep_recent/_spaced=0).
#      The hardcoded prune(4, 4) DELETED weight files after every snapshot: a
#      2,192-update run ended holding 8 of the 21 snapshots it wrote. That is
#      not a throughput measure -- act cost scales with the number of DISTINCT
#      policies in a rollout batch, which --max_live_opponents bounds on its
#      own. What it cost was the ability to ask where a run peaked. At ~3 MB a
#      snapshot and --snapshot_every=25, a 72h run keeps ~700 MB.
#
#      SIDE EFFECT, STATED BECAUSE IT IS A REAL CHANGE AND NOT A FREE ONE:
#      Matchmaker._live_opponents draws its 4 live slots uniformly from
#      roster.opponent_ids(), so keeping the whole history also widens the
#      opponent DISTRIBUTION from "4 near-adjacent recent policies" to "4 drawn
#      from the run's entire past" -- i.e. fictitious self-play. That is the
#      classical anti-cycling choice, but it is not free: late in a long run
#      most sampled opponents are much weaker than main, so a share of the
#      mixed-lineup games are lopsided and carry little signal.
#      --selfplay_fraction=0.5 keeps half the envs on the current policy, which
#      is what bounds that cost. Note this was NOT adopted to fix forgetting:
#      the T1 pair table refutes forgetting outright (its final policy lost
#      uniformly to old AND new alike). It is adopted for peak selection, and
#      the wider distribution comes along with it.
#   3. DENSE SNAPSHOTS (25, not 100). T1's peak sat at u1700 and its final
#      policy rated 0.18 BELOW it. The policy to ship is the peak, and you can
#      only ship a peak you snapshotted.
#   4. NO PERIODIC VERDICT EVALS (--verdict_every_sec=0 disables them; the code
#      tests the flag for truthiness). They measure utility vs Greedy, which
#      every trained net beat 127/128 in the T1 ladder -- a saturated signal
#      that costs training time. The chunk-boundary verdict is unconditional in
#      the trainer, so --noeval_greedy/--noeval_random keep that one cheap too.
#
# WHAT IS DELIBERATELY UNCHANGED
#   1,024 envs / 128 steps / 16 workers / 16 minibatches (8,192 rows per
#   minibatch), --amp, --compile_encoder, --overlap_record, device obs buffer,
#   --max_live_opponents=4, --lr_schedule=fixed, --ent_coef=0.05,
#   --aux_coef=0.1, update_epochs=4. All measured on BIG; see
#   docs/eclipse_rl_todo.md. update_epochs=4 over 1: T1's ue=1 arm is the one
#   that regressed, its 1.35x sample-rate advantage bought nothing measurable,
#   and head-to-head of the two FINAL policies favoured ue=4.
#
# OPERATING IT
#   ./run_long.sh                      # defaults: 72h in chunks, GPU 3
#   RUN=runs/main_v2 WIDTH=128 ./run_long.sh
#   touch runs/main_v1/STOP            # stop cleanly after the current chunk
#   tail -f runs/main_v1/gates.log     # the only file worth watching
#
# GPU 3 ONLY. GPUs 1-2 hold two resident vLLM workers at 96.8 GB each and 0
# drives the display. Never run two arms at once -- two once thrashed the card
# to ~45 SPS and three OOM'd.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
export CUDA_VISIBLE_DEVICES=3

RUN="${RUN:-runs/main_v1}"
WIDTH="${WIDTH:-256}"          # set from run_t4_capacity.sh's table
DEPTH="${DEPTH:-2}"
CHUNK1="${CHUNK1:-7200}"       # first chunk short: an early read on the bet
CHUNK="${CHUNK:-21600}"        # 6h thereafter; the gate is ~4% on top
CHUNKS="${CHUNKS:-13}"         # 2h + 12x6h = 74h of training
GATE_KEEP="${GATE_KEEP:-3}"    # snapshots subsampled per gate (+ main)
GATE_GAMES="${GATE_GAMES:-24}" # per pair per seat direction
SEED="${SEED:-1}"

ENVS=1024; STEPS=128; MB=16; WORKERS=16; UE=4; K=4

mkdir -p "$RUN"
GATES="$RUN/gates.log"

say() { echo "$@" | tee -a "$GATES"; }

say "=== production run  $(date +%F' '%H:%M:%S) ==="
say "run=$RUN width=$WIDTH depth=$DEPTH seed=$SEED"
say "chunks=$CHUNKS (first ${CHUNK1}s, then ${CHUNK}s)  gate: keep=$GATE_KEEP games=$GATE_GAMES"
say "commit $(git rev-parse --short HEAD)"
say ""

for i in $(seq 1 "$CHUNKS"); do
  if [ -f "$RUN/STOP" ]; then
    say "[chunk $i] STOP file present — exiting cleanly."
    break
  fi

  SECS="$CHUNK"; [ "$i" = "1" ] && SECS="$CHUNK1"

  # Resume whenever a main.pt exists. On chunk 1 there is none, so the run
  # starts fresh; the arch change means an older roster could not be loaded
  # anyway (a width mismatch is a SIZE mismatch, which load_state_dict raises
  # on even with strict=False -- see runs/roster, dead for exactly this reason).
  RESUME=()
  [ -f "$RUN/main.pt" ] && RESUME=(--resume=main)

  say "[chunk $i/$CHUNKS] train ${SECS}s  $(date +%F' '%H:%M:%S)  ${RESUME[*]:-fresh}"
  .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
    --game='eclipse(players=4)' --seed="$SEED" --cuda \
    --encoder=spatial --nn_activation=tanh \
    --nn_width="$WIDTH" --nn_depth="$DEPTH" \
    --factored_actions --league --max_live_opponents="$K" \
    --ent_coef=0.05 --aux_target_mode=rank --aux_coef=0.1 \
    --num_envs="$ENVS" --num_steps="$STEPS" --num_workers="$WORKERS" \
    --num_minibatches="$MB" --update_epochs="$UE" \
    --learning_rate=2.5e-4 --lr_schedule=fixed \
    --amp --compile_encoder --overlap_record --obs_buffer_device=cuda \
    --snapshot_every=25 --roster_keep_recent=0 --roster_keep_spaced=0 \
    --verdict_every_sec=0 --noeval_greedy --noeval_random --eval_games=8 \
    --timing --timing_every=100 \
    --total_timesteps=100000000000 --max_seconds="$SECS" \
    "${RESUME[@]}" \
    --roster_dir="$RUN" --run_dir="$RUN" --track="$(basename "$RUN")" \
    >> "$RUN/train.log" 2>&1
  rc=$?

  if [ "$rc" != "0" ]; then
    say "[chunk $i] TRAIN FAILED rc=$rc — stopping. Last log lines:"
    tail -20 "$RUN/train.log" | sed 's/^/    /' | tee -a "$GATES"
    break
  fi

  # A crashed run holds the GPU indefinitely: async workers block forever on
  # their semaphores and nothing releases VRAM. Never start the gate (or the
  # next chunk) into a zombie -- a 91 GB one once made a clean config look like
  # a fresh OOM. Poll a file/driver, never pgrep on this loop's own cmdline.
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    [ "$used" -lt 2000 ] && break
    sleep 10
  done

  # The trainer's own "[update N]" counter is CHUNK-LOCAL (the loop is
  # `for update in range(num_updates)` and restarts at 0 every process), so it
  # is not the age. Snapshot ids are keyed on agent.updates_done, which does
  # survive a resume -- read the age off the newest snapshot instead.
  AGE=$(ls "$RUN"/snap_u*.pt 2>/dev/null | sed 's/.*snap_u//;s/\.pt//' \
        | sort -n | tail -1)
  NSNAP=$(ls "$RUN"/snap_*.pt 2>/dev/null | wc -l)
  say "[chunk $i] trained to update ${AGE:-?}  ($NSNAP snapshots kept)  gpu3=${used}MiB  $(date +%F' '%H:%M:%S)"

  # ── Gate ────────────────────────────────────────────────────────────────
  # Rate a subsample of THIS run's own history in one tournament. Ratings are
  # not comparable across ladder runs, so each gate is read on its internal
  # ordering only -- never by diffing a number against the previous gate.
  # prune_roster writes a roster.json pointing at the ORIGINAL .pt files, so
  # this copies nothing.
  GDIR="$RUN/gate_$i"
  rm -rf "$GDIR"
  .venv/bin/python tools/prune_roster.py "$RUN" "$GDIR" --keep="$GATE_KEEP" \
    >> "$RUN/gate.log" 2>&1
  .venv/bin/python -m open_spiel.python.eclipse.roster_ladder \
    --game='eclipse(players=4)' --cuda --seed="$SEED" \
    --ladder_roster_dir="$GDIR" \
    --ladder_games_per_dir="$GATE_GAMES" \
    --ladder_include_bots --noladder_include_heuristic \
    --eval_envs=64 --num_workers="$WORKERS" \
    --ladder_out="$RUN/gate_$i.json" \
    >> "$RUN/gate.log" 2>&1
  grc=$?

  if [ "$grc" != "0" ] || [ ! -f "$RUN/gate_$i.json" ]; then
    say "[gate $i] LADDER FAILED rc=$grc — training continues, but this chunk"
    say "          is unjudged. Check $RUN/gate.log."
  else
    say "[gate $i] $(date +%F' '%H:%M:%S)"
    .venv/bin/python tools/gate_report.py "$RUN/gate_$i.json" | tee -a "$GATES"
  fi
  say ""
done

say "=== run finished $(date +%F' '%H:%M:%S) ==="
say "Snapshots: $(ls "$RUN"/snap_*.pt 2>/dev/null | wc -l) kept (nothing pruned)."
say "SHIP THE PEAK, NOT main.pt — the last gate's best-in-tournament line names it."
