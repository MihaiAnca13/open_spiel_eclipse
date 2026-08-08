#!/usr/bin/env bash
# Measure REAL training throughput (envstep/s) for the new throughput flags, and
# find the largest --num_envs this 12GB card actually sustains.
#
# Microbenchmarks of the trunk are not enough: they miss env stepping, the act
# path, GAE and the optimizer. These are short real training runs.
#
# Baseline for comparison: ~3.0-3.3k envstep/s at 128 envs on the config that
# produced the 1.2204 policy.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"

SECS="${1:-180}"
SCRATCH="${SCRATCH_DIR:-/tmp/eclipse_bench}"
mkdir -p "$SCRATCH"

probe() {  # probe <label> <num_envs> <extra flags...>
  local label="$1" envs="$2"; shift 2
  local dir="$SCRATCH/$label"
  rm -rf "$dir"; mkdir -p "$dir"
  timeout $((SECS + 180)) .venv/bin/python -m open_spiel.python.examples.ppo_eclipse \
    --game='eclipse(players=4)' --seed=1 --cuda \
    --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2 \
    --ent_coef=0.01 --aux_target_mode=rank --aux_coef=0.1 \
    --num_envs="$envs" --num_steps=128 --num_workers=16 --num_minibatches=4 \
    --snapshot_every=0 --lr_schedule=fixed \
    --total_timesteps=1000000000 --max_seconds="$SECS" \
    --roster_dir="$dir" --run_dir="$dir" --track=bench "$@" \
    > "$dir/log" 2>&1
  local sps oom peak
  # take the LAST reported rate: the first few updates include warmup/compile
  sps="$(tr '\r' '\n' < "$dir/log" | grep -oE "[0-9.]+envstep/s" | tail -1 | tr -d 'envstep/s')"
  oom="$(grep -c "OutOfMemoryError" "$dir/log")"
  peak="$(tr '\r' '\n' < "$dir/log" | grep -oE "[0-9]+/1000000000" | tail -1 | cut -d/ -f1)"
  if [ "$oom" != "0" ]; then
    printf "%-26s envs=%-4s  OOM\n" "$label" "$envs"
  else
    printf "%-26s envs=%-4s  SPS=%-8s steps=%s\n" "$label" "$envs" "${sps:-FAIL}" "${peak:-0}"
  fi
}

echo "=== throughput flags (128 envs, ${SECS}s each) ==="
probe base              128
probe amp               128 --amp
probe channels_last     128 --channels_last
probe amp+chlast        128 --amp --channels_last
probe amp+chlast+compile 128 --amp --channels_last --compile_encoder

echo
echo "=== max parallel envs (best flags from above are NOT auto-applied; using amp+chlast) ==="
for e in 192 256 384 512; do
  probe "envs_$e" "$e" --amp --channels_last
done
echo "done"
