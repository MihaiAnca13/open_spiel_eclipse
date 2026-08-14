#!/usr/bin/env bash
# Everything on the pre-flight path that is CHEAP and DECISION-FORMING, in order.
#
# Deliberately stops before T1. T1 is ~10 GPU-hours across two sequential arms
# and it inherits three decisions from the steps below (--max_live_opponents,
# --num_workers, and whether its judgement step is even affordable). Those get
# reviewed before the expensive thing starts, not after.
#
# Sequential by necessity, not tidiness: two training arms once thrashed this
# card to ~45 SPS and three OOM'd. The one CPU-only step runs BETWEEN GPU steps
# rather than alongside them -- it would steal cores from the env workers and
# corrupt the very env-phase numbers it exists to corroborate.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="build/open_spiel/python:$PWD"
export CUDA_VISIBLE_DEVICES=3

OUT=runs/preflight
mkdir -p "$OUT"

step() {
  echo ""
  echo "############ $* ############"
  echo "start $(date +%F' '%H:%M:%S)"
}

{
  echo "=== pre-flight remainder, BIG/behemoth, GPU 3 ==="
  echo "commit $(git rev-parse --short HEAD)  $(date +%F' '%H:%M:%S)"

  step "A. observation writer check (CPU only)"
  .venv/bin/python tools/t0_obs_writer_check.py

  step "B. T3 -- act cost vs K distinct league policies"
  .venv/bin/python tools/t3_opponent_curve.py \
    --num_rows=256 --num_workers=16 --ks=1,2,4,6,8,12,16,24,32 --repeats=25

  step "C. worker probe at the long run's env count"
  bash run_t0_workers.sh

  step "D. ladder tournament sizing"
  bash run_ladder_sizing.sh 3 2

  echo ""
  echo "=== pre-flight remainder done $(date +%F' '%H:%M:%S) ==="
  echo "STOP. Review, then launch T1 with the values these produced:"
  echo "  ./run_t1_update_epochs.sh <secs-per-arm> <K> <workers> "
} 2>&1 | tee "$OUT/preflight.log"
