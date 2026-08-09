#!/usr/bin/env bash
# One monitoring tick for the long8h run. Prints a compact status block.
# Read-only; safe to call repeatedly.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
DIR="${1:-runs/long8h}"

echo "### tick $(date +%F' '%H:%M:%S)"
if pgrep -f "roster_dir=$DIR" > /dev/null; then
  echo "state: RUNNING"
else
  echo "state: NOT RUNNING"
fi
echo "gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"
echo "oom_lines: $(grep -c 'OutOfMemoryError' "$DIR/train.log" 2>/dev/null | head -1)"
echo "progress: $(tr '\r' '\n' < "$DIR/train.log" 2>/dev/null | grep -oE '[0-9]+/1000000000' | tail -1) steps"
echo "sps: $(tr '\r' '\n' < "$DIR/train.log" 2>/dev/null | grep -oE '[0-9.]+envstep/s' | tail -1)"
echo "snapshots: $(ls "$DIR"/snap_u*.pt 2>/dev/null | wc -l)"

.venv/bin/python - "$DIR" <<'PY' 2>/dev/null
import glob, sys, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d = sys.argv[1]
fs = sorted(glob.glob(f"{d}/events*"))
if not fs:
    print("scalars: (no tfevents yet)"); raise SystemExit
ea = EventAccumulator(fs[-1], size_guidance={"scalars": 400000}); ea.Reload()
tags = ea.Tags()["scalars"]
def series(t):
    return [e.value for e in ea.Scalars(t)] if t in tags else []
out = []
for t, label in [("losses/entropy","entropy"), ("losses/approx_kl","kl"),
                 ("losses/explained_variance","ev"), ("losses/returns_out_of_band","oob"),
                 ("charts/SPS","sps")]:
    s = series(t)
    if s: out.append(f"{label}={s[-1]:.4f}")
print("scalars: " + "  ".join(out))
PY
echo
# Wall-clock sanity: steps / elapsed seconds since run start.
ACT=$(date +%s); START=$(stat -c %Y "$DIR/train.log" 2>/dev/null || echo "$ACT")
echo "elapsed_sec: $((ACT - START))"
