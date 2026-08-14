#!/usr/bin/env python3
"""Parse one T0 rung's log into a single comparable row.

Why a parser and not grep: the two numbers that matter are (a) the wall-clock
per-update breakdown, which only exists on the `[timing uN]` lines, and (b) real
throughput, which is final_steps/elapsed and is NOT the logged sps. Update 0
carries process warmup and the torch.compile trace, so it is discarded rather
than averaged in -- including it made the compile rung look like a regression.
"""
import re
import sys

TIMING = re.compile(
    r"\[timing u(?P<u>\d+)\]\s+"
    r"act=(?P<act>[\d.]+)ms/env\s+"
    r"act\+phi=(?P<actphi>[\d.]+)\s+"
    r"env=(?P<env>[\d.]+)\s+"
    r"shape\+refresh=(?P<shape>[\d.]+)\s+"
    r"post=(?P<post>[\d.]+)\s+"
    r"\|\|\s+per update:\s+"
    r"rollout=(?P<rollout>[\d.]+)s\s+"
    r"\(phases\s+(?P<phases>[\d.]+)s,\s+overlap\s+(?P<overlap>[\d.]+)x\)\s+"
    r"learn=(?P<learn>[\d.]+)s\s+"
    r"update=(?P<update>[\d.]+)s\s+"
    r"sps=(?P<sps>[\d.]+)\s+"
    r"learn_share=(?P<learn_share>[\d.]+)%")

STEPS = re.compile(r"\[update \d+\] steps=(\d+)")
OBSBUF = re.compile(r"obs_buffer=(\S+)\s+\((?P<gb>[\d.]+) GB\)")
FELL_BACK = re.compile(r"compil\w*.*(fail|fell back|eager)", re.I)


def mean(xs):
  return sum(xs) / len(xs) if xs else float("nan")


def main():
  label, path, t0, t1, rc, total = sys.argv[1:7]
  elapsed = float(t1) - float(t0)
  total = int(total)
  text = open(path, errors="replace").read()

  rows = [m.groupdict() for m in TIMING.finditer(text)]
  # Discard update 0: warmup + compile trace.
  hot = [r for r in rows if int(r["u"]) > 0]

  steps = [int(m.group(1)) for m in STEPS.finditer(text)]
  final_steps = max(steps) if steps else 0
  # The step counter only prints at eval_every, so it can lag the true end. When
  # the run exited cleanly the requested total is the honest denominator-mate.
  if rc == "0" and final_steps < total:
    final_steps = total

  obs = OBSBUF.search(text)
  oom = "OutOfMemoryError" in text or "out of memory" in text.lower()
  fallback = bool(FELL_BACK.search(text))

  print(f"  rc={rc}  elapsed={elapsed:.1f}s  steps={final_steps}")
  if obs:
    print(f"  obs_buffer={obs.group(1)} ({obs.group('gb')} GB)")
  if oom:
    print("  !! OOM in log")
  if fallback:
    print("  !! possible torch.compile fallback to eager -- grep the log")
  if not hot:
    print(f"  !! no usable [timing] lines (found {len(rows)} total) -- FAILED")
    return

  real_sps = final_steps / elapsed if elapsed > 0 else 0.0
  act = mean([float(r["act"]) for r in hot])
  env = mean([float(r["env"]) for r in hot])
  ovl = [float(r["overlap"]) for r in hot]

  print(f"  n_hot_updates={len(hot)} (u={','.join(r['u'] for r in hot)})")
  print(f"  act={act:.2f}ms/env  act+phi={mean([float(r['actphi']) for r in hot]):.2f}"
        f"  env={env:.2f}  shape+refresh={mean([float(r['shape']) for r in hot]):.2f}"
        f"  post={mean([float(r['post']) for r in hot]):.2f}")
  print(f"  act:env = {act / env:.2f}:1   ({act:.1f} : {env:.1f})")
  print(f"  rollout={mean([float(r['rollout']) for r in hot]):.2f}s"
        f"  learn={mean([float(r['learn']) for r in hot]):.2f}s"
        f"  update={mean([float(r['update']) for r in hot]):.2f}s"
        f"  learn_share={mean([float(r['learn_share']) for r in hot]):.0f}%")
  print(f"  overlap={min(ovl):.2f}-{max(ovl):.2f}x"
        f"  {'OK (serialized)' if max(ovl) <= 1.005 else '!! NOT 1.00x -- INSTRUMENT SUSPECT'}")
  print(f"  logged_sps={mean([float(r['sps']) for r in hot]):.0f}"
        f"   REAL_SPS={real_sps:.0f}  (real includes startup/compile/eval)")
  # The per-update series, not just its mean: a torch.compile rung that is still
  # warming up at the first hot sample is only visible as a trend, and averaging
  # it away is exactly how a compile win gets misread as a regression.
  series = "  ".join(f"u{r['u']}:{float(r['update']):.2f}s" for r in rows)
  print(f"  update series: {series}")
  # torch.compile keeps warming up past update 0 -- measured 2.74 -> 2.34 -> 2.26 s
  # of learn over u4/u8/u12 on BIG -- so the mean over all hot samples understates
  # the compiled steady state. The last three are the number to compare rungs on.
  tail = hot[-3:]
  s_upd = mean([float(r["update"]) for r in tail])
  s_learn = mean([float(r["learn"]) for r in tail])
  s_roll = mean([float(r["rollout"]) for r in tail])
  print(f"  STEADY (last {len(tail)}): "
        f"update={s_upd:.2f}s  learn={s_learn:.2f}s  rollout={s_roll:.2f}s"
        f"  sps={mean([float(r['sps']) for r in tail]):.0f}")

  # What T2 (act/env overlap) would be worth here, computed rather than eyeballed.
  # Overlapping group B's act with group A's env step saves min(act+phi, env) of
  # every rollout step; that is a fraction of the rollout phase sum, and the
  # rollout is only part of the update -- which is exactly why the item's value is
  # contingent on how much of the update `learn` eats.
  a_phase = mean([float(r["actphi"]) for r in tail])
  e_phase = mean([float(r["env"]) for r in tail])
  p_sum = a_phase + e_phase + mean([float(r["shape"]) for r in tail]) \
      + mean([float(r["post"]) for r in tail])
  roll_frac = min(a_phase, e_phase) / p_sum
  print(f"  T2 sizing: min(act+phi,env)={min(a_phase, e_phase):.2f}ms of "
        f"{p_sum:.2f}ms phase sum = {100 * roll_frac:.0f}% of rollout, "
        f"{100 * roll_frac * s_roll / s_upd:.0f}% of wall clock at this "
        f"update_epochs")


if __name__ == "__main__":
  main()
