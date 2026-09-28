#!/usr/bin/env python3
"""One-screen verdict for a mid-run gate ladder.

The question a gate answers is narrow: *within this one tournament*, is the
newest policy rated above the older ones? Ratings are NOT comparable across
ladder runs, so nothing here may be compared to a previous gate's numbers --
only each gate's own internal ordering is meaningful. That is why this prints
an ordering and a CI-clear comparison rather than a headline number to track.

Two failure shapes it is built to name, both of which the T1 run produced and
neither of which any training-side metric showed:

  FLAT       -- no snapshot clears any other's CI. 200M steps bought nothing.
  REGRESSING -- a later snapshot rates CI-clear BELOW an earlier one.

The headline verdict is RECENT: newest against the snapshot nearest
--baseline_age, which run_long.sh sets to the previous gate's `main`, i.e. what
the last chunk bought. The newest-vs-oldest line is kept as an anchor, but it
cannot detect a plateau: prune_roster pins the oldest snapshot (u25), so it
reads IMPROVING for as long as the run still beats a 25-update policy, whether
or not the last chunk bought anything.

Usage:  gate_report.py <ladder.json> [--baseline_age N]
"""
import argparse
import json
import sys


def _compare(new, old):
  """IMPROVING / REGRESSING / FLAT for `new` against `old`, by CI clearance."""
  if new["rating_ci"][0] > old["rating_ci"][1]:
    return "IMPROVING", (f"{new['rating_ci'][0]:.3f} > "
                         f"{old['rating_ci'][1]:.3f}")
  if new["rating_ci"][1] < old["rating_ci"][0]:
    return "REGRESSING", (f"{new['rating_ci'][1]:.3f} < "
                          f"{old['rating_ci'][0]:.3f}")
  return "FLAT", "CIs overlap"


def _age(pid):
  """Birth update from a policy id, or None for bots."""
  tail = pid.split(":")[-1]
  if tail.startswith("snap_u"):
    try:
      return int(tail[len("snap_u"):])
    except ValueError:
      return None
  return None


def main():
  ap = argparse.ArgumentParser(usage=__doc__.strip().splitlines()[-1])
  ap.add_argument("ladder_json")
  ap.add_argument("--baseline_age", type=int, default=None,
                  help="age of the previous gate's main; the RECENT verdict "
                       "compares newest against the net nearest it")
  args = ap.parse_args()
  with open(args.ladder_json) as f:
    d = json.load(f)

  pols = d["policies"]
  nets = [p for p in pols if p.get("kind") == "net"]
  bots = [p for p in pols if p.get("kind") != "net"]

  def sort_key(p):
    a = _age(p["id"])
    # `main` is the newest thing in the roster; birth_update carries it when
    # present, otherwise sort it last.
    return (a if a is not None else p.get("birth_update", 10**9))

  nets.sort(key=sort_key)

  print(f"  {'policy':>26} {'age':>7} {'rating':>8} {'95% CI':>18}")
  for p in nets:
    lo, hi = p["rating_ci"]
    a = sort_key(p)
    print(f"  {p['id']:>26} {a:>7} {p['rating']:8.3f}   [{lo:6.3f},{hi:6.3f}]")
  for p in bots:
    lo, hi = p["rating_ci"]
    print(f"  {p['id']:>26} {'-':>7} {p['rating']:8.3f}   [{lo:6.3f},{hi:6.3f}]")

  if len(nets) < 2:
    print("\n  GATE: only one net rated — nothing to compare.")
    return 0

  first, last = nets[0], nets[-1]
  best = max(nets, key=lambda p: p["rating"])

  # Any CI-clear regression from an earlier peak to the newest policy is the
  # shape that matters most: it is what makes the final policy the wrong one to
  # ship, and it is invisible in vp_all / mean_episode_return / vs-Greedy.
  peak_drop = (best is not last
               and last["rating_ci"][1] < best["rating_ci"][0])

  print()
  if args.baseline_age is None:
    print("  GATE: RECENT n/a — no previous gate to compare against")
  else:
    base = min(nets[:-1], key=lambda p: abs(sort_key(p) - args.baseline_age))
    verdict, why = _compare(last, base)
    print(f"  GATE: RECENT {verdict} — newest ({last['id']}) vs one chunk ago "
          f"({base['id']}): {why}")
    if sort_key(base) != args.baseline_age:
      print(f"  GATE: WARNING — no net aged {args.baseline_age} in this "
            f"tournament; RECENT used the nearest, age {sort_key(base)}")

  verdict, why = _compare(last, first)
  print(f"  GATE: vs anchor {verdict} — newest vs oldest ({first['id']}): "
        f"{why}  (cannot detect a plateau; read RECENT)")

  if peak_drop:
    print(f"  GATE: PAST PEAK — best is {best['id']} "
          f"({best['rating']:.3f}), newest is {last['rating']:.3f}, CI-clear. "
          f"Ship the peak, not main.")
  print(f"  GATE: best-in-tournament = {best['id']} ({best['rating']:.3f})")

  # The bot anchors saturate: every trained net beat Greedy 127/128 in the T1
  # ladder, so a big margin here means nothing about progress. Say so, rather
  # than let a healthy-looking anchor gap read as evidence.
  if bots:
    gap = min(p["rating"] for p in nets) - max(p["rating"] for p in bots)
    print(f"  (bot anchors {gap:+.2f} below the weakest net — saturated, "
          f"not a progress signal)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
