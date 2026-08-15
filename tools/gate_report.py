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

Usage:  gate_report.py <ladder.json>
"""
import json
import sys


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
  if len(sys.argv) != 2:
    print(__doc__.strip().splitlines()[-1])
    return 2
  with open(sys.argv[1]) as f:
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

  # CI-clear in either direction, against the OLDEST net in the tournament.
  gained = last["rating_ci"][0] > first["rating_ci"][1]
  lost = last["rating_ci"][1] < first["rating_ci"][0]

  # Any CI-clear regression from an earlier peak to the newest policy is the
  # shape that matters most: it is what makes the final policy the wrong one to
  # ship, and it is invisible in vp_all / mean_episode_return / vs-Greedy.
  peak_drop = (best is not last
               and last["rating_ci"][1] < best["rating_ci"][0])

  print()
  if gained:
    verdict = "IMPROVING"
    why = (f"newest ({last['id']}) clears oldest ({first['id']}) CI: "
           f"{last['rating_ci'][0]:.3f} > {first['rating_ci'][1]:.3f}")
  elif lost:
    verdict = "REGRESSING"
    why = (f"newest ({last['id']}) rates CI-clear BELOW oldest "
           f"({first['id']}): {last['rating_ci'][1]:.3f} < "
           f"{first['rating_ci'][0]:.3f}")
  else:
    verdict = "FLAT"
    why = (f"newest ({last['id']}) and oldest ({first['id']}) CIs overlap — "
           f"no measurable gain over this span")
  print(f"  GATE: {verdict} — {why}")

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
