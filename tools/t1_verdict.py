#!/usr/bin/env python3
"""Apply T1's pass rule to a two-arm ladder.json, and refuse to guess.

The rule from next_work.md, verbatim: "update_epochs=1's rating lower bound
clears update_epochs=4's upper bound. Anything less and keep 4 -- it is what
produced `long8h`, the strongest model measured."

So this compares each arm's BEST-RATED policy by CI, not by point estimate, and
prints PASS only on a strict non-overlap. Everything else is KEEP-4. The point
estimate ordering is printed too, but it does not decide anything -- an arm can
lead on the mean and still lose this test, and that is the intended outcome.
"""
import json
import sys


def arm_of(pid):
  return pid.split(":", 1)[0] if ":" in pid else None


def main():
  path = sys.argv[1]
  with open(path) as f:
    payload = json.load(f)

  nets = [p for p in payload["policies"] if p["kind"] == "net"]
  bots = [p for p in payload["policies"] if p["kind"] != "net"]
  arms = {}
  for p in nets:
    arms.setdefault(arm_of(p["id"]), []).append(p)

  print(f"=== T1 verdict from {path} ===")
  print(f"games/dir={payload['games_per_dir']}  anchor={payload['anchor']}")
  for name in sorted(arms):
    print(f"\n--- {name} ({len(arms[name])} policies) ---")
    for p in sorted(arms[name], key=lambda q: -q["rating"]):
      lo, hi = p["rating_ci"]
      print(f"  {p['id']:<28} birth={str(p['birth_update']):>6}  "
            f"rating={p['rating']:+.3f}  ci=[{lo:+.3f},{hi:+.3f}]  "
            f"n={p['games']}")
  if bots:
    print("\n--- anchors ---")
    for p in sorted(bots, key=lambda q: -q["rating"]):
      lo, hi = p["rating_ci"]
      print(f"  {p['id']:<28} rating={p['rating']:+.3f} "
            f"ci=[{lo:+.3f},{hi:+.3f}]")

  ue1 = arms.get("t1_ue1") or arms.get("t1_ue1_lr")
  ue4 = arms.get("t1_ue4") or arms.get("t1_ue4_lr")
  if not ue1 or not ue4:
    print(f"\n!! could not find both arms; saw {sorted(arms)}")
    return 2

  # "Best" = highest rating point estimate within the arm; its CI is then what
  # the rule is applied to. Using the arm's best rather than its final snapshot
  # is deliberate: an arm whose peak was mid-run has still demonstrated that
  # strength, and next_work.md asks for mid AND final snapshots precisely so a
  # late collapse cannot masquerade as the arm's ceiling.
  b1 = max(ue1, key=lambda p: p["rating"])
  b4 = max(ue4, key=lambda p: p["rating"])
  lo1, hi1 = b1["rating_ci"]
  lo4, hi4 = b4["rating_ci"]

  print("\n=== the rule ===")
  print(f"  best ue=1 : {b1['id']}  rating={b1['rating']:+.3f} "
        f"ci=[{lo1:+.3f},{hi1:+.3f}]")
  print(f"  best ue=4 : {b4['id']}  rating={b4['rating']:+.3f} "
        f"ci=[{lo4:+.3f},{hi4:+.3f}]")
  print(f"  ue=1 lower bound {lo1:+.3f} vs ue=4 upper bound {hi4:+.3f}")

  if lo1 > hi4:
    print("\n  PASS -- update_epochs=1 clears update_epochs=4. Use ue=1 for the "
          "long run, and note that T2 (act/env overlap) roughly triples in "
          "value because learn no longer dominates the update.")
    return 0
  print("\n  KEEP 4 -- ue=1 did not clear ue=4's upper bound.")
  if b1["rating"] > b4["rating"]:
    print("  (ue=1 leads on the point estimate but the intervals overlap. That "
          "is not a pass; the rule is deliberately strict.)")
  print("  Before discarding ue=1, re-run it ONCE with a raised LR -- a null "
        "result at an LR tuned for 4x the gradient steps is not a result:")
  print("      ./run_t1_update_epochs.sh <secs> <K> <workers> <raised-lr>  "
        "with ARMS=1 TAG=_lr")
  print("  Also check explained_variance: if only the critic degraded, a "
        "separate epoch count for the value head is the fix, not abandoning "
        "ue=1.")
  return 1


if __name__ == "__main__":
  sys.exit(main())
