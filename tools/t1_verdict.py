#!/usr/bin/env python3
"""Apply T1's pass rule to a two-arm ladder.json, and refuse to guess.

The rule from docs/eclipse_rl_todo.md, verbatim: "update_epochs=1's rating lower bound
clears update_epochs=4's upper bound. Anything less and keep 4 -- it is what
produced `long8h`, the strongest model measured."

So this compares each arm's BEST-RATED policy by CI, not by point estimate, and
prints PASS only on a strict non-overlap. Everything else is KEEP-4.

BUT THE RULE IS UNDERSPECIFIED AND THIS SCRIPT SAYS SO. It never states WHICH
policy per arm. On the 2026-08-14 data best-of-arm and final-of-arm give OPPOSITE
answers (ue=1 wins best-of-arm by 0.004; ue=4 wins on finals), so reporting only
the reading this script implements would let the ambiguity quietly pick the
flattering side. Both are printed, always, along with:

  * whether either arm improved at all -- a verdict comparing two FLAT arms is a
    comparison of noise, and the pass/fail line alone cannot show that;
  * whether an arm REGRESSED after its peak, which means a long run at that
    setting spends its last hours losing rating;
  * whether the margin is inside the best-of-N selection bias.

When any of those fire, a technical pass is reported as "PASS ON THE LETTER OF THE
RULE, BUT DO NOT ACT ON IT ALONE" with the reasons listed, and the exit code is
non-zero. A flat or regressing arm is a reason to fix training, not to crown a
winner.
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
  # strength, and the plan asks for mid AND final snapshots precisely so a
  # late collapse cannot masquerade as the arm's ceiling.
  b1 = max(ue1, key=lambda p: p["rating"])
  b4 = max(ue4, key=lambda p: p["rating"])
  lo1, hi1 = b1["rating_ci"]
  lo4, hi4 = b4["rating_ci"]

  # BOTH readings, always. The rule never said WHICH policy per arm,
  # and on the 2026-08-14 data best-of-arm and final-of-arm give OPPOSITE answers
  # (ue=1 by 0.004 on best-of-arm; ue=4 head-to-head on finals). Printing only the
  # one this script happens to implement would let an underspecified rule quietly
  # pick the flattering side.
  f1 = max(ue1, key=lambda p: p["birth_update"] or 0)
  f4 = max(ue4, key=lambda p: p["birth_update"] or 0)
  caveats = []
  print("\n=== final-of-arm (the policy a long run would actually ship) ===")
  print(f"  ue=1 final: {f1['id']}  rating={f1['rating']:+.3f} "
        f"ci=[{f1['rating_ci'][0]:+.3f},{f1['rating_ci'][1]:+.3f}]")
  print(f"  ue=4 final: {f4['id']}  rating={f4['rating']:+.3f} "
        f"ci=[{f4['rating_ci'][0]:+.3f},{f4['rating_ci'][1]:+.3f}]")
  if f1["rating_ci"][0] > f4["rating_ci"][1]:
    print("  on FINALS: ue=1 clears ue=4")
  elif f4["rating_ci"][0] > f1["rating_ci"][1]:
    print("  on FINALS: ue=4 clears ue=1  <- OPPOSITE of a best-of-arm verdict")
    caveats.append("finals disagree with best-of-arm")
  else:
    print("  on FINALS: intervals overlap, no separation")

  # Did either arm improve at all? A verdict comparing two flat arms is a
  # comparison of noise, and that is not visible from the pass/fail line.
  print("\n=== did each arm improve with training? ===")
  for name, arm in (("ue=4", ue4), ("ue=1", ue1)):
    ordered = sorted(arm, key=lambda p: p["birth_update"] or 0)
    first, last = ordered[0], ordered[-1]
    peak = max(arm, key=lambda p: p["rating"])
    print(f"  {name}: u{first['birth_update']} {first['rating']:+.3f} -> "
          f"final u{last['birth_update']} {last['rating']:+.3f} "
          f"(delta {last['rating'] - first['rating']:+.3f}); "
          f"peak u{peak['birth_update']} {peak['rating']:+.3f}")
    if last["rating_ci"][1] < peak["rating_ci"][0]:
      print(f"    !! REGRESSED: the final policy is CI-clear WORSE than the peak. "
            f"A long run at this setting spends its last hours losing rating.")
      caveats.append(f"{name} regresses after its peak")
    spread = max(p["rating"] for p in arm) - min(p["rating"] for p in arm)
    width = max(p["rating_ci"][1] - p["rating_ci"][0] for p in arm)
    if spread < width:
      print(f"    !! FLAT: whole-arm rating spread {spread:.3f} is smaller than a "
            f"single CI ({width:.3f}). This arm did not measurably learn.")
      caveats.append(f"{name} did not measurably learn")

  print("\n=== the rule (best-of-arm) ===")
  print(f"  best ue=1 : {b1['id']}  rating={b1['rating']:+.3f} "
        f"ci=[{lo1:+.3f},{hi1:+.3f}]")
  print(f"  best ue=4 : {b4['id']}  rating={b4['rating']:+.3f} "
        f"ci=[{lo4:+.3f},{hi4:+.3f}]")
  print(f"  ue=1 lower bound {lo1:+.3f} vs ue=4 upper bound {hi4:+.3f}")
  if b1["birth_update"] and b4["birth_update"]:
    gap = abs(b1["birth_update"] - b4["birth_update"])
    if gap > 500:
      print(f"  !! the two maxima are {gap} updates apart, so this is not a "
            f"controlled comparison of epoch counts")
      caveats.append(f"maxima {gap} updates apart")
  if lo1 - hi4 < 0.02:
    print(f"  !! margin is only {lo1 - hi4:+.3f}; best-of-N selection over "
        f"{len(ue1)}/{len(ue4)} policies inflates both arms by more than that")
    caveats.append(f"margin {lo1 - hi4:+.3f} inside best-of-N selection noise")

  if lo1 > hi4 and not caveats:
    print("\n  PASS -- update_epochs=1 clears update_epochs=4 cleanly. Use ue=1.")
    return 0
  if lo1 > hi4:
    print("\n  PASS ON THE LETTER OF THE RULE, BUT DO NOT ACT ON IT ALONE.")
    print("  ue=1's best clears ue=4's best, yet:")
    for c in caveats:
      print(f"    - {c}")
    print("  The rule does not say WHICH policy per arm, and best-of-arm vs "
          "final-of-arm\n  disagree here. Decide which you meant BEFORE looking "
          "at the ratings, and treat\n  a flat or regressing arm as a reason to "
          "fix training rather than to pick a winner.")
    return 1
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
