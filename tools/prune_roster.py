#!/usr/bin/env python3
"""Build a subsampled view of a roster dir, for affordable ladder tournaments.

WHY THIS IS NEEDED
  roster_ladder runs a FULL round-robin: p*(p-1)/2 pairs x 2 directions x
  --ladder_games_per_dir games. p grows linearly with snapshots, so the game
  count grows quadratically. Two T1 arms at --snapshot_every=100 over 5h produce
  ~10 snapshots each; with main and the three bots that is p=25, i.e. 300 pairs
  and 76,800 full Eclipse games. That is not a judgement step, it is a second
  experiment.

  The plan asks for "final and mid snapshots", not every snapshot. This
  writes a roster.json holding a chosen subset -- always main and the extreme
  snapshots, plus evenly-spaced interior ones -- and leaves the weights where
  they are (roster.json paths are repo-relative, so nothing is copied).

  Subsampling is NOT the same as lowering --ladder_games_per_dir. Dropping games
  widens every rating CI, which directly weakens T1's pass test (a lower bound
  must clear an upper bound). Dropping redundant near-adjacent snapshots costs
  almost no information -- adjacent snapshots inside one run are nearly
  indistinguishable, which is why the ladder has --ladder_min_sep at all.
"""
import argparse
import json
import os
import shutil


def pick(entries, keep):
  """`keep` snapshots spanning the birth-update range, endpoints included."""
  snaps = sorted((e for e in entries if e["role"] == "snapshot"),
                 key=lambda e: e["birth_update"])
  if keep >= len(snaps) or keep <= 0:
    return snaps
  if keep == 1:
    return [snaps[-1]]
  # Evenly spaced by POSITION in birth order, endpoints pinned. Spacing by
  # birth_update value instead would cluster wherever the snapshot cadence
  # changed mid-run, which is exactly what runs/roster does (u25, u50, then a
  # dense u1450..u1575 block).
  idx = [round(i * (len(snaps) - 1) / (keep - 1)) for i in range(keep)]
  return [snaps[i] for i in sorted(set(idx))]


def pin(entries, chosen, birth):
  """Adds the snapshot born nearest `birth` to `chosen`, unless already there.

  The gate pins the previous gate's `main` here so each tournament holds the
  policy from one chunk ago: newest-vs-oldest (the endpoints `pick` keeps) only
  ever says "still beats u25", which stays true long after progress stops.
  """
  snaps = [e for e in entries if e["role"] == "snapshot"]
  if not snaps:
    return chosen
  near = min(snaps, key=lambda e: abs(e["birth_update"] - birth))
  if any(e["policy_id"] == near["policy_id"] for e in chosen):
    return chosen
  return sorted(chosen + [near], key=lambda e: e["birth_update"])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--keep", type=int, default=4,
                  help="snapshots to keep per roster (main is always kept)")
  ap.add_argument("--pin_birth", type=int, default=None,
                  help="also keep the snapshot born nearest this update, on "
                       "top of the --keep budget")
  args = ap.parse_args()

  with open(os.path.join(args.src, "roster.json")) as f:
    entries = json.load(f)
  main_e = [e for e in entries if e["role"] == "main"]
  # `main` is always kept, and the final snapshot shares its birth_update -- the
  # two are byte-identical weights. roster_ladder detects that (its
  # identical-policy gate requires direction 2 to mirror direction 1 exactly), but
  # rating them against each other still spends a full pair's games on a
  # comparison whose answer is known to be zero. Drop the duplicate first so the
  # `keep` budget buys `keep` DISTINCT policies.
  main_births = {e["birth_update"] for e in main_e}
  candidates = [e for e in entries
                if not (e["role"] == "snapshot"
                        and e["birth_update"] in main_births)]
  dropped = len(entries) - len(candidates)
  chosen = pick(candidates, args.keep)
  if args.pin_birth is not None:
    chosen = pin(candidates, chosen, args.pin_birth)

  os.makedirs(args.dst, exist_ok=True)
  arch = os.path.join(args.src, "arch.json")
  if os.path.exists(arch):
    shutil.copy(arch, os.path.join(args.dst, "arch.json"))
  out = chosen + main_e
  with open(os.path.join(args.dst, "roster.json"), "w") as f:
    json.dump(out, f, indent=2)

  print(f"{args.src} -> {args.dst}")
  print(f"  {len(entries)} entries -> {len(out)} "
        f"({len(chosen)} snapshots + {len(main_e)} main"
        f"{f'; dropped {dropped} duplicating main' if dropped else ''})")
  for e in out:
    print(f"    {e['policy_id']:<16} birth={e['birth_update']}")


if __name__ == "__main__":
  main()
