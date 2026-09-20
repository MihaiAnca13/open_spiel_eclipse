#!/usr/bin/env python3
"""Stage one roster that points at several runs' policies, for a cross-arm FFA.

`ffa_metagame.py` takes exactly four policy ids from a SINGLE --metagame_roster_dir,
so arms that trained in separate run dirs cannot be compared as they sit. Roster
entries carry an explicit `path`, though, so the staged roster is just an index
pointing at the existing weight files -- nothing is copied, and the arms keep
running while this is built.

All staged policies must share one architecture; arch.json is copied from the
first source and the others are checked against it, because _load_policies builds
ONE agent_fn for all four ids and a mismatch would surface as a confusing
load error rather than "these are not comparable".

Usage:
  python tools/stage_crossarm_roster.py --out runs/crossarm \
      ctrl=runs/main_v3/main.pt anneal=runs/lr_anneal/main.pt \
      refresh=runs/league_refresh/main.pt ref=runs/main_v3/snap_u275.pt
"""
import argparse
import json
import os
import shutil


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True, help="staged roster dir to create")
  ap.add_argument("policies", nargs="+", metavar="id=path/to/weights.pt")
  args = ap.parse_args()

  entries = []
  arch_src = None
  for spec in args.policies:
    if "=" not in spec:
      raise SystemExit(f"expected id=path, got {spec!r}")
    policy_id, path = spec.split("=", 1)
    if not os.path.exists(path):
      raise SystemExit(f"no weights at {path}")
    # arch.json lives beside the run dir that owns the weights.
    run_dir = os.path.dirname(path)
    arch_path = os.path.join(run_dir, "arch.json")
    if not os.path.exists(arch_path):
      raise SystemExit(f"no arch.json beside {path}")
    with open(arch_path) as f:
      arch = json.load(f)
    if arch_src is None:
      arch_src, arch_first = arch_path, arch
    elif arch != arch_first:
      raise SystemExit(
          f"architecture mismatch: {arch_path} differs from {arch_src}; "
          "these policies cannot share one agent_fn")
    entries.append({
        "policy_id": policy_id,
        "role": "snapshot",
        "birth_update": 0,
        "path": path,
        "win_rate": None,
    })

  ids = [e["policy_id"] for e in entries]
  if len(set(ids)) != len(ids):
    raise SystemExit(f"duplicate policy ids: {ids}")

  os.makedirs(args.out, exist_ok=True)
  shutil.copy(arch_src, os.path.join(args.out, "arch.json"))
  with open(os.path.join(args.out, "roster.json"), "w") as f:
    json.dump(entries, f, indent=2)
  print(f"staged {len(entries)} policies in {args.out}: {','.join(ids)}")
  print(f"arch.json copied from {arch_src}")


if __name__ == "__main__":
  main()
