#!/usr/bin/env python3
"""Distinct trainable parameter count for a roster checkpoint.

Why this is not `sum(v.numel() for v in state_dict.values())`: with
``--separate_critic=False`` the ONE shared encoder is registered under four
names -- ``shared``, ``critic_trunk``, ``critic.0`` and ``actor.0`` -- and
``state_dict()`` emits an entry for each. Naively summing therefore counts the
encoder four times. On the T1 checkpoints that reads 1,897,461 against a true
~657k, i.e. it overstates by 2.9x, which is exactly the kind of error that makes
a small network look adequately sized.

De-duplication is by (shape, bit-pattern) rather than ``data_ptr()`` because
``torch.load`` of a saved state_dict materializes each entry separately: the
aliases share storage in the live module but not in the file. Two genuinely
distinct tensors colliding would need identical shape AND identical bytes.

Usage:  count_params.py <roster_dir | checkpoint.pt>
"""
import hashlib
import os
import sys

import torch


def _key(t):
  # reshape(-1) first: `mask_value` is 0-dim, and a byte view of a 0-dim
  # tensor raises rather than returning an empty buffer.
  flat = t.detach().cpu().contiguous().reshape(-1)
  return (tuple(t.shape), hashlib.blake2b(
      flat.numpy().tobytes(), digest_size=16).hexdigest())


def main():
  if len(sys.argv) != 2:
    print(__doc__.strip().splitlines()[-1])
    return 2
  path = sys.argv[1]
  if os.path.isdir(path):
    cand = [os.path.join(path, n) for n in ("main.pt",)]
    cand += sorted(os.path.join(path, n)
                   for n in os.listdir(path) if n.endswith(".pt"))
    path = next((c for c in cand if os.path.exists(c)), None)
    if path is None:
      print("params: no checkpoint found")
      return 0

  sd = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(sd, dict):
    print("params: unexpected checkpoint format")
    return 0

  raw = 0
  seen = {}
  for name, t in sd.items():
    if not torch.is_tensor(t):
      continue
    raw += t.numel()
    seen.setdefault(_key(t), []).append(name)

  trainable = 0
  buffers = 0
  aliased = []
  for (shape, _), names in seen.items():
    n = 1
    for d in shape:
      n *= d
    # Integer tensors in this checkpoint are the action-factorization lookup
    # tables (actor.1.decode, .cell_id, .unit_id, .slot_id, .seat_id,
    # .direction_id, .family_id -- 111,170 entries at 11,117 actions). They are
    # registered buffers, carry no gradient, and are NOT capacity. Reporting
    # them as parameters overstates a width-64 net by 20%.
    if sd[names[0]].dtype in (torch.int64, torch.int32, torch.int16,
                              torch.uint8, torch.bool):
      buffers += n
    else:
      trainable += n
    if len(names) > 1:
      aliased.append((names[0], len(names)))

  distinct = trainable + buffers
  print(f"params: trainable={trainable:,}  +int_buffers={buffers:,}  "
        f"= distinct {distinct:,}   state_dict_sum={raw:,} "
        f"(x{raw / max(distinct, 1):.2f})  from {os.path.basename(path)}")
  if aliased:
    top = sorted(aliased, key=lambda x: -x[1])[:3]
    print("        aliased tensors: "
          + ", ".join(f"{n} x{c}" for n, c in top))
  return 0


if __name__ == "__main__":
  sys.exit(main())
