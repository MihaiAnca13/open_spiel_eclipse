#!/usr/bin/env python3
"""Equivalence gate for the _pairs / rows_for rewrite.

Rather than importing the pre-change module (which re-defines every absl flag and
raises DuplicateFlagError), this carries a verbatim reference implementation of the
ORIGINAL _pairs and rows_for and runs both against the SAME head instance -- so the
weights are identical by construction, not by a state_dict copy.

The two changes have different expectations and are asserted separately:

  * The combined cell-table pick is a refactor with no reassociation: the same
    values, accumulated in the same order. It must be BITWISE identical, and that
    is checked with the original rows_for restored so the two changes cannot mask
    each other.
  * rows_for's embedding_bag sums the same `slots` rows in a possibly different
    order, so it is checked against the bf16 autocast error --amp already accepts
    (4.5e-3), not against zero.

Also checks GRADIENTS, not just forward values -- the entire point of the change is
the backward pass, so a forward-only gate would miss a wrong gradient completely.
"""
import sys

import numpy as np
import torch
from absl import flags as absl_flags

import pyspiel


def old_rows_for(head, idx):
  """Original: materialises (len(idx), slots, width) then sums."""
  return head.embedding[head.decode[idx]].sum(dim=1)


def old_pairs(head, features, context, rows, cols, rows_for):
  """Verbatim pre-change _pairs: three separate cell-table picks, interleaved."""
  from open_spiel.python.eclipse import obs_layout
  f = features[rows]
  result = (f * rows_for(head, cols)).sum(-1) + head.bias[cols]
  zero = torch.zeros_like(result)
  family = head.family_id[cols]
  cells_t = context.cells.transpose(1, 2)

  cell = head.cell_id[cols]
  cell_q = head.cell_query(f) + head.cell_family(family)
  result = result + torch.where(
      cell >= 0, (cell_q * head._pick(cells_t, rows, cell)).sum(-1), zero)

  unit = head.unit_id[cols]
  unit_q = head.unit_query(f) + head.unit_family(family)
  result = result + torch.where(
      unit >= 0, (unit_q * head._pick(context.units, rows, unit)).sum(-1), zero)

  direction = head.direction_id[cols]
  route = context.routes[rows, unit.clamp(min=0), direction.clamp(min=0)]
  destination = (route * obs_layout.GALAXY_CELLS).round().long() - 1
  has_route = (unit >= 0) & (direction >= 0) & (route > 0)
  result = result + torch.where(
      has_route, (cell_q * head._pick(cells_t, rows, destination)).sum(-1), zero)

  slot = head.slot_id[cols]
  pop_cell = context.pop_cell[rows]
  keyed_slot = torch.where(
      cell >= 0, cell * obs_layout.PLANET_SLOTS_PER_CELL + slot,
      pop_cell * obs_layout.PLANET_SLOTS_PER_CELL + slot)
  has_slot = ((slot >= 0) & (slot < obs_layout.PLANET_SLOTS_PER_CELL)
              & ((cell >= 0) | (pop_cell >= 0)))
  slot_row = head._pick(context.slots, rows, keyed_slot)
  ptype = (slot_row[:, 1] * (obs_layout.PLANET_TYPE_COUNT - 1)
           ).round().long().clamp(0, obs_layout.PLANET_TYPE_COUNT - 1)
  slot_cell = head._pick(
      cells_t, rows,
      torch.div(keyed_slot, obs_layout.PLANET_SLOTS_PER_CELL,
                rounding_mode="floor"))
  slot_h = head.slot_target(
      torch.cat([slot_row, head.planet_type_embed(ptype), slot_cell], dim=-1))
  slot_q = head.slot_query(f) + head.slot_family(family)
  result = result + torch.where(has_slot, (slot_q * slot_h).sum(-1), zero)

  target_seat = head.seat_id[cols]
  matches = (context.seat_abs[rows].eq(target_seat.unsqueeze(-1))
             & context.seat_valid[rows])
  seat_q = head.seat_query(f) + head.seat_family(family)
  seat_h = head._pick(context.seats, rows, matches.float().argmax(dim=1))
  result = result + torch.where(
      (target_seat >= 0) & matches.any(dim=1),
      (seat_q * seat_h).sum(-1), zero)
  return result


def real_batch(game, ish, B, seed):
  """Real observations from real mid-game states, with their real legal sets."""
  rng = np.random.RandomState(seed)
  obs_list, rows_l, cols_l = [], [], []
  for b in range(B):
    s = game.new_initial_state()
    depth = rng.randint(30, 140)
    for _ in range(depth):
      if s.is_terminal():
        break
      if s.is_chance_node():
        oc, pr = zip(*s.chance_outcomes())
        s.apply_action(int(rng.choice(oc, p=np.asarray(pr))))
        continue
      la = s.legal_actions()
      s.apply_action(int(la[rng.randint(len(la))]))
    while s.is_terminal() or s.is_chance_node():
      if s.is_terminal():
        s = game.new_initial_state()
      oc, pr = zip(*s.chance_outcomes())
      s.apply_action(int(rng.choice(oc, p=np.asarray(pr))))
    seat = max(s.current_player(), 0)
    buf = np.zeros(int(np.prod(ish)), dtype=np.float64)
    s.observation_tensor_into(seat, buf)
    obs_list.append(buf.astype(np.float32))
    la = s.legal_actions()
    rows_l.extend([b] * len(la))
    cols_l.extend(int(a) for a in la)
  return (np.stack(obs_list), np.asarray(rows_l, dtype=np.int64),
          np.asarray(cols_l, dtype=np.int64))


def main():
  absl_flags.FLAGS(["pairs_equiv"])
  from open_spiel.python.examples import ppo_eclipse as pe
  from open_spiel.python.eclipse.action_factors import factorization_from_game
  from open_spiel.python.pytorch.ppo import shared_and_cells

  dev = "cuda"
  game = pyspiel.load_game("eclipse(players=4)")
  na = game.num_distinct_actions()
  ish = tuple(game.observation_tensor_shape())
  torch.manual_seed(1234)
  net = pe.make_agent_fn(64, 2, ("final_rank",), activation="tanh",
                         factored_actions=factorization_from_game(game),
                         encoder="spatial")(na, ish, dev).to(dev)
  net.eval()
  head = net.actor[1]
  print(f"head: {type(head).__name__}  embedding={tuple(head.embedding.shape)}"
        f"  decode={tuple(head.decode.shape)}")

  obs_np, rows_np, cols_np = real_batch(game, ish, 64, 0)
  obs = torch.from_numpy(obs_np).to(dev)
  rows = torch.from_numpy(rows_np).to(dev)
  cols = torch.from_numpy(cols_np).to(dev)
  uniq = len(np.unique(head.decode.cpu().numpy()[cols_np]))
  print(f"batch={obs.shape[0]} rows  pairs={len(cols_np)}  "
        f"mean legal/row={len(cols_np) / obs.shape[0]:.1f}  "
        f"distinct factor rows touched={uniq}/{head.embedding.shape[0]}")

  ok = True

  # ---- 1. forward: full change ------------------------------------------------
  with torch.no_grad():
    feats, ctx = shared_and_cells(net, obs)
    out_new = head.logits_for(feats, ctx, rows, cols)
    out_old = old_pairs(head, feats, ctx, rows, cols, old_rows_for)
  scale = out_old.abs().max().item()
  d_full = (out_new - out_old).abs().max().item()
  rel = d_full / max(scale, 1e-12)
  print(f"\n1. forward, both changes:  max|diff|={d_full:.3e}  "
        f"rel={rel:.2e}  (bitwise={torch.equal(out_new, out_old)})")
  print(f"   bf16 autocast error --amp already accepts: 4.5e-3")
  ok &= rel < 1e-3

  # ---- 2. forward: pick change ALONE must be bitwise --------------------------
  # Restore the original rows_for so the embedding_bag change cannot hide a
  # non-neutral pick refactor (or vice versa).
  saved = head.rows_for
  head.rows_for = lambda idx: old_rows_for(head, idx)
  with torch.no_grad():
    out_pick = head.logits_for(feats, ctx, rows, cols)
  head.rows_for = saved
  pick_bitwise = torch.equal(out_pick, out_old)
  print(f"\n2. forward, combined-pick change ALONE: bitwise={pick_bitwise} "
        f"<- must be True")
  if not pick_bitwise:
    print(f"   max|diff|={(out_pick - out_old).abs().max().item():.3e}  "
          f"!! the pick refactor is NOT value-neutral")
  ok &= pick_bitwise

  # ---- 3. GRADIENTS -- the whole point of the change --------------------------
  # A forward-only gate would pass a completely wrong backward.
  def grads(fn):
    net.zero_grad(set_to_none=True)
    feats_g, ctx_g = shared_and_cells(net, obs)
    out = fn(feats_g, ctx_g)
    out.square().mean().backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in net.named_parameters()}

  g_new = grads(lambda f_, c_: head.logits_for(f_, c_, rows, cols))
  g_old = grads(lambda f_, c_: old_pairs(head, f_, c_, rows, cols, old_rows_for))
  worst_name, worst_rel = None, 0.0
  missing = []
  for n in g_old:
    a, b = g_new[n], g_old[n]
    if a is None or b is None:
      if (a is None) != (b is None):
        missing.append(n)
      continue
    denom = max(b.abs().max().item(), 1e-12)
    r = (a - b).abs().max().item() / denom
    if r > worst_rel:
      worst_rel, worst_name = r, n
  print(f"\n3. gradients over {len(g_old)} params: worst relative diff "
        f"{worst_rel:.2e} on {worst_name}")
  if missing:
    print(f"   !! grad presence differs for: {missing}")
    ok = False
  ok &= worst_rel < 1e-3

  print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
