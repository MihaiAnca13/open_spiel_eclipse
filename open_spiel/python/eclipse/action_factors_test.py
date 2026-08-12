# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the factored Eclipse action head."""

from absl.testing import absltest
import numpy as np
import torch

import pyspiel
from open_spiel.python.eclipse.action_factors import build_action_factorization
from open_spiel.python.eclipse.action_factors import factorization_from_game
from open_spiel.python.eclipse.action_factors import NUM_SLOTS
from open_spiel.python.eclipse import obs_layout
from open_spiel.python.examples.ppo_eclipse import EclipsePPOAgent
from open_spiel.python.examples.ppo_eclipse import build_aux_targets
from open_spiel.python.pytorch.ppo import CategoricalMasked
from open_spiel.python.pytorch.ppo import PPO


class ActionFactorizationTest(absltest.TestCase):

  def test_decode_is_injective(self):
    """Distinct actions must decode to distinct row sets.

    If two actions shared all their factor rows the head could not express a
    different preference between them -- the factorization would silently reduce
    the policy class.
    """
    game = pyspiel.load_game("eclipse(players=4)")
    fz = factorization_from_game(game)
    num_actions = game.num_distinct_actions()
    self.assertEqual(fz.decode.shape, (num_actions, NUM_SLOTS))
    self.assertGreaterEqual(int(fz.decode.min()), 0)
    self.assertLess(int(fz.decode.max()), fz.num_rows)
    distinct = len(set(map(tuple, fz.decode.tolist())))
    self.assertEqual(distinct, num_actions)

  def test_shrinks_the_head_substantially(self):
    game = pyspiel.load_game("eclipse(players=4)")
    fz = factorization_from_game(game)
    self.assertLess(fz.num_rows, game.num_distinct_actions() // 5)
    # The big product families must actually be recognized, else the win is lost.
    for family, at_least in (("colony", 5000), ("upgrade", 1500),
                             ("build", 1300), ("move_unit", 700)):
      self.assertGreaterEqual(fz.stats.get(family, 0), at_least, family)

  def test_unparsed_actions_get_their_own_row(self):
    fz = build_action_factorization(["PASS", "RESEARCH", "SOMETHING_ODD"])
    self.assertEqual(fz.stats["atom"], 3)
    self.assertEqual(len(set(map(tuple, fz.decode.tolist()))), 3)


class FactoredActorHeadTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.game = pyspiel.load_game("eclipse(players=4)")
    self.num_actions = self.game.num_distinct_actions()
    self.obs_size = obs_layout.validate(self.game)
    self.fz = factorization_from_game(self.game)

  def _flat_net(self):
    return EclipsePPOAgent(
        self.num_actions, (self.obs_size,), "cpu", width=64, depth=2,
        aux_tasks=("final_rank",), factored_actions=self.fz, encoder="flat")

  def _sparse_agent(self, net):
    return PPO(input_shape=(self.obs_size,), num_actions=self.num_actions,
               num_players=4, num_envs=3, steps_per_batch=4, device="cpu",
               agent_fn=lambda n, s, d: net, value_mode="win")

  def _assert_sparse_matches_dense(self, net, agent, batch):
    context_fn = getattr(net.shared, "forward_with_context", None)
    if context_fn is None:
      feats, context = net.shared(batch), None
    else:
      feats, context = context_fn(batch)
    legal = [sorted(np.random.RandomState(k).choice(
        self.num_actions, size=40, replace=False).tolist())
        for k in range(batch.shape[0])]
    rows = np.concatenate([[i] * len(l) for i, l in enumerate(legal)])
    cols = np.concatenate(legal).astype(np.int64)

    packed = agent._pack_logits(feats, context,
                               torch.from_numpy(rows.astype(np.int64)),
                               torch.from_numpy(cols), net.actor[-1])
    dense = net.dense_logits(batch)
    reference = dense[torch.from_numpy(rows.astype(np.int64)),
                      torch.from_numpy(cols)]
    self.assertLess(float((packed - reference).abs().max().detach()), 1e-5)

    mask = torch.zeros(batch.shape[0], self.num_actions, dtype=torch.bool)
    for i, l in enumerate(legal):
      mask[i, l] = True
    dist = CategoricalMasked(logits=dense, masks=mask,
                             mask_value=torch.tensor(-1e6))
    _, entropy = agent._segment_lse_entropy(
        packed, torch.from_numpy(rows.astype(np.int64)), batch.shape[0])
    self.assertLess(float((entropy - dist.entropy()).abs().max().detach()), 1e-4)

  def test_rows_for_matches_materialized_weight(self):
    net = self._flat_net()
    head = net.actor[-1]
    idx = torch.tensor([0, 332, 5731, 7541, 9142, self.num_actions - 1])
    self.assertTrue(torch.allclose(head.rows_for(idx),
                                   head.full_weight()[idx], atol=1e-6))

  def test_forward_is_the_implied_linear_map(self):
    """Flat encoder: logits are exactly features @ W^T + b."""
    net = self._flat_net()
    x = torch.randn(3, self.obs_size)
    feats = net.shared(x)
    expected = feats @ net.actor[-1].full_weight().t() + net.actor[-1].bias
    self.assertTrue(torch.allclose(net.actor(x), expected, atol=1e-5))

  def test_sparse_path_matches_dense_logits_and_entropy(self):
    """The whole point: masking and the distribution must be unchanged."""
    net = self._flat_net()
    agent = self._sparse_agent(net)
    self.assertTrue(agent._sparse_supported())
    self._assert_sparse_matches_dense(net, agent, torch.randn(3, self.obs_size))

  def test_spatial_encoder_sparse_and_shape(self):
    """The spatial encoder (the default run path) must also feed the sparse
    factored head, and must slice the real flat tensor without a shape error."""
    # ponytail: seeded -- the spatial branch MLPs have no input normalization
    # (real observations are bounded [0,1]; unseeded torch.randn synthetic
    # input is not), so an unlucky random draw sends logits to ~1e8 and the
    # dense-vs-sparse comparison's *absolute* 1e-5 tolerance below sees plain
    # float32 rounding as a mismatch -- pre-existing on git HEAD too (repro'd
    # with seeds 1,4,5,7,9,13 before this test covered typed pointers).
    # Fix the *test's* tolerance/scale if this ever needs unseeding.
    torch.manual_seed(0)
    net = EclipsePPOAgent(
        self.num_actions, (self.obs_size,), "cpu", width=64, depth=2,
        aux_tasks=("final_rank",), factored_actions=self.fz, encoder="spatial")
    agent = self._sparse_agent(net)
    self.assertTrue(agent._sparse_supported())
    latent = net.shared(torch.randn(3, self.obs_size))
    self.assertEqual(latent.shape, (3, 64))
    self._assert_sparse_matches_dense(net, agent, torch.randn(3, self.obs_size))

  def test_gradient_reaches_shared_factor_rows(self):
    """A cell row must receive gradient from every action targeting that cell."""
    net = self._flat_net()
    x = torch.randn(2, self.obs_size)
    logits = net.actor(x)
    # Two colony-ship actions on the same cell differ only in slot/track.
    logits[:, 332].sum().backward()
    grad = net.actor[-1].embedding.grad
    touched = set(self.fz.decode[332].tolist())
    self.assertTrue(all(float(grad[r].abs().sum()) > 0 for r in touched))
    untouched = set(range(self.fz.num_rows)) - touched
    self.assertTrue(all(float(grad[r].abs().sum()) == 0
                        for r in list(untouched)[:50]))

  def test_v2_typed_dense_sparse_equivalence(self):
    """Every action's typed V2 dense and sparse logit must agree exactly."""
    net = EclipsePPOAgent(
        self.num_actions, (self.obs_size,), "cpu", width=8, depth=1,
        aux_tasks=("final_rank",), factored_actions=self.fz, encoder="spatial")
    net.actor[-1].DENSE_CHUNK_SIZE = 32
    agent = self._sparse_agent(net)
    state = self.game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    x = torch.tensor([state.observation_tensor(0)], dtype=torch.float32)
    with torch.no_grad():
      features, context = net.shared.forward_with_context(x)
      dense = net.dense_logits(x)
      for start in range(0, self.num_actions, 32):
        cols = torch.arange(start, min(start + 32, self.num_actions))
        rows = torch.arange(x.shape[0]).repeat_interleave(cols.numel())
        sparse = agent._pack_logits(
            features, context, rows, cols.repeat(x.shape[0]), net.actor[-1])
        self.assertTrue(torch.allclose(
            sparse, dense[:, start:start + cols.numel()].reshape(-1), atol=1e-5))


class TypedPointerGroundingTest(absltest.TestCase):
  """Does each pointer read the entity its action actually names?

  Dense-vs-sparse agreement (above) is NOT this test: both paths call the same
  ``_pairs``, so both are wrong together if a pointer reads the wrong row. These
  are falsifiers -- each swaps two entities' state and requires the two actions'
  pointer contributions to swap with them. They fail against a head with no
  pointer term, against one that feeds the query a globally-pooled vector, and
  against one that keys the wrong entity.

  The comparison is always on the pointer term alone (full logit minus the
  base-only logit): two different actions keep their own factored base weights,
  so the full logits have no reason to become numerically equal.
  """

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.game = pyspiel.load_game("eclipse(players=4)")
    self.num_actions = self.game.num_distinct_actions()
    self.obs_size = obs_layout.validate(self.game)
    self.fz = factorization_from_game(self.game)
    self.net = EclipsePPOAgent(
        self.num_actions, (self.obs_size,), "cpu", width=32, depth=1,
        aux_tasks=("final_rank",), factored_actions=self.fz, encoder="spatial")
    self.head = self.net.actor[-1]
    state = self.game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    self.x = torch.tensor([state.observation_tensor(0)] * 1, dtype=torch.float32)

  def _pointer_terms(self, context, actions):
    """Pointer-only contribution for each action id, as a (len(actions),) tensor."""
    rows = torch.zeros(len(actions), dtype=torch.long)
    cols = torch.tensor(actions, dtype=torch.long)
    features, _ = self.net.shared.forward_with_context(self.x)
    full = self.head.logits_for(features, context, rows, cols)
    base = (features[rows] * self.head.rows_for(cols)).sum(-1) + self.head.bias[cols]
    return full - base

  def _two_actions_keyed_by(self, table, a_key, b_key):
    ids = np.flatnonzero(table == a_key), np.flatnonzero(table == b_key)
    self.assertTrue(ids[0].size and ids[1].size)
    return int(ids[0][0]), int(ids[1][0])

  def _assert_swap(self, act_a, act_b, mutate):
    """Pointer terms of act_a/act_b must exchange when their entities do."""
    with torch.no_grad():
      _, ctx = self.net.shared.forward_with_context(self.x)
      before = self._pointer_terms(ctx, [act_a, act_b])
      # Distinguishable to begin with, else the swap proves nothing.
      self.assertNotAlmostEqual(float(before[0]), float(before[1]), places=5)
      after = self._pointer_terms(mutate(ctx), [act_a, act_b])
    self.assertAlmostEqual(float(after[0]), float(before[1]), places=5)
    self.assertAlmostEqual(float(after[1]), float(before[0]), places=5)

  def test_cell_pointer_reads_the_targeted_cell(self):
    a, b = self._two_actions_keyed_by(self.head.cell_id.numpy(), 0, 1)

    def mutate(ctx):
      cells = ctx.cells.clone()
      cells[0, :, 0], cells[0, :, 1] = ctx.cells[0, :, 1], ctx.cells[0, :, 0]
      return ctx._replace(cells=cells)

    self._assert_swap(a, b, mutate)

  def test_unit_pointer_reads_the_targeted_registry_row(self):
    """Isolated to COMBAT_TARGET_UNIT: those key a unit and nothing else.

    A MOVE_UNIT action keys a unit AND a route destination, so swapping only the
    unit embeddings would leave its route term behind and the swap would not be
    clean -- that would test the harness, not the pointer.
    """
    unit_id = self.head.unit_id.numpy()
    unit_only = np.flatnonzero((unit_id >= 0) &
                               (self.head.direction_id.numpy() < 0))
    self.assertTrue(unit_only.size > 1)
    by_unit = {int(unit_id[a]): int(a) for a in unit_only}
    self.assertIn(0, by_unit)
    self.assertIn(1, by_unit)
    a, b = by_unit[0], by_unit[1]

    def mutate(ctx):
      units = ctx.units.clone()
      units[0, 0], units[0, 1] = ctx.units[0, 1], ctx.units[0, 0]
      return ctx._replace(units=units)

    self._assert_swap(a, b, mutate)

  def test_slot_pointer_reads_the_targeted_planet_slot(self):
    """A colony action keyed (cell, slot) must read cell*8+slot, not cell."""
    cell_id, slot_id = self.head.cell_id.numpy(), self.head.slot_id.numpy()
    with torch.no_grad():
      _, ctx = self.net.shared.forward_with_context(self.x)
      valid = (ctx.slots[0, :, 0] >= 0.5).numpy()
    # A cell that really carries two distinct planet slots. Cell 0 is a corner
    # of the dense 15x15 grid and holds no sector at all, so its slot rows are
    # both zero and no correct pointer could tell them apart.
    per_cell = valid.reshape(obs_layout.GALAXY_CELLS,
                             obs_layout.PLANET_SLOTS_PER_CELL)
    cells = np.flatnonzero(per_cell[:, 0] & per_cell[:, 1])
    self.assertTrue(cells.size, "no cell with two populated planet slots")
    cell = int(cells[0])
    same = np.flatnonzero((cell_id == cell) & (slot_id >= 0))
    by_slot = {int(slot_id[a]): int(a) for a in same}
    a, b = by_slot[0], by_slot[1]
    row_a = cell * obs_layout.PLANET_SLOTS_PER_CELL
    row_b = row_a + 1

    def mutate(ctx):
      slots = ctx.slots.clone()
      slots[0, row_a], slots[0, row_b] = ctx.slots[0, row_b], ctx.slots[0, row_a]
      return ctx._replace(slots=slots)

    self._assert_swap(a, b, mutate)

  def test_seat_pointer_reads_the_valid_seat_row_and_ignores_padding(self):
    """Padding seat rows decode to absolute seat 0 and must never be read.

    In a 4-player game slots 4 and 5 are empty but still decode to absolute
    seat 0, so the pre-fix reduction -- a ``max`` over the seat *logits* -- let
    'propose to seat 0' be won by a constant padding embedding. Two assertions
    pin the fix down: perturbing only the padding rows must change nothing, and
    perturbing the row that really holds the target seat must change something
    (otherwise the first assertion could pass on a pointer that reads no seat
    at all).

    Note what actually kills the bug: selecting the FIRST matching slot rather
    than the arg-max over logits. Valid slots always precede padding, so
    ``seat_valid`` is a second line of defence, not the primary one.
    """
    seat_id = self.head.seat_id.numpy()
    actions = [int(a) for a in np.flatnonzero(seat_id >= 0)]
    self.assertTrue(actions)
    with torch.no_grad():
      _, ctx = self.net.shared.forward_with_context(self.x)
      self.assertFalse(bool(ctx.seat_valid[0, 4]), "slot 4 should be padding")
      self.assertFalse(bool(ctx.seat_valid[0, 5]), "slot 5 should be padding")
      self.assertEqual(int(ctx.seat_abs[0, 4]), 0,
                       "padding decodes to absolute seat 0 -- that is the trap")
      before = self._pointer_terms(ctx, actions)
      seats = ctx.seats.clone()
      seats[0, 4] += 50.0
      seats[0, 5] -= 50.0
      after = self._pointer_terms(ctx._replace(seats=seats), actions)
      self.assertLess(float((after - before).abs().max()), 1e-6)
      # Sensitivity: the seat rows that ARE valid must reach these logits, or
      # the assertion above would hold for a pointer that reads no seat at all.
      real = ctx.seats.clone()
      real[0, :4] += 10.0
      moved = self._pointer_terms(ctx._replace(seats=real), actions)
    self.assertGreater(float((moved - before).abs().max()), 1e-3)

  def test_pop_target_slots_beyond_the_table_are_rejected(self):
    """COMBAT_POP_TARGET_8..15 must not alias into the next cell's slot rows."""
    slot_id = self.head.slot_id.numpy()
    cell_id = self.head.cell_id.numpy()
    over = [int(a) for a in np.flatnonzero(
        (slot_id >= obs_layout.PLANET_SLOTS_PER_CELL) & (cell_id < 0))]
    self.assertTrue(over, "expected COMBAT_POP_TARGET ids with slot >= 8")
    with torch.no_grad():
      _, ctx = self.net.shared.forward_with_context(self.x)
      # Force a live pop-attack cell so the slot branch is otherwise reachable.
      ctx = ctx._replace(pop_cell=torch.zeros_like(ctx.pop_cell))
      before = self._pointer_terms(ctx, over)
      slots = ctx.slots.clone()
      slots += 25.0            # perturb EVERY slot row
      mutated = ctx._replace(slots=slots)
      after = self._pointer_terms(mutated, over)
      # Rejected, so no slot row can move these logits at all.
      self.assertLess(float((after - before).abs().max()), 1e-6)
      # Sensitivity: the in-range pop targets on the same cell DO move under the
      # identical perturbation, so the perturbation is real and it is the bound
      # -- not an inert test -- that stops the out-of-range ones.
      in_range = [int(a) for a in np.flatnonzero(
          (slot_id >= 0) & (slot_id < obs_layout.PLANET_SLOTS_PER_CELL)
          & (cell_id < 0))]
      self.assertTrue(in_range)
      moved = (self._pointer_terms(mutated, in_range)
               - self._pointer_terms(ctx, in_range))
    self.assertGreater(float(moved.abs().max()), 1e-3)

  def test_move_direction_convention_matches_the_engine(self):
    """MOVE_UNIT_<i>_<NAME> must resolve to the neighbour the engine means.

    The direction NAME -> index mapping lives in two files that share nothing:
    ``kDirNames`` in eclipse.cc and the tuple in action_factors.py. Reorder
    either and every move pointer silently reads the wrong neighbour, with no
    shape error and no test failing. This walks the whole chain -- action string
    -> direction_id -> V2 route row -> cell id -- against hex arithmetic done
    independently here.
    """
    dirs = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))  # E NE NW W SW SE
    obs = self.x[0].numpy()
    units = obs[obs_layout.V2_UNITS_START:
                obs_layout.V2_UNITS_START
                + obs_layout.UNIT_ROWS * obs_layout.UNIT_ROW_SIZE].reshape(
                    obs_layout.UNIT_ROWS, obs_layout.UNIT_ROW_SIZE)
    routes = obs[obs_layout.V2_UNIT_ROUTES_START:
                 obs_layout.V2_UNIT_ROUTES_START
                 + obs_layout.UNIT_ROWS * obs_layout.UNIT_ROUTE_SIZE].reshape(
                     obs_layout.UNIT_ROWS, obs_layout.UNIT_ROUTE_SIZE)
    radius = obs_layout.GALAXY_DIM // 2
    checked = 0
    for i in range(obs_layout.UNIT_ROWS):
      if units[i, obs_layout.U_VALID] < 0.5:
        continue
      q = int(round(float(units[i, obs_layout.U_Q]) * radius))
      r = int(round(float(units[i, obs_layout.U_R]) * radius))
      for d, (dq, dr) in enumerate(dirs):
        raw = float(routes[i, d])
        if raw <= 0.0:
          continue                      # explicit "no neighbour" sentinel
        got = int(round(raw * obs_layout.GALAXY_CELLS)) - 1
        self.assertEqual(got, obs_layout.hex_to_index(q + dq, r + dr),
                         f"unit {i} direction {d} resolves to the wrong cell")
        checked += 1
    self.assertGreater(checked, 0, "no unit routes available to check")

    # And the factorization must agree with that same index order.
    strings = {}
    state = self.game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    for a in range(self.num_actions):
      strings[state.action_to_string(state.current_player(), a)] = a
    for name, idx in (("E", 0), ("NE", 1), ("NW", 2),
                      ("W", 3), ("SW", 4), ("SE", 5)):
      action = strings.get(f"MOVE_UNIT_0_{name}")
      self.assertIsNotNone(action, f"MOVE_UNIT_0_{name} not in the action space")
      self.assertEqual(int(self.fz.direction_id[action]), idx,
                       f"{name} must map to HEX_DIRECTIONS index {idx}")

  def test_every_v2_block_reaches_the_features(self):
    """No V2 sub-block may be write-only.

    1,835 of the 12,882 V2 floats (14.2%) were written every step and read by
    nothing -- the tech-bag histogram, the revealed-discovery ledger, the keyed
    seat tech tracks and the whole combat queue. Perturb each block in turn; the
    fused features must move, or that block is dead weight in the tensor again.
    """
    blocks = {
        "v2_global": (obs_layout.V2_GLOBAL_START, obs_layout.V2_GLOBAL_SIZE),
        "v2_cells": (obs_layout.V2_CELLS_START,
                     obs_layout.GALAXY_CELLS * obs_layout.V2_CELL_SIZE),
        "v2_seat_tech": (obs_layout.V2_SEATS_START + obs_layout.VS_TECH_TRACKS,
                         obs_layout.TECH_TRACK_COUNT * obs_layout.TECH_BIT_COUNT),
        "v2_combat": (obs_layout.V2_COMBAT_START, obs_layout.V2_COMBAT_SIZE),
        "v2_units": (obs_layout.V2_UNITS_START,
                     obs_layout.UNIT_ROWS * obs_layout.UNIT_ROW_SIZE),
    }
    with torch.no_grad():
      base, _ = self.net.shared.forward_with_context(self.x)
      for name, (start, size) in blocks.items():
        x = self.x.clone()
        x[0, start:start + size] += 0.25
        moved, _ = self.net.shared.forward_with_context(x)
        self.assertGreater(float((moved - base).abs().max()), 1e-5,
                           f"{name} does not reach the features")

  def test_unit_attention_is_nan_safe_with_no_valid_units(self):
    """A sample with zero valid units must not poison the batch with NaN.

    Masking every key of a query row makes MultiheadAttention emit NaN, which
    would silently destroy a whole minibatch's loss.
    """
    with torch.no_grad():
      x = self.x.clone()
      units_at = obs_layout.V2_UNITS_START
      x[0, units_at:units_at
        + obs_layout.UNIT_ROWS * obs_layout.UNIT_ROW_SIZE] = 0.0
      features, ctx = self.net.shared.forward_with_context(x)
    self.assertTrue(torch.isfinite(features).all())
    self.assertTrue(torch.isfinite(ctx.units).all())

  def test_npc_units_do_not_borrow_a_player_seat_block(self):
    """kRelNpc(6)/kRelNone(7) owners need their own rows, not seat slot 5.

    The old head clamped the owner index to 5, so every NPC unit -- measured at
    61% of live units -- read an empty padding player block. Perturbing seat
    slot 5 must not move an NPC unit's embedding.
    """
    with torch.no_grad():
      x = self.x.clone()
      units_at = obs_layout.V2_UNITS_START
      # Row 0: valid, owner = kRelNpc (index 6 of the 8-wide one-hot).
      row = units_at + 0 * obs_layout.UNIT_ROW_SIZE
      x[0, row + obs_layout.U_VALID] = 1.0
      x[0, row + obs_layout.U_OWNER:row + obs_layout.U_OWNER
        + obs_layout.REL_SEAT_WIDTH] = 0.0
      x[0, row + obs_layout.U_OWNER + 6] = 1.0
      _, ctx = self.net.shared.forward_with_context(x)
      npc_before = ctx.units[0, 0].clone()
      # Corrupt seat slot 5's V1 block; an NPC unit must not react.
      x2 = x.clone()
      slot5 = obs_layout.player_block_start(5)
      x2[0, slot5:slot5 + obs_layout.PLAYER_SIZE] += 5.0
      _, ctx2 = self.net.shared.forward_with_context(x2)
      npc_after = ctx2.units[0, 0]
    self.assertLess(float((npc_after - npc_before).abs().max()), 1e-6)


class BuildAuxTargetsBreakdownTest(absltest.TestCase):
  """The ``breakdown`` aux mode reads the 9 VP categories from terminal obs."""

  def setUp(self):
    super().setUp()
    self.game = pyspiel.load_game("eclipse(players=4)")
    self.obs_size = obs_layout.validate(self.game)
    self.tasks, self.fn = build_aux_targets("breakdown", 30.0)

  def _terminal_obs(self, acting_seat):
    """A synthetic terminal obs with a distinct breakdown per seat."""
    obs = np.zeros(self.obs_size, dtype=np.float32)
    for s in range(4):
      slot = obs_layout.slot_for_seat(s, acting_seat, 4)
      base = (obs_layout.player_block_start(slot)
              + obs_layout.P_VP_BREAKDOWN)
      # Seat s's categories = s + 1...s + 9 (all within [0,1]).
      obs[base:base + 9] = np.arange(1.0, 10.0) * 0.01 + s
    return obs

  def test_task_names_are_the_nine_categories(self):
    self.assertEqual(len(self.tasks), 9)
    self.assertIn("bd_sector", self.tasks)
    self.assertIn("bd_traitor", self.tasks)

  def test_missing_obs_returns_none_for_unmasking(self):
    out = self.fn(np.array([40.0, 25.0, 10.0, 0.0], dtype=np.float32))
    self.assertIsNone(out)

  def test_reads_terminal_breakdown_per_seat(self):
    acting = 2
    obs = self._terminal_obs(acting)
    out = self.fn(np.zeros(4, dtype=np.float32), terminal_obs=obs,
                  acting_seat=acting)
    self.assertEqual(out.shape, (4, 9))
    for s in range(4):
      expected = np.arange(1.0, 10.0) * 0.01 + s
      np.testing.assert_allclose(out[s], expected, atol=1e-6)

  def test_seat_canonicalisation(self):
    # Each terminal obs is canonicalised to its own acting seat; the extractor
    # must therefore yield the same per-absolute-seat values regardless of who
    # acted last.
    for acting in range(4):
      obs = self._terminal_obs(acting)
      out = self.fn(np.zeros(4, dtype=np.float32), terminal_obs=obs,
                    acting_seat=acting)
      for s in range(4):
        np.testing.assert_allclose(
            out[s], np.arange(1.0, 10.0) * 0.01 + s, atol=1e-6, err_msg=f"a={acting} s={s}")


if __name__ == "__main__":
  absltest.main()
