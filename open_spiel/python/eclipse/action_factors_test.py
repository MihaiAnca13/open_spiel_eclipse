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

"""Tests for the factored Eclipse action head and the spatial encoder."""

from absl.testing import absltest
import numpy as np
import torch

import pyspiel
from open_spiel.python.eclipse.action_factors import build_action_factorization
from open_spiel.python.eclipse.action_factors import factorization_from_game
from open_spiel.python.eclipse.action_factors import NUM_SLOTS
from open_spiel.python.eclipse import obs_layout
from open_spiel.python.examples.ppo_eclipse import EclipsePPOAgent
from open_spiel.python.examples.ppo_eclipse import _argmax_over_legal
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
    feats = net.shared(batch)
    legal = [sorted(np.random.RandomState(k).choice(
        self.num_actions, size=40, replace=False).tolist())
        for k in range(batch.shape[0])]
    rows = np.concatenate([[i] * len(l) for i, l in enumerate(legal)])
    cols = np.concatenate(legal).astype(np.int64)

    packed = agent._pack_logits(feats,
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
    expected = dense.masked_fill(~mask, float("-inf")).argmax(dim=1).numpy()
    actual = _argmax_over_legal(
        net, batch.numpy(), rows, cols, np.arange(batch.shape[0]),
        torch.device("cpu"))
    np.testing.assert_array_equal(actual, expected)

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
    # float32 rounding as a mismatch.
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

  def test_spatial_norm_flag_reaches_branch_mlps(self):
    net = EclipsePPOAgent(
        self.num_actions, (self.obs_size,), "cpu", width=32, depth=1,
        aux_tasks=(), norm=True, factored_actions=self.fz, encoder="spatial")
    for name in ("self_mlp", "tail_mlp", "seat_mlp", "unit_mlp"):
      self.assertTrue(
          any(isinstance(layer, torch.nn.LayerNorm)
              for layer in getattr(net.shared, name)), name)

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

  def test_move_direction_convention_matches_the_engine(self):
    """MOVE_UNIT_<i>_<NAME> must resolve to the neighbour the engine means.

    The direction NAME -> index mapping lives in two files that share nothing:
    ``kDirNames`` in eclipse.cc and the tuple in action_factors.py. Reorder
    either and every move action silently reads the wrong neighbour, with no
    shape error and no test failing. This walks the whole chain -- action string
    -> direction_id -> V2 route row -> cell id -- against hex arithmetic done
    independently here.
    """
    game = self.game
    fz = self.fz
    num_actions = self.num_actions
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    x = torch.tensor([state.observation_tensor(0)], dtype=torch.float32)
    dirs = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))  # E NE NW W SW SE
    obs = x[0].numpy()
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
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    for a in range(num_actions):
      strings[state.action_to_string(state.current_player(), a)] = a
    for name, idx in (("E", 0), ("NE", 1), ("NW", 2),
                      ("W", 3), ("SW", 4), ("SE", 5)):
      action = strings.get(f"MOVE_UNIT_0_{name}")
      self.assertIsNotNone(action, f"MOVE_UNIT_0_{name} not in the action space")
      self.assertEqual(int(fz.direction_id[action]), idx,
                       f"{name} must map to HEX_DIRECTIONS index {idx}")

  def test_spatial_encoder_is_differentiable_and_finite(self):
    """A spatial encoder forward/backward over a real observation is finite.

    The unit self-attention block must not emit NaN when fed a real (mostly
    padding) unit table, and gradients must flow back through the fused
    features to the encoder parameters.
    """
    net = EclipsePPOAgent(
        self.num_actions, (self.obs_size,), "cpu", width=32, depth=1,
        aux_tasks=("final_rank",), factored_actions=self.fz, encoder="spatial")
    state = self.game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    x = torch.tensor([state.observation_tensor(0)], dtype=torch.float32)
    logits = net.actor(x)
    self.assertTrue(torch.isfinite(logits).all())
    logits.sum().backward()
    for name, p in net.shared.named_parameters():
      if p.grad is not None:
        self.assertTrue(torch.isfinite(p.grad).all(), name)

  def test_unit_attention_is_nan_safe_with_no_valid_units(self):
    """A sample with zero valid units must not poison the batch with NaN.

    Masking every key of a query row makes MultiheadAttention emit NaN, which
    would silently destroy a whole minibatch's loss.
    """
    net = EclipsePPOAgent(
        self.num_actions, (self.obs_size,), "cpu", width=32, depth=1,
        aux_tasks=("final_rank",), factored_actions=self.fz, encoder="spatial")
    state = self.game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    x = torch.tensor([state.observation_tensor(0)], dtype=torch.float32)
    with torch.no_grad():
      x = x.clone()
      units_at = obs_layout.V2_UNITS_START
      x[0, units_at:units_at
        + obs_layout.UNIT_ROWS * obs_layout.UNIT_ROW_SIZE] = 0.0
      features = net.shared(x)
    self.assertTrue(torch.isfinite(features).all())

  def test_every_v2_block_reaches_the_features(self):
    """No V2 sub-block may be write-only.

    1,835 of the 12,882 V2 floats (14.2%) were written every step and read by
    nothing -- the tech-bag histogram, the revealed-discovery ledger, the keyed
    seat tech tracks and the whole combat queue. Perturb each block in turn; the
    fused features must move, or that block is dead weight in the tensor again.
    """
    net = self._flat_net()
    state = self.game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    x = torch.tensor([state.observation_tensor(0)], dtype=torch.float32)
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
      base = net.shared(x)
      for name, (start, size) in blocks.items():
        x2 = x.clone()
        x2[0, start:start + size] += 0.25
        moved = net.shared(x2)
        self.assertGreater(float((moved - base).abs().max()), 1e-5,
                           f"{name} does not reach the features")


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
