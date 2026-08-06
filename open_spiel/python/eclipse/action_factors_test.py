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

    packed = agent._pack_logits(feats, torch.from_numpy(rows.astype(np.int64)),
                               torch.from_numpy(cols), net.actor[-1],
                               net.actor[-1].bias)
    dense = net.actor(batch)
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


if __name__ == "__main__":
  absltest.main()
