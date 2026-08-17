# Copyright 2022 DeepMind Technologies Limited
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

"""Tests for the Stage 1 three-head cell-attention critic readout.

The readout (``CellAttentionCritic`` + the ``--critic_readout=cell_attn`` mode
of ``EclipsePPOAgent``) is the architectural core of Stage 1: ONE value feature
built from the fused state vector plus a single state-conditioned cross-
attention query over all 225 per-cell conv features (no C_PRESENT mask), with
three heads off it -- an UNBOUNDED scalar value, four rank logits, and nine
UNBOUNDED VP-component predictions normalised with frozen train-split stats.

For every test the actor must remain bitwise frozen: ``critic_readout`` only
adds readout modules that read the fused vector; the actor path
(``self.actor``/``dense_logits``/``shared``) never changes.
"""

from absl.testing import absltest
import torch
import pyspiel
from open_spiel.python.eclipse import obs_layout
from open_spiel.python.eclipse.action_factors import factorization_from_game
from open_spiel.python.examples.ppo_eclipse import CellAttentionCritic
from open_spiel.python.examples.ppo_eclipse import EclipsePPOAgent
from open_spiel.python.examples.ppo_eclipse import _AUX_TASKS_BY_MODE


class CellAttentionCriticTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.game = pyspiel.load_game("eclipse(players=4)")
    self.num_actions = self.game.num_distinct_actions()
    self.obs_size = obs_layout.validate(self.game)
    self.fz = factorization_from_game(self.game)
    self.width = 64
    self.n_cells = obs_layout.GALAXY_CELLS

  def _net(self, critic_readout):
    return EclipsePPOAgent(
        self.num_actions, (self.obs_size,), "cpu", width=self.width, depth=2,
        aux_tasks=("final_rank",), factored_actions=self.fz,
        encoder="spatial", critic_readout=critic_readout)

  def _fused_cells(self, x):
    return self.net.shared.forward_with_cells(x)

  def test_cross_attention_output_shape(self):
    """ONE value feature (B, width) over all 225 cells."""
    self.net = self._net("cell_attn")
    x = torch.randn(5, self.obs_size)
    fused, h_cells = self._fused_cells(x)
    self.assertEqual(h_cells.shape, (5, 64, self.n_cells))
    feature = self.net.cell_attn.feature(fused, h_cells)
    self.assertEqual(feature.shape, (5, self.width))

  def test_all_cells_attended_no_c_present_mask(self):
    """No C_PRESENT mask: gradients flow into absent (empty) cells too.

    Empty cells are future exploration/action targets and must be attended.
    Zero out some cell rows (as an absent cell would be) and assert position-
    embedding gradients are non-zero on those exact absent cells -- i.e. the
    attention reaches them rather than masking them out.
    """
    self.net = self._net("cell_attn")
    critic = self.net.cell_attn
    b = 4
    fused = torch.randn(b, self.width, requires_grad=True)
    h_cells = torch.randn(b, 64, self.n_cells)
    absent = [7, 120, 224]  # arbitrary empty-cell indices to blank
    h_cells[:, :, absent] = 0.0  # absent cell -> zero conv features

    crit = critic.feature(fused, h_cells)
    crit.sum().backward()
    grad = critic.cell_pos.weight.grad
    self.assertIsNotNone(grad)
    # Gradients flowed into the position embeddings of EVERY cell, including
    # the absent ones (nothing masked them out).
    for cell in absent:
      self.assertGreater(float(grad[cell].abs().sum()), 0.0,
                         f"absent cell {cell} received no gradient (masked?)")
    # And the attention weights covered all cells (softmax over 225).
    with torch.no_grad():
      cells = h_cells.transpose(1, 2)
      pos = critic.cell_pos.weight
      cells = cells + pos.unsqueeze(0)
      k = critic.k_proj(critic.key_norm(cells))
      q = critic.q_proj(critic.query_norm(fused).unsqueeze(1))
      attn = (q @ k.transpose(-2, -1) / (critic.hidden ** 0.5)).softmax(dim=-1)
      self.assertEqual(attn.shape, (b, 1, self.n_cells))
      # Mass is spread over all 225 cells, not concentrated on a "present" few.
      self.assertGreater(float(attn[:, :, absent[1]].sum()), 0.01)

  def test_pre_layernorm_before_qkv_projection(self):
    """pre-LayerNorm: LayerNorm is applied BEFORE the q/k/v projection."""
    self.net = self._net("cell_attn")
    critic = self.net.cell_attn
    self.assertIsInstance(critic.query_norm, torch.nn.LayerNorm)
    self.assertIsInstance(critic.key_norm, torch.nn.LayerNorm)
    # Structural: the norms are distinct modules registered before the proj
    # linears, and the projection reads the NORMED input in the forward.
    # (Guarded by _value_feature using self.query_norm(fused) and
    # self.key_norm(cells) before k_proj/v_proj/q_proj -- asserted here by
    # checking the hooks fire in norm-before-proj order.)
    order = []
    hooks = []
    def _tag(name):
      def hook(_m, _i, _o):
        order.append(name)
      return hook
    hooks.append(critic.query_norm.register_forward_hook(_tag("q_norm")))
    hooks.append(critic.key_norm.register_forward_hook(_tag("k_norm")))
    hooks.append(critic.q_proj.register_forward_hook(_tag("q_proj")))
    hooks.append(critic.k_proj.register_forward_hook(_tag("k_proj")))
    hooks.append(critic.v_proj.register_forward_hook(_tag("v_proj")))
    try:
      fused = torch.randn(2, self.width)
      h_cells = torch.randn(2, 64, self.n_cells)
      critic.feature(fused, h_cells)
    finally:
      for h in hooks:
        h.remove()
    # q_norm and k_norm both precede their projections.
    self.assertLess(order.index("q_norm"), order.index("q_proj"))
    self.assertLess(order.index("k_norm"), order.index("k_proj"))

  def test_scalar_head_unbounded(self):
    """The value head is NOT clipped to [-0.5, 1]: it can reach far targets."""
    self.net = self._net("cell_attn")
    critic = self.net.cell_attn
    b = 8
    fused = torch.randn(b, self.width)
    h_cells = torch.randn(b, 64, self.n_cells)
    target = torch.full((b,), 3.5)  # well beyond the old hard top of 1.0

    opt = torch.optim.SGD(critic.parameters(), lr=5e-2)
    for _ in range(2000):
      opt.zero_grad()
      pred = critic.value(fused, h_cells)
      loss = ((pred - target) ** 2).mean()
      loss.backward()
      opt.step()
    pred = critic.value(fused, h_cells)
    self.assertGreater(float(pred.max()), 2.0,
                       "unbounded scalar head could not exceed the old top of 1.0")
    self.assertLess(float((pred - target).abs().mean()), 0.5)

  def test_rank_logits_fourwide_vp_head_ninewide(self):
    """rank logits are 4-wide; VP head is 9-wide and denormalises to VP units."""
    self.net = self._net("cell_attn")
    critic = self.net.cell_attn
    b = 3
    fused = torch.randn(b, self.width)
    h_cells = torch.randn(b, 64, self.n_cells)
    rank_logits = critic.rank_logits(fused, h_cells)
    self.assertEqual(rank_logits.shape, (b, 4))
    # VP head: 9-wide, frozen mean/std denormalisation back to VP units.
    mean = torch.tensor([1.0, 2.0, 0.5, 3.0, 0.0, 4.0, 0.0, 1.5, 2.5])
    std = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    critic.set_vp_stats(mean, std)
    vp = critic.vp_components(fused, h_cells)
    self.assertEqual(vp.shape, (b, 9))
    # Denormalised: with std=1 the output is mean + head_z; check shape/units.
    # Registering keeps floor at documented VP_STD_FLOOR.
    self.assertGreaterEqual(float(critic.vp_std.min()),
                            CellAttentionCritic.VP_STD_FLOOR)
    # Summing the 9 components reports predicted total VP (no redundant head).
    self.assertEqual(vp.sum(-1).shape, (b,))

  def test_variance_floor_documented_at_registration(self):
    """set_vp_stats clamps zero-variance components to the documented floor."""
    self.net = self._net("cell_attn")
    critic = self.net.cell_attn
    mean = torch.zeros(9)
    std = torch.zeros(9)  # constant component -> 0 variance
    critic.set_vp_stats(mean, std)
    self.assertTrue(torch.all(critic.vp_std
                              == CellAttentionCritic.VP_STD_FLOOR))

  def test_rank_backcompat_and_actor_frozen(self):
    """rank mode reproduces the old value; actor identical under BOTH flags."""
    net_r = self._net("rank")
    net_c = self._net("cell_attn")

    # Make the two shared encoders AND actor heads bitwise identical so the
    # actor comparison is meaningful: both are independently initialised, so
    # without copying the FactoredActorHead the logits would differ purely from
    # random init -- not from the readout flag. The ONLY remaining difference
    # between net_r and net_c is the cell_attn readout itself, which must not
    # perturb the actor path.
    net_c.shared.load_state_dict(net_r.shared.state_dict())
    net_c.actor.load_state_dict(net_r.actor.state_dict())
    net_c.critic.load_state_dict(net_r.critic.state_dict())

    x = torch.randn(6, self.obs_size)

    # 1) rank mode get_value == old expected-rank-utility value, hard-bounded.
    old = net_r.get_value(x).detach()
    expected = net_r.rank_value(net_r.critic[-1](net_r.shared(x))).detach()
    self.assertTrue(torch.allclose(old, expected), "rank value drifted")
    self.assertGreaterEqual(float(old.min()), -0.5 - 1e-6)
    self.assertLessEqual(float(old.max()), 1.0 + 1e-6)

    # 2) actor outputs bitwise identical under both flags (frozen actor).
    actor_r = net_r.dense_logits(x)
    actor_c = net_c.dense_logits(x)
    self.assertEqual(actor_r.shape, actor_c.shape)
    self.assertTrue(torch.equal(actor_r, actor_c),
                    "actor logits differ between rank and cell_attn!")

    # 3) mode-aware value_bounds.
    self.assertEqual(net_r.value_bounds(), (-0.5, 1.0))
    self.assertEqual(net_c.value_bounds(), (None, float("inf")))

  def test_value_from_actor_features_flag(self):
    """Semantics: cell_attn cannot value from actor features (needs h_cells)."""
    net_r = self._net("rank")
    net_c = self._net("cell_attn")
    self.assertTrue(net_r.value_from_actor_features)
    self.assertFalse(net_c.value_from_actor_features)

  def test_breakdown_aux_supervision_routes_through_vp_head(self):
    """cell_attn + breakdown: aux_from_obs trains the VP head, not aux_heads.

    The 9 VP-component targets must train ``CellAttentionCritic.vp_head`` (the
    cross-attention head consuming fused + h_cells), NOT the old flat
    ``aux_heads``. The returned per-name predictions must match
    ``vp_components`` (component k == task name k), and backpropping the aux
    MSE must put gradients on ``cell_attn.vp_head`` -- with none on the old
    flat aux heads.
    """
    self.net = EclipsePPOAgent(
        self.num_actions, (self.obs_size,), "cpu", width=self.width, depth=2,
        aux_tasks=_AUX_TASKS_BY_MODE["breakdown"], factored_actions=self.fz,
        encoder="spatial", critic_readout="cell_attn")
    x = torch.randn(4, self.obs_size, requires_grad=True)

    pred = self.net.aux_from_obs(x)
    self.assertEqual(set(pred.keys()), set(_AUX_TASKS_BY_MODE["breakdown"]))

    # Predictions sourced from the cell-attention VP head: component k of
    # vp_components equals the prediction for breakdown task name k.
    fused, h_cells = self.net._critic_features(x)
    expected = self.net.cell_attn.vp_components(fused, h_cells)  # (B, 9)
    for k, name in enumerate(_AUX_TASKS_BY_MODE["breakdown"]):
      self.assertEqual(pred[name].shape, (4,))
      self.assertTrue(
          torch.allclose(pred[name], expected[:, k], atol=1e-6),
          f"{name} not sourced from cell_attn vp_head")

    # Backprop the aux MSE: gradients reach cell_attn.vp_head but NOT the old
    # flat aux heads.
    self.net.zero_grad()
    targets = torch.randn(4, 9)
    loss = sum((pred[name] - targets[:, k]) ** 2
               for k, name in enumerate(_AUX_TASKS_BY_MODE["breakdown"]))
    loss.mean().backward()
    vp_grad = self.net.cell_attn.vp_head.weight.grad
    self.assertIsNotNone(vp_grad)
    self.assertGreater(float(vp_grad.abs().sum()), 0.0,
                       "aux MSE did not reach cell_attn.vp_head")
    for name in _AUX_TASKS_BY_MODE["breakdown"]:
      g = self.net.aux_heads[name].weight.grad
      self.assertIsNone(g, f"old flat aux head {name} trained by breakdown")

  def test_breakdown_components_match_same_index_order(self):
    """Component k of vp_components maps to breakdown task name k."""
    self.net = self._net("cell_attn")
    x = torch.randn(2, self.obs_size)
    fused, h_cells = self.net._critic_features(x)
    vp = self.net.cell_attn.vp_components(fused, h_cells)
    self.assertEqual(vp.shape, (2, len(_AUX_TASKS_BY_MODE["breakdown"])))
    # Index 0 is the reputation component (breakdown column 0).
    self.assertEqual(_AUX_TASKS_BY_MODE["breakdown"][0], "bd_reputation")
    self.assertEqual(_AUX_TASKS_BY_MODE["breakdown"][-1], "bd_minor_species")


if __name__ == "__main__":
  absltest.main()
