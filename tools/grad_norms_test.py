# Copyright 2026 DeepMind Technologies Limited
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
"""TDD tests for tools/grad_norms.py (RED -> GREEN)."""

import numpy as np
import torch

from open_spiel.python.eclipse import frozen_dataset
from open_spiel.python.examples import ppo_eclipse as pe

# Small, CPU-only image of the frozen dataset schema so these tests run without
# a collection run or a GPU. The spatial encoder only needs obs rows of length
# OBS_LEN (it slices inside that); values are random but shape-valid.
OBS_LEN = 37596
NUM_ACTIONS = 11117
BATCH = 8


def _make_batch():
  """Synthetic rows matching the frozen_dataset schema."""
  np.random.seed(0)
  obs = np.random.rand(BATCH, OBS_LEN).astype(np.float32)
  rank_target = np.random.rand(BATCH, 4).astype(np.float32)
  rank_target /= rank_target.sum(axis=1, keepdims=True)
  return {
      "obs": torch.from_numpy(obs),
      "return_": torch.rand(BATCH).float(),
      "rank_target": torch.from_numpy(rank_target),
      "vp_components": torch.rand(BATCH, 9).float(),
  }


def _make_net():
  """A cell_attn EclipsePPOAgent on CPU with a legit small encoder width."""
  agent_fn = pe.make_agent_fn(
      width=64, depth=1, aux_tasks=("final_rank",), activation="tanh",
      encoder="spatial", critic_readout="cell_attn")
  net = agent_fn(NUM_ACTIONS, (OBS_LEN,), torch.device("cpu"))
  net.train()
  return net


def test_each_objective_contributes_independently():
  """Gradients are additive: dropping one objective removes exactly its share.

  aux_coef tunes how much the VP objective contributes to the shared trunk
  gradient. The scan must measure each objective independently so that scaling
  one (or zeroing it) changes the total by exactly its own contribution. This
  pins gradient additivity and that the VP objective carries its own, non-zero,
  separable trunk gradient distinct from value/rank/policy.
  """
  import grad_norms as gn

  net = _make_net()
  batch = _make_batch()
  trunk_params = list(net.shared.parameters())
  x = batch["obs"].requires_grad_(True)

  g_vp = gn.objective_trunk_grad("vp", net, x, batch, trunk_params)
  assert float(torch.norm(g_vp)) > 0.0

  # Gradient of the SUM equals the SUM of the independent ones (linearity).
  # => each objective's trunk contribution is separable and correctly attributed.
  summed = gn.objective_trunk_grad("value", net, x, batch, trunk_params)
  for name in ("rank", "policy", "vp"):
    summed = summed + gn.objective_trunk_grad(name, net, x, batch, trunk_params)
  g_all = torch.autograd.grad(
      sum(gn.objective_loss(name, net, x, batch) for name in
          ("value", "rank", "policy", "vp")),
      trunk_params, retain_graph=True)
  g_all = torch.cat([g.reshape(-1) for g in g_all])
  assert torch.allclose(summed, g_all, atol=1e-3), (
      "per-objective grads must sum to the joint gradient")

  # Dropping the VP objective (aux_coef -> 0) removes exactly the VP share.
  g_no_vp = gn.objective_trunk_grad("value", net, x, batch, trunk_params)
  for name in ("rank", "policy"):
    g_no_vp = g_no_vp + gn.objective_trunk_grad(name, net, x, batch,
                                                trunk_params)
  assert torch.allclose(g_all - g_vp, g_no_vp, atol=1e-3), (
      "removing VP must subtract exactly its trunk gradient")


def test_shares_sum_to_one():
  """Attributed trunk-gradient shares sum to ~1.0 regardless of scaling."""
  import grad_norms as gn

  net = _make_net()
  batch = _make_batch()
  x = batch["obs"].requires_grad_(True)
  shares, _ = gn.objective_shares(net, x, batch)
  total = sum(shares[k] for k in shares)
  assert abs(total - 1.0) < 1e-3, f"shares must sum to 1, got {total}"
  for k, v in shares.items():
    assert 0.0 <= v <= 1.0 + 1e-3, f"{k} share {v} out of range"
