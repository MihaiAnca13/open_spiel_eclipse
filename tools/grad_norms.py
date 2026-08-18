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
"""Gradient-norm scan: per-objective L2 share of the shared trunk gradient.

Picks ``--aux_coef`` for the VP-component aux objective under
``--critic_readout=cell_attn``. For a minibatch of the frozen diagnostic
dataset it measures the gradient each objective (policy / scalar value / rank
CE / aggregate VP) places on the SHARED encoder (trunk) parameters -- before
any global clip -- and reports each objective's L2 share of the total trunk
gradient. The recommendation is drawn from that share: VP should be material
but not dominant (~10-40%).

Objects are measured independently so the TDD invariant holds: freezing one
head (e.g. the VP head) zeroes its trunk contribution while the others keep
flowing.

Run (measurement only, no training):
  PYTHONPATH=build/open_spiel/python:. .venv/bin/python tools/grad_norms.py \
    --dataset runs/frozen_diag --batch 256
"""

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from open_spiel.python.eclipse import frozen_dataset
from open_spiel.python.examples import ppo_eclipse as pe

_OBJECTIVES = ("policy", "value", "rank", "vp")

# Tunable scan hyperparameters. The encoder width must match what the cell_attn
# critic is trained with (the settled 1024-env config builds width=64 depth=2).
_WIDTH = 64
_DEPTH = 2
_NUM_ACTIONS = 11117
_OBS_LEN = 37596


def make_net():
  """A cell_attn EclipsePPOAgent (current V2 spatial encoder), train mode."""
  agent_fn = pe.make_agent_fn(
      width=_WIDTH, depth=_DEPTH, aux_tasks=("final_rank",), activation="tanh",
      encoder="spatial", critic_readout="cell_attn")
  net = agent_fn(_NUM_ACTIONS, (_OBS_LEN,), torch.device("cpu"))
  net.train()
  return net


def _policy_loss(net, x, batch):
  """PPO-ish policy surrogate: self-sampled action, reward proxy as advantage.

  The frozen schema stores no recorded action, so the actor logits are sampled
  to build a log-prob; the advantage proxy is ``return_ - value.detach()``.
  This reproduces the magnitude of the policy-gradient term the optimizer sees.
  """
  logits = net.actor(x)
  dist = torch.distributions.Categorical(logits=logits)
  a = dist.sample()
  log_prob = dist.log_prob(a)
  value = net.value_from_obs(x)
  advantage = (batch["return_"] - value.detach())
  return -(advantage * log_prob).mean()


def _value_loss(net, x, batch):
  """0.5 * mean((scalar_value - return_)^2) -- the PPO value head."""
  return 0.5 * ((net.value_from_obs(x) - batch["return_"])**2).mean()


def _rank_loss(net, x, batch):
  """Soft-label cross-entropy of the 4 rank logits vs the soft rank target."""
  logits = net.rank_logits_from_obs(x)
  target = batch["rank_target"]
  per_row = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
  return per_row.mean()


def _vp_loss(net, x, batch):
  """Mean over the 9 VP-component heads of the MSE vs the recorded targets."""
  fused, h_cells = net._critic_features(x)
  pred = net.cell_attn.vp_components(fused, h_cells)
  target = batch["vp_components"]
  return ((pred - target)**2).mean()


def objective_loss(name, net, x, batch):
  """Forward loss for one objective on ``batch``."""
  fn = {"policy": _policy_loss, "value": _value_loss, "rank": _rank_loss,
        "vp": _vp_loss}[name]
  return fn(net, x, batch)


def objective_trunk_grad(name, net, x, batch, trunk_params):
  """L2 norm of the shared-trunk gradient attributable to one objective."""
  loss = objective_loss(name, net, x, batch)
  grads = torch.autograd.grad(loss, trunk_params, retain_graph=True,
                              allow_unused=True)
  return torch.cat([g.reshape(-1) for g in grads if g is not None])


def objective_shares(net, x, batch):
  """{objective: L2 share of total trunk grad} summing to ~1."""
  trunk_params = list(net.shared.parameters())
  norms = {}
  for name in _OBJECTIVES:
    g = objective_trunk_grad(name, net, x, batch, trunk_params)
    norms[name] = float(torch.norm(g))
  total = sum(norms.values()) or 1.0
  return {k: v / total for k, v in norms.items()}, norms


def _load_minibatch(dataset_dir, batch_size, device):
  """Sample `batch_size` rows (whole-obs rows) from the frozen dataset."""
  with open(f"{dataset_dir}/manifest.json") as f:
    manifest = json.load(f)
  rows = frozen_dataset.read_rows(dataset_dir, manifest)
  n = len(rows["obs"])
  idx = np.random.RandomState(0).choice(n, size=min(batch_size, n),
                                        replace=False)
  order = np.argsort(idx)
  idx = idx[order]
  return {
      "obs": torch.from_numpy(rows["obs"][idx]).to(device),
      "return_": torch.from_numpy(rows["return_"][idx]).float().to(device),
      "rank_target": torch.from_numpy(rows["rank_target"][idx]).float().to(
          device),
      "vp_components": torch.from_numpy(rows["vp_components"][idx]).float().to(
          device),
  }


def recommend_aux_coef(shares, floor=0.10, ceiling=0.40, baseline=1.0):
  """Scale ``baseline`` so VP lands at the midpoint of [floor, ceiling].

  If VP already dominates (>= ceiling at aux_coef=1.0), returns a value < 1; if
  it is negligible returns a value > 1, capped so the clip does not saturate.
  """
  target = 0.5 * (floor + ceiling)
  vp = shares["vp"]
  if vp <= 0.0:
    return baseline
  return baseline * (target / vp)


def main():
  ap = argparse.ArgumentParser(
      description=__doc__.split("\n")[0] if __doc__ else "grad_norms scan")
  ap.add_argument("--dataset", required=True)
  ap.add_argument("--batch", type=int, default=256)
  args = ap.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = make_net().to(device)
  batch = _load_minibatch(args.dataset, args.batch, device)
  x = batch["obs"].requires_grad_(True)

  shares, norms = objective_shares(net, x, batch)
  total_norm = sum(norms.values())
  print("\n=== cell_attn trunk gradient shares (pre-clip, per-objective) ===")
  for name in _OBJECTIVES:
    print(f"  {name:6s}: share={shares[name]*100:6.2f}%  "
          f"L2={norms[name]:9.3f}")
  print(f"  total  : L2={total_norm:9.3f}  "
        f"(max_grad_norm=0.5 -> {'SATURATING' if total_norm > 0.5 else 'ok'})")
  coef = recommend_aux_coef(shares)
  print(f"\nRecommended --aux_coef = {coef:.3f} "
        f"(targets VP ~ {0.5*(0.10+0.40)*100:.0f}% of trunk grad)")


if __name__ == "__main__":
  main()
