# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity tests for the sparse step-side acting path.

``PPO._act_sparse`` samples from packed legal logits only (no dense
(num_envs, num_actions) mask/logits in the per-step hot loop). It must be
distributionally identical to the dense ``CategoricalMasked`` path: same
log_prob / entropy / value, and sampled actions always legal.
"""

import numpy as np
import torch

from absl.testing import absltest
from open_spiel.python.pytorch import ppo
from open_spiel.python.pytorch import ppo_win_test

NUM_ACTIONS = 60
OBS_SIZE = 12
BATCH = 16


def _masked(maximal=True, logits=True):
  pass


class SparseActParityTest(absltest.TestCase):

  def _make_agent(self):
    return ppo.PPO(
        input_shape=(OBS_SIZE,),
        num_actions=NUM_ACTIONS,
        num_players=2,
        device="cpu",
        num_envs=BATCH,
        steps_per_batch=8,
        agent_fn=ppo_win_test._EclipseLikeAgent,  # pylint: disable=protected-access
        value_mode="win",
        aux_tasks=None,
    )

  def _random_legal(self, rng, num_rows=None):
    """Random (mask_rows, mask_cols) packed legal entries per row (contiguous)."""
    num_rows = BATCH if num_rows is None else num_rows
    rows = []
    cols = []
    for i in range(num_rows):
      n = int(rng.randint(1, 12))
      c = rng.choice(NUM_ACTIONS, size=n, replace=False)
      rows.append(np.full(n, i, dtype=np.int64))
      cols.append(c.astype(np.int64))
    return (np.concatenate(rows), np.concatenate(cols))

  def test_sparse_matches_dense(self):
    rng = np.random.RandomState(3)
    torch.manual_seed(3)
    agent = self._make_agent()
    obs = torch.from_numpy(rng.randn(BATCH, OBS_SIZE).astype(np.float32))
    mask_rows, mask_cols = self._random_legal(rng)
    seats = np.tile(np.array([0, 1]), BATCH // 2)

    with torch.no_grad():
      action, logprob, entropy, value = agent._act_sparse(  # pylint: disable=protected-access
          obs, mask_rows, mask_cols, seats)

    # Sampled actions must be legal (in-row) and never the sentinel.
    for i in range(BATCH):
      legal = set(mask_cols[mask_rows == i])
      self.assertIn(int(action[i]), legal)
    self.assertTrue(bool((action < NUM_ACTIONS).all()))

    # Dense reference.
    mask = torch.zeros((BATCH, NUM_ACTIONS), dtype=torch.bool)
    mask[torch.from_numpy(mask_rows), torch.from_numpy(mask_cols)] = True
    _, d_logprob_batch, d_entropy, d_value, d_probs = agent.network.get_action_and_value(
        obs, legal_actions_mask=mask, action=action)

    d_logprob = d_logprob_batch.detach()
    self.assertTrue(np.allclose(logprob.numpy(), d_logprob.numpy(), atol=1e-4),
                    f"logprob mismatch\n{logprob.numpy()}\n{d_logprob.numpy()}")
    self.assertTrue(np.allclose(entropy.numpy(), d_entropy.detach().numpy(), atol=1e-3),
                    f"entropy mismatch\n{entropy.numpy()}\n{d_entropy.detach().numpy()}")
    self.assertTrue(np.allclose(value.numpy(), d_value.detach().numpy(), atol=1e-5),
                    f"value mismatch\n{value.numpy()}\n{d_value.detach().numpy()}")

  def test_sparse_empty_legal_row_does_not_oob(self):
    """A row with no legal entries must not leak the num_actions sentinel into
    the gather kernel (an OOB device-side assert that killed a long run)."""
    rng = np.random.RandomState(11)
    agent = self._make_agent()
    obs = torch.from_numpy(rng.randn(BATCH, OBS_SIZE).astype(np.float32))
    # Rows 0..BATCH-2 have legal actions; the LAST row has none.
    rows = []
    cols = []
    for i in range(BATCH - 1):
      n = int(rng.randint(1, 12))
      c = rng.choice(NUM_ACTIONS, size=n, replace=False)
      rows.append(np.full(n, i, dtype=np.int64))
      cols.append(c.astype(np.int64))
    mask_rows = np.concatenate(rows)
    mask_cols = np.concatenate(cols)
    seats = np.tile(np.array([0, 1]), BATCH // 2)

    # Must not throw: the empty row keeps the sentinel which is clamped to a
    # valid index so the head_logits gather stays in bounds.
    action, logprob, entropy, _ = agent._act_sparse(  # pylint: disable=protected-access
        obs, mask_rows, mask_cols, seats)
    self.assertTrue(bool((action < NUM_ACTIONS).all()))
    # Every non-empty row still samples a legal action.
    for i in range(BATCH - 1):
      legal = set(mask_cols[mask_rows == i])
      self.assertIn(int(action[i]), legal)

  def test_sparse_sample_distribution_matches_dense(self):
    """Empirical sampling distribution matches the dense masked categorical."""
    rng = np.random.RandomState(7)
    agent = self._make_agent()
    obs = torch.from_numpy(rng.randn(4, OBS_SIZE).astype(np.float32))
    mask_rows, mask_cols = self._random_legal(rng, num_rows=4)
    b = 4
    mask = torch.zeros((b, NUM_ACTIONS), dtype=torch.bool)
    mask[torch.from_numpy(mask_rows), torch.from_numpy(mask_cols)] = True
    _, _, _, _, d_probs = agent.network.get_action_and_value(
        obs, legal_actions_mask=mask)
    d_probs = d_probs.detach().numpy()

    n_samples = 4000
    counts = np.zeros((b, NUM_ACTIONS))
    seats = np.zeros(b, dtype=np.int64)
    for _ in range(n_samples):
      with torch.no_grad():
        a, _, _, _ = agent._act_sparse(  # pylint: disable=protected-access
            obs[:b], mask_rows, mask_cols, seats)
      counts[np.arange(b), a.numpy()] += 1
    empirical = counts / n_samples
    for i in range(b):
      legal = mask_cols[mask_rows == i]
      expected = d_probs[i, legal]
      got = empirical[i, legal]
      # Loose absolute agreement on a well-separated small support.
      max_err = np.abs(got - expected).max()
      self.assertLess(max_err, 0.03,
                      f"row {i} empirical vs expected max err {max_err:.4f}")


if __name__ == "__main__":
  absltest.main()
