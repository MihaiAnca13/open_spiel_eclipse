# Copyright 2026
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

"""The fp16 rollout observation buffer must not cost real precision.

``PPO._resolve_obs_buffer_dtype`` stores the rollout observations in float16
when ``--amp`` is on, halving a buffer that is gigabytes at realistic env
counts. The argument for why that is free is precision-based: the buffer is read
in exactly one place (the ``_learn_core`` minibatch gather) and every consumer of
that minibatch runs inside a **bfloat16** autocast region. fp16 keeps 10 mantissa
bits, bf16 keeps 7 (plus implicit) -- so fp16 storage is strictly finer than the
arithmetic that immediately consumes it.

That argument was load-bearing for a merge and nothing tested it. Comparing two
training runs cannot test it either: this stack's loss series is bit-identical
for two updates and then diverges chaotically from GPU reduction order alone, so
a diff at update 20 says nothing. What *is* measurable is the thing the argument
actually claims, at the point of consumption:

    error(fp16 storage) << error(bf16 compute, which is already accepted)

If that inequality ever flips, fp16 storage has started discarding something the
autocast was not already discarding, and the buffer dtype must be pinned to
float32 before any long run.
"""

from absl.testing import absltest
import numpy as np
import torch

import pyspiel
from open_spiel.python.eclipse import obs_layout
from open_spiel.python.examples.ppo_eclipse import SpatialEclipseEncoder

SEED = 20260813
BATCH = 96


def _real_observations(n):
  """`n` observation rows from actual mid-game Eclipse states.

  Synthetic ``randn`` input would not test this: the observation is 97.5% zeros
  with the rest one-hots and small ratios, and rounding behaves differently on
  that distribution than on unit-variance noise.
  """
  game = pyspiel.load_game("eclipse(players=4)")
  rng = np.random.RandomState(SEED)
  rows = []
  buf = np.zeros(obs_layout.TOTAL, dtype=np.float32)
  while len(rows) < n:
    state = game.new_initial_state()
    steps = 0
    while not state.is_terminal() and steps < 250 and len(rows) < n:
      if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        probs = np.asarray(probs, dtype=np.float64)
        state.apply_action(int(rng.choice(outcomes, p=probs / probs.sum())))
      else:
        legal = state.legal_actions()
        state.apply_action(int(legal[rng.randint(len(legal))]))
      steps += 1
      if steps % 5 == 0 and not state.is_terminal():
        state.observation_tensor_into(max(state.current_player(), 0), buf)
        rows.append(buf.copy())
  return np.stack(rows[:n])


class ObsBufferDtypeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(SEED)
    self.obs = torch.from_numpy(_real_observations(BATCH))
    # Production shape (run_v2.sh): width 64, depth 2, tanh.
    self.encoder = SpatialEclipseEncoder(
        width=64, depth=2, activation="tanh").eval()

  def _features(self, stored_dtype, compute_dtype):
    """Encode after a round trip through `stored_dtype`, under `compute_dtype`.

    Mirrors the real path: the buffer stores `stored_dtype`, ``_ObsRows`` upcasts
    the gathered minibatch back to float32, and the autocast region then does the
    arithmetic in `compute_dtype`.
    """
    x = self.obs.to(stored_dtype).to(torch.float32)
    with torch.no_grad():
      if compute_dtype is torch.float32:
        return self.encoder.forward_with_context(x)[0].float()
      with torch.autocast("cpu", dtype=compute_dtype):
        return self.encoder.forward_with_context(x)[0].float()

  @staticmethod
  def _rel(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(a)).item()

  def test_fp16_storage_error_is_dominated_by_the_bf16_autocast_error(self):
    exact = self._features(torch.float32, torch.float32)
    storage_only = self._features(torch.float16, torch.float32)
    compute_only = self._features(torch.float32, torch.bfloat16)
    both = self._features(torch.float16, torch.bfloat16)

    e_storage = self._rel(exact, storage_only)
    e_compute = self._rel(exact, compute_only)
    e_both = self._rel(exact, both)

    # The claim being defended: what fp16 storage costs is small next to what
    # the autocast already costs. 4x margin, not 1x, so a marginal regression
    # trips this before it reaches a long run.
    self.assertLess(
        e_storage * 4.0, e_compute,
        f"fp16 storage error {e_storage:.2e} is no longer negligible against "
        f"the bf16 autocast error {e_compute:.2e} that the learn path already "
        f"accepts -- pin --obs_buffer_dtype=float32 and re-derive the argument "
        f"in PPO._resolve_obs_buffer_dtype.")

    # And adding fp16 storage on top of the autocast must not compound: the
    # combined error should still be about the autocast's own.
    self.assertLess(e_both, 2.0 * e_compute,
                    f"combined error {e_both:.2e} exceeds 2x the autocast-only "
                    f"error {e_compute:.2e}; the two are interacting.")

  def test_fp16_round_trip_preserves_the_observation_itself(self):
    """No observation value may be destroyed outright by fp16 storage.

    The tensor is one-hots and ``Frac`` ratios clamped to [-1, 1]. The failure
    this guards is a future field written with a divisor that pushes real values
    below fp16's subnormal range, where they would land on exact zero and become
    invisible to the network without anything erroring.
    """
    rt = self.obs.to(torch.float16).to(torch.float32)
    nonzero = self.obs != 0
    self.assertGreater(int(nonzero.sum()), 0)
    lost = nonzero & (rt == 0)
    self.assertEqual(
        int(lost.sum()), 0,
        f"{int(lost.sum())} nonzero observation values became exactly 0 under "
        f"fp16 storage; smallest magnitude present is "
        f"{self.obs[nonzero].abs().min().item():.3e}")

    rel = ((rt - self.obs).abs() / self.obs.abs().clamp(min=1e-12))[nonzero]
    self.assertLess(rel.max().item(), 1e-2)


if __name__ == "__main__":
  absltest.main()
