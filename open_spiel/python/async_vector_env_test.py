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

"""Tests for AsyncVectorEnv, focused on legal-action fidelity.

The legal-action buffer is fixed-width shared memory, so a too-small width
silently drops legal actions: the agent simply never sees them, the mask is
consistent, and nothing errors anywhere downstream. These tests pin the
invariant that the published legal set is exactly ``state.legal_actions(seat)``.
"""

from absl.testing import absltest
import numpy as np

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.async_vector_env import AsyncVectorEnv


def _game_string(seed):
  return (f"eclipse(players=4,rng_seed={seed},randomize_races=true,"
          f"randomize_npc_difficulty=true,randomize_warped=true)")


def _make_env(seed):
  return rl_environment.Environment(
      game=pyspiel.load_game(_game_string(seed)),
      chance_event_sampler=rl_environment.ChanceEventSampler(seed=seed),
      observation_type=rl_environment.ObservationType.OBSERVATION,
      observations_as_numpy=True)


class AsyncVectorEnvTest(absltest.TestCase):

  def test_max_legal_defaults_to_full_action_space(self):
    """No probing: a probed bound cannot see decision nodes it never sampled.

    Eclipse's initial state has ~13 legal actions while mid-game states reach
    ~130, so a bound probed from initial states truncated ~2/3 of all legal
    entries -- systematically the highest action ids (MOVE, UPGRADE).
    """
    envs = [_make_env(1 + i) for i in range(2)]
    num_actions = envs[0]._game.num_distinct_actions()  # pylint: disable=protected-access
    env = AsyncVectorEnv(envs, num_workers=1,
                         sampler_seeds=[1, 2],
                         game_strs=[_game_string(1), _game_string(2)])
    try:
      self.assertEqual(env._max_legal, num_actions)  # pylint: disable=protected-access
    finally:
      env.close()

  def test_published_legal_sets_match_ground_truth(self):
    """Published (rows, cols) must equal state.legal_actions for the acting seat.

    Ground truth is independent sequential envs, seeded identically and driven
    with the same actions.
    """
    num_envs, num_workers, num_steps = 4, 2, 250
    envs = [_make_env(1 + i) for i in range(num_envs)]
    num_actions = envs[0]._game.num_distinct_actions()  # pylint: disable=protected-access
    vec = AsyncVectorEnv(
        envs, num_workers=num_workers,
        sampler_seeds=[1 + i for i in range(num_envs)],
        game_strs=[_game_string(1 + i) for i in range(num_envs)],
        max_legal=num_actions)

    reference = [_make_env(1 + i) for i in range(num_envs)]
    ref_steps = [e.reset(players="current") for e in reference]
    vec.reset(players="current")  # consume the workers' startup publish
    arrays = vec.reset_np()

    rng = np.random.RandomState(7)
    widest = 0
    compared = 0
    try:
      for _ in range(num_steps):
        counts = np.bincount(arrays.legal_rows.astype(np.int64),
                             minlength=num_envs)
        offsets = np.zeros(num_envs, dtype=np.int64)
        np.cumsum(counts[:-1], out=offsets[1:])
        actions = np.zeros(num_envs, dtype=np.int32)
        for i in range(num_envs):
          seat = int(ref_steps[i].observations["current_player"])
          self.assertEqual(int(arrays.seats[i]), seat)
          published = sorted(
              int(c) for c in
              arrays.legal_cols[offsets[i]:offsets[i] + counts[i]])
          truth = sorted(
              int(a) for a in
              ref_steps[i].observations["legal_actions"][seat])
          self.assertEqual(published, truth)
          widest = max(widest, len(truth))
          compared += 1
          actions[i] = int(rng.choice(truth))
        for i in range(num_envs):
          ref_steps[i] = reference[i].step([int(actions[i])])
          if ref_steps[i].last():
            ref_steps[i] = reference[i].reset(players="current")
        arrays = vec.step_np(actions, reset_if_done=True)
    finally:
      vec.close()

    self.assertGreater(compared, 500)
    # Guards the regression: if this stays inside the initial-state legal count
    # the test is no longer exercising the truncation path at all.
    self.assertGreater(widest, 50)


if __name__ == "__main__":
  absltest.main()
