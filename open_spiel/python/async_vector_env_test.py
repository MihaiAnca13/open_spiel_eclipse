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

  def test_collect_keeps_exactly_two_live_generations(self):
    """Consecutive step_np results must be independent; the third reuses buffer 0.

    ``_collect`` writes into two preallocated generations instead of allocating
    a fresh 38.5 MB observation array per step. The PPO loop depends on depth
    exactly 2: it reads the PREVIOUS step's observations (``last_obs_batch`` ->
    ``_terminal_obs_for``) after the current step has been collected. If someone
    reduces this to one buffer, terminal aux targets silently read the wrong
    state; if the alternation breaks, the same happens intermittently. Both
    failure modes are invisible in reward curves, so pin the invariant here.
    """
    num_envs, num_workers = 4, 2
    envs = [_make_env(1 + i) for i in range(num_envs)]
    num_actions = envs[0]._game.num_distinct_actions()  # pylint: disable=protected-access
    vec = AsyncVectorEnv(
        envs, num_workers=num_workers,
        sampler_seeds=[1 + i for i in range(num_envs)],
        game_strs=[_game_string(1 + i) for i in range(num_envs)],
        max_legal=num_actions)
    try:
      vec.reset(players="current")
      a0 = vec.reset_np()
      seen = [a0]
      rng = np.random.RandomState(11)
      for _ in range(3):
        prev = seen[-1]
        counts = np.bincount(prev.legal_rows.astype(np.int64),
                             minlength=num_envs)
        offsets = np.zeros(num_envs, dtype=np.int64)
        np.cumsum(counts[:-1], out=offsets[1:])
        actions = np.zeros(num_envs, dtype=np.int32)
        for i in range(num_envs):
          choices = prev.legal_cols[offsets[i]:offsets[i] + counts[i]]
          if len(choices):
            actions[i] = int(rng.choice(choices))
        seen.append(vec.step_np(actions, reset_if_done=True))

      # Adjacent generations must be distinct storage -- this is the property
      # the PPO loop actually relies on.
      for j in range(len(seen) - 1):
        for field in ("obs", "seats", "rewards", "dones"):
          self.assertIsNot(getattr(seen[j], field), getattr(seen[j + 1], field),
                           f"{field}: generations {j} and {j+1} alias")

      # ...and depth is exactly 2, not 3: generation j and j+2 share storage.
      # Documented, not incidental. A caller reaching back two steps is a bug.
      self.assertIs(seen[0].obs, seen[2].obs)
      self.assertIs(seen[1].obs, seen[3].obs)
    finally:
      vec.close()


if __name__ == "__main__":
  absltest.main()
