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
"""Tests for N-player self-play support in open_spiel.python.pytorch.ppo."""

import random
from absl.testing import absltest
import numpy as np
import torch

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import PPOAgent
from open_spiel.python.vector_env import SyncVectorEnv

SEED = 24261711
STEPS_PER_BATCH = 64


class PPOSelfPlayTest(absltest.TestCase):

  def _make_agent(self, num_envs, env, game):
    info_state_shape = tuple(
        np.array(env.observation_spec()["info_state"]).flatten())
    return PPO(
        input_shape=info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        player_id=0,
        num_envs=num_envs,
        steps_per_batch=STEPS_PER_BATCH,
        num_minibatches=4,
        update_epochs=3,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
        agent_fn=PPOAgent,
    )

  def test_colored_trails_selfplay_smoke(self):
    game = pyspiel.load_game("colored_trails")
    num_envs = 4
    envs = SyncVectorEnv(
        [rl_environment.Environment(game=game) for _ in range(num_envs)])
    agent = self._make_agent(num_envs, envs, game)

    time_step = envs.reset()
    terminals = 0
    for _ in range(10):
      for _ in range(STEPS_PER_BATCH):
        agent_output = agent.step(time_step)
        time_step, reward, done, _ = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(reward, done)
        terminals += sum(done)
      agent.learn(time_step)
    self.assertGreater(terminals, 0)
    self.assertEqual(len(agent._extra_samples), 0)
    self.assertEqual(agent.cur_batch_idx, 0)

  def test_terminal_reward_attribution(self):
    """Every seat that acted is closed out with its own payoff, exactly once.

    Invariant per terminal event: the acting seat's payoff lands on its own done
    row, and each *other* seat that made a decision this episode is closed out
    with its slot of the terminal returns vector via exactly one of
      * an in-buffer write on that seat's last row of the episode, when the row
        is still in this batch (also marks it done, cutting the GAE chain), or
      * an independent extra sample, when the seat last acted in an earlier
        batch and its row is gone.
    Being closed out through *both* would leave two contradictory targets for
    the same (obs, action), which is what the previous implementation did.
    """
    game = pyspiel.load_game("colored_trails")
    num_players = game.num_players()
    num_envs = 2
    envs = SyncVectorEnv(
        [rl_environment.Environment(game=game) for _ in range(num_envs)])
    agent = self._make_agent(num_envs, envs, game)

    time_step = envs.reset()
    terminals_checked = 0
    # Seats that have made a decision in the episode currently in progress.
    acted = [set() for _ in range(num_envs)]
    for _ in range(6):
      for _ in range(STEPS_PER_BATCH):
        agent_output = agent.step(time_step)
        row = agent.cur_batch_idx
        for i in range(num_envs):
          acted[i].add(int(agent.players_cpu[row, i]))
        n_extras = len(agent._extra_samples)
        n_writes = len(agent._closeout_writes)
        time_step, reward, done, unreset = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(reward, done)
        new_extras = agent._extra_samples[n_extras:]
        new_writes = agent._closeout_writes[n_writes:]
        for i in range(num_envs):
          if not done[i]:
            continue
          terminals_checked += 1
          acting = int(agent.players_cpu[row, i])
          terminal_returns = unreset[i].rewards
          # Acting seat: payoff on the done row of the transition it made.
          self.assertEqual(float(agent.rewards[row, i]),
                           terminal_returns[acting])
          self.assertEqual(float(agent.dones[row, i]), 1.0)

          extra_seats = [e[1] for e in new_extras if e[0] == i]
          write_seats = [int(agent.players_cpu[r, i])
                         for (r, env_idx, _) in new_writes if env_idx == i]
          closed = extra_seats + write_seats
          # Exactly once each: no seat appears via both mechanisms.
          self.assertCountEqual(closed, set(closed))
          self.assertCountEqual(closed, acted[i] - {acting})

          for e in new_extras:
            if e[0] == i:
              self.assertEqual(e[7], terminal_returns[e[1]])
          for (r, env_idx, target) in new_writes:
            if env_idx == i:
              self.assertAlmostEqual(
                  target, terminal_returns[int(agent.players_cpu[r, i])],
                  places=5)
          # Last-decision bookkeeping cleared for this env.
          self.assertEqual(len(agent._last_decision[i]), 0)
          acted[i] = set()
      agent.learn(time_step)
    self.assertGreater(terminals_checked, 0)
    self.assertGreater(num_players, 1)

  def test_no_gae_chain_crosses_an_episode_boundary(self):
    """A seat's last row of a finished episode must carry done=1.

    ``dones`` has one flag per (row, env), so only the seat that made the final
    move used to get one. Every other seat's chain then bootstrapped through the
    reset into the *next* episode's values, so the terminal payoff never reached
    the trajectory that earned it.
    """
    game = pyspiel.load_game("colored_trails")
    num_envs = 2
    envs = SyncVectorEnv(
        [rl_environment.Environment(game=game) for _ in range(num_envs)])
    agent = self._make_agent(num_envs, envs, game)

    time_step = envs.reset()
    boundaries_checked = 0
    for _ in range(6):
      for _ in range(STEPS_PER_BATCH):
        agent_output = agent.step(time_step)
        time_step, reward, done, _ = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(reward, done)
      # Flush the deferred terminal writes exactly as _learn_core does, then
      # inspect the batch the optimizer is about to see.
      agent._apply_closeout_writes()
      players = agent.players_cpu.numpy()
      dones = agent.dones.cpu().numpy()
      for i in range(num_envs):
        terminal_rows = np.flatnonzero(dones[:, i] > 0.5)
        start = 0
        for t in terminal_rows:
          for seat in range(agent.num_players):
            seat_rows = [r for r in range(start, t + 1)
                         if players[r, i] == seat]
            if not seat_rows:
              continue
            boundaries_checked += 1
            self.assertEqual(
                dones[seat_rows[-1], i], 1.0,
                msg=(f"env {i} seat {seat}: last row {seat_rows[-1]} of the "
                     f"episode ending at {t} is not marked done, so its GAE "
                     f"chain bootstraps into the next episode"))
          start = t + 1
      agent.learn(time_step)
    self.assertGreater(boundaries_checked, 0)


if __name__ == "__main__":
  random.seed(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  absltest.main()
