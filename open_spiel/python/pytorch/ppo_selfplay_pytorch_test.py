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
    """Every seat, not just the mover, is closed out with its own payoff.

    Invariant checked per terminal event: the acting seat's payoff is stored on
    its own done row, and each other seat's slot of the terminal returns vector
    appears as an independent extra-sample target closed out with that seat's
    last decision.
    """
    game = pyspiel.load_game("colored_trails")
    num_players = game.num_players()
    num_envs = 2
    envs = SyncVectorEnv(
        [rl_environment.Environment(game=game) for _ in range(num_envs)])
    agent = self._make_agent(num_envs, envs, game)

    time_step = envs.reset()
    terminals_checked = 0
    for _ in range(6):
      for _ in range(STEPS_PER_BATCH):
        agent_output = agent.step(time_step)
        n0 = len(agent._extra_samples)
        time_step, reward, done, unreset = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(reward, done)
        row = agent.cur_batch_idx - 1
        new_extras = agent._extra_samples[n0:]
        for i in range(num_envs):
          if not done[i]:
            continue
          terminals_checked += 1
          acting = int(agent.players[row, i])
          terminal_returns = unreset[i].rewards
          # Acting seat: payoff on the done row of the transition it made.
          self.assertEqual(float(agent.rewards[row, i]),
                           terminal_returns[acting])
          self.assertEqual(float(agent.dones[row, i]), 1.0)
          # Other seats: their own last decisions closed out with their payoff.
          closeouts = [e for e in new_extras if e[0] == i]
          expected_seats = set(range(num_players)) - {acting}
          closeout_seats = set(e[1] for e in closeouts)
          self.assertEqual(closeout_seats, expected_seats)
          for e in closeouts:
            self.assertEqual(e[7], terminal_returns[e[1]])
          # Last-decision bookkeeping cleared for this env.
          self.assertEqual(len(agent._last_decision[i]), 0)
      agent.learn(time_step)
    self.assertGreater(terminals_checked, 0)


if __name__ == "__main__":
  random.seed(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  absltest.main()
