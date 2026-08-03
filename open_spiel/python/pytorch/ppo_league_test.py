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
"""Tests for league (multi-policy) self-play support in pytorch.ppo."""

import random
from absl.testing import absltest
import numpy as np
import torch

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch import ppo_win_test
from open_spiel.python.vector_env import SyncVectorEnv

SEED = 20260804
STEPS_PER_BATCH = 48


class PPLeagueTest(absltest.TestCase):

  def _make_agent(self, num_envs, env, game, value_mode="win"):
    info_state_shape = tuple(
        np.array(env.observation_spec()["info_state"]).flatten())
    return PPO(
        input_shape=info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        player_id=0,
        num_envs=num_envs,
        steps_per_batch=STEPS_PER_BATCH,
        num_minibatches=2,
        update_epochs=2,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
        agent_fn=ppo_win_test._EclipseLikeAgent,
        value_mode=value_mode,
        aux_tasks=["final_vp"],
        aux_target_fn=(lambda rvec: np.asarray(rvec, dtype=np.float32).reshape(
            -1, 1)),
        aux_coef=0.1,
    )

  def test_main_only_gradients(self):
    game = pyspiel.load_game("colored_trails")
    num_players = game.num_players()
    num_envs = 4
    envs = SyncVectorEnv(
        [rl_environment.Environment(game=game) for _ in range(num_envs)])
    agent = self._make_agent(num_envs, envs, game)

    # League: env 0/1 are pure main; env 2/3 pit main against frozen clones.
    snap = ppo_win_test._EclipseLikeAgent(
        game.num_distinct_actions(),
        tuple(np.array(envs.envs[0].observation_spec()["info_state"]).flatten()),
        "cpu")
    snap.load_state_dict(agent.network.state_dict())
    snap.eval()
    lineup = np.array([["main"] * num_players, ["main"] * num_players,
                       ["main", "snap", "snap"], ["main", "snap", "snap"]],
                      dtype=object)
    agent.setup_league({"main": agent.network, "snap": snap}, lineup, "main")

    snap_before = {k: v.detach().clone() for k, v in snap.state_dict().items()}
    main_before = {k: v.detach().clone()
                   for k, v in agent.network.state_dict().items()}

    time_step = envs.reset()
    terminals = 0
    snaps_closed_out = 0
    for _ in range(6):
      for _ in range(STEPS_PER_BATCH):
        agent_output = agent.step(time_step)
        time_step, reward, done, _ = envs.step(
            agent_output, reset_if_done=True)
        done = np.asarray(done)
        agent.post_step(reward, done)
        if done.any():
          terminals += int(done.sum())
          # Extras must only close out trainable (main) seats.
          for e in agent._extra_samples:
            if e[0] in (2, 3) and e[1] != 0:
              snaps_closed_out += 1
      agent.learn(time_step)
    self.assertGreater(terminals, 0)

    # Non-trainable opponent network must be untouched by training.
    snap_changed = any(
        not torch.allclose(snap.state_dict()[k], v)
        for k, v in snap_before.items())
    self.assertFalse(snap_changed)
    # Main must have moved.
    main_changed = any(
        not torch.allclose(agent.network.state_dict()[k], v)
        for k, v in main_before.items())
    self.assertTrue(main_changed)
    # No extras closed out non-trainable seats.
    self.assertEqual(snaps_closed_out, 0)
    self.assertTrue(agent.league)

  def test_trainable_flags(self):
    game = pyspiel.load_game("colored_trails")
    num_players = game.num_players()
    num_envs = 2
    envs = SyncVectorEnv(
        [rl_environment.Environment(game=game) for _ in range(num_envs)])
    agent = self._make_agent(num_envs, envs, game)
    snap = ppo_win_test._EclipseLikeAgent(
        game.num_distinct_actions(),
        tuple(np.array(envs.envs[0].observation_spec()["info_state"]).flatten()),
        "cpu")
    snap.load_state_dict(agent.network.state_dict())
    lineup = np.array([["main"] * num_players, ["main", "snap", "snap"]],
                      dtype=object)
    agent.setup_league({"main": agent.network, "snap": snap}, lineup, "main")

    time_step = envs.reset()
    agent.step(time_step)
    reward = [np.zeros(num_players) for _ in range(num_envs)]
    done = [False] * num_envs
    agent.post_step(reward, done)
    row = agent.cur_batch_idx - 1
    # Env 0 all seats are main; env 1 only seat 0 is main.
    self.assertTrue(bool(agent.trainable[row, 0]))
    acting = int(agent.players_cpu[row, 1])
    expected = acting == 0
    self.assertEqual(bool(agent.trainable[row, 1]), expected)


if __name__ == "__main__":
  random.seed(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  absltest.main()
