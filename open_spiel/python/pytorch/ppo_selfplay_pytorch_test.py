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
from open_spiel.python.async_vector_env import _StepArrays
from open_spiel.python import rl_environment
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import PPOAgent
from open_spiel.python.pytorch.ppo import PPOIntegrityError
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

  def test_batch_boundary_never_bootstraps_from_another_seat(self):
    agent = PPO(input_shape=(4,), num_actions=3, num_players=2, num_envs=1,
                steps_per_batch=4, num_minibatches=1, gamma=1.0,
                gae_lambda=1.0, device="cpu", agent_fn=PPOAgent)
    agent.players[:, 0] = torch.tensor([0, 1, 0, 1])
    agent.values[:, 0] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    agent.rewards.zero_()
    agent.dones.zero_()

    returns = agent._compute_returns(torch.tensor([100.0]),
                                     np.array([0], dtype=np.int32))

    # Seat 0 owns the next observation and may bootstrap from 100. Seat 1's
    # boundary row is neutral; its preceding row chains through value 40, not
    # through seat 0's value 100.
    torch.testing.assert_close(returns[:, 0],
                               torch.tensor([100.0, 40.0, 100.0, 40.0]))


class _SparseDummy(torch.nn.Module):
  """Minimal shared-trunk network, enough for PPO's buffer bookkeeping."""

  def __init__(self, num_actions, observation_shape, device):
    del device
    super().__init__()
    self.num_actions = num_actions
    size = int(np.prod(observation_shape))
    self.shared = torch.nn.Sequential(torch.nn.Linear(size, 8), torch.nn.Tanh())
    self.actor = torch.nn.Sequential(self.shared,
                                     torch.nn.Linear(8, num_actions))
    self.critic = torch.nn.Sequential(self.shared, torch.nn.Linear(8, 1))
    self.register_buffer("mask_value", torch.tensor(-1e6))

  def get_value(self, x):
    return self.critic(x).view(-1)


class TelescopeShapingTest(absltest.TestCase):
  """post_step(phi=...) must produce an exact potential-based telescope."""

  GAMMA = 0.97

  def test_shaped_reward_telescopes_across_own_decisions(self):
    """shaped(r_j) == gamma*phi(next own decision) - phi(this one).

    Potential-based shaping is only policy-invariant when the difference spans
    the same transition the discount does. In self-play GAE that is a seat's own
    consecutive decisions -- its chain skips other seats' rows -- so differencing
    across env steps (--phi=banked/soft) does not telescope, and differencing two
    *different* seats' values (--phi=learned) is not a potential difference at
    all.
    """
    steps, num_envs, num_players = 12, 2, 3
    agent = PPO(input_shape=(4,), num_actions=3, num_players=num_players,
                num_envs=num_envs, steps_per_batch=steps, num_minibatches=2,
                gamma=self.GAMMA, device="cpu", agent_fn=_SparseDummy,
                value_mode="vp")

    rng = np.random.RandomState(0)
    seats = np.array([[0, 1], [1, 2], [0, 0], [2, 1], [0, 2], [1, 0],
                      [2, 1], [0, 0], [1, 1], [2, 2], [0, 0], [1, 1]])
    phis = rng.uniform(0, 5, size=(steps, num_envs)).astype(np.float32)
    dones = np.zeros((steps, num_envs), dtype=bool)
    dones[7, 1] = True  # env 1 ends mid-batch; env 0 never terminates
    payoff = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    for row in range(steps):
      seat_row = torch.from_numpy(seats[row].astype(np.int64))
      agent.players_cpu[row] = seat_row
      agent.players[row] = seat_row
      agent.trainable_cpu[row] = True
      agent.trainable[row] = True
      agent.cur_batch_idx = row
      rewards = np.zeros((num_envs, num_players), dtype=np.float32)
      if dones[row].any():
        rewards[1] = payoff
      agent.post_step_np(rewards, dones[row], phi=phis[row])

    agent._apply_closeout_writes()
    stored = agent.rewards.numpy()

    checked = 0
    for env in range(num_envs):
      episode_start = 0
      for row in range(steps):
        if dones[row, env]:
          episode_start = row + 1
      for seat in range(num_players):
        rows = [r for r in range(steps) if seats[r, env] == seat]
        for cur, nxt in zip(rows, rows[1:]):
          if cur < episode_start or nxt < episode_start or dones[cur, env]:
            continue
          checked += 1
          self.assertAlmostEqual(
              float(stored[cur, env]),
              self.GAMMA * phis[nxt, env] - phis[cur, env], places=4,
              msg=f"env {env} seat {seat}: rows {cur}->{nxt} do not telescope")
    self.assertGreater(checked, 5)

    # The terminal transition carries the true payoff, never a shaping delta.
    terminal_seat = int(seats[7, 1])
    self.assertAlmostEqual(float(stored[7, 1]), float(payoff[terminal_seat]),
                           places=5)

  def test_pending_potentials_do_not_survive_the_batch(self):
    """A potential recorded in a previous batch has no row left to credit."""
    agent = PPO(input_shape=(4,), num_actions=3, num_players=2, num_envs=1,
                steps_per_batch=2, num_minibatches=1, gamma=self.GAMMA,
                device="cpu", agent_fn=_SparseDummy, value_mode="vp")
    for row in range(2):
      agent.players_cpu[row] = torch.zeros(1, dtype=torch.long)
      agent.players[row] = torch.zeros(1, dtype=torch.long)
      agent.trainable_cpu[row] = True
      agent.trainable[row] = True
      agent.cur_batch_idx = row
      agent.post_step_np(np.zeros((1, 2), dtype=np.float32),
                         np.zeros(1, dtype=bool),
                         phi=np.array([1.0], dtype=np.float32))
    agent.learn_np(np.zeros((1, 4), dtype=np.float32), np.zeros(1, np.int32))
    self.assertEqual(agent._pending_phi[0], {})
    self.assertEqual(agent._shaping_adds, [])

  def test_sync_and_array_native_shaping_are_identical(self):
    """The two collectors must write the same exact telescope."""
    def run(array_native, phi):
      agent = PPO(input_shape=(4,), num_actions=5, num_players=2, num_envs=1,
                  steps_per_batch=4, num_minibatches=1, gamma=self.GAMMA,
                  device="cpu", agent_fn=_SparseDummy, value_mode="vp")
      seats = (0, 1, 0, 1)
      potentials = (1.0, 4.0, 2.0, 8.0)
      for row, seat in enumerate(seats):
        agent.players_cpu[row, 0] = seat
        agent.players[row, 0] = seat
        agent.trainable_cpu[row, 0] = True
        agent.trainable[row, 0] = True
        agent.cur_batch_idx = row
        done = np.asarray([row == 3])
        rewards = (np.asarray([[3.0, 1.0]], dtype=np.float32)
                   if done[0] else np.zeros((1, 2), dtype=np.float32))
        kwargs = {"phi": np.asarray([potentials[row]], dtype=np.float32)} if phi else {}
        if array_native:
          agent.post_step_np(rewards, done, **kwargs)
        else:
          agent.post_step([rewards[0]], done.tolist(), **kwargs)
      agent._apply_closeout_writes()
      return agent.rewards[:, 0].numpy(), agent.dones[:, 0].numpy()

    for shaped in (False, True):
      sync_rewards, sync_dones = run(False, shaped)
      async_rewards, async_dones = run(True, shaped)
      np.testing.assert_allclose(sync_rewards, async_rewards)
      np.testing.assert_array_equal(sync_dones, async_dones)
      if shaped:
        self.assertAlmostEqual(sync_rewards[0], self.GAMMA * 2.0 - 1.0)
      else:
        self.assertEqual(sync_rewards[0], 0.0)
      # Terminal closeout targets seat 0's last decision and seat 1's terminal
      # action. Seat 1's earlier potential has no successor before termination.
      np.testing.assert_allclose(sync_rewards[1:], [0.0, 3.0, 1.0])


class PPOIntegrityTest(absltest.TestCase):

  def test_rejects_invalid_legal_sets_and_nonfinite_rewards(self):
    agent = PPO(input_shape=(4,), num_actions=3, num_players=2, num_envs=1,
                steps_per_batch=1, num_minibatches=1, device="cpu",
                agent_fn=_SparseDummy)
    with self.assertRaises(PPOIntegrityError):
      agent._validate_legal_actions([[]])
    with self.assertRaises(PPOIntegrityError):
      agent._validate_packed_legal_actions(
          np.asarray([0, 0]), np.asarray([1, 1]))
    with self.assertRaises(PPOIntegrityError):
      agent.post_step_np(np.asarray([[np.nan, 0.0]], dtype=np.float32),
                         np.asarray([False]))


class CrossBatchTerminalAttributionTest(absltest.TestCase):

  def test_terminal_extras_keep_each_seats_last_decision(self):
    """Closeout after a batch boundary cannot reuse another seat's action."""
    agent = PPO(input_shape=(4,), num_actions=5, num_players=3, num_envs=1,
                steps_per_batch=2, num_minibatches=1, device="cpu",
                agent_fn=_SparseDummy, value_mode="vp")

    def arrays(seat, tag):
      return _StepArrays(
          obs=np.asarray([[tag, 0.0, 0.0, 0.0]], dtype=np.float32),
          seats=np.asarray([seat], dtype=np.int32),
          legal_rows=np.asarray([0, 0], dtype=np.int64),
          legal_cols=np.asarray([0, 1], dtype=np.int64),
          rewards=np.zeros((1, 3), dtype=np.float32),
          dones=np.zeros(1, dtype=bool), terminal_obs=None)

    # Seats 0 and 1 act in batch A. Save their unique records before calling
    # learn, which resets the buffer but intentionally preserves episode state.
    step = arrays(0, 10.0)
    agent.step_np(step)
    agent.post_step_np(step.rewards, step.dones)
    step = arrays(1, 20.0)
    agent.step_np(step)
    agent.post_step_np(step.rewards, step.dones)
    expected = {seat: agent._last_decision[0][seat] for seat in (0, 1)}
    agent.learn_np(arrays(2, 30.0).obs, np.asarray([2], dtype=np.int32))

    # Seat 2 terminates in batch B. Its own row gets the terminal reward;
    # seats 0 and 1 must be represented once each by their original records.
    terminal = arrays(2, 30.0)
    agent.step_np(terminal)
    terminal_rewards = np.asarray([[7.0, 4.0, 1.0]], dtype=np.float32)
    agent.post_step_np(terminal_rewards, np.asarray([True]))

    self.assertEqual(len(agent._extra_samples), 2)
    for sample in agent._extra_samples:
      seat = sample[1]
      self.assertIn(seat, expected)
      old = expected[seat]
      np.testing.assert_array_equal(sample[2], old[0])
      np.testing.assert_array_equal(sample[3], old[1])
      self.assertEqual(sample[4], old[2])
      self.assertEqual(sample[5], old[3])
      self.assertEqual(sample[7], terminal_rewards[0, seat])

    # Fill the remainder with the first decision of the reset episode so learn
    # consumes both closeout records and cannot silently leave stale extras.
    reset = arrays(0, 40.0)
    agent.step_np(reset)
    agent.post_step_np(reset.rewards, reset.dones)
    agent.learn_np(arrays(1, 50.0).obs, np.asarray([1], dtype=np.int32))
    self.assertEqual(agent._extra_samples, [])
    self.assertTrue(np.isfinite(list(agent.last_metrics.values())).all())


if __name__ == "__main__":
  random.seed(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  absltest.main()
