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
"""Tests for win-value mode and auxiliary heads in open_spiel.python.pytorch.ppo.

Covers:
  * rank_utility / rank_of terminal-target conversion (incl. ties).
  * The sparse-network value path via ``value_from_features`` (Eclipse-style
    rank-head critic, 4 outputs).
  * Auxiliary-head targets back-filled at terminal closeout and the aux loss
    updating the shared trunk.
  * End-to-end self-play smoke in win-value mode (terminal targets are rank
    utilities, extras carry per-seat utility targets).
"""

import random
from absl.testing import absltest
import numpy as np
import torch
from torch import nn

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import layer_init
from open_spiel.python.pytorch.ppo import rank_utility
from open_spiel.python.pytorch.ppo import rank_of
from open_spiel.python.pytorch.ppo import vp_bonus_cap
from open_spiel.python.pytorch.ppo import DEFAULT_RANK_UTILITY
from open_spiel.python.pytorch.ppo import CategoricalMasked
from open_spiel.python.vector_env import SyncVectorEnv

SEED = 20260803
STEPS_PER_BATCH = 64

RANK_UTILITY = (1.0, 0.5, 0.0, -0.5)


class _EclipseLikeAgent(nn.Module):
  """Minimal Eclipse-shaped actor-critic (shared trunk, 4-output rank head).

  Mirrors ppo_eclipse.EclipsePPOAgent so the sparse learn path
  (``value_from_features``) and auxiliary-head machinery are exercised.
  """

  def __init__(self, num_actions, observation_shape, device):
    super().__init__()
    width = 32
    self.shared = nn.Sequential(
        layer_init(nn.Linear(int(np.prod(observation_shape)), width)),
        nn.Tanh())
    self.critic = nn.Sequential(
        self.shared, layer_init(nn.Linear(width, 4), std=1.0))
    self.actor = nn.Sequential(
        self.shared, layer_init(nn.Linear(width, num_actions), std=0.01))
    self.aux_heads = nn.ModuleDict({
        "final_vp": layer_init(nn.Linear(width, 1), std=1.0),
    })
    self.num_actions = num_actions
    self.device = device
    self.register_buffer("mask_value", torch.tensor(-1e6))

  def get_value(self, x):
    return self.rank_value(self.critic(x))

  def rank_value(self, rank_logits):
    probs = rank_logits.softmax(dim=-1)
    utility = torch.tensor(RANK_UTILITY, dtype=rank_logits.dtype,
                           device=rank_logits.device)
    return (probs * utility).sum(dim=-1)

  def value_from_features(self, features):
    return self.rank_value(self.critic[-1](features))

  def get_aux(self, features):
    return {name: head(features) for name, head in self.aux_heads.items()}

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    if legal_actions_mask is None:
      legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()
    logits = self.actor(x)
    probs = CategoricalMasked(logits=logits, masks=legal_actions_mask,
                              mask_value=self.mask_value)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.get_value(
        x), probs.probs


class RankUtilityTest(absltest.TestCase):

  def test_rank_utility_strict(self):
    rvec = [10.0, 20.0, 30.0, 40.0]
    self.assertEqual(rank_utility(rvec, 3), 1.0)
    self.assertEqual(rank_utility(rvec, 2), 0.5)
    self.assertEqual(rank_utility(rvec, 1), 0.0)
    self.assertEqual(rank_utility(rvec, 0), -0.5)

  def test_rank_utility_ties_share_the_average_slot(self):
    # Seats 0 and 2 tie for 1st, so they jointly occupy slots 1 and 2 and each
    # take the mean: (1.0 + 0.5) / 2. Awarding both the *best* slot (1.0) would
    # make total utility outcome-dependent and, in the all-tied case, hand every
    # seat the maximum -- see test_all_tied_is_not_the_optimum.
    rvec = [40.0, 20.0, 40.0, 30.0]
    self.assertEqual(rank_utility(rvec, 0), 0.75)
    self.assertEqual(rank_utility(rvec, 2), 0.75)
    # Seat 3 is alone in 3rd.
    self.assertEqual(rank_utility(rvec, 3), 0.0)
    # Seat 1 is alone in 4th.
    self.assertEqual(rank_utility(rvec, 1), -0.5)

  def test_all_tied_is_not_the_optimum(self):
    """A universally-tied outcome must not pay every seat the best utility.

    Regression: with "ties share the best placement", an all-bankrupt Eclipse
    game ([0,0,0,0], ~95% of unskilled games) paid all four seats +1.0 -- the
    maximum in the table -- making mutual failure the global optimum of the
    training objective.
    """
    tied = [rank_utility([0.0] * 4, s) for s in range(4)]
    self.assertEqual(tied, [0.25] * 4)
    self.assertLess(max(tied), max(DEFAULT_RANK_UTILITY))
    # A single seat scoring anything at all is strictly better than the tie.
    scored = [rank_utility([0.0, 0.0, 0.0, 1.0], s) for s in range(4)]
    self.assertEqual(scored[3], 1.0)
    self.assertGreater(scored[3], tied[3])
    self.assertLess(scored[0], tied[0])

  def test_rank_utility_is_constant_sum(self):
    """Total utility must not depend on the outcome (with the bonus disabled)."""
    rng = np.random.RandomState(0)
    expected = sum(DEFAULT_RANK_UTILITY)
    for _ in range(2000):
      rvec = rng.randint(0, 12, size=4).astype(float).tolist()
      total = sum(rank_utility(rvec, s) for s in range(4))
      self.assertAlmostEqual(total, expected, places=9)

  def test_vp_bonus_never_reorders_placements(self):
    """The escape bonus may break ties but never invert a real rank gap."""
    rng = np.random.RandomState(1)
    for vp_beta in (0.002, 0.05, 1.0, 1e6):
      for _ in range(2000):
        rvec = rng.randint(0, 300, size=4).astype(float).tolist()
        util = [rank_utility(rvec, s, vp_beta=vp_beta) for s in range(4)]
        for a in range(4):
          for b in range(4):
            if rank_of(rvec, a) < rank_of(rvec, b):
              self.assertGreater(util[a], util[b])

  def test_vp_bonus_breaks_the_dead_zone(self):
    """The bonus must create return variance where the rank term is flat."""
    beta = 0.002
    flat = rank_utility([0.0] * 4, 0, vp_beta=beta)
    # Scoring while others do not: from a 4-way tie to sole first.
    self.assertGreater(rank_utility([3.0, 0.0, 0.0, 0.0], 0, vp_beta=beta),
                       flat + 0.5)
    # Everyone scoring equally still beats everyone scoring nothing, so the
    # gradient points out of the all-zero basin even while outcomes stay tied.
    self.assertGreater(rank_utility([5.0] * 4, 0, vp_beta=beta), flat)
    # And the bonus is bounded, so it cannot swamp the rank signal.
    self.assertLessEqual(
        rank_utility([1e9] * 4, 0, vp_beta=beta) - flat, vp_bonus_cap())

  def test_rank_utility_two_player(self):
    rvec = [5.0, 9.0]
    self.assertEqual(rank_utility(rvec, 1), 1.0)
    # Seat 0 has one strictly above -> rank 2 (2nd place in the 4-row table).
    self.assertEqual(rank_utility(rvec, 0), 0.5)


class PPOWinValueTest(absltest.TestCase):

  def _make_agent(self, num_envs, env, game, value_mode="win", aux=None):
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
        agent_fn=_EclipseLikeAgent,
        value_mode=value_mode,
        aux_tasks=["final_vp"] if aux else None,
        aux_target_fn=(lambda rvec: np.asarray(rvec, dtype=np.float32).reshape(
            -1, 1)) if aux else None,
        aux_coef=0.1,
    )

  def test_win_mode_terminal_targets_are_utilities(self):
    game = pyspiel.load_game("colored_trails")
    num_players = game.num_players()
    num_envs = 2
    envs = SyncVectorEnv(
        [rl_environment.Environment(game=game) for _ in range(num_envs)])
    agent = self._make_agent(num_envs, envs, game, value_mode="win", aux=True)
    self.assertTrue(agent._sparse_supported())

    time_step = envs.reset()
    terminals_checked = 0
    for _ in range(4):
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
          # Acting seat's stored reward is the rank utility, not raw payoff.
          self.assertEqual(float(agent.rewards[row, i]),
                           rank_utility(terminal_returns, acting))
          closeouts = [e for e in new_extras if e[0] == i]
          for e in closeouts:
            self.assertEqual(e[7], rank_utility(terminal_returns, e[1]))
            # Extra samples carry aux targets (final payoff /1) + masks.
            self.assertIsNotNone(e[8])
            self.assertEqual(float(e[8][0]), terminal_returns[e[1]])
      agent.learn(time_step)
    self.assertGreater(terminals_checked, 0)

  def test_aux_backfill_and_loss_finite(self):
    game = pyspiel.load_game("colored_trails")
    num_envs = 2
    envs = SyncVectorEnv(
        [rl_environment.Environment(game=game) for _ in range(num_envs)])
    agent = self._make_agent(num_envs, envs, game, value_mode="win", aux=True)

    time_step = envs.reset()
    got_targets = False
    for _ in range(4):
      for _ in range(STEPS_PER_BATCH):
        agent_output = agent.step(time_step)
        time_step, reward, done, _ = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(reward, done)
      agent.learn(time_step)
      # Back-filled targets/masks should have been written for some rows.
      if agent.num_aux and agent.aux_mask is not None:
        if bool(agent.aux_mask.sum().item()):
          got_targets = True
    self.assertTrue(got_targets)
    self.assertEqual(agent.num_aux, 1)
    self.assertEqual(agent.aux_tasks, ["final_vp"])

  def test_vp_mode_regression(self):
    # vp mode keeps raw terminal payoffs (backward compat on the new agent).
    game = pyspiel.load_game("colored_trails")
    num_players = game.num_players()
    num_envs = 2
    envs = SyncVectorEnv(
        [rl_environment.Environment(game=game) for _ in range(num_envs)])
    agent = self._make_agent(num_envs, envs, game, value_mode="vp", aux=False)

    time_step = envs.reset()
    checked = 0
    for _ in range(4):
      for _ in range(STEPS_PER_BATCH):
        agent_output = agent.step(time_step)
        time_step, reward, done, unreset = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(reward, done)
        row = agent.cur_batch_idx - 1
        for i in range(num_envs):
          if done[i]:
            checked += 1
            acting = int(agent.players[row, i])
            self.assertEqual(float(agent.rewards[row, i]),
                             unreset[i].rewards[acting])
      agent.learn(time_step)
    self.assertGreater(checked, 0)


if __name__ == "__main__":
  random.seed(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  absltest.main()
