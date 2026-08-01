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

"""An implementation of PPO.

Note: code adapted (with permission) from
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and
https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py.

Supports both the original single-agent case (a fixed ``player_id``) and
N-player general-sum self-play with a single shared network dispatched by seat:

* ``step`` gathers, per environment, the observations of whichever seat is
  currently to move (``observations["current_player"]``) when ``num_players > 1``.
* Per-seat terminal reward attribution: when an environment reaches a terminal
  state, the acting seat's payoff is stored on its own final transition, and
  every other seat is closed out with an independent terminal sample carrying
  that seat's slot of the terminal returns vector (collected across batches, so
  seats that last acted in a previous rollout batch are still closed out
  correctly).
* The shaped-reward variant of ``post_step`` (``shaped_reward`` argument) lets
  callers add per-step potential-based shaping (``gamma * phi(s') - phi(s)``) to
  the acting seat's transition while leaving terminal payoff attribution intact.
"""

import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

from open_spiel.python.rl_agent import StepOutput

INVALID_ACTION_PENALTY = -1e6


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class CategoricalMasked(Categorical):
  """A masked categorical."""

  # pylint: disable=dangerous-default-value
  def __init__(self,
               probs=None,
               logits=None,
               validate_args=None,
               masks=[],
               mask_value=None):
    logits = torch.where(masks.bool(), logits, mask_value)
    super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class PPOAgent(nn.Module):
  """A PPO agent module."""

  def __init__(self, num_actions, observation_shape, device):
    super().__init__()
    self.critic = nn.Sequential(
        layer_init(nn.Linear(np.array(observation_shape).prod(), 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0),
    )
    self.actor = nn.Sequential(
        layer_init(nn.Linear(np.array(observation_shape).prod(), 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01),
    )
    self.device = device
    self.num_actions = num_actions
    self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

  def get_value(self, x):
    return self.critic(x)

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    if legal_actions_mask is None:
      legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

    logits = self.actor(x)
    probs = CategoricalMasked(
        logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(
        x), probs.probs


class PPOAtariAgent(nn.Module):
  """A PPO Atari agent module."""

  def __init__(self, num_actions, observation_shape, device):
    super(PPOAtariAgent, self).__init__()
    # Note: this network is intended for atari games, taken from
    # https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py
    self.network = nn.Sequential(
        layer_init(nn.Conv2d(4, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )
    self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
    self.critic = layer_init(nn.Linear(512, 1), std=1)
    self.num_actions = num_actions
    self.device = device
    self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

  def get_value(self, x):
    return self.critic(self.network(x / 255.0))

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    if legal_actions_mask is None:
      legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

    hidden = self.network(x / 255.0)
    logits = self.actor(hidden)
    probs = CategoricalMasked(
        logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)

    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(
        hidden), probs.probs


def legal_actions_to_mask(legal_actions_list, num_actions, device="cpu"):
  """Converts a list of legal actions to a mask.

  The mask has size num actions with a 1 in a legal positions.

  Args:
    legal_actions_list: the list of legal actions
    num_actions: number of actions (width of mask)
    device: device to build the mask on.

  Returns:
    legal actions mask.
  """
  mask = torch.zeros((len(legal_actions_list), num_actions),
                     dtype=torch.bool, device=device)
  row_ids = []
  actions = []
  for i, legal_actions in enumerate(legal_actions_list):
    if not legal_actions:
      continue
    n = len(legal_actions)
    actions.extend(legal_actions)
    row_ids.extend([i] * n)
  if row_ids:
    rows = torch.tensor(row_ids, dtype=torch.long, device=device)
    cols = torch.tensor(actions, dtype=torch.long, device=device)
    mask[rows, cols] = True
  return mask


class PPO(nn.Module):
  """PPO Agent implementation in PyTorch.

  See open_spiel/python/examples/ppo_example.py for an usage example.

  Note that PPO runs multiple environments concurrently on each step (see
  open_spiel/python/vector_env.py). In practice, this tends to improve PPO's
  performance. The number of parallel environments is controlled by the
  num_envs argument.

  When ``num_players > 1`` the agent acts as a single shared policy for all
  seats (self-play): ``step`` dispatches each environment's current player and
  terminal rewards are attributed per seat (see module docstring).
  """

  def __init__(
      self,
      input_shape,
      num_actions,
      num_players,
      player_id=0,
      num_envs=1,
      steps_per_batch=128,
      num_minibatches=4,
      update_epochs=4,
      learning_rate=2.5e-4,
      gae=True,
      gamma=0.99,
      gae_lambda=0.95,
      normalize_advantages=True,
      clip_coef=0.2,
      clip_vloss=True,
      entropy_coef=0.01,
      value_coef=0.5,
      max_grad_norm=0.5,
      target_kl=None,
      device="cpu",
      writer=None,  # Tensorboard SummaryWriter
      agent_fn=PPOAtariAgent,
  ):
    super().__init__()

    self.input_shape = input_shape
    self.num_actions = num_actions
    self.num_players = num_players
    self.player_id = player_id
    self.selfplay = num_players > 1
    self.device = device

    # Training settings
    self.num_envs = num_envs
    self.steps_per_batch = steps_per_batch
    self.batch_size = self.num_envs * self.steps_per_batch
    self.num_minibatches = num_minibatches
    self.update_epochs = update_epochs
    self.learning_rate = learning_rate

    # Loss function
    self.gae = gae
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.normalize_advantages = normalize_advantages
    self.clip_coef = clip_coef
    self.clip_vloss = clip_vloss
    self.entropy_coef = entropy_coef
    self.value_coef = value_coef
    self.max_grad_norm = max_grad_norm
    self.target_kl = target_kl

    # Logging
    self.writer = writer

    # Initialize networks
    self.network = agent_fn(self.num_actions, self.input_shape,
                            device).to(device)
    self.optimizer = optim.Adam(
        self.parameters(), lr=self.learning_rate, eps=1e-5)

    # Initialize training buffers
    self.legal_actions_mask = torch.zeros(
        (self.steps_per_batch, self.num_envs, self.num_actions),
        dtype=torch.bool).to(device)
    self.obs = torch.zeros((self.steps_per_batch, self.num_envs) +
                           self.input_shape).to(device)
    self.actions = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
    self.logprobs = torch.zeros(
        (self.steps_per_batch, self.num_envs)).to(device)
    self.rewards = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
    self.dones = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
    self.values = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)

    # Self-play bookkeeping: which seat acted on each (step, env), the last
    # decision per (env, seat) so seats unaffected by the final move can still
    # be closed out with their slot of the terminal rewards, and independent
    # terminal closeout samples collected during the batch.
    self.players = torch.full((self.steps_per_batch, self.num_envs),
                              self.player_id,
                              dtype=torch.long).to(device)
    self.players_cpu = torch.full((self.steps_per_batch, self.num_envs),
                                  self.player_id, dtype=torch.long)
    self._extra_samples = []
    self._last_decision = [{} for _ in range(self.num_envs)]

    # Sparse legal-action storage for the batch: per-row packed column indices
    # (tiny vs. the dense (S, N, num_actions) mask, and the basis of the
    # sparse learn path).
    self.legal_rows_packed = [None] * self.steps_per_batch
    self.legal_cols_packed = [None] * self.steps_per_batch

    # CPU views of the most recent step (avoid re-syncing from GPU in the
    # shaping / logging loop).
    self.last_obs_batch = None
    self.last_seats = None

    # Initialize counters
    self.cur_batch_idx = 0
    self.total_steps_done = 0
    self.updates_done = 0
    self.start_time = time.time()

  def get_value(self, x):
    return self.network.get_value(x)

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    return self.network.get_action_and_value(x, legal_actions_mask, action)

  def _current_seats(self, time_step):
    """Returns the acting seat for each environment in this time step."""
    if self.selfplay:
      return [
          int(ts.observations["current_player"]) for ts in time_step
      ]
    return [self.player_id for _ in time_step]

  def _obs_cpu(self, time_step, seats):
    """Batched CPU numpy observations for the acting seats."""
    return np.asarray(
        [
            np.asarray(ts.observations["info_state"][seats[i]],
                       dtype=np.float32)
            for i, ts in enumerate(time_step)
        ],
        dtype=np.float32,
    )

  def _gather_obs(self, time_step, seats):
    return torch.from_numpy(self._obs_cpu(time_step, seats)).to(self.device)

  def _build_mask(self, num_envs, mask_rows, mask_cols):
    """Builds a dense legal-action mask directly on the compute device.

    Args:
      num_envs: number of envs (rows).
      mask_rows: (M,) int64 env indices for each legal action (or None).
      mask_cols: (M,) int64 legal action ids (or None).

    Returns:
      (num_envs, num_actions) bool tensor on self.device.
    """
    mask = torch.zeros((num_envs, self.num_actions),
                       dtype=torch.bool, device=self.device)
    if mask_rows is not None and mask_rows.size:
      rows = torch.from_numpy(mask_rows).to(self.device)
      cols = torch.from_numpy(mask_cols).to(self.device)
      mask[rows, cols] = True
    return mask

  def _gather_legal_actions_mask(self, time_step, seats):
    return legal_actions_to_mask([
        ts.observations["legal_actions"][seats[i]]
        for i, ts in enumerate(time_step)
    ], self.num_actions, device=self.device)

  def step(self, time_step, is_evaluation=False):
    seats = self._current_seats(time_step)
    legal_actions_mask_cpu = legal_actions_to_mask([
        ts.observations["legal_actions"][seats[i]]
        for i, ts in enumerate(time_step)
    ], self.num_actions)
    legal_actions_mask = legal_actions_mask_cpu.to(self.device)
    obs_cpu = self._obs_cpu(time_step, seats)
    self.last_obs_batch = obs_cpu
    self.last_seats = list(seats)
    if is_evaluation:
      with torch.no_grad():
        obs = torch.from_numpy(obs_cpu).to(self.device)
        action, _, _, value, probs = self.get_action_and_value(
            obs, legal_actions_mask=legal_actions_mask)
        action_list = action.detach().cpu().tolist()
        return [
            StepOutput(action=a, probs=p)
            for (a, p) in zip(action_list, probs)
        ]
    else:
      with torch.no_grad():
        # act
        obs = torch.from_numpy(obs_cpu).to(self.device)
        action, logprob, _, value, probs = self.get_action_and_value(
            obs, legal_actions_mask=legal_actions_mask)

        # store
        row = self.cur_batch_idx
        self.players[row] = torch.tensor(seats, dtype=torch.long).to(
            self.device)
        self.players_cpu[row] = torch.tensor(seats, dtype=torch.long)
        self.legal_actions_mask[row] = legal_actions_mask
        self.obs[row] = obs
        self.actions[row] = action
        self.logprobs[row] = logprob
        self.values[row] = value.flatten()

        rows_p = []
        cols_p = []
        for i, ts in enumerate(time_step):
          for c in ts.observations["legal_actions"][seats[i]]:
            rows_p.append(i)
            cols_p.append(c)
        self.legal_rows_packed[row] = np.asarray(rows_p, dtype=np.int64)
        self.legal_cols_packed[row] = np.asarray(cols_p, dtype=np.int64)

        if self.selfplay:
          mask_cpu = legal_actions_mask_cpu.numpy()
          action_np = action.detach().cpu().numpy()
          logprob_np = logprob.detach().cpu().numpy()
          value_np = value.detach().cpu().numpy().ravel()
          for i, s in enumerate(seats):
            self._last_decision[i][s] = (
                obs_cpu[i].copy(), mask_cpu[i].copy(), int(action_np[i]),
                float(logprob_np[i]), float(value_np[i]))
          action_view = action_np
        else:
          action_view = action.detach().cpu().tolist()

        agent_output = [
            StepOutput(action=int(a), probs=p)
            for (a, p) in zip(action_view, probs)
        ]
        return agent_output

  def step_np(self, step_arrays, is_evaluation=False):
    """Array-native ``step``: consumes ``async_vector_env._StepArrays``.

    Mirrors ``step`` occupancy-for-occupancy (stores obs/action/logprob/value/
    mask into the same rolling buffers and updates the same self-play
    ``_last_decision`` bookkeeping) but works directly from numpy arrays, so no
    per-env ``TimeStep``/``StepOutput`` objects are built. Trajectory and
    stored tensors are identical to ``step`` given the same env actions.

    Args:
      step_arrays: ``_StepArrays`` from ``AsyncVectorEnv.step_np/reset_np``.
      is_evaluation: if True, skip storing and return StepOutput objects.

    Returns:
      numpy int32 array of selected actions (training path), or a list of
      ``StepOutput`` (evaluation path).
    """
    seats = step_arrays.seats
    obs_cpu = step_arrays.obs
    mask_rows = step_arrays.legal_rows
    mask_cols = step_arrays.legal_cols
    self.last_obs_batch = obs_cpu
    self.last_seats = [int(s) for s in seats]
    if is_evaluation:
      with torch.no_grad():
        obs = torch.from_numpy(obs_cpu).to(self.device)
        legal_actions_mask = self._build_mask(
            self.num_envs, mask_rows, mask_cols)
        action, _, _, value, probs = self.get_action_and_value(
            obs, legal_actions_mask=legal_actions_mask)
        action_list = action.detach().cpu().tolist()
        return [
            StepOutput(action=a, probs=p)
            for (a, p) in zip(action_list, probs)
        ]
    with torch.no_grad():
      obs = torch.from_numpy(obs_cpu).to(self.device)
      legal_actions_mask = self._build_mask(
          self.num_envs, mask_rows, mask_cols)
      action, logprob, _, value, probs = self.get_action_and_value(
          obs, legal_actions_mask=legal_actions_mask)

      row = self.cur_batch_idx
      self.players[row] = torch.from_numpy(
          seats.astype(np.int64)).to(self.device)
      self.players_cpu[row] = torch.from_numpy(seats.astype(np.int64))
      self.legal_actions_mask[row] = legal_actions_mask
      self.obs[row] = obs
      self.actions[row] = action
      self.logprobs[row] = logprob
      self.values[row] = value.flatten()

      self.legal_rows_packed[row] = mask_rows.astype(np.int64)
      self.legal_cols_packed[row] = mask_cols.astype(np.int64)

      if self.selfplay:
        action_np = action.detach().cpu().numpy()
        logprob_np = logprob.detach().cpu().numpy().ravel()
        value_np = value.detach().cpu().numpy().ravel()
        # Precompute per-env column offsets so each env's mask row can be
        # sliced out of the packed legal buffers without a scan.
        lens = np.bincount(mask_rows, minlength=self.num_envs)
        offsets = np.zeros(self.num_envs, dtype=np.int64)
        np.cumsum(lens[:-1], out=offsets[1:])
        for i, s in enumerate(seats):
          n = lens[i]
          mrow = np.zeros(self.num_actions, dtype=bool)
          if n:
            mrow[mask_cols[offsets[i]:offsets[i] + n]] = True
          self._last_decision[i][s] = (
              obs_cpu[i].copy(), mrow, int(action_np[i]),
              float(logprob_np[i]), float(value_np[i]))
        return np.asarray(action_np, dtype=np.int32)

      return action.detach().cpu().numpy().astype(np.int32)

  def post_step(self, reward, done, shaped_reward=None):
    """Stores rewards/dones for the action taken at the current batch step.

    Args:
      reward: list (one entry per environment) of per-player reward vectors, as
        returned by ``SyncVectorEnv.step`` (``ts.rewards``).
      done: list of booleans, one per environment.
      shaped_reward: optional list of floats, one per environment, holding the
        potential-based shaping delta for the acting seat's transition this
        step. Shaping is skipped for terminal transitions (the true payoff is
        used instead), which keeps the shaped reward telescope consistent.
    """
    row = self.cur_batch_idx
    if self.selfplay:
      seats = self.players_cpu[row].tolist()
      rew_row = np.empty(self.num_envs, dtype=np.float32)
      done_row = np.empty(self.num_envs, dtype=np.float32)
      for i in range(self.num_envs):
        rvec = reward[i]
        seat = seats[i]
        is_done = bool(done[i])
        shaped = 0.0 if shaped_reward is None else shaped_reward[i]
        rew_row[i] = rvec[seat] + (0.0 if is_done else shaped)
        done_row[i] = 1.0 if is_done else 0.0
        if is_done:
          for s in range(self.num_players):
            if s == seat:
              continue
            pair = self._last_decision[i][s]
            if pair is None:
              continue
            obs_, mask_, action_, logprob_, value_ = pair
            # Independent closed-out sample: target = seat's terminal payoff.
            self._extra_samples.append(
                (i, s, obs_, mask_, int(action_), logprob_, value_,
                 float(rvec[s])))
          self._last_decision[i].clear()
      self.rewards[row] = torch.from_numpy(rew_row).to(self.device)
      self.dones[row] = torch.from_numpy(done_row).to(self.device)
    else:
      self.rewards[row] = torch.tensor(reward).to(self.device).view(-1)
      self.dones[row] = torch.tensor(done).to(self.device).view(-1)

    self.total_steps_done += self.num_envs
    self.cur_batch_idx += 1

  def post_step_np(self, reward, done, shaped_reward=None):
    """Array-native ``post_step``; identical semantics, numpy inputs.

    Args:
      reward: (num_envs, num_players) float32 numpy array of per-player
        rewards.
      done: (num_envs,) bool numpy array.
      shaped_reward: optional (num_envs,) float32 numpy array with the
        potential-based shaping delta for the acting seat's transition.
    """
    row = self.cur_batch_idx
    if self.selfplay:
      seats = self.players_cpu[row].tolist()
      rew_row = np.empty(self.num_envs, dtype=np.float32)
      done_row = np.empty(self.num_envs, dtype=np.float32)
      shaped = (np.zeros(self.num_envs, dtype=np.float32)
                if shaped_reward is None else shaped_reward)
      for i in range(self.num_envs):
        rvec = reward[i]
        seat = seats[i]
        is_done = bool(done[i])
        rew_row[i] = rvec[seat] + (0.0 if is_done else shaped[i])
        done_row[i] = 1.0 if is_done else 0.0
        if is_done:
          for s in range(self.num_players):
            if s == seat:
              continue
            pair = self._last_decision[i][s]
            if pair is None:
              continue
            obs_, mask_, action_, logprob_, value_ = pair
            self._extra_samples.append(
                (i, s, obs_, mask_, int(action_), logprob_, value_,
                 float(rvec[s])))
          self._last_decision[i].clear()
      self.rewards[row] = torch.from_numpy(rew_row).to(self.device)
      self.dones[row] = torch.from_numpy(done_row).to(self.device)
    else:
      self.rewards[row] = torch.from_numpy(
          np.asarray(reward, dtype=np.float32).ravel()).to(self.device)
      self.dones[row] = torch.from_numpy(
          np.asarray(done, dtype=np.float32)).to(self.device)

    self.total_steps_done += self.num_envs
    self.cur_batch_idx += 1

  def _compute_returns(self, next_value_per_env):
    """Computes returns (and GAE advantages) for every stored transition.

    For the single-agent case this is the standard fixed-timeline GAE. For
    self-play, advantage estimation is done per (env, seat) subsequence, so a
    seat's bootstrapping chains only through that seat's own decision values
    (different seats observe different views and have different objectives).
    The only cross-seat approximation is the final-row bootstrap at the batch
    boundary when a seat has not terminated by the end of the batch.
    """
    returns = torch.zeros_like(self.rewards).to(self.device)
    if not self.selfplay:
      with torch.no_grad():
        next_value = next_value_per_env.reshape(1, -1)
        if self.gae:
          advantages = torch.zeros_like(self.rewards).to(self.device)
          lastgaelam = 0
          for t in reversed(range(self.steps_per_batch)):
            nextvalues = next_value if t == self.steps_per_batch - 1 else self.values[
                t + 1]
            nextnonterminal = 1.0 - self.dones[t]
            delta = self.rewards[
                t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[
                t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
          returns = advantages + self.values
        else:
          for t in reversed(range(self.steps_per_batch)):
            next_return = next_value if t == self.steps_per_batch - 1 else returns[
                t + 1]
            nextnonterminal = 1.0 - self.dones[t]
            returns[t] = self.rewards[
                t] + self.gamma * nextnonterminal * next_return
      return returns
    # self-play: per (env, seat) subsequence GAE.
    # Computed on CPU numpy: the per-(env,seat) chains are scalar-linked, so the
    # original pure-torch version issued ~1 tiny GPU kernel per (row, ops);
    # numpy scalar ops are several orders of magnitude cheaper here.
    players_np = self.players.cpu().numpy()
    rewards_np = self.rewards.cpu().numpy()
    dones_np = self.dones.cpu().numpy()
    values_np = self.values.cpu().numpy()
    nv = next_value_per_env.cpu().numpy()
    rets = np.zeros_like(rewards_np)
    for i in range(self.num_envs):
      for s in range(self.num_players):
        rows = np.flatnonzero(players_np[:, i] == s)
        if rows.size == 0:
          continue
        lastgaelam = 0.0
        for k in range(rows.size - 1, -1, -1):
          idx = int(rows[k])
          if k == rows.size - 1:
            nextvalues = nv[i]
          else:
            nextvalues = values_np[int(rows[k + 1]), i]
          nextnonterminal = 1.0 - dones_np[idx, i]
          delta = (
              rewards_np[idx, i] +
              self.gamma * nextvalues * nextnonterminal - values_np[idx, i])
          lastgaelam = (
              delta + self.gamma * self.gae_lambda * nextnonterminal *
              lastgaelam)
          rets[idx, i] = lastgaelam + values_np[idx, i]
    return torch.from_numpy(rets).to(self.device)

  def learn(self, time_step):
    seats = self._current_seats(time_step)
    next_obs = self._gather_obs(time_step, seats)
    self._learn_core(next_obs)

  def learn_np(self, next_obs_np, next_seats):
    """Array-native :meth:`learn` for the async vector-env path.

    Identical semantics to :meth:`learn` but takes the next-step observations
    for the acting seats directly as a (num_envs, obs_size) float32 numpy
    array, avoiding per-env ``TimeStep`` construction.
    """
    next_obs = torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32)
                               ).to(self.device)
    self._learn_core(next_obs)

  def _sparse_supported(self):
    """True when the network exposes shared features + a Linear actor head.

    Structure (see EclipsePPOAgent): ``actor = Sequential(shared,
    Linear(width, num_actions))`` so logits = features @ W^T + b with
    ``features = shared(x)`` of small width. This lets learn compute logits /
    log_prob / entropy only over the ~legal actions instead of all 11117.
    """
    net = self.network
    return (hasattr(net, "shared") and isinstance(net.actor, nn.Sequential)
            and isinstance(net.actor[-1], nn.Linear)
            and net.actor[-1].out_features == self.num_actions)

  def _pack_legal_batch(self, n_extra):
    """Builds (rows, cols, counts) over the whole flattened batch.

    rows: (M,) int64 global sample index of each legal entry (sample index =
      step * num_envs + env).
    cols: (M,) int64 legal action id.
    counts: (B_total,) int64 legal count per sample.
    Extras (terminal closeouts) are appended at the end, their cols derived
    from the stored dense mask.
    """
    rows_all = []
    cols_all = []
    num_envs = self.num_envs
    for r in range(self.steps_per_batch):
      rr = self.legal_rows_packed[r]
      cc = self.legal_cols_packed[r]
      if rr is None or cc is None or rr.size == 0:
        continue
      rows_all.append(r * num_envs + rr)
      cols_all.append(cc)
    for k, e in enumerate(self._extra_samples):
      cols_all.append(np.nonzero(e[3])[0].astype(np.int64))
      rows_all.append(np.full(len(cols_all[-1]), self.steps_per_batch *
                              num_envs + k, dtype=np.int64))
    if rows_all:
      rows = np.concatenate(rows_all)
      cols = np.concatenate(cols_all)
    else:
      rows = np.zeros(0, dtype=np.int64)
      cols = np.zeros(0, dtype=np.int64)
    total = self.steps_per_batch * num_envs + n_extra
    counts = np.bincount(rows, minlength=total)
    return rows, cols, counts

  def _sparse_minibatch(self, features, rows, cols, actions_mb, weight, bias):
    """Per-minibatch masked log-prob + entropy from sparse legal entries.

    Args:
      features: (B_mb, H) shared features of the minibatch.
      rows: (M_mb,) local sample index (into the minibatch) per legal entry.
      cols: (M_mb,) legal action id per entry.
      actions_mb: (B_mb,) chosen action per sample.
      weight: (num_actions, H) actor head weight.
      bias: (num_actions,) actor head bias.

    Returns:
      (logprob, entropy) each (B_mb,), computed only over legal entries.
    """
    M = rows.shape[0]
    if M == 0:
      b = features.shape[0]
      return (torch.zeros(b, device=self.device),
              torch.zeros(b, device=self.device))
    rows_gpu = torch.from_numpy(rows).to(self.device)
    cols_gpu = torch.from_numpy(cols).to(self.device)
    w = weight[cols_gpu]             # (M, H)
    f = features[rows_gpu]           # (M, H)
    logits_e = (f * w).sum(-1) + bias[cols_gpu]   # (M,)
    # segment reductions per block (entries of a sample are contiguous)
    b_mb = features.shape[0]
    m = torch.full((b_mb,), float("-inf"), device=self.device)
    m.scatter_reduce_(0, rows_gpu, logits_e, reduce="amax", include_self=False)
    e = (logits_e - m[rows_gpu]).exp()
    s = torch.zeros(b_mb, device=self.device)
    s.scatter_add_(0, rows_gpu, e)
    lse = m + s.log()
    num = torch.zeros(b_mb, device=self.device)
    num.scatter_add_(0, rows_gpu, e * logits_e)
    entropy = lse - num / s
    # log_prob of the chosen action: logit_a = features @ W[action] + b[action]
    wa = weight[actions_mb]          # (B_mb, H)
    logit_a = (features * wa).sum(-1) + bias[actions_mb]
    logprob = logit_a - lse
    return logprob, entropy

  def _learn_core(self, next_obs):
    # bootstrap value if not done
    with torch.no_grad():
      next_value_per_env = self.get_value(next_obs).reshape(-1)

    returns = self._compute_returns(next_value_per_env)

    # flatten the batch
    b_legal_actions_mask = self.legal_actions_mask.reshape(
        (-1, self.num_actions))
    b_obs = self.obs.reshape((-1,) + self.input_shape)
    b_logprobs = self.logprobs.reshape(-1)
    b_actions = self.actions.reshape(-1)
    b_advantages = (returns - self.values).reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = self.values.reshape(-1)

    use_sparse = self._sparse_supported()

    # Packed legal structures for the sparse learn path (built regardless of
    # extras; n_extra appended at the end of the batch below).
    sparse_rows = None
    if use_sparse:
      n_extra0 = len(self._extra_samples)
      sparse_rows, sparse_cols, sparse_counts = self._pack_legal_batch(
          n_extra0)
      total_n = self.steps_per_batch * self.num_envs + n_extra0
      sparse_offsets = np.zeros(total_n + 1, dtype=np.int64)
      np.cumsum(sparse_counts, out=sparse_offsets[1:])

    # Append independent terminal-closeout samples (self-play only). Each is a
    # done transition: returns target is that seat's terminal payoff directly,
    # with no bootstrapping.
    n_extra = len(self._extra_samples)
    if n_extra:
      ex_obs = torch.Tensor(np.array([e[2] for e in self._extra_samples])
                           ).to(self.device)
      ex_mask = torch.BoolTensor(
          np.array([e[3] for e in self._extra_samples])).to(self.device)
      ex_actions = torch.tensor([e[4]
                                 for e in self._extra_samples]).to(self.device)
      ex_logprobs = torch.tensor([e[5]
                                  for e in self._extra_samples]).to(self.device)
      ex_values = torch.tensor([e[6]
                                for e in self._extra_samples]).to(self.device)
      ex_targets = torch.tensor([e[7]
                                 for e in self._extra_samples]).to(self.device)
      b_obs = torch.cat([b_obs, ex_obs])
      b_legal_actions_mask = torch.cat([b_legal_actions_mask, ex_mask])
      b_actions = torch.cat([b_actions, ex_actions])
      b_logprobs = torch.cat([b_logprobs, ex_logprobs])
      b_values = torch.cat([b_values, ex_values])
      b_returns = torch.cat([b_returns, ex_targets])
      b_advantages = torch.cat([b_advantages, ex_targets - ex_values])
      if use_sparse:
        # Packed (rows, cols, counts) including the extra samples' legal cols.
        sparse_rows, sparse_cols, sparse_counts = self._pack_legal_batch(
            n_extra)
        total_n = self.steps_per_batch * self.num_envs + n_extra
        # per-sample cumulative offsets into the packed arrays (contiguous
        # within each sample) for fast per-minibatch slicing.
        sparse_offsets = np.zeros(total_n + 1, dtype=np.int64)
        np.cumsum(sparse_counts, out=sparse_offsets[1:])
    self._extra_samples = []

    # Keep the batch a whole number of minibatches: with a trailing remainder
    # the final minibatch can be a single sample, whose std() is NaN and
    # poisons advantage normalization. The dropped tail is only ever a small
    # number of terminal closeout samples.
    batch_size = (len(b_obs) // self.num_minibatches) * self.num_minibatches
    if batch_size != len(b_obs):
      b_obs = b_obs[:batch_size]
      b_legal_actions_mask = b_legal_actions_mask[:batch_size]
      b_actions = b_actions[:batch_size]
      b_logprobs = b_logprobs[:batch_size]
      b_values = b_values[:batch_size]
      b_returns = b_returns[:batch_size]
      b_advantages = b_advantages[:batch_size]
    minibatch_size = max(1, batch_size // self.num_minibatches)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    for _ in range(self.update_epochs):
      np.random.shuffle(b_inds)
      for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]

        if use_sparse:
          # Slice the packed legal entries for this minibatch (vectorized via
          # the per-sample cumulative offsets).
          cnt_mb = sparse_counts[mb_inds].astype(np.int64)
          m_size = int(cnt_mb.sum())
          newvalue = self.network.critic[-1](
              self.network.shared(b_obs[mb_inds])).view(-1)
          if m_size > 0:
            starts = sparse_offsets[mb_inds]
            borders = np.concatenate([[0], np.cumsum(cnt_mb)])
            rep = np.repeat(np.arange(len(mb_inds)), cnt_mb)
            within = np.arange(m_size) - borders[rep]
            col_idx = np.repeat(starts, cnt_mb) + within
            features = self.network.shared(b_obs[mb_inds])
            logprob, entropy = self._sparse_minibatch(
                features, rep, sparse_cols[col_idx],
                b_actions.long()[mb_inds], self.network.actor[-1].weight,
                self.network.actor[-1].bias)
          else:
            # Degenerate empty minibatch: uniform, zero-entropy losses.
            b = b_obs[mb_inds].shape[0]
            logprob = torch.zeros(b, device=self.device)
            entropy = torch.zeros(b, device=self.device)
        else:
          _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(
              b_obs[mb_inds],
              legal_actions_mask=b_legal_actions_mask[mb_inds],
              action=b_actions.long()[mb_inds])
          logprob = newlogprob
        logratio = logprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
          # calculate approx_kl http://joschu.net/blog/kl-approx.html
          old_approx_kl = (-logratio).mean()
          approx_kl = ((ratio - 1) - logratio).mean()
          clipfracs += [
              ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
          ]

        mb_advantages = b_advantages[mb_inds]
        if self.normalize_advantages:
          mb_advantages = (mb_advantages - mb_advantages.mean()) / (
              mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef,
                                                1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self.clip_vloss:
          v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
          v_clipped = b_values[mb_inds] + torch.clamp(
              newvalue - b_values[mb_inds],
              -self.clip_coef,
              self.clip_coef,
          )
          v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
          v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
          v_loss = 0.5 * v_loss_max.mean()
        else:
          v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_coef

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

      if self.target_kl is not None:
        if approx_kl > self.target_kl:
          break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true -
                                                         y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    if self.writer is not None:
      self.writer.add_scalar("charts/learning_rate",
                             self.optimizer.param_groups[0]["lr"],
                             self.total_steps_done)
      self.writer.add_scalar("losses/value_loss", v_loss.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/policy_loss", pg_loss.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/entropy", entropy_loss.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/approx_kl", approx_kl.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs),
                             self.total_steps_done)
      self.writer.add_scalar("losses/explained_variance", explained_var,
                             self.total_steps_done)
      self.writer.add_scalar(
          "charts/SPS",
          int(self.total_steps_done / (time.time() - self.start_time)),
          self.total_steps_done)

    # Update counters
    self.updates_done += 1
    self.cur_batch_idx = 0

  def anneal_learning_rate(self, update, num_total_updates):
    # Annealing the rate
    frac = 1.0 - (update / num_total_updates)
    if frac <= 0:
      raise ValueError("Annealing learning rate to <= 0")
    lrnow = frac * self.learning_rate
    self.optimizer.param_groups[0]["lr"] = lrnow
