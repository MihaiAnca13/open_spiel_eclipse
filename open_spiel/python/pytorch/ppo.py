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

# Rank-to-utility table used in win-value mode: the value/return of finishing
# at each placement (1st..4th). Optimizing this (rather than raw VP) is the
# 4-player general-sum objective: "finish first", not "maximize score".
DEFAULT_RANK_UTILITY = (1.0, 0.5, 0.0, -0.5)

# Safety margin for the VP escape bonus (see rank_utility): the bonus is capped
# at this fraction of the smallest gap between adjacent utility slots, so it can
# never reorder two different placements no matter how large the payoffs get.
_VP_BONUS_GAP_FRACTION = 0.49


def rank_of(per_agent_vp, seat):
  """Competition placement (1-based) of `seat` given a per-agent VP vector.

  Ties share the best placement: rank = 1 + (# agents strictly above `seat`).
  This is the reporting convention; see ``rank_utility`` for the *training*
  target, which must handle ties differently.
  """
  return 1 + sum(1 for v in per_agent_vp if v > per_agent_vp[seat])


def min_utility_gap(utility_table=DEFAULT_RANK_UTILITY):
  """Smallest gap between adjacent slots of ``utility_table``."""
  if len(utility_table) < 2:
    return float("inf")
  return min(utility_table[i] - utility_table[i + 1]
             for i in range(len(utility_table) - 1))


def vp_bonus_cap(utility_table=DEFAULT_RANK_UTILITY):
  """Largest VP bonus that provably cannot reorder two placements."""
  return _VP_BONUS_GAP_FRACTION * min_utility_gap(utility_table)


def rank_utility(per_agent_vp, seat, utility_table=DEFAULT_RANK_UTILITY,
                 vp_beta=0.0):
  """Terminal utility of `seat`, from a per-agent final VP vector.

  Two properties matter here, and the naive "ties share the best placement"
  rule (what ``rank_of`` does, correct for *reporting*) breaks both:

  1. **Ties get the mean of the slots they occupy** (fractional ranking). With
     the best-placement rule an all-tied outcome pays *every* seat
     ``utility_table[0]`` -- the maximum -- which in Eclipse makes "everybody
     goes bankrupt scoring nothing" the global optimum of the objective, since
     ~95% of unskilled games end exactly there. Averaging instead makes the
     rank term exactly constant-sum (it always sums to ``sum(utility_table)``),
     so no outcome is jointly better for everyone.

  2. **Optional VP escape bonus** ``vp_beta * own_vp``. Fixing (1) removes the
     *reward* for mutual failure but not the *flatness*: if every game ends
     all-tied at zero, every seat scores the same constant, the return variance
     is zero and there is no advantage signal to learn from. A small monotone
     term in own VP gives the optimizer a direction inside that dead zone.
     The bonus is clamped to ``vp_bonus_cap()`` so it can never reorder two
     different placements -- it only ever breaks ties and gradates within them.

  Args:
    per_agent_vp: per-seat terminal payoff vector.
    seat: seat whose utility to compute.
    utility_table: utility by placement, best first.
    vp_beta: slope of the escape bonus in utility per VP. 0 disables it and
      restores the pure constant-sum objective.

  Returns:
    Scalar utility for `seat`.
  """
  own = per_agent_vp[seat]
  above = sum(1 for v in per_agent_vp if v > own)
  tied = sum(1 for v in per_agent_vp if v == own)
  lo = min(above, len(utility_table) - 1)
  hi = min(above + max(tied, 1), len(utility_table))
  utility = sum(utility_table[lo:hi]) / max(1, hi - lo)
  if vp_beta:
    cap = vp_bonus_cap(utility_table)
    utility += max(-cap, min(cap, vp_beta * float(own)))
  return utility


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
      value_mode="vp",
      aux_tasks=None,
      aux_target_fn=None,
      aux_coef=0.1,
      rank_vp_beta=0.0,
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

    # Value / auxiliary-head options.
    #   value_mode == "vp"  : value predicts the agent's final raw payoff (the
    #     original behaviour, backward compatible).
    #   value_mode == "win" : terminal targets are rank-utilities (1st/2nd/...)
    #     so the value (and the shaped-reward scale) optimizes "finish first"
    #     rather than raw score. rank_utility() converts a per-agent terminal
    #     payoff vector into each seat's target.
    self.value_mode = value_mode
    # Slope of the VP escape bonus added to the rank utility (win mode only).
    # Mutable: callers anneal it to 0 once games stop ending in the degenerate
    # all-tied-at-zero outcome, recovering the pure constant-sum objective.
    self.rank_vp_beta = float(rank_vp_beta)
    self.rank_vp_beta_initial = float(rank_vp_beta)
    self.aux_tasks = list(aux_tasks) if aux_tasks else None
    self.num_aux = len(self.aux_tasks) if self.aux_tasks else 0
    self.aux_target_fn = aux_target_fn
    self.aux_coef = aux_coef

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
    # First row of the episode currently in progress, per env. Rows before it
    # belong to an episode that already ended inside this same batch, so they
    # must not be swept up by terminal attribution or aux back-fill.
    self._episode_start_row = np.zeros(self.num_envs, dtype=np.int64)
    # Deferred (row, env, target) terminal writes, applied as one vectorized
    # scatter in _learn_core. Writing them one scalar at a time would issue a
    # GPU kernel per (env, seat) per episode in the hot loop.
    self._closeout_writes = []

    # Auxiliary-head buffers (self-play/win mode only): per (step, env) target
    # vectors and availability masks for the aux tasks. Targets are back-filled
    # for every row of a seat the moment its episode closes (rows are
    # overwritten each batch, so rows 0..current-row of a seat all get the
    # same terminal-derived target).
    if self.num_aux:
      self.aux_targets = torch.zeros(
          (self.steps_per_batch, self.num_envs, self.num_aux),
          dtype=torch.float32).to(device)
      self.aux_mask = torch.zeros(
          (self.steps_per_batch, self.num_envs, self.num_aux),
          dtype=torch.float32).to(device)
    else:
      self.aux_targets = None
      self.aux_mask = None

    # League (population self-play) state. When enabled, the acting policy per
    # (env, seat) comes from ``lineup`` and ``networks``; only ``train_pid``'s
    # rows receive gradients (other policies generate actions but their
    # transitions are filtered out of the loss).
    self.league = False
    self.networks = None
    self.lineup = None  # (num_envs, num_players) policy ids, or None.
    self.train_pid = None
    self.trainable = torch.zeros(
        (self.steps_per_batch, self.num_envs), dtype=torch.bool).to(device)
    # CPU mirror: terminal attribution needs the trainability recorded *at act
    # time*, not a re-query of the lineup (which the caller may already have
    # re-sampled for the next episode).
    self.trainable_cpu = torch.zeros(
        (self.steps_per_batch, self.num_envs), dtype=torch.bool)

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

    # Per-update loss diagnostics (filled by _learn_core), so outer loops can
    # report value/policy/entropy/aux loss + KL/clip/explained-variance without
    # pulling them from a tensorboard writer.
    self.last_metrics = None

  def setup_league(self, networks, lineup, train_pid):
    """Enables population self-play.

    Args:
      networks: dict policy_id -> nn.Module acting as that policy. ``train_pid``
        must be present and is the module that owns ``self.network``.
      lineup: (num_envs, num_players) int/str array of policy ids, per env per
        seat.
      train_pid: policy id of the trainable network (only its rows get
        gradients).
    """
    if train_pid not in networks:
      raise ValueError(f"train_pid {train_pid} not in networks: "
                       f"{list(networks)}")
    self.networks = networks
    self.lineup = np.asarray(lineup)
    if self.lineup.shape != (self.num_envs, self.num_players):
      raise ValueError(
          f"lineup must be ({self.num_envs},{self.num_players}), got "
          f"{self.lineup.shape}")
    self.train_pid = train_pid
    if self.networks[train_pid] is not self.network:
      raise ValueError("networks[train_pid] must be self.network (the module "
                       "PPO owns and optimizes)")
    self.league = True

  def _acts_trainable(self, env_idx, seat):
    """True when (env, seat) is driven by the trainable policy (league)."""
    if not self.league:
      return True
    return self.lineup[env_idx, seat] == self.train_pid

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

  def _act_batch(self, obs, legal_actions_mask, seats):
    """One batched forward over ``obs``, routing rows by acting policy.

    In league mode each (env, seat) is driven by the policy in ``lineup``, so
    rows are grouped by policy id and forwarded through that policy's network
    (main stays ``self.network``). Non-league is the single (shared) network.
    """
    if not self.league:
      return self.get_action_and_value(
          obs, legal_actions_mask=legal_actions_mask)
    pids = np.asarray([
        self.lineup[i, int(seats[i])] for i in range(obs.shape[0])])
    action = torch.empty(obs.shape[0], dtype=torch.long, device=self.device)
    logprob = torch.empty(obs.shape[0], device=self.device)
    value = torch.empty(obs.shape[0], device=self.device)
    probs = torch.empty((obs.shape[0], self.num_actions),
                        device=self.device)
    entropy = torch.empty(obs.shape[0], device=self.device)
    for pid in np.unique(pids):
      idx = np.flatnonzero(pids == pid)
      net = self.networks.get(pid)
      if net is None:
        raise ValueError(f"league lineup references unknown policy {pid}")
      a, lp, ent, v, pr = net.get_action_and_value(
          obs[idx], legal_actions_mask=legal_actions_mask[idx])
      i_t = torch.from_numpy(idx.astype(np.int64)).to(self.device)
      action[i_t] = a
      logprob[i_t] = lp
      value[i_t] = v
      probs[i_t] = pr
      entropy[i_t] = ent
    return action, logprob, entropy, value, probs

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
        action, logprob, _, value, probs = self._act_batch(
            obs, legal_actions_mask, seats)

        # store
        row = self.cur_batch_idx
        self.players[row] = torch.tensor(seats, dtype=torch.long).to(
            self.device)
        self.players_cpu[row] = torch.tensor(seats, dtype=torch.long)
        trainable_row = torch.tensor(
            [self._acts_trainable(i, seats[i]) for i in range(self.num_envs)],
            dtype=torch.bool)
        self.trainable_cpu[row] = trainable_row
        self.trainable[row] = trainable_row.to(self.device)
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
          action_np = action.detach().cpu().numpy()
          logprob_np = logprob.detach().cpu().numpy().ravel()
          value_np = value.detach().cpu().numpy().ravel()
          for i, s in enumerate(seats):
            if not self._acts_trainable(i, s):
              continue
            # Packed legal cols for this env's seat (no dense mask materialized);
            # matches the format stored by step_np.
            la = ts.observations["legal_actions"][seats[i]]
            cols = (np.asarray(la, dtype=np.int64) if la else
                    np.zeros(0, dtype=np.int64))
            self._last_decision[i][s] = (
                obs_cpu[i].copy(), cols, int(action_np[i]),
                float(logprob_np[i]), float(value_np[i]))
          action_view = action.detach().cpu().numpy()
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
      row = self.cur_batch_idx
      if self._sparse_supported():
        # Sparse acting: no dense (num_envs, num_actions) logits/mask built or
        # transferred; sample from packed legal entries only. ``legal_actions_mask``
        # buffer is left stale (unused by the sparse learn path).
        action, logprob, _, value = self._act_sparse(
            obs, mask_rows, mask_cols, seats)
      else:
        legal_actions_mask = self._build_mask(
            self.num_envs, mask_rows, mask_cols)
        action, logprob, _, value, _ = self._act_batch(
            obs, legal_actions_mask, seats)
        self.legal_actions_mask[row] = legal_actions_mask

      self.players[row] = torch.from_numpy(
          seats.astype(np.int64)).to(self.device)
      self.players_cpu[row] = torch.from_numpy(seats.astype(np.int64))
      trainable_row = torch.tensor(
          [self._acts_trainable(i, int(sv)) for i, sv in enumerate(seats)],
          dtype=torch.bool)
      self.trainable_cpu[row] = trainable_row
      self.trainable[row] = trainable_row.to(self.device)
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
        # Per-env column offsets so each env's legal cols can be sliced out of
        # the packed buffers (no dense 11k mask build) for the closeout sample.
        lens = np.bincount(np.asarray(mask_rows, dtype=np.int64),
                           minlength=self.num_envs)
        offsets = np.zeros(self.num_envs, dtype=np.int64)
        np.cumsum(lens[:-1], out=offsets[1:])
        for i, s in enumerate(seats):
          if not self._acts_trainable(i, int(s)):
            continue
          n = int(lens[i])
          cols = (mask_cols[offsets[i]:offsets[i] + n].astype(np.int64)
                  if n else np.zeros(0, dtype=np.int64))
          self._last_decision[i][int(s)] = (
              obs_cpu[i].copy(), cols, int(action_np[i]),
              float(logprob_np[i]), float(value_np[i]))
        return np.asarray(action_np, dtype=np.int32)

      return action.detach().cpu().numpy().astype(np.int32)

  def _resolve_last_decision(self, env_idx, seat):
    """(obs, packed_cols, action, logprob, value) for a seat's last decision.

    Returns the committed CPU record stored per step by ``step``/``step_np``
    (obs copy + legal cols + action + logprob + value). None if the seat never
    had a recorded decision.
    """
    return self._last_decision[env_idx].get(seat)

  def _terminal_target(self, per_agent_reward, seat):
    """Scalar terminal target for `seat` given the per-agent payoff vector."""
    if self.value_mode == "win":
      return rank_utility(per_agent_reward, seat,
                          vp_beta=self.rank_vp_beta)
    return per_agent_reward[seat]

  def _aux_targets_for(self, per_agent_reward):
    """Per-seat aux-target matrix (num_players, num_aux), or None.

    Uses ``aux_target_fn`` if supplied, else defaults to the raw per-agent
    payoff (one-column aux = predict a seat's final payoff).
    """
    if not self.num_aux:
      return None
    if self.aux_target_fn is None:
      arr = np.asarray(per_agent_reward, dtype=np.float32)
      if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    else:
      arr = np.asarray(self.aux_target_fn(per_agent_reward), dtype=np.float32)
    try:
      return arr.reshape(self.num_players, self.num_aux)
    except ValueError:
      return None

  def _attribute_terminal(self, env_idx, row, per_agent_reward, acting_seat):
    """Gives every seat other than the mover its slot of the terminal payoff.

    ``dones`` is one flag per (row, env), so only the seat that happened to make
    the final move gets a terminal row. Left alone, every *other* seat's last
    decision keeps ``done=0``, and its per-seat GAE chain therefore bootstraps
    straight across the episode boundary into the values of the *next* episode
    (envs auto-reset), so the terminal payoff never reaches the trajectory that
    earned it. Previously those rows were also re-emitted as independent
    ``_extra_samples`` carrying the true target, leaving two contradictory
    targets for the same (obs, action).

    Fix: when the seat's last decision is a row in this batch, overwrite that
    row's reward with the seat's terminal target and mark it done -- which both
    attributes the payoff and cuts the chain. Only when the seat last acted in
    an *earlier* batch (its row is gone) do we fall back to an extra sample.
    """
    start = int(self._episode_start_row[env_idx])
    players_col = self.players_cpu[start:row + 1, env_idx].numpy()
    trainable_col = self.trainable_cpu[start:row + 1, env_idx].numpy()
    for seat in range(self.num_players):
      if seat == acting_seat:
        continue
      target = self._terminal_target(per_agent_reward, seat)
      rows_seat = np.flatnonzero((players_col == seat) & trainable_col)
      if rows_seat.size:
        self._closeout_writes.append(
            (start + int(rows_seat[-1]), env_idx, float(target)))
        continue
      # No row in this batch: the seat last acted before the batch boundary, so
      # carry its stored decision as an independent terminal sample.
      if not self._acts_trainable(env_idx, seat):
        continue
      pair = self._resolve_last_decision(env_idx, seat)
      if pair is None:
        continue
      obs_, cols_, action_, logprob_, value_ = pair
      ex_aux, ex_aux_mask = self._extras_aux(per_agent_reward, seat)
      self._extra_samples.append(
          (env_idx, seat, obs_, cols_, int(action_), logprob_, value_,
           target, ex_aux, ex_aux_mask))

  def _apply_closeout_writes(self):
    """Flushes deferred terminal rewards/dones as one vectorized scatter."""
    if not self._closeout_writes:
      return
    rows = np.fromiter((w[0] for w in self._closeout_writes), dtype=np.int64,
                       count=len(self._closeout_writes))
    envs = np.fromiter((w[1] for w in self._closeout_writes), dtype=np.int64,
                       count=len(self._closeout_writes))
    tgts = np.fromiter((w[2] for w in self._closeout_writes), dtype=np.float32,
                       count=len(self._closeout_writes))
    rows_t = torch.from_numpy(rows).to(self.device)
    envs_t = torch.from_numpy(envs).to(self.device)
    self.rewards[rows_t, envs_t] = torch.from_numpy(tgts).to(self.device)
    self.dones[rows_t, envs_t] = 1.0
    self._closeout_writes = []

  def _backfill_aux(self, env_idx, row, per_agent_reward):
    """Fills aux targets for every stored row (<=row) of each seat in `env_idx`.

    Terminal payoff fixes the final target for all seats of a finished episode,
    so every row of a seat in this batch (which came from the same episode)
    inherits that seat's terminal-derived aux target.
    """
    if not self.num_aux:
      return
    tgt = self._aux_targets_for(per_agent_reward)
    if tgt is None:
      return
    start = int(self._episode_start_row[env_idx])
    players_col = self.players_cpu[start:row + 1, env_idx]
    trainable_col = self.trainable_cpu[start:row + 1, env_idx].numpy()
    for s in range(self.num_players):
      rows_s = np.flatnonzero((players_col.numpy() == s) & trainable_col)
      if rows_s.size == 0:
        continue
      rows_t = torch.from_numpy(
          (rows_s + start).astype(np.int64)).to(self.device)
      target_t = torch.from_numpy(tgt[s]).to(self.device)
      self.aux_targets[rows_t, env_idx, :] = target_t
      self.aux_mask[rows_t, env_idx, :] = 1.0

  def _extras_aux(self, per_agent_reward, seat):
    """(aux_target, aux_mask) 1-D tensors for one extra sample's seat."""
    if not self.num_aux:
      return None, None
    tgt = self._aux_targets_for(per_agent_reward)
    if tgt is None:
      return None, None
    return (torch.from_numpy(tgt[seat]).to(self.device),
            torch.ones((self.num_aux,), dtype=torch.float32).to(self.device))

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
        rew_row[i] = (self._terminal_target(rvec, seat) if is_done else
                      rvec[seat] + shaped)
        done_row[i] = 1.0 if is_done else 0.0
        if is_done:
          self._backfill_aux(i, row, rvec)
          self._attribute_terminal(i, row, rvec, seat)
          self._last_decision[i].clear()
          self._episode_start_row[i] = row + 1
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
      seats = self.players_cpu[row].numpy()
      reward_np = np.asarray(reward, dtype=np.float32)      # (N, num_players)
      done_np = np.asarray(done, dtype=bool)
      shaped = (np.zeros(self.num_envs, dtype=np.float32)
                if shaped_reward is None else shaped_reward)
      chosen = reward_np[np.arange(self.num_envs), seats]
      terminal = np.empty(self.num_envs, dtype=np.float32)
      terminal.fill(0.0)
      done_idx = np.flatnonzero(done_np)
      if done_idx.size:
        if self.value_mode == "win":
          for i in done_idx:
            terminal[i] = rank_utility(reward_np[i], int(seats[i]),
                                       vp_beta=self.rank_vp_beta)
        else:
          for i in done_idx:
            terminal[i] = reward_np[i, int(seats[i])]
      rew_row = np.where(done_np, terminal, chosen + shaped).astype(np.float32)
      done_row = done_np.astype(np.float32)
      # Terminal-closeout bookkeeping only for envs that actually finished.
      for i in done_idx:
        rvec = reward_np[i]
        seat = int(seats[i])
        self._backfill_aux(i, row, rvec)
        self._attribute_terminal(int(i), row, rvec, seat)
        self._last_decision[i].clear()
        self._episode_start_row[i] = row + 1
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

  def _pack_legal_batch(self, n_extra, remap=None, extra_base=None):
    """Builds (rows, cols, counts) over the whole flattened batch.

    rows: (M,) int64 global sample index of each legal entry (sample index =
      step * num_envs + env, or the filtered index when ``remap`` is given).
    cols: (M,) int64 legal action id.
    counts: (B_total,) int64 legal count per sample.
    Extras (terminal closeouts) are appended at the end, their cols carried
    directly as packed legal column indices (slot ``e[3]``).

    Args:
      remap: optional (steps * num_envs,) int64 array mapping old global
        sample index to the filtered index, with -1 for dropped (non-trainable
        league) samples.
      extra_base: sample index where extras begin (defaults to
        steps * num_envs; in league w/ remap this is the filtered row count).
    """
    rows_all = []
    cols_all = []
    num_envs = self.num_envs
    for r in range(self.steps_per_batch):
      rr = self.legal_rows_packed[r]
      cc = self.legal_cols_packed[r]
      if rr is None or cc is None or rr.size == 0:
        continue
      old = r * num_envs + rr
      if remap is not None:
        new = remap[old]
        keep_idx = new >= 0
        if not np.any(keep_idx):
          continue
        rows_all.append(new[keep_idx])
        cols_all.append(cc[keep_idx])
      else:
        rows_all.append(old)
        cols_all.append(cc)
    if extra_base is None:
      extra_base = self.steps_per_batch * num_envs
    for k, e in enumerate(self._extra_samples):
      cols_all.append(e[3].astype(np.int64))
      rows_all.append(
          np.full(len(cols_all[-1]), extra_base + k, dtype=np.int64))
    if rows_all:
      rows = np.concatenate(rows_all)
      cols = np.concatenate(cols_all)
    else:
      rows = np.zeros(0, dtype=np.int64)
      cols = np.zeros(0, dtype=np.int64)
    total = (extra_base if remap is not None else
             self.steps_per_batch * num_envs) + n_extra
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

  def _pack_logits(self, features, rows, cols, weight, bias):
    """Logits at the legal entries. ``rows`` are torch sample indices (a
    row's entries contiguous); ``cols`` the legal action ids; ``features`` the
    shared features for the batch. Returns (M,) logits and the (M,) rows torch
    tensor for downstream segment ops."""
    w = weight[cols]                 # (M, H)
    f = features[rows]               # (M, H)
    return (f * w).sum(-1) + bias[cols]

  def _segment_lse_entropy(self, logits_e, rows, batch_size):
    """Per-row logsumexp + entropy of the (masked) distribution given the
    (M,) logits of legal entries grouped contiguously by row."""
    m = torch.full((batch_size,), float("-inf"), device=self.device)
    m.scatter_reduce_(0, rows, logits_e, reduce="amax", include_self=False)
    e = (logits_e - m[rows]).exp()
    s = torch.zeros(batch_size, device=self.device)
    s.scatter_add_(0, rows, e)
    lse = m + s.log()
    num = torch.zeros(batch_size, device=self.device)
    num.scatter_add_(0, rows, e * logits_e)
    entropy = lse - num / s
    return lse, entropy

  def _gumbel_sample(self, logits_e, rows, cols, batch_size):
    """Sample one action per row by Gumbel-max over the legal entries.

    The Gumbel trick is exactly how ``torch.distributions.Categorical.sample``
    samples (logits + Gumbel argmax), so this reproduces the dense
    ``CategoricalMasked`` distribution without materializing the full
    (B, num_actions) logits. Returns (B,) chosen action ids.
    """
    g = -torch.log(-torch.log(torch.rand(logits_e.shape, device=self.device)))
    u = logits_e + g
    mx = torch.full((batch_size,), float("-inf"), device=self.device)
    mx.scatter_reduce_(0, rows, u, reduce="amax", include_self=False)
    maxima = (u == mx[rows])
    chosen = torch.full((batch_size,), self.num_actions, dtype=torch.long,
                        device=self.device)
    mrows = rows[maxima]
    mcols = cols[maxima]
    if mrows.size(0):
      # 'amin' over the (measure-zero) tied maxima columns; include_self=False
      # so healthy rows get exactly one valid col.
      chosen.scatter_reduce_(0, mrows, mcols, reduce="amin", include_self=False)
    return chosen

  def _log_prob_chosen(self, features, chosen, lse, weight, bias):
    """(B,) log_prob of ``chosen`` under the masked distribution with known
    per-row logsumexp ``lse``."""
    wa = weight[chosen]
    logit_a = (features * wa).sum(-1) + bias[chosen]
    return logit_a - lse

  def _act_sparse(self, obs, mask_rows, mask_cols, seats):
    """Array-native, sparse acting: sample from packed legal logits only.

    Replaces the dense (num_envs, num_actions) logits + mask + softmax in the
    per-step hot loop with one shared-trunk forward plus logits/entropy/logprob
    computed only at the legal columns. Distributionally identical to dense
    ``CategoricalMasked`` sampling (Gumbel-max over the legal set). League
    traffic is dispatched per policy as in ``_act_batch``.

    Args:
      obs: (num_envs, obs_size) torch tensor on device.
      mask_rows: (M,) numpy env (== row) indices per legal action.
      mask_cols: (M,) numpy legal action ids.
      seats: (num_envs,) numpy acting seat per env.

    Returns:
      (action, logprob, entropy, value) torch tensors, each (num_envs,).
    """
    batch = obs.shape[0]
    if batch == 0:
      return (torch.zeros(0, dtype=torch.long, device=self.device),
              torch.zeros(0, device=self.device),
              torch.zeros(0, device=self.device),
              torch.zeros(0, device=self.device))
    if not self.league:
      nets = [self.network]
      groups = [np.arange(batch)]
    else:
      pids = np.asarray(
          [self.lineup[i, int(seats[i])] for i in range(batch)])
      groups = [np.flatnonzero(pids == p) for p in np.unique(pids)]
      nets = [self.networks[p] for p in np.unique(pids)]
    action = torch.empty(batch, dtype=torch.long, device=self.device)
    logprob = torch.empty(batch, device=self.device)
    entropy = torch.empty(batch, device=self.device)
    values = torch.empty(batch, device=self.device)
    for net, idx in zip(nets, groups):
      n = len(idx)
      features = net.shared(obs[idx])
      # Remap env (== row) indices in this group to local 0..n-1.
      local = np.full(batch, -1, dtype=np.int64)
      local[idx] = np.arange(n)
      keep = local[mask_rows] >= 0
      rows_np = local[mask_rows[keep]]
      cols = torch.from_numpy(mask_cols[keep].astype(np.int64)).to(self.device)
      rows = torch.from_numpy(rows_np.astype(np.int64)).to(self.device)
      logits_e = self._pack_logits(features, rows, cols,
                                   net.actor[-1].weight, net.actor[-1].bias)
      chosen = self._gumbel_sample(logits_e, rows, cols, n)
      lse, ent = self._segment_lse_entropy(logits_e, rows, n)
      lp = self._log_prob_chosen(features, chosen, lse,
                                 net.actor[-1].weight, net.actor[-1].bias)
      if hasattr(net, "value_from_features"):
        v = net.value_from_features(features).view(-1)
      else:
        v = net.critic[-1](features).view(-1)
      idx_t = torch.from_numpy(idx.astype(np.int64)).to(self.device)
      action[idx_t] = chosen
      logprob[idx_t] = lp
      entropy[idx_t] = ent
      values[idx_t] = v
    return action, logprob, entropy, values

  def _value_from_features(self, features):
    """Scalar value from shared features (sparse-path value computation).

    Networks may implement ``value_from_features`` (e.g. Eclipse's rank-head
    win value); otherwise the dense path's critic tail is applied to the
    features and squeezed to a scalar.
    """
    net = self.network
    if hasattr(net, "value_from_features"):
      return net.value_from_features(features)
    return net.critic[-1](features).view(-1)

  def _learn_core(self, next_obs):
    # Terminal attribution for non-acting seats, deferred during the rollout to
    # keep it off the hot loop. Must land before returns are computed.
    self._apply_closeout_writes()

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

    if self.num_aux:
      b_aux = self.aux_targets.reshape(-1, self.num_aux)
      b_aux_mask = self.aux_mask.reshape(-1, self.num_aux)
    else:
      b_aux = None
      b_aux_mask = None

    use_sparse = self._sparse_supported()

    # Append independent terminal-closeout samples (self-play only). Each is a
    # done transition: returns target is that seat's terminal payoff directly,
    # with no bootstrapping.
    n_extra = len(self._extra_samples)
    if n_extra:
      ex_obs = torch.Tensor(np.array([e[2] for e in self._extra_samples])
                           ).to(self.device)
      ex_actions = torch.tensor([e[4]
                                 for e in self._extra_samples]).to(self.device)
      ex_logprobs = torch.tensor([e[5]
                                  for e in self._extra_samples]).to(self.device)
      ex_values = torch.tensor([e[6]
                                for e in self._extra_samples]).to(self.device)
      ex_targets = torch.tensor([e[7]
                                 for e in self._extra_samples]).to(self.device)
      b_obs = torch.cat([b_obs, ex_obs])
      if not use_sparse:
        # Dense-path extras mask, reconstructed from packed legal cols.
        ex_mask = torch.zeros((n_extra, self.num_actions), dtype=torch.bool,
                              device=self.device)
        for k, e in enumerate(self._extra_samples):
          if e[3].size:
            ex_mask[k, torch.from_numpy(e[3]).to(self.device)] = True
        b_legal_actions_mask = torch.cat([b_legal_actions_mask, ex_mask])
      b_actions = torch.cat([b_actions, ex_actions])
      b_logprobs = torch.cat([b_logprobs, ex_logprobs])
      b_values = torch.cat([b_values, ex_values])
      b_returns = torch.cat([b_returns, ex_targets])
      b_advantages = torch.cat([b_advantages, ex_targets - ex_values])
      if self.num_aux:
        ex_aux = torch.stack([e[8] for e in self._extra_samples])
        ex_aux_mask = torch.stack([e[9] for e in self._extra_samples])
        b_aux = torch.cat([b_aux, ex_aux])
        b_aux_mask = torch.cat([b_aux_mask, ex_aux_mask])

    # League: drop transitions generated by non-trainable (opponent) policies
    # before any loss math. Only ``train_pid`` rows keep gradients; the packed
    # legal structures are remapped to the filtered sample indices (extras are
    # already trainable-only, so they pass through at the tail).
    remap = None
    extra_base = self.steps_per_batch * self.num_envs
    if self.league:
      keep_train = self.trainable.reshape(-1)
      keep = torch.cat([
          keep_train,
          torch.ones(n_extra, dtype=torch.bool, device=self.device)
      ])
      keep_np = keep.cpu().numpy()
      # Only main-region rows get remapped into the filtered index space; a
      # non-trainable (opponent) row is dropped (remap -1). Extras are already
      # trainable-only and are appended after all main rows.
      nb_main = int(keep_train.cpu().numpy().sum())
      b_obs = b_obs[keep]
      if not use_sparse:
        # In the sparse path b_legal_actions_mask is unused (packed legal
        # structures drive the loss) and has no extra rows, so it must not be
        # indexed by the extras-extended keep vector.
        b_legal_actions_mask = b_legal_actions_mask[keep]
      b_actions = b_actions[keep]
      b_logprobs = b_logprobs[keep]
      b_values = b_values[keep]
      b_returns = b_returns[keep]
      b_advantages = b_advantages[keep]
      if self.num_aux:
        b_aux = b_aux[keep]
        b_aux_mask = b_aux_mask[keep]
      if use_sparse:
        remap = -np.ones(keep_train.shape[0], dtype=np.int64)
        remap[keep_train.cpu().numpy()] = np.arange(nb_main, dtype=np.int64)
      extra_base = nb_main

    # Packed legal structures for the sparse learn path (built after the
    # league filter if active).
    total_n = extra_base + n_extra
    sparse_rows = sparse_cols = sparse_counts = None
    sparse_offsets = None
    if use_sparse:
      sparse_rows, sparse_cols, sparse_counts = self._pack_legal_batch(
          n_extra, remap=remap, extra_base=extra_base)
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
      if self.num_aux:
        b_aux = b_aux[:batch_size]
        b_aux_mask = b_aux_mask[:batch_size]
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
          features = self.network.shared(b_obs[mb_inds])
          newvalue = self._value_from_features(features)
          cnt_mb = sparse_counts[mb_inds].astype(np.int64)
          m_size = int(cnt_mb.sum())
          if m_size > 0:
            starts = sparse_offsets[mb_inds]
            borders = np.concatenate([[0], np.cumsum(cnt_mb)])
            rep = np.repeat(np.arange(len(mb_inds)), cnt_mb)
            within = np.arange(m_size) - borders[rep]
            col_idx = np.repeat(starts, cnt_mb) + within
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
          features = None
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

        # Auxiliary-head loss: supervise the trunk's auxiliary predictors
        # (e.g. "final total VP") on the terminal-derived targets back-filled
        # into this batch. Only rows whose episode closed get a target; others
        # are masked out, so the loss is over the available fraction.
        aux_loss = torch.zeros((), device=self.device)
        aux_hit = 0
        if self.num_aux and features is not None:
          pred = self.network.get_aux(features)
          tgt = b_aux[mb_inds]
          msk = b_aux_mask[mb_inds]
          for k, name in enumerate(self.aux_tasks):
            p = pred[name]
            if p.dim() > 1 and p.size(1) == 1:
              p = p.view(-1)
            m = msk[:, k].float()
            denom = m.sum()
            if denom > 0:
              aux_loss = aux_loss + (((p - tgt[:, k])**2) * m).sum() / denom
              aux_hit += 1
          if aux_hit:
            loss = loss + self.aux_coef * aux_loss

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

    # Per-update diagnostics for the outer loop (tqdm progress / console).
    self.last_metrics = {
        "policy_loss": float(pg_loss.detach()),
        "value_loss": float(v_loss.detach()),
        "entropy": float(entropy_loss.detach()),
        "aux_loss": float(aux_loss.detach()),
        "clipfrac": float(np.mean(clipfracs)),
        "old_kl": float(old_approx_kl.detach()),
        "kl": float(approx_kl.detach()),
        "explained_variance": float(explained_var),
    }


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
      if self.num_aux and features is not None:
        self.writer.add_scalar("losses/aux_loss", aux_loss.item(),
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
    # Rows are overwritten from 0 next batch, so every env's in-progress episode
    # now starts at row 0 again.
    self._episode_start_row[:] = 0

  def anneal_rank_vp_beta(self, update, num_total_updates, beta_to=0.0):
    """Linearly moves the VP escape bonus from its initial slope to ``beta_to``.

    The bonus exists to create return variance in the degenerate all-tied-at-
    zero regime; once real outcomes differ it is no longer needed, and holding
    it at a nonzero value leaves the objective mildly general-sum (total utility
    becomes ``sum(utility_table) + beta * sum(VP)``). Annealing it out restores
    the pure constant-sum "finish first" objective for the long run.
    """
    if num_total_updates <= 0:
      return self.rank_vp_beta
    frac = min(1.0, max(0.0, update / num_total_updates))
    self.rank_vp_beta = (
        self.rank_vp_beta_initial +
        (float(beta_to) - self.rank_vp_beta_initial) * frac)
    return self.rank_vp_beta

  def anneal_learning_rate(self, update, num_total_updates):
    # Annealing the rate
    frac = 1.0 - (update / num_total_updates)
    if frac <= 0:
      raise ValueError("Annealing learning rate to <= 0")
    lrnow = frac * self.learning_rate
    self.optimizer.param_groups[0]["lr"] = lrnow
