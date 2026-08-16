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


def rank_distribution(per_agent_vp, seat,
                      num_ranks=len(DEFAULT_RANK_UTILITY)):
  """Soft placement target, uniform over every rank occupied by a tie."""
  own = per_agent_vp[seat]
  above = sum(1 for v in per_agent_vp if v > own)
  tied = sum(1 for v in per_agent_vp if v == own)
  lo = min(above, num_ranks - 1)
  hi = min(above + max(tied, 1), num_ranks)
  target = np.zeros(num_ranks, dtype=np.float32)
  target[lo:hi] = 1.0 / max(1, hi - lo)
  return target


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


def head_logits(head, features, rows, cols):
  """(M,) logits for the (rows[i], cols[i]) pairs, for any actor head type.

  An action's logit factors as ``features[row] . weight[col] + bias[col]``, and
  the weight row for ``col`` comes from the head's ``rows_for`` (for a
  ``FactoredActorHead``) or directly from ``head.weight[col]`` (for a plain
  ``nn.Linear``). ``cells`` formerly threaded a spatial pointer term into the
  head; that mechanism was removed.
  """
  rows_for = getattr(head, "rows_for", None)
  w = rows_for(cols) if rows_for is not None else head.weight[cols]
  return (features[rows] * w).sum(-1) + head.bias[cols]


class _ObsRows:
  """Row-addressable view over the rollout obs buffer plus the closeout extras.

  ``_learn_core`` needs to treat (rollout rows, terminal-closeout extra rows) as
  one batch that it filters and then draws minibatches from. Doing that with
  ``torch.cat`` allocates a second copy of the *entire* rollout buffer in order
  to append a few hundred rows -- 39.4 GB at 2,048 envs to bolt on ~30 MB of
  extras -- and that doubled peak is what forces the buffer off the GPU at high
  env counts, costing ~34% throughput.

  So the two tensors stay separate and this carries an int64 row-id vector
  instead: filtering is a cheap permutation of ids, and only the minibatch is
  ever materialized. Rows are addressed in the *concatenated* space, so ids in
  [0, len(main)) index ``main`` and ids >= len(main) index ``extra`` -- which is
  exactly the indexing the packed-legal remap in ``_pack_legal_batch`` already
  assumes, so nothing downstream has to change.

  Row ids are kept on the host on purpose: the per-minibatch index math then
  costs no GPU sync, and only a ~65 KB index vector crosses to the device per
  minibatch (the same transfer the old ``torch.from_numpy(mb_inds)`` did).
  """

  def __init__(self, main, extra=None, row_ids=None, out_dtype=torch.float32):
    self.main = main
    self.extra = (extra if extra is not None and extra.shape[0] else None)
    self.n_main = int(main.shape[0])
    self.out_dtype = out_dtype
    if row_ids is None:
      n_extra = 0 if self.extra is None else int(self.extra.shape[0])
      row_ids = torch.arange(self.n_main + n_extra, dtype=torch.int64)
    self.row_ids = row_ids

  def __len__(self):
    return int(self.row_ids.shape[0])

  @property
  def device(self):
    return self.main.device

  def select(self, sel):
    """A filtered view. ``sel`` is a slice or a bool/index tensor over logical
    rows; the underlying data is never copied."""
    if isinstance(sel, slice):
      ids = self.row_ids[sel]
    else:
      t = sel if isinstance(sel, torch.Tensor) else torch.as_tensor(sel)
      ids = self.row_ids[t.to(self.row_ids.device)]
    return _ObsRows(self.main, self.extra, ids, self.out_dtype)

  def minibatch(self, mb_inds, device):
    """Materialize the rows at logical positions ``mb_inds`` on ``device``.

    The gather happens in the buffer's storage dtype and the upcast to
    ``out_dtype`` happens last, on the destination device -- so a 16-bit buffer
    also halves the bytes crossing the bus on the host-buffer fallback path,
    rather than only saving memory.
    """
    ids = self.row_ids[torch.as_tensor(np.asarray(mb_inds), dtype=torch.int64)]
    src = self.main
    if self.extra is None:
      rows = src[ids.to(src.device)]
    else:
      is_extra = ids >= self.n_main
      if not bool(is_extra.any()):
        rows = src[ids.to(src.device)]
      else:
        rows = torch.empty((ids.shape[0],) + tuple(src.shape[1:]),
                           dtype=src.dtype, device=src.device)
        from_main = ~is_extra
        if bool(from_main.any()):
          rows[from_main.to(src.device)] = src[ids[from_main].to(src.device)]
        rows[is_extra.to(src.device)] = self.extra[
            (ids[is_extra] - self.n_main).to(self.extra.device)]
    return rows.to(device=device).to(self.out_dtype)


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
      rank_ce_coef=0.0,
      amp=False,
      obs_buffer_device="auto",
      obs_buffer_dtype="auto",
  ):
    super().__init__()

    self.input_shape = input_shape
    self.num_actions = num_actions
    self.num_players = num_players
    self.player_id = player_id
    self.selfplay = num_players > 1
    self.device = device
    # bf16 autocast on the _learn_core minibatch loop only -- see the comment
    # at its call site for why the act path is excluded.
    self.amp = amp

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
    # Distributional critic: cross-entropy on the *realized* placement. The 4
    # rank logits are otherwise trained only through the MSE of the scalar
    # expected utility they produce, which is a very thin use of a
    # distributional head. 0 disables it.
    self.rank_ce_coef = float(rank_ce_coef)
    self.obs_buffer_device_pref = obs_buffer_device
    self.obs_buffer_dtype = self._resolve_obs_buffer_dtype(
        obs_buffer_dtype, device)

    # Logging
    self.writer = writer

    # Initialize networks
    self.network = agent_fn(self.num_actions, self.input_shape,
                            device).to(device)
    self.optimizer = optim.Adam(
        self.parameters(), lr=self.learning_rate, eps=1e-5)

    # Initialize training buffers. The dense legal-action mask is only read by
    # the dense learn path; when the sparse path is active the packed
    # rows/cols drive the loss instead, so allocating it wastes real memory --
    # (steps, envs, num_actions) bool is 182 MB at 128 envs for Eclipse's 11117
    # actions, and scales linearly with num_envs.
    mask_steps = 0 if self._sparse_supported() else self.steps_per_batch
    self.legal_actions_mask = torch.zeros(
        (mask_steps, self.num_envs, self.num_actions),
        dtype=torch.bool).to(device)
    # Obs rollout storage device. Keeping it on CPU bounds VRAM to one
    # minibatch, which is what lifted the 128-env cap on a 12 GiB card -- but it
    # makes _learn_core pay a host-side gather plus an unpinned H2D copy of the
    # whole minibatch on every one of its update_epochs*num_minibatches
    # iterations, with the GPU stalled throughout. For Eclipse's 37,596-float
    # observation at 256 envs that is 1.23 GB moved 16 times = 19.7 GB per
    # update, measured at 219 ms per minibatch (115 ms gather + 112 ms copy) or
    # 3.5 s per update -- 44% of learn(). The same gather from a device-resident
    # buffer measures 2 ms, so on a card with room to hold the batch this is a
    # ~100x cheaper place to keep it.
    #
    # "auto" therefore keeps the buffer on the training device when it fits in
    # a conservative slice of *currently free* VRAM, and silently falls back to
    # CPU when it does not (preserving the small-card behaviour). "cpu"/"cuda"
    # force the choice.
    self.obs_buffer_device = self._resolve_obs_buffer_device(
        obs_buffer_device, device)
    self.obs = torch.zeros((self.steps_per_batch, self.num_envs) +
                           self.input_shape,
                           dtype=self.obs_buffer_dtype,
                           device=self.obs_buffer_device)
    # True when the rollout obs buffer and the network live on the same device,
    # so a stored row needs no host round-trip and a minibatch needs no copy.
    # Compared through real tensors rather than the requested device strings:
    # torch.device("cuda") != torch.device("cuda:0") even though both resolve to
    # the same physical device, and a stale False here silently reinstates the
    # slow copy path while a stale True would gather on the wrong device.
    self.obs_on_device = (
        self.obs.device == torch.empty(0, device=self.device).device)
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
    # Set by step_np(defer_record=True), consumed by flush_selfplay_record.
    self._pending_record = None
    # First row of the episode currently in progress, per env. Rows before it
    # belong to an episode that already ended inside this same batch, so they
    # must not be swept up by terminal attribution or aux back-fill.
    self._episode_start_row = np.zeros(self.num_envs, dtype=np.int64)
    # Deferred (row, env, target) terminal writes, applied as one vectorized
    # scatter in _learn_core. Writing them one scalar at a time would issue a
    # GPU kernel per (env, seat) per episode in the hot loop.
    self._closeout_writes = []
    # Per-own-decision potential shaping (post_step(phi=...)): the last
    # (row, phi) recorded for each (env, seat), and the deferred additive reward
    # writes that pair a seat's consecutive decisions.
    self._pending_phi = [{} for _ in range(self.num_envs)]
    self._shaping_adds = []

    # Auxiliary-head buffers (self-play/win mode only): per (step, env) target
    # vectors and availability masks for the aux tasks. Targets are back-filled
    # for every row of a seat the moment its episode closes (rows are
    # overwritten each batch, so rows 0..current-row of a seat all get the
    # same terminal-derived target).
    # Tie-aware realized-placement distributions, back-filled exactly where
    # aux targets are. A tie occupies multiple rank slots, so a one-hot label
    # would conflict with the tie-averaged scalar utility.
    self.rank_labels = torch.zeros(
        (self.steps_per_batch, self.num_envs, len(DEFAULT_RANK_UTILITY)),
        dtype=torch.float32).to(device)
    self.rank_label_mask = torch.zeros(
        (self.steps_per_batch, self.num_envs), dtype=torch.float32).to(device)
    self._extra_ranks = []

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

    self._aux_unsupported_warned = False

    # Per-update loss diagnostics (filled by _learn_core), so outer loops can
    # report value/policy/entropy/aux loss + KL/clip/explained-variance without
    # pulling them from a tensorboard writer.
    self.last_metrics = None

  def _resolve_obs_buffer_dtype(self, pref, device):
    """Storage dtype for the rollout obs buffer: 'auto' | a torch dtype name.

    'auto' picks float16 when --amp is on and training runs on a GPU. The
    rollout buffer is read in exactly one place -- the minibatch gather in
    _learn_core -- and every consumer of that minibatch runs inside the bf16
    autocast region, so the fp32 mantissa bits being stored are discarded by the
    very next op. Halving the buffer therefore costs precision the learn phase
    was already throwing away, and it doubles the env count that fits on the
    device (measured: 4,096 envs resident at 39.4 GB).

    float16 rather than bfloat16 on purpose: the observation tensor is one-hots
    and ratios in [0, 1], so what matters is mantissa (fp16 has 10 bits, bf16
    has 8), not bf16's wider exponent range.

    Two cases where 'auto' deliberately stays float32:
      * no --amp: the learn path is genuinely fp32, so 16-bit storage would
        reduce precision that is actually in use.
      * CPU training (the small-machine / smoke-test path): there is no VRAM to
        save, and the fp16 -> fp32 conversion on every gather is pure cost.
    Both can still be overridden explicitly.
    """
    if pref is not None and str(pref).lower() != "auto":
      dt = getattr(torch, str(pref), None)
      if not isinstance(dt, torch.dtype):
        raise ValueError(f"obs_buffer_dtype must be 'auto' or a torch dtype "
                         f"name (float32/float16/bfloat16), got {pref!r}")
      return dt
    on_gpu = torch.device(device).type == "cuda"
    return torch.float16 if (self.amp and on_gpu) else torch.float32

  def _resolve_obs_buffer_device(self, pref, device):
    """Device for the rollout obs buffer: 'auto' | 'cpu' | a device string.

    'auto' places the buffer on ``device`` only when it comfortably fits in
    currently-free memory there, so a big card gets the fast path and a small
    one keeps the CPU-streaming behaviour instead of OOMing at startup. The
    headroom factor leaves room for the network, optimizer state, activations
    of one minibatch and the other (small) rollout tensors; the buffer is by
    far the largest single allocation, so a coarse budget is enough.
    """
    if pref is not None and str(pref).lower() != "auto":
      return str(pref)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
      return "cpu"
    nbytes = (self.steps_per_batch * self.num_envs *
              int(np.prod(self.input_shape)) *
              self.obs_buffer_dtype.itemsize)
    free, _ = torch.cuda.mem_get_info(dev)
    # The buffer is counted once, not twice: _ObsRows removed the torch.cat that
    # used to materialize a second copy of the whole batch during learn. The
    # remaining headroom covers one minibatch of encoder activations (measured
    # 20.8 GB peak at 8,192 rows for Eclipse's spatial encoder), the network and
    # its optimizer state, and the small rollout tensors.
    if nbytes <= 0.5 * free:
      return dev
    print(f"PPO: rollout obs buffer needs {nbytes / 1e9:.1f} GB "
          f"({self.obs_buffer_dtype}), over the {0.5 * free / 1e9:.1f} GB budget "
          f"(50% of {free / 1e9:.1f} GB free) on {dev}. Keeping it on CPU and "
          f"streaming minibatches, which costs roughly 220 ms per minibatch. "
          f"Lower num_envs/num_steps, use obs_buffer_dtype=float16, or force "
          f"obs_buffer_device=cuda if you know the peak fits.")
    return "cpu"

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
        if self.legal_actions_mask.shape[0]:
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
            # matches the format stored by step_np. NOTE: index time_step[i] --
            # reading the leaked loop variable from the packing loop above meant
            # every env got the *last* env's timestep, which under
            # players="current" is None for any env whose acting seat differs, so
            # cols came out empty. An empty legal set gives a zero-length segment
            # in _segment_lse_entropy -> lse = -inf -> NaN entropy -> NaN loss.
            la = time_step[i].observations["legal_actions"][s]
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

  def flush_selfplay_record(self):
    """Runs the deferred ``_last_decision`` bookkeeping, if any is pending.

    Split out of ``step_np`` so the caller can start the env workers first and pay
    this ~17 ms/step (at 1,024 envs) against the ~27 ms the workers take, instead
    of after them. Idempotent and a no-op when nothing is pending, so a caller that
    does not defer needs no changes.

    Nothing here reads shared memory: ``obs_cpu`` and ``seats`` point into a
    ``_collect`` generation buffer, and ``mask_rows``/``mask_cols`` are freshly
    allocated by ``_legal_indices`` every step. That is what makes it safe to run
    concurrently with the workers.
    """
    pending = self._pending_record
    if pending is None:
      return
    self._pending_record = None
    (obs_cpu, seats, mask_rows, mask_cols, action, logprob, value) = pending
    action_np = action.detach().to(torch.int32).cpu().numpy()
    lpv = torch.stack((logprob.detach().reshape(-1),
                       value.detach().reshape(-1))).cpu().numpy()
    lens = np.bincount(np.asarray(mask_rows, dtype=np.int64),
                       minlength=self.num_envs)
    offsets = np.zeros(self.num_envs, dtype=np.int64)
    np.cumsum(lens[:-1], out=offsets[1:])
    seats_i = np.asarray(seats, dtype=np.int64)
    if self.league:
      trainable = (self.lineup[np.arange(self.num_envs), seats_i]
                   == self.train_pid)
    else:
      trainable = np.ones(self.num_envs, dtype=bool)
    acts_l = action_np.tolist()
    lp_l = lpv[0].tolist()
    v_l = lpv[1].tolist()
    lens_l = lens.tolist()
    off_l = offsets.tolist()
    seats_l = seats_i.tolist()
    for i in np.flatnonzero(trainable).tolist():
      o = off_l[i]
      self._last_decision[i][seats_l[i]] = (
          obs_cpu[i].copy(), mask_cols[o:o + lens_l[i]],
          acts_l[i], lp_l[i], v_l[i])

  def step_np(self, step_arrays, is_evaluation=False, defer_record=False):
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
        if self.legal_actions_mask.shape[0]:
          self.legal_actions_mask[row] = legal_actions_mask

      self.players[row] = torch.from_numpy(
          seats.astype(np.int64)).to(self.device)
      self.players_cpu[row] = torch.from_numpy(seats.astype(np.int64))
      trainable_row = torch.tensor(
          [self._acts_trainable(i, int(sv)) for i, sv in enumerate(seats)],
          dtype=torch.bool)
      self.trainable_cpu[row] = trainable_row
      self.trainable[row] = trainable_row.to(self.device)
      # Store the acting observations into the rollout buffer. When the buffer
      # is device-resident, reuse the `obs` tensor already uploaded above for
      # the forward pass -- a device-to-device copy -- instead of a second
      # host-side copy of the same 38 MB (at 256 envs) out of shared memory.
      # When the buffer is on CPU, copy from the numpy view as before.
      self.obs[row] = obs if self.obs_on_device else torch.from_numpy(obs_cpu)
      self.actions[row] = action
      self.logprobs[row] = logprob
      self.values[row] = value.flatten()

      self.legal_rows_packed[row] = mask_rows.astype(np.int64)
      self.legal_cols_packed[row] = mask_cols.astype(np.int64)

      if self.selfplay and defer_record:
        # Hand the bookkeeping to flush_selfplay_record so the caller can start the
        # env workers first and overlap it. Only references are stashed -- no work
        # and no copies happen here.
        self._pending_record = (obs_cpu, seats, mask_rows, mask_cols,
                                action, logprob, value)
        return action.detach().to(torch.int32).cpu().numpy()

      if self.selfplay:
        # This block was 47.5% of the `act` phase at 1,024 envs (19.06 of 40.10
        # ms/step, ~2.4 s of an 18.87 s update) and every term in it scales with
        # num_envs. What follows removes the per-env Python overhead; the obs
        # copies below are the remaining half and need the (row, env) reference
        # design to go away.
        #
        # TWO device->host transfers instead of three. action is integral while
        # logprob/value are float, so they cannot share one tensor, but the two
        # floats can be stacked -- and each separate .cpu() is its own transfer
        # and its own sync. action goes as int32, halving those bytes too.
        action_np = action.detach().to(torch.int32).cpu().numpy()
        lpv = torch.stack((logprob.detach().reshape(-1),
                           value.detach().reshape(-1))).cpu().numpy()
        # Per-env column offsets so each env's legal cols can be sliced out of
        # the packed buffers (no dense 11k mask build) for the closeout sample.
        lens = np.bincount(np.asarray(mask_rows, dtype=np.int64),
                           minlength=self.num_envs)
        offsets = np.zeros(self.num_envs, dtype=np.int64)
        np.cumsum(lens[:-1], out=offsets[1:])
        # Trainability in one vectorised compare rather than num_envs calls to
        # _acts_trainable, each of which indexed an object array.
        seats_i = np.asarray(seats, dtype=np.int64)
        if self.league:
          trainable = (self.lineup[np.arange(self.num_envs), seats_i]
                       == self.train_pid)
        else:
          trainable = np.ones(self.num_envs, dtype=bool)
        # One .tolist() each, rather than num_envs numpy scalar extractions plus
        # int()/float() conversions inside the loop.
        acts_l = action_np.tolist()
        lp_l = lpv[0].tolist()
        v_l = lpv[1].tolist()
        lens_l = lens.tolist()
        off_l = offsets.tolist()
        seats_l = seats_i.tolist()
        # mask_cols is ALREADY int64 and freshly allocated by
        # async_vector_env._legal_indices on every step, and its only consumer
        # copies it (`e[3].astype(np.int64)` in _packed_legal_with_extras), so a
        # slice view is safe to retain. The old per-env `.astype(np.int64)` was a
        # redundant copy -- num_envs small allocations per step for nothing.
        for i in np.flatnonzero(trainable).tolist():
          o = off_l[i]
          self._last_decision[i][seats_l[i]] = (
              obs_cpu[i].copy(), mask_cols[o:o + lens_l[i]],
              acts_l[i], lp_l[i], v_l[i])
        return action_np

      return action.detach().to(torch.int32).cpu().numpy()

  def _resolve_last_decision(self, env_idx, seat):
    """(obs, packed_cols, action, logprob, value) for a seat's last decision.

    Returns the committed CPU record stored per step by ``step``/``step_np``
    (obs copy + legal cols + action + logprob + value). None if the seat never
    had a recorded decision.
    """
    return self._last_decision[env_idx].get(seat)

  def _terminal_rank(self, per_agent_reward, seat):
    """Tie-aware soft target over the four realized placement slots."""
    return rank_distribution(per_agent_reward, seat)

  def _terminal_target(self, per_agent_reward, seat):
    """Scalar terminal target for `seat` given the per-agent payoff vector."""
    if self.value_mode == "win":
      return rank_utility(per_agent_reward, seat,
                          vp_beta=self.rank_vp_beta)
    return per_agent_reward[seat]

  def _aux_targets_for(self, per_agent_reward, terminal_obs=None,
                       acting_seat=None, direct_targets=None):
    """Per-seat aux-target matrix (num_players, num_aux), or None.

    Uses ``aux_target_fn`` if supplied, else defaults to the raw per-agent
    payoff (one-column aux = predict a seat's final payoff). ``terminal_obs`` /
    ``acting_seat`` are forwarded to the extractor so e.g. the VP-breakdown mode
    can read the per-category targets out of the terminal observation.
    """
    if not self.num_aux:
      return None
    if direct_targets is not None:
      arr = np.asarray(direct_targets, dtype=np.float32)
    elif self.aux_target_fn is None:
      arr = np.asarray(per_agent_reward, dtype=np.float32)
      if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif getattr(self.aux_target_fn, "needs_terminal_obs", False):
      # Breakdown-style extractor: without a terminal obs there is no real
      # target, so leave the rows unmasked instead of training toward zeros.
      if terminal_obs is None or acting_seat is None:
        return None
      arr = np.asarray(
          self.aux_target_fn(per_agent_reward, terminal_obs, acting_seat),
          dtype=np.float32)
    else:
      arr = np.asarray(self.aux_target_fn(per_agent_reward), dtype=np.float32)
    try:
      return arr.reshape(self.num_players, self.num_aux)
    except ValueError:
      return None

  def _attribute_terminal(self, env_idx, row, per_agent_reward, acting_seat,
                          aux_targets=None):
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
      ex_aux, ex_aux_mask = self._extras_aux(
          per_agent_reward, seat, direct_targets=aux_targets)
      self._extra_samples.append(
          (env_idx, seat, obs_, cols_, int(action_), logprob_, value_,
           target, ex_aux, ex_aux_mask))
      self._extra_ranks.append(self._terminal_rank(per_agent_reward, seat))

  def _record_phi(self, env_idx, seat, row, phi):
    """Pairs a seat's consecutive decisions into one telescoping shaped reward.

    Potential-based shaping needs ``gamma * phi(s') - phi(s)`` across the same
    transition the discount is applied to. In self-play GAE that transition is
    between a seat's *own* consecutive decisions (a seat's chain skips the other
    seats' rows and applies one gamma per own decision) -- not between
    consecutive env steps. Shaping across env steps therefore does not telescope,
    and comparing potentials of *different* seats (what --phi=learned does) is
    not a potential difference at all.

    So the delta is attributed to the seat's *previous* decision row, which is
    only known once the seat acts again -- hence the deferred additive write.

    A seat whose previous decision fell in an earlier rollout batch has no row
    left to credit, so that one delta per (env, seat) per batch is dropped.
    """
    prev = self._pending_phi[env_idx].get(seat)
    self._pending_phi[env_idx][seat] = (row, float(phi))
    if prev is None:
      return
    prev_row, prev_phi = prev
    if prev_row >= self._episode_start_row[env_idx]:
      self._shaping_adds.append(
          (prev_row, env_idx, self.gamma * float(phi) - prev_phi))

  def _apply_closeout_writes(self):
    """Flushes deferred shaping adds, then terminal rewards/dones.

    Order matters: a terminal row's reward is the true payoff, so the overwrite
    must land after any additive shaping on that row.
    """
    if self._shaping_adds:
      rows = np.fromiter((w[0] for w in self._shaping_adds), dtype=np.int64,
                         count=len(self._shaping_adds))
      envs = np.fromiter((w[1] for w in self._shaping_adds), dtype=np.int64,
                         count=len(self._shaping_adds))
      vals = np.fromiter((w[2] for w in self._shaping_adds), dtype=np.float32,
                         count=len(self._shaping_adds))
      self.rewards.index_put_(
          (torch.from_numpy(rows).to(self.device),
           torch.from_numpy(envs).to(self.device)),
          torch.from_numpy(vals).to(self.device),
          accumulate=True)
      self._shaping_adds = []
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

  def _backfill_rank_labels(self, env_idx, row, per_agent_reward):
    """Writes each seat's realized placement onto every row it acted on.

    Same rows as the aux back-fill, but kept independent so the distributional
    critic works with --aux_target_mode=none.
    """
    if not self.rank_ce_coef:
      return
    start = int(self._episode_start_row[env_idx])
    players_col = self.players_cpu[start:row + 1, env_idx].numpy()
    trainable_col = self.trainable_cpu[start:row + 1, env_idx].numpy()
    for seat in range(self.num_players):
      rows_seat = np.flatnonzero((players_col == seat) & trainable_col)
      if rows_seat.size == 0:
        continue
      idx = torch.from_numpy((rows_seat + start).astype(np.int64)).to(
          self.device)
      target = torch.from_numpy(
          self._terminal_rank(per_agent_reward, seat)).to(self.device)
      self.rank_labels[idx, env_idx] = target
      self.rank_label_mask[idx, env_idx] = 1.0

  def _backfill_aux(self, env_idx, row, per_agent_reward,
                    terminal_obs=None, acting_seat=None, direct_targets=None):
    """Fills aux targets for every stored row (<=row) of each seat in `env_idx`.

    Terminal payoff fixes the final target for all seats of a finished episode,
    so every row of a seat in this batch (which came from the same episode)
    inherits that seat's terminal-derived aux target. ``terminal_obs`` /
    ``acting_seat`` (the observation the terminal decision was taken from, and
    the seat that took it) are forwarded to a breakdown aux extractor.
    """
    if not self.num_aux:
      return
    tgt = self._aux_targets_for(
        per_agent_reward, terminal_obs, acting_seat, direct_targets)
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

  def _extras_aux(self, per_agent_reward, seat, terminal_obs=None,
                  acting_seat=None, direct_targets=None):
    """(aux_target, aux_mask) 1-D tensors for one extra sample's seat."""
    if not self.num_aux:
      return None, None
    tgt = self._aux_targets_for(
        per_agent_reward, terminal_obs, acting_seat, direct_targets)
    if tgt is None:
      return None, None
    return (torch.from_numpy(tgt[seat]).to(self.device),
            torch.ones((self.num_aux,), dtype=torch.float32).to(self.device))

  def post_step(self, reward, done, shaped_reward=None, phi=None,
                terminal_aux=None):
    """Stores rewards/dones for the action taken at the current batch step.

    Args:
      reward: list (one entry per environment) of per-player reward vectors, as
        returned by ``SyncVectorEnv.step`` (``ts.rewards``).
      done: list of booleans, one per environment.
      shaped_reward: optional list of floats, one per environment, holding the
        potential-based shaping delta for the acting seat's transition this
        step. Shaping is skipped for terminal transitions (the true payoff is
        used instead), which keeps the shaped reward telescope consistent.
      terminal_aux: optional exact terminal targets with shape
        ``(num_envs, num_players, num_aux)`` captured before environment reset.
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
        if phi is not None and not is_done:
          self._record_phi(i, seat, row, phi[i])
        if is_done:
          needs_direct = bool(
              self.num_aux and
              getattr(self.aux_target_fn, "needs_terminal_obs", False))
          direct = (terminal_aux[i]
                    if needs_direct and terminal_aux is not None else None)
          if needs_direct and direct is None:
            raise ValueError("terminal auxiliary targets were not captured")
          self._backfill_aux(i, row, rvec, direct_targets=direct)
          self._backfill_rank_labels(i, row, rvec)
          self._attribute_terminal(i, row, rvec, seat, direct)
          self._last_decision[i].clear()
          self._pending_phi[i].clear()
          self._episode_start_row[i] = row + 1
      self.rewards[row] = torch.from_numpy(rew_row).to(self.device)
      self.dones[row] = torch.from_numpy(done_row).to(self.device)
    else:
      self.rewards[row] = torch.tensor(reward).to(self.device).view(-1)
      self.dones[row] = torch.tensor(done).to(self.device).view(-1)

    self.total_steps_done += self.num_envs
    self.cur_batch_idx += 1

  def post_step_np(self, reward, done, shaped_reward=None, phi=None,
                   terminal_aux=None):
    """Array-native ``post_step``; identical semantics, numpy inputs.

    Args:
      reward: (num_envs, num_players) float32 numpy array of per-player
        rewards.
      done: (num_envs,) bool numpy array.
      shaped_reward: optional (num_envs,) float32 numpy array with the
        potential-based shaping delta for the acting seat's transition.
      terminal_aux: optional exact terminal targets with shape
        ``(num_envs, num_players, num_aux)`` captured before environment reset.
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
      if phi is not None:
        for i in range(self.num_envs):
          if not done_np[i]:
            self._record_phi(i, int(seats[i]), row, phi[i])
      # Terminal-closeout bookkeeping only for envs that actually finished.
      for i in done_idx:
        rvec = reward_np[i]
        seat = int(seats[i])
        needs_direct = bool(
            self.num_aux and
            getattr(self.aux_target_fn, "needs_terminal_obs", False))
        direct = (terminal_aux[i]
                  if needs_direct and terminal_aux is not None else None)
        if needs_direct and direct is None:
          raise ValueError("terminal auxiliary targets were not captured")
        self._backfill_aux(i, row, rvec, direct_targets=direct)
        self._backfill_rank_labels(i, row, rvec)
        self._attribute_terminal(int(i), row, rvec, seat, direct)
        self._last_decision[i].clear()
        self._pending_phi[i].clear()
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

  def _compute_returns(self, next_value_per_env, next_seats=None):
    """Computes returns (and GAE advantages) for every stored transition.

    For the single-agent case this is the standard fixed-timeline GAE. For
    self-play, advantage estimation is done per (env, seat) subsequence, so a
    seat's bootstrapping chains only through that seat's own decision values
    (different seats observe different views and have different objectives).
    At a batch boundary only the next acting seat has a valid next-state value.
    Other seats' final rows are truncated with zero advantage instead of being
    bootstrapped from another player's observation; earlier rows still
    bootstrap through that seat's final stored value.
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
    if next_seats is None:
      raise ValueError("self-play returns require next_seats")
    next_seats_np = np.asarray(next_seats, dtype=np.int64)
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
            if not dones_np[idx, i] and s != next_seats_np[i]:
              # No observation for this seat exists at the rollout boundary.
              # Make its boundary transition neutral rather than using the
              # next actor's value, which belongs to a different objective.
              rets[idx, i] = values_np[idx, i]
              lastgaelam = 0.0
              continue
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
    self._learn_core(next_obs, seats)

  def learn_np(self, next_obs_np, next_seats):
    """Array-native :meth:`learn` for the async vector-env path.

    Identical semantics to :meth:`learn` but takes the next-step observations
    for the acting seats directly as a (num_envs, obs_size) float32 numpy
    array, avoiding per-env ``TimeStep`` construction.
    """
    next_obs = torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32)
                               ).to(self.device)
    self._learn_core(next_obs, next_seats)

  def _sparse_supported(self):
    """True when the network exposes shared features + a Linear actor head.

    Structure (see EclipsePPOAgent): ``actor = Sequential(shared,
    Linear(width, num_actions))`` so logits = features @ W^T + b with
    ``features = shared(x)`` of small width. This lets learn compute logits /
    log_prob / entropy only over the ~legal actions instead of all 11117.
    """
    net = self.network
    if not (hasattr(net, "shared") and isinstance(net.actor, nn.Sequential)):
      return False
    head = net.actor[-1]
    return (getattr(head, "out_features", None) == self.num_actions
            and (isinstance(head, nn.Linear) or hasattr(head, "rows_for")))

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

  def _sparse_minibatch(self, features, rows, cols, actions_mb, head):
    """Per-minibatch masked log-prob + entropy from sparse legal entries.

    Args:
      features: (B_mb, H) shared features of the minibatch.
      rows: (M_mb,) local sample index (into the minibatch) per legal entry.
      cols: (M_mb,) legal action id per entry.
      actions_mb: (B_mb,) chosen action per sample.
      head: actor head module (``net.actor[-1]``); carries its own bias.

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
    logits_e = head_logits(head, features, rows_gpu, cols_gpu)  # (M,)
    # segment reductions per block (entries of a sample are contiguous)
    b_mb = features.shape[0]
    m = torch.full((b_mb,), float("-inf"), device=self.device)
    m.scatter_reduce_(0, rows_gpu, logits_e, reduce="amax", include_self=False)
    e = (logits_e - m[rows_gpu]).exp()
    s = torch.zeros(b_mb, device=self.device)
    s.scatter_add_(0, rows_gpu, e)
    num = torch.zeros(b_mb, device=self.device)
    num.scatter_add_(0, rows_gpu, e * logits_e)
    # See _segment_lse_entropy: guard the zero-legal-entry row against NaN.
    empty = s == 0
    safe_s = torch.where(empty, torch.ones_like(s), s)
    lse = torch.where(empty, torch.zeros_like(m), m + safe_s.log())
    entropy = torch.where(empty, torch.zeros_like(m), lse - num / safe_s)
    # log_prob of the chosen action: each sample's own feature row paired with
    # its own chosen action, so rows == arange(b_mb) here.
    own_rows = torch.arange(b_mb, device=self.device)
    logit_a = head_logits(head, features, own_rows, actions_mb)
    logprob = logit_a - lse
    return logprob, entropy

  def _pack_logits(self, features, rows, cols, head):
    """Logits at the legal entries. ``rows`` are torch sample indices (a
    row's entries contiguous); ``cols`` the legal action ids; ``features``
    the shared features for the batch. Returns (M,) logits."""
    return head_logits(head, features, rows, cols)

  def _segment_lse_entropy(self, logits_e, rows, batch_size):
    """Per-row logsumexp + entropy of the (masked) distribution given the
    (M,) logits of legal entries grouped contiguously by row."""
    m = torch.full((batch_size,), float("-inf"), device=self.device)
    m.scatter_reduce_(0, rows, logits_e, reduce="amax", include_self=False)
    e = (logits_e - m[rows]).exp()
    s = torch.zeros(batch_size, device=self.device)
    s.scatter_add_(0, rows, e)
    num = torch.zeros(batch_size, device=self.device)
    num.scatter_add_(0, rows, e * logits_e)
    # A row with no legal entries leaves m = -inf and s = 0, giving
    # lse = -inf and entropy = -inf - 0/0 = NaN, which poisons the whole loss.
    # It should not happen (a decision node always has a legal action) but a
    # bookkeeping slip upstream has produced it twice, and silently, so the
    # degenerate row contributes zero rather than NaN.
    empty = s == 0
    safe_s = torch.where(empty, torch.ones_like(s), s)
    lse = torch.where(empty, torch.zeros_like(m), m + safe_s.log())
    entropy = torch.where(empty, torch.zeros_like(m), lse - num / safe_s)
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
    # Sentinel num_actions marks a row with an empty legal segment or NaN logits
    # (the bookkeeping slip documented at _segment_lse_entropy). Clamped below to
    # a valid index so the gather can never OOB; healthy rows keep a real col.
    chosen = torch.full((batch_size,), self.num_actions, dtype=torch.long,
                        device=self.device)
    mrows = rows[maxima]
    mcols = cols[maxima]
    if mrows.size(0):
      # 'amin' over the (measure-zero) tied maxima columns; include_self=False
      # so healthy rows get exactly one valid col.
      chosen.scatter_reduce_(0, mrows, mcols, reduce="amin", include_self=False)
    # Map any surviving sentinel to a valid action id. A degenerate row's logprob
    # is discarded by the same zero-guard _segment_lse_entropy applies to lse, so
    # this keeps the downstream gather in bounds without changing healthy rows at
    # all (their chosen is already < num_actions).
    return chosen.clamp(max=self.num_actions - 1)

  def _log_prob_chosen(self, features, chosen, lse, head):
    """(B,) log_prob of ``chosen`` under the masked distribution with known
    per-row logsumexp ``lse``."""
    own_rows = torch.arange(features.shape[0], device=self.device)
    logit_a = head_logits(head, features, own_rows, chosen)
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
      pids = self.lineup[np.arange(batch), np.asarray(seats, dtype=np.int64)]
      unique_pids = np.unique(pids)
      groups = [np.flatnonzero(pids == p) for p in unique_pids]
      nets = [self.networks[p] for p in unique_pids]
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
      logits_e = self._pack_logits(features, rows, cols, net.actor[-1])
      chosen = self._gumbel_sample(logits_e, rows, cols, n)
      lse, ent = self._segment_lse_entropy(logits_e, rows, n)
      lp = self._log_prob_chosen(features, chosen, lse, net.actor[-1])
      if getattr(net, "value_from_actor_features", False):
        # Critic shares the actor trunk: features are already computed, so read
        # the value from them instead of re-running the whole encoder via
        # value_from_obs (a ~2x encoder cost on the hot act path).
        v = net.value_from_features(features).view(-1)
      elif hasattr(net, "value_from_obs"):
        v = net.value_from_obs(obs[idx]).view(-1)
      elif hasattr(net, "value_from_features"):
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

  def _learn_core(self, next_obs, next_seats):
    # Terminal attribution for non-acting seats, deferred during the rollout to
    # keep it off the hot loop. Must land before returns are computed.
    self._apply_closeout_writes()

    # bootstrap value if not done
    with torch.no_grad():
      next_value_per_env = self.get_value(next_obs).reshape(-1)

    returns = self._compute_returns(next_value_per_env, next_seats)

    # flatten the batch
    b_legal_actions_mask = self.legal_actions_mask.reshape(
        (-1, self.num_actions))
    # A VIEW of the rollout buffer -- never copied wholesale. Logical batch rows
    # are addressed through the `_ObsRows` built below, which indexes this view
    # for ids < len(main) and the closeout extras beyond that. Filtering or
    # concatenating this tensor directly costs a full buffer copy.
    b_obs_main = self.obs.reshape((-1,) + self.input_shape)
    ex_obs = None
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
    b_ranks = self.rank_labels.reshape(-1, len(DEFAULT_RANK_UTILITY))
    b_rank_mask = self.rank_label_mask.reshape(-1)

    use_sparse = self._sparse_supported()

    # Append independent terminal-closeout samples (self-play only). Each is a
    # done transition: returns target is that seat's terminal payoff directly,
    # with no bootstrapping.
    n_extra = len(self._extra_samples)
    if n_extra:
      # Extras are collected as numpy rows; match the rollout buffer's device and
      # dtype so _ObsRows can serve both halves without a per-row conversion.
      ex_obs = torch.as_tensor(
          np.array([e[2] for e in self._extra_samples]),
          dtype=torch.float32).to(device=b_obs_main.device,
                                  dtype=b_obs_main.dtype)
      ex_actions = torch.tensor([e[4]
                                 for e in self._extra_samples]).to(self.device)
      ex_logprobs = torch.tensor([e[5]
                                  for e in self._extra_samples]).to(self.device)
      ex_values = torch.tensor([e[6]
                                for e in self._extra_samples]).to(self.device)
      ex_targets = torch.tensor([e[7]
                                 for e in self._extra_samples]).to(self.device)
      # NOT torch.cat for the observations -- see _ObsRows. b_obs_main is a view
      # of the whole rollout buffer, so concatenating a handful of closeout rows
      # onto it copied the entire buffer (4.9 GB at 256 envs x 128 steps, 39.4 GB
      # at 2,048) on EVERY learn call. The other extras are per-row scalars, so
      # concatenating them is cheap and stays as-is.
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
      if self._extra_ranks:
        b_ranks = torch.cat([b_ranks, torch.tensor(
            np.asarray(self._extra_ranks), dtype=torch.float32,
            device=self.device)])
        b_rank_mask = torch.cat([b_rank_mask, torch.ones(
            len(self._extra_ranks), device=self.device)])
      else:
        b_ranks = torch.cat([b_ranks, torch.zeros(
            (n_extra, len(DEFAULT_RANK_UTILITY)), dtype=torch.float32,
            device=self.device)])
        b_rank_mask = torch.cat([b_rank_mask, torch.zeros(
            n_extra, device=self.device)])

    # Rollout rows and closeout extras as one row-addressable batch, without
    # materializing their concatenation. Slicing the int64 row-id vector (260 KB)
    # is what replaces copying gigabytes of observations. Minibatches come out as
    # float32 no matter how the buffer is stored, so a 16-bit buffer changes only
    # the rounding of the stored observations and nothing downstream of the
    # gather.
    b_obs = _ObsRows(b_obs_main, ex_obs, out_dtype=torch.float32)

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
      # Filter the obs INDEX, never the obs. Two bugs lived on this line: it
      # indexed a CPU rollout buffer with this CUDA mask ("indices should be
      # either on cpu or on the same device"), which broke --league outright from
      # the commit that moved the buffer off the GPU; and even when it worked it
      # copied the whole buffer a second time. `_ObsRows.select` moves the mask to
      # wherever the row ids live, so it is correct for either buffer device.
      b_obs = b_obs.select(keep)
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
      b_ranks = b_ranks[keep]
      b_rank_mask = b_rank_mask[keep]
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
    self._extra_ranks = []

    # Keep the batch a whole number of minibatches: with a trailing remainder
    # the final minibatch can be a single sample, whose std() is NaN and
    # poisons advantage normalization. The dropped tail is only ever a small
    # number of terminal closeout samples.
    batch_size = (len(b_obs) // self.num_minibatches) * self.num_minibatches
    if batch_size != len(b_obs):
      b_obs = b_obs.select(slice(None, batch_size))
      b_legal_actions_mask = b_legal_actions_mask[:batch_size]
      b_actions = b_actions[:batch_size]
      b_logprobs = b_logprobs[:batch_size]
      b_values = b_values[:batch_size]
      b_returns = b_returns[:batch_size]
      b_advantages = b_advantages[:batch_size]
      if self.num_aux:
        b_aux = b_aux[:batch_size]
        b_aux_mask = b_aux_mask[:batch_size]
      b_ranks = b_ranks[:batch_size]
      b_rank_mask = b_rank_mask[:batch_size]
    minibatch_size = max(1, batch_size // self.num_minibatches)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    # --amp: bf16 autocast around this minibatch loop's forward+loss only. The
    # rollout/act path (self.step/step_np) is deliberately never autocast --
    # it's ~128 rows/step (not the bottleneck) and its fp32 log-probs are what
    # b_logprobs below was recorded with, so mixing precisions there would put
    # the PPO ratio between two different numeric paths from the start.
    # torch.device(...).type normalizes self.device, which callers pass as
    # either a string ("cpu") or a torch.device.
    amp_device_type = torch.device(self.device).type
    # Per-update accumulators: kl / old_kl / entropy are computed inside the
    # minibatch loop; without these, last_metrics fell through with the FINAL
    # minibatch's values, so controllers and TB saw noise instead of the mean.
    _mb_count = 0
    _kl_acc = torch.zeros((), device=self.device)
    _old_kl_acc = torch.zeros((), device=self.device)
    _entropy_acc = torch.zeros((), device=self.device)
    _clipfrac_acc = torch.zeros((), device=self.device)
    for _ in range(self.update_epochs):
      np.random.shuffle(b_inds)
      for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]
        # Gather this minibatch's observations. A device-resident buffer makes
        # this a pure on-device gather (~2 ms at 8192x37596); a host buffer falls
        # back to a host gather plus an H2D copy, ~100x more per minibatch but
        # bounded in VRAM (which is what decoupled the env count from a 12 GiB
        # card's capacity). _ObsRows picks the right one from where `main` lives.
        mb_obs = b_obs.minibatch(mb_inds, self.device)

        with torch.autocast(device_type=amp_device_type, dtype=torch.bfloat16,
                            enabled=self.amp):
          if use_sparse:
            # Slice the packed legal entries for this minibatch (vectorized via
            # the per-sample cumulative offsets).
            features = self.network.shared(mb_obs)
            # A network with its own critic trunk must not have its value read off
            # the *actor* features; value_from_obs runs the right trunk. When the
            # critic shares the actor trunk, reuse the already-computed features.
            if getattr(self.network, "value_from_actor_features", False):
              newvalue = self._value_from_features(features)
            elif hasattr(self.network, "value_from_obs"):
              newvalue = self.network.value_from_obs(mb_obs)
            else:
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
                  b_actions.long()[mb_inds], self.network.actor[-1])
            else:
              # Degenerate empty minibatch: uniform, zero-entropy losses.
              b = mb_obs.shape[0]
              logprob = torch.zeros(b, device=self.device)
              entropy = torch.zeros(b, device=self.device)
          else:
            _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(
                mb_obs,
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
            _kl_acc += approx_kl.detach()
            _old_kl_acc += old_approx_kl.detach()
            _mb_count += 1
            _clipfrac_acc += (
                (ratio - 1.0).abs() > self.clip_coef).float().mean()

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
          _entropy_acc += entropy_loss.detach()
          loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_coef

          # Auxiliary-head loss: supervise the trunk's auxiliary predictors
          # (e.g. "final total VP") on the terminal-derived targets back-filled
          # into this batch. Only rows whose episode closed get a target; others
          # are masked out, so the loss is over the available fraction.
          aux_loss = torch.zeros((), device=self.device)
          aux_hit = 0
          if self.num_aux and features is None:
            # Aux heads need the shared trunk features, which only the sparse path
            # produces. Silently training no aux head while aux_coef > 0 hid a dead
            # head for an entire 408M-step run; say so once.
            if not self._aux_unsupported_warned:
              self._aux_unsupported_warned = True
              print("WARNING: aux_coef>0 but this network has no shared-feature "
                    "path (needs `shared` + `get_aux`); aux heads are NOT being "
                    "trained.")
          if self.num_aux and features is not None:
            # aux_from_obs runs the *critic* trunk, which is only a distinct
            # network when separate_critic is set. With a shared trunk it
            # recomputes exactly the `features` already in hand -- a second full
            # encoder forward *and backward* per minibatch, i.e. ~2x the whole
            # learn-phase network cost for a bit-identical result. The act path
            # got this treatment in the value head (value_from_actor_features);
            # the learn path's aux heads did not.
            if (not getattr(self.network, "value_from_actor_features", False)
                and hasattr(self.network, "aux_from_obs")):
              pred = self.network.aux_from_obs(mb_obs)
            else:
              pred = self.network.get_aux(features)
            tgt = b_aux[mb_inds]
            msk = b_aux_mask[mb_inds]
            head_losses = []
            head_present = []
            for k, name in enumerate(self.aux_tasks):
              p = pred[name]
              if p.dim() > 1 and p.size(1) == 1:
                p = p.view(-1)
              m = msk[:, k].float()
              denom = m.sum()
              head_losses.append(
                  (((p - tgt[:, k])**2) * m).sum() / denom.clamp_min(1.0))
              head_present.append((denom > 0).float())
            if head_losses:
              # Keep aux_coef independent of how many related targets a mode
              # exposes. Summing made nine-way VP breakdown supervision roughly
              # nine times stronger than a single target at the same setting.
              present = torch.stack(head_present)
              aux_loss = ((torch.stack(head_losses) * present).sum() /
                          present.sum().clamp_min(1.0))
              aux_hit = len(head_losses)
              loss = loss + self.aux_coef * aux_loss

          # Distributional critic: cross-entropy on the realized placement.
          rank_ce = torch.zeros((), device=self.device)
          if self.rank_ce_coef and hasattr(self.network, "rank_logits_from_obs"):
            msk = b_rank_mask[mb_inds]
            # Same shared-trunk recompute as the aux heads above: when the
            # critic shares the actor trunk, the rank logits are just the
            # critic's tail applied to `features`. `features is not None` is
            # load-bearing: the dense path leaves it None, and only the sparse
            # path produces trunk features to reuse.
            if (features is not None
                and getattr(self.network, "value_from_actor_features", False)
                and hasattr(self.network, "rank_logits_from_features")):
              logits_r = self.network.rank_logits_from_features(features)
            else:
              logits_r = self.network.rank_logits_from_obs(mb_obs)
            per_row = -(b_ranks[mb_inds] * nn.functional.log_softmax(
                logits_r, dim=-1)).sum(dim=-1)
            rank_ce = (per_row * msk).sum() / msk.sum().clamp_min(1.0)
            loss = loss + self.rank_ce_coef * rank_ce

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

    # Share of the total loss magnitude contributed by the auxiliary term.
    #
    # Caveat on interpretation: pg_loss is the *clipped surrogate mean*, which
    # sits near zero at ratio ~ 1 by construction, so this ratio is only a proxy
    # for the thing that matters (the gradient-norm share). Measured on Eclipse
    # at aux_coef=0.1 the two agreed closely -- loss share 0.76-0.85 against a
    # gradient-norm share of 82% aux / 11% policy / 6% value -- so it is a useful
    # early warning, but confirm with an explicit gradient measurement before
    # concluding anything from it. Note also that the mechanism is direct
    # gradient dominance on the shared trunk, not grad-norm clipping: the
    # combined norm stayed under max_grad_norm throughout.
    # Anything sustained above ~0.5 here wants attention.
    # Fraction of value targets the critic cannot represent. A bounded value
    # head (e.g. an expected-rank-utility read-out, hard-limited to
    # [-0.5, 1.0]) silently caps explained variance once shaping pushes returns
    # outside its range, and every advantage built on it is then wrong.
    out_of_band = 0.0
    bounds = getattr(self.network, "value_bounds", None)
    if callable(bounds):
      lo, hi = bounds()
      out_of_band = float(
          ((b_returns < lo) | (b_returns > hi)).float().mean().detach())

    _pg = abs(float(pg_loss.detach()))
    _vf = self.value_coef * abs(float(v_loss.detach()))
    _aux = self.aux_coef * abs(float(aux_loss.detach())) if aux_hit else 0.0
    aux_share = _aux / (_pg + _vf + _aux + 1e-12)

    # Per-update diagnostics for the outer loop (tqdm progress / console).
    # _mb_count >= 1 whenever batch_size > 0, so it is safe as the divisor.
    mean_kl, mean_old_kl, mean_entropy, mean_clipfrac = torch.stack([
        _kl_acc, _old_kl_acc, _entropy_acc, _clipfrac_acc
    ]).div(_mb_count).cpu().tolist()
    self.last_metrics = {
        "policy_loss": float(pg_loss.detach()),
        "value_loss": float(v_loss.detach()),
        "entropy": mean_entropy,
        "aux_loss": float(aux_loss.detach()),
        "aux_share": aux_share,
        "returns_out_of_band": out_of_band,
        "rank_ce": float(rank_ce.detach()),
        "clipfrac": mean_clipfrac,
        "old_kl": mean_old_kl,
        "kl": mean_kl,
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
        self.writer.add_scalar("losses/aux_share", aux_share,
                               self.total_steps_done)
      self.writer.add_scalar("losses/returns_out_of_band", out_of_band,
                             self.total_steps_done)
      self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/approx_kl", approx_kl.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/clipfrac", mean_clipfrac,
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
    # now starts at row 0 again, and any pending potential refers to a row that
    # no longer exists.
    self._episode_start_row[:] = 0
    for pending in self._pending_phi:
      pending.clear()
    self.rank_label_mask.zero_()

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

  def set_learning_rate(self, learning_rate):
    """Sets the base LR that annealing decays from, and applies it now.

    Callers that override the LR after construction (exploiter mode) must go
    through this: ``anneal_learning_rate`` recomputes from ``self.learning_rate``,
    so writing ``param_groups[0]["lr"]`` directly was silently reverted by the
    first anneal call, and the override never took effect for the run.
    """
    self.learning_rate = float(learning_rate)
    for group in self.optimizer.param_groups:
      group["lr"] = self.learning_rate

  def anneal_learning_rate(self, update, num_total_updates):
    # Annealing the rate
    frac = 1.0 - (update / num_total_updates)
    if frac <= 0:
      raise ValueError("Annealing learning rate to <= 0")
    lrnow = frac * self.learning_rate
    for group in self.optimizer.param_groups:
      group["lr"] = lrnow

  def kl_step_lr(self, approx_kl, target, tau, lr_min, lr_max):
    """Closed-loop LR: react to the realized KL every step.

    A fixed anneal-to-zero shuts off learning past convergence (Item 5 showed
    post-plateau updates buy nothing) and, on resume, a fresh loop restarted the
    anneal at the base rate. Instead, keep exploration/update strength where
    the policy can actually absorb it: if the per-update KL is above the target,
    scale LR down (protect against over-large updates); if below, ease it back
    up (recover speed that was cut prematurely). Uses an EMA so a single noisy
    KL reading does not swing the LR. All mutation goes through
    ``set_learning_rate`` so the base ``self.learning_rate`` stays aligned with
    the optimizer's param groups (the same requirement ``anneal_learning_rate``
    has). Returns the applied multiplier (1.0 = unchanged).
    """
    if target is None or target <= 0:
      return 1.0
    decay = getattr(self, "_kl_ema", None)
    if decay is None:
      decay = approx_kl
    decay = tau * approx_kl + (1.0 - tau) * decay
    self._kl_ema = float(decay)
    # Proportional error; keeps the step size dimensionless across targets.
    err = (decay - target) / target
    # Logistic-style update clamped to keep a single step from halving/tripling
    # the rate. err<0 (KL under target) raises LR, err>0 lowers it.
    mult = float(np.exp(-np.clip(err, -1.0, 1.0)))
    new_lr = float(np.clip(self.learning_rate * mult, lr_min, lr_max))
    self.set_learning_rate(new_lr)
    return mult

  def entropy_band_step(self, entropy, lo, hi, step, every):
    """Entropy-band controller for ``ent_coef``.

    Maintains an EMA of the realized entropy and nudges ``ent_coef`` up when the
    band is too low (risk of premature determinism / policy collapse) and down
    when too high (paying exploration noise with nothing to show). Rate-limited
    to fire ``every`` updates so it does not react to per-update noise, giving a
    far longer time constant than the KL/LR loop. Returns the new ``ent_coef``.
    """
    if lo is None or hi is None or lo >= hi:
      return self.entropy_coef
    decay = getattr(self, "_ent_ema", None)
    if decay is None:
      decay = entropy
    decay = 0.9 * entropy + 0.1 * decay
    self._ent_ema = float(decay)
    count = getattr(self, "_ent_control_count", 0) + 1
    self._ent_control_count = count
    if count % every != 0:
      return self.entropy_coef
    if decay < lo:
      self.entropy_coef = self.entropy_coef * (1.0 + step)
    elif decay > hi:
      self.entropy_coef = self.entropy_coef * (1.0 - step)
    self.entropy_coef = float(np.clip(self.entropy_coef, 1e-6, 1e6))
    return self.entropy_coef
