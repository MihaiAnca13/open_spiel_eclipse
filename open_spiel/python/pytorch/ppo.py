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


def legal_actions_to_mask(legal_actions_list, num_actions):
  """Converts a list of legal actions to a mask.

  The mask has size num actions with a 1 in a legal positions.

  Args:
    legal_actions_list: the list of legal actions
    num_actions: number of actions (width of mask)

  Returns:
    legal actions mask.
  """
  legal_actions_mask = torch.zeros((len(legal_actions_list), num_actions),
                                   dtype=torch.bool)
  for i, legal_actions in enumerate(legal_actions_list):
    legal_actions_mask[i, legal_actions] = 1
  return legal_actions_mask


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
    self._extra_samples = []
    self._last_decision = [{} for _ in range(self.num_envs)]

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

  def _gather_obs(self, time_step, seats):
    return torch.Tensor(
        np.array([
            np.reshape(ts.observations["info_state"][seats[i]], self.input_shape)
            for i, ts in enumerate(time_step)
        ])).to(self.device)

  def _gather_legal_actions_mask(self, time_step, seats):
    return legal_actions_to_mask([
        ts.observations["legal_actions"][seats[i]]
        for i, ts in enumerate(time_step)
    ], self.num_actions).to(self.device)

  def step(self, time_step, is_evaluation=False):
    seats = self._current_seats(time_step)
    if is_evaluation:
      with torch.no_grad():
        legal_actions_mask = self._gather_legal_actions_mask(time_step, seats)
        obs = self._gather_obs(time_step, seats)
        action, _, _, value, probs = self.get_action_and_value(
            obs, legal_actions_mask=legal_actions_mask)
        return [
            StepOutput(action=a.item(), probs=p)
            for (a, p) in zip(action, probs)
        ]
    else:
      with torch.no_grad():
        # act
        obs = self._gather_obs(time_step, seats)
        legal_actions_mask = self._gather_legal_actions_mask(time_step, seats)
        action, logprob, _, value, probs = self.get_action_and_value(
            obs, legal_actions_mask=legal_actions_mask)

        # store
        row = self.cur_batch_idx
        self.players[row] = torch.tensor(seats, dtype=torch.long).to(
            self.device)
        self.legal_actions_mask[row] = legal_actions_mask
        self.obs[row] = obs
        self.actions[row] = action
        self.logprobs[row] = logprob
        self.values[row] = value.flatten()

        if self.selfplay:
          obs_cpu = obs.detach().cpu().numpy()
          mask_cpu = legal_actions_mask.detach().cpu().numpy()
          for i, s in enumerate(seats):
            self._last_decision[i][s] = (obs_cpu[i].copy(), mask_cpu[i].copy(),
                                         int(action[i].item()),
                                         float(logprob[i].item()),
                                         float(value[i].item()))

        agent_output = [
            StepOutput(action=a.item(), probs=p)
            for (a, p) in zip(action, probs)
        ]
        return agent_output

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
      seats = self.players[row].tolist()
      for i in range(self.num_envs):
        rvec = torch.tensor(reward[i], dtype=torch.float).to(self.device)
        seat = seats[i]
        is_done = bool(done[i])
        shaped = 0.0 if shaped_reward is None else shaped_reward[i]
        self.rewards[row, i] = rvec[seat].item() + (0.0 if is_done else shaped)
        self.dones[row, i] = 1.0 if is_done else 0.0
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
                 rvec[s].item()))
          self._last_decision[i].clear()
    else:
      self.rewards[row] = torch.tensor(reward).to(self.device).view(-1)
      self.dones[row] = torch.tensor(done).to(self.device).view(-1)

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
    advantages = torch.zeros_like(self.rewards).to(self.device)
    with torch.no_grad():
      for i in range(self.num_envs):
        next_value = next_value_per_env[i]
        for s in range(self.num_players):
          rows = (self.players[:, i] == s).nonzero(as_tuple=True)[0]
          if len(rows) == 0:
            continue
          lastgaelam = 0.0
          for k in reversed(range(len(rows))):
            idx = int(rows[k])
            if k == len(rows) - 1:
              nextvalues = next_value
            else:
              nextvalues = self.values[int(rows[k + 1]), i]
            nextnonterminal = 1.0 - self.dones[idx, i]
            delta = (
                self.rewards[idx, i] + self.gamma * nextvalues * nextnonterminal -
                self.values[idx, i])
            lastgaelam = (
                delta + self.gamma * self.gae_lambda * nextnonterminal *
                lastgaelam)
            advantages[idx, i] = lastgaelam
            returns[idx, i] = lastgaelam + self.values[idx, i]
    return returns

  def learn(self, time_step):
    seats = self._current_seats(time_step)
    next_obs = self._gather_obs(time_step, seats)

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

        _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(
            b_obs[mb_inds],
            legal_actions_mask=b_legal_actions_mask[mb_inds],
            action=b_actions.long()[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
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
