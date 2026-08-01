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

"""Shared-policy PPO self-play for Eclipse with potential-based reward shaping.

This is the Stage 0.5/1 training path (see ppo_eclipse_plan.md). A single
shared network acts for whichever seat is to move in each of `num_envs` parallel
games (the N-player self-play generalization of open_spiel.python.pytorch.ppo).

Reward shaping (optional, default on) uses the banked-VP potential that the game
already writes into the observation tensor's "score" slots:

  phi(s)  = my-seat banked VP (current VP if the game ended right now, /200)
  shaped  = gamma * phi(s') - phi(s), added to the acting seat's transition

Shaping is skipped on the terminal transition (true payoff is used), keeping
the potential-based telescope consistent. This addresses the sparse terminal
reward problem without access to expert/human demonstration data.
"""

from datetime import datetime
import os
import random
import time
from absl import app
from absl import flags
import numpy as np
import torch
from torch import nn

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import layer_init
from open_spiel.python.vector_env import SyncVectorEnv

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  SummaryWriter = None


class NullWriter(object):
  """Fallback no-op writer when tensorboard is unavailable."""

  def add_text(self, *args, **kwargs):
    pass

  def add_scalar(self, *args, **kwargs):
    pass

  def close(self):
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "eclipse(players=4)", "Name of the game.")
flags.DEFINE_integer("seed", 1, "Seed of the experiment.")
flags.DEFINE_bool("cuda", True, "If True, cuda will be enabled by default.")
flags.DEFINE_bool("torch_deterministic", True, "Deterministic torch.")
flags.DEFINE_string("track", None, "Experiment tracking run id.")
flags.DEFINE_string("run_dir", "runs", "Root dir for tensorboard runs.")

flags.DEFINE_integer("num_envs", 8, "Number of parallel game environments.")
flags.DEFINE_integer("num_steps", 128, "Rollout steps per update per env.")
flags.DEFINE_integer("total_timesteps", 100_000, "Total environment steps.")
flags.DEFINE_integer("eval_every", 10, "Log every N updates.")

flags.DEFINE_bool("shaping", True,
                  "Potential-based shaping from the obs 'score' slot.")
flags.DEFINE_enum("phi", "soft", ["banked", "soft", "none"],
                  "Potential definition. 'banked' = current VP if the game "
                  "ended now. 'soft' = banked plus in-progress presence terms "
                  "(colony ships, disks on sectors, orbitals/monoliths, "
                  "ambassadors) so the shaped reward is non-zero even before "
                  "any VP is banked. 'none' disables shaping.")
flags.DEFINE_float("phi_w_colony", 0.5,
                   "Soft-Phi weight per colony ship (in VP-equivalent units).")
flags.DEFINE_float("phi_w_disk", 1.0,
                   "Soft-Phi weight per influence disk committed to sectors.")
flags.DEFINE_float("phi_w_structure", 1.0,
                   "Soft-Phi weight per orbital/monolith built.")
flags.DEFINE_float("phi_w_ambassador", 1.0,
                   "Soft-Phi weight per ambassador tile held.")
flags.DEFINE_float("gamma", 0.99, "Discount factor (used for shaping too).")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda.")
flags.DEFINE_bool("gae", True, "Use GAE.")
flags.DEFINE_float("learning_rate", 2.5e-4, "Learning rate.")
flags.DEFINE_integer("num_minibatches", 4, "Number of minibatches.")
flags.DEFINE_integer("update_epochs", 4, "Number of updates epochs.")
flags.DEFINE_bool("norm_adv", True, "Normalize advantages.")
flags.DEFINE_float("clip_coef", 0.2, "Surrogate clipping coefficient.")
flags.DEFINE_bool("clip_vloss", True, "Clipped value loss.")
flags.DEFINE_float("ent_coef", 0.01, "Entropy coefficient.")
flags.DEFINE_float("vf_coef", 0.5, "Value coefficient.")
flags.DEFINE_float("max_grad_norm", 0.5, "Max gradient norm.")
flags.DEFINE_float("target_kl", None, "Target KL divergence threshold.")

flags.DEFINE_integer("nn_width", 64, "Hidden width of actor/critic MLPs.")
flags.DEFINE_integer("nn_depth", 2, "Number of hidden layers in each MLP.")


class EclipsePPOAgent(nn.Module):
  """MLP actor-critic for Eclipse's flat observation vector."""

  def __init__(self, num_actions, observation_shape, device, width=64,
               depth=2):
    super().__init__()
    layers = []
    in_features = np.array(observation_shape).prod()
    for _ in range(depth):
      layers.append(layer_init(nn.Linear(in_features, width)))
      layers.append(nn.Tanh())
      in_features = width
    self.shared = nn.Sequential(*layers)

    self.critic = nn.Sequential(
        self.shared,
        layer_init(nn.Linear(width, 1), std=1.0),
    )
    self.actor = nn.Sequential(
        self.shared,
        layer_init(nn.Linear(width, num_actions), std=0.01),
    )
    self.num_actions = num_actions
    self.device = device
    self.register_buffer("mask_value", torch.tensor(-1e6))

  def get_value(self, x):
    return self.critic(x)

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    if legal_actions_mask is None:
      legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()
    logits = self.actor(x)
    from open_spiel.python.pytorch.ppo import CategoricalMasked
    probs = CategoricalMasked(logits=logits, masks=legal_actions_mask,
                              mask_value=self.mask_value)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(
        x), probs.probs


def make_agent_fn(width, depth):
  def agent_fn(num_actions, observation_shape, device):
    return EclipsePPOAgent(num_actions, observation_shape, device,
                           width=width, depth=depth)

  return agent_fn


# Observation tensor score slots written by the C++ game (see eclipse.cc):
#   self score slot  B0 + 10   => index 45 + 10 = 55
#   opponent k slot  Bn + 0    => index 45 + 135 + k * 25, k = other index
SCORE_SELF_SLOT = 45 + 10
OPP_BASE = 45 + 135
SCORE_DIVISOR = 200.0  # total_vp normalized by /200 in the tensor.


def opponent_block_index(seat, viewer):
  """Block index of `seat` as seen by `viewer`, matchingC++ opponent ordering."""
  if seat < viewer:
    return seat
  return seat - 1


def phi_from_obs_slot(obs_full, slot):
  """Banked-VP potential read from a score slot of a full observation."""
  return float(obs_full[slot]) * SCORE_DIVISOR


# Lazy caches: the phi functions run once per (env, observed state) in the hot
# loop; absl FLAGS lookups (2.5M+ calls in profiling) are replaced by plain
# locals resolved after flag parsing.
_PHI_WEIGHTS = None
_PHI_VARS = None


def _phi_cached():
  """Returns (mode, weights) resolved once from FLAGS."""
  global _PHI_WEIGHTS, _PHI_VARS
  if _PHI_VARS is None:
    _PHI_WEIGHTS = (FLAGS.phi_w_colony, FLAGS.phi_w_disk,
                    FLAGS.phi_w_structure, FLAGS.phi_w_ambassador)
    _PHI_VARS = (FLAGS.phi, FLAGS.gamma)
  return _PHI_VARS[0], _PHI_WEIGHTS, _PHI_VARS[1]


def phi_soft_self(obs):
  """Soft potential read from a seat's own (self) observation block."""
  _, (w_colony, w_disk, w_struct, w_amb), _ = _phi_cached()
  base = phi_from_obs_slot(obs, SCORE_SELF_SLOT)
  colony = float(obs[45 + 21]) * 12.0
  disks = float(obs[45 + 22]) * 16.0
  orbitals = float(obs[45 + 27]) * 10.0
  monoliths = float(obs[45 + 28]) * 6.0
  amb = float(obs[45 + 29]) * 3.0
  return base + (w_colony * colony + w_disk * disks +
                 w_struct * (orbitals + monoliths) +
                 w_amb * amb)


def phi_soft_opponent(obs_viewer, block):
  """Soft potential for a seat read from another seat's observation block."""
  _, (w_colony, w_disk, w_struct, w_amb), _ = _phi_cached()
  base = phi_from_obs_slot(obs_viewer, block)
  colony = float(obs_viewer[block + 12]) * 12.0
  disks = float(obs_viewer[block + 13]) * 16.0
  orbitals = float(obs_viewer[block + 14]) * 10.0
  monoliths = float(obs_viewer[block + 15]) * 6.0
  amb = float(obs_viewer[block + 11]) * 3.0
  return base + (w_colony * colony + w_disk * disks +
                 w_struct * (orbitals + monoliths) +
                 w_amb * amb)


def potential_self(obs):
  """Potential of the acting seat from its own observation."""
  mode, _, _ = _phi_cached()
  if mode == "banked":
    return phi_from_obs_slot(obs, SCORE_SELF_SLOT)
  if mode == "soft":
    return phi_soft_self(obs)
  return 0.0


def potential_opponent(obs_viewer, block):
  """Potential of a non-acting seat read from the viewer's observation."""
  mode, _, _ = _phi_cached()
  if mode == "banked":
    return phi_from_obs_slot(obs_viewer, block)
  if mode == "soft":
    return phi_soft_opponent(obs_viewer, block)
  return 0.0


def main(_):
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)
  torch.backends.cudnn.deterministic = FLAGS.torch_deterministic

  device = torch.device(
      "cuda" if torch.cuda.is_available() and FLAGS.cuda else "cpu")

  run_name = f"{FLAGS.game}__{FLAGS.seed}__{datetime.now().strftime('%Y%m%d%H%M%S')}"
  if SummaryWriter is None:
    writer = NullWriter()
  elif FLAGS.track:
    writer = SummaryWriter(os.path.join(FLAGS.run_dir, FLAGS.track))
  else:
    writer = SummaryWriter(os.path.join(FLAGS.run_dir, run_name))
  writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" %
      ("\n".join([f"|{key}|{value}|" for key, value in
                  sorted(vars(FLAGS).items())])),
  )

  game = pyspiel.load_game(FLAGS.game)
  envs = SyncVectorEnv([
      rl_environment.Environment(
          game=pyspiel.load_game(FLAGS.game),
          chance_event_sampler=rl_environment.ChanceEventSampler(
              seed=FLAGS.seed + i),
          observation_type=rl_environment.ObservationType.OBSERVATION,
          observations_as_numpy=True)
      for i in range(FLAGS.num_envs)
  ])
  game = envs.envs[0]._game  # pylint: disable=protected-access
  input_shape = tuple(game.observation_tensor_shape())
  num_players = game.num_players()

  agent = PPO(
      input_shape=input_shape,
      num_actions=game.num_distinct_actions(),
      num_players=num_players,
      player_id=0,
      num_envs=FLAGS.num_envs,
      steps_per_batch=FLAGS.num_steps,
      num_minibatches=FLAGS.num_minibatches,
      update_epochs=FLAGS.update_epochs,
      learning_rate=FLAGS.learning_rate,
      gae=FLAGS.gae,
      gamma=FLAGS.gamma,
      gae_lambda=FLAGS.gae_lambda,
      normalize_advantages=FLAGS.norm_adv,
      clip_coef=FLAGS.clip_coef,
      clip_vloss=FLAGS.clip_vloss,
      entropy_coef=FLAGS.ent_coef,
      value_coef=FLAGS.vf_coef,
      max_grad_norm=FLAGS.max_grad_norm,
      target_kl=FLAGS.target_kl,
      device=device,
      writer=writer,
      agent_fn=make_agent_fn(FLAGS.nn_width, FLAGS.nn_depth),
  )

  batch_size = FLAGS.num_envs * FLAGS.num_steps
  num_updates = FLAGS.total_timesteps // batch_size

  # Per-player episode return logging.
  episode_returns = {i: [] for i in range(FLAGS.num_envs)}
  recent_returns = []

  time_step = envs.reset(players="current")
  for update in range(num_updates):
    for step in range(FLAGS.num_steps):
      agent_output = agent.step(time_step)
      # phi(s) for the acting seat from its own obs (this row).
      # Uses the CPU numpy obs batch already gathered by agent.step (no second
      # GPU->CPU round trip).
      obs_batch = agent.last_obs_batch
      phi_prev = np.fromiter(
          (potential_self(obs_batch[i]) for i in range(FLAGS.num_envs)),
          dtype=np.float64, count=FLAGS.num_envs)

      time_step, reward, done, unreset = envs.step(
          agent_output, reset_if_done=True, players="current")

      shaped = np.zeros(FLAGS.num_envs)
      if FLAGS.shaping:
        seats = agent.last_seats
        for i, ts in enumerate(time_step):
          if done[i]:
            continue
          viewer = ts.observations["current_player"]
          seat = seats[i]
          k = opponent_block_index(seat, viewer)
          phi_next = potential_opponent(
              ts.observations["info_state"][viewer], OPP_BASE + k * 25)
          shaped[i] = FLAGS.gamma * phi_next - phi_prev[i]

      agent.post_step(reward, done, shaped_reward=shaped.tolist())

      # Episode return logging.
      for i, ts in enumerate(unreset):
        if ts.last():
          ret = float(ts.rewards[0])
          episode_returns[i].append(ret)
          recent_returns.append(ret)

    agent.learn(time_step)

    if update % FLAGS.eval_every == 0:
      n_completed = sum(len(r) for r in episode_returns.values())
      recent = recent_returns[-200:]
      nonzero = sum(1 for r in recent if r > 0.5)
      summary = ""
      if recent:
        summary = (f"  mean_term_return={np.mean(recent):.2f}  "
                   f"nonzero_episodes={nonzero}/{len(recent)}")
      print(f"[update {update}] steps={agent.total_steps_done}"
            f"  total_episodes={n_completed}{summary}")
      if writer is not None:
        writer.add_scalar("charts/num_episodes", n_completed,
                          agent.total_steps_done)
        if recent:
          writer.add_scalar("charts/mean_episode_return", np.mean(recent),
                            agent.total_steps_done)
          writer.add_scalar("charts/nonzero_episodes", nonzero,
                            agent.total_steps_done)

  writer.close()
  print("pilot done")


if __name__ == "__main__":
  app.run(main)
