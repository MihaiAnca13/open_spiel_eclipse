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

Reward shaping (default on, --phi=learned) uses a learned potential: the
network's own win-value prediction, which the Sprint-1 grid validated as the
only configuration that stays monotonic against its own snapshots while
crushing fixed Random/Greedy baselines:

  phi(s)  = my-seat predicted final win-value (rank utility scale, /200)
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
from open_spiel.python.examples.league import Matchmaker
from open_spiel.python.examples.league import PolicyRoster
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import layer_init
from open_spiel.python.pytorch.ppo import rank_utility
from open_spiel.python.vector_env import SyncVectorEnv

try:
  from tqdm import tqdm
except ImportError:
  tqdm = None

try:
  from open_spiel.python.async_vector_env import AsyncVectorEnv
except ImportError:
  AsyncVectorEnv = None

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

  def flush(self):
    pass

  def close(self):
    pass


# Module-level handle to the active tqdm bar (set by main()). All non-bar
# console output routes through _emit so it draws above (not through) the bar.
_ACTIVE_PBAR = None


def _emit(message):
  """Print outside the progress bar (above it) when one is active."""
  bar = _ACTIVE_PBAR
  if bar is not None:
    bar.write(message)
  else:
    print(message)

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "eclipse(players=4)", "Name of the game.")
flags.DEFINE_integer("seed", 1, "Seed of the experiment.")
flags.DEFINE_bool("cuda", True, "If True, cuda will be enabled by default.")
flags.DEFINE_bool("torch_deterministic", True, "Deterministic torch.")
flags.DEFINE_string("track", None, "Experiment tracking run id.")
flags.DEFINE_string("run_dir", "runs", "Root dir for tensorboard runs.")
flags.DEFINE_bool(
    "no_tb", False,
    "Disable the tensorboard writer entirely (NullWriter). Use for raw "
    "throughput benchmarking; no scalar diagnostics are persisted.")
flags.DEFINE_bool(
    "timing", False,
    "Emit per-phase wall-clock timings (ms/env-step) for the async loop "
    "every --timing_every updates. Phases: act (PPO step_np), env "
    "(AsyncVectorEnv.step_np nearest draw), shape (potential), learn.")
flags.DEFINE_integer("timing_every", 10,
                     "Cadence (updates) for --timing per-phase report.")

flags.DEFINE_bool(
    "randomize_races", True,
    "Randomize per-episode race assignment (unique alien draft, Terran as "
    "filler) instead of using the fixed species_p* game params.")
flags.DEFINE_float(
    "race_alien_prob", 0.8,
    "Per-seat probability of drawing a (unique) alien species when "
    "--randomize_races; the remainder are Terran Factions.")
flags.DEFINE_bool(
    "randomize_npc_difficulty", True,
    "Randomize the NPC (GCDS/guardian/ancient) difficulty per episode.")
flags.DEFINE_bool(
    "randomize_warped", True,
    "Randomize the warped-universe module flag per episode.")
flags.DEFINE_float("warped_prob", 0.5,
                   "Probability the warped-universe module is on when "
                   "--randomize_warped.")

flags.DEFINE_integer("num_envs", 8, "Number of parallel game environments.")
flags.DEFINE_integer("num_steps", 128, "Rollout steps per update per env.")
flags.DEFINE_integer("total_timesteps", 100_000, "Total environment steps.")
flags.DEFINE_integer("eval_every", 10, "Log every N updates.")
flags.DEFINE_bool("progress", True,
                  "Show a tqdm-style progress bar with it/s (env steps/sec). "
                  "Disabled automatically if tqdm is not installed.")
flags.DEFINE_bool("anneal_lr", True,
                  "Linearly anneal the learning rate to 0 over the run "
                  "(standard PPO practice for long training).")

flags.DEFINE_integer(
    "num_workers", 0,
    "Async process-pool workers for the vector env (0 = sync). When >0 the "
    "envs run in a fork pool with shared-memory buffers and the array-native "
    "PPO path (step_np/post_step_np/learn_np) is used, so no per-env "
    "TimeStep/StepOutput objects are built. Recommended 8-16 with >= 512 "
    "envs.")

flags.DEFINE_bool("shaping", True,
                  "Potential-based shaping from the obs 'score' slot.")
flags.DEFINE_enum("phi", "learned", ["banked", "soft", "none", "learned"],
                  "Potential definition. 'learned' (default, validated by the "
                  "Sprint-1 grid) = the network's own predicted win-value, "
                  "i.e. a learned critic potential (requires "
                  "--value_mode=win). 'banked' = current VP if the game ended "
                  "now. 'soft' = banked plus in-progress presence terms (colony "
                  "ships, disks on sectors, orbitals/monoliths, ambassadors) so "
                  "the shaped reward is non-zero even before any VP is banked. "
                  "'none' disables shaping.")
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
flags.DEFINE_enum("value_mode", "win", ["win", "vp"],
                  "Value/return objective. 'win' = terminal targets are "
                  "per-seat rank utilities (1st/2nd/3rd/4th), so the agent "
                  "optimizes 'finish first'. 'vp' = raw final VP (the original "
                  "behaviour).")
flags.DEFINE_float("aux_coef", 0.1,
                   "Weight of auxiliary-head losses (e.g. final-VP regression).")
flags.DEFINE_enum(
    "aux_target_mode", "vp", ["vp", "rank", "none"],
    "Aux-head regression target. 'vp' = raw final VP / 200 (legacy, saturates "
    "to 0 on zero-VP games). 'rank' = per-seat rank-utility (nonzero signal "
    "even when every seat banks ~0 VP). 'none' disables the aux head.")
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

flags.DEFINE_bool(
    "league", False,
    "Population self-play: train main against a roster of snapshots/"
    "exploiters sampled into mixed lineups (requires --roster_dir).")
flags.DEFINE_string("roster_dir", "runs/roster",
                    "Directory backing the policy roster (checkpoints + JSON).")
flags.DEFINE_string(
    "resume", None,
    "Seed the network from a saved checkpoint before training. Accepts a "
    "roster policy id (e.g. 'main' or 'snap_u100', loaded from --roster_dir "
    "if --league/--exploit_victim is on) or an explicit .pt path. Lets an "
    "interrupted/extended run continue from where it left off.")
flags.DEFINE_integer("snapshot_every", 25,
                     "Snapshot the main policy into the roster every N updates.")
flags.DEFINE_float("selfplay_fraction", 0.5,
                   "Fraction of (re)spawned lineups that are pure self-play.")
flags.DEFINE_float("old_fraction", 0.125,
                   "Within mixed lineups, chance a seat is a weak/old policy.")
flags.DEFINE_bool(
    "eval_squad", False,
    "At the eval cadence, pit main against a snapshots-only eval squad and "
    "report win rate / avg rank (no heuristics in v1).")
flags.DEFINE_integer(
    "verdict_every_sec", 1800,
    "Minimum wall-clock gap (seconds) between full verdict evals (main "
    "seats 0,1 vs fixed Random / fixed Greedy / snapshot squad).")
flags.DEFINE_integer(
    "max_seconds", 0,
    "Hard wall-clock cap (seconds) for fail-fast runs: at this deadline emit "
    "a final verdict + snapshot the roster and exit 0. 0 disables.")
flags.DEFINE_bool("eval_random", True,
                  "In the verdict eval, include main vs fixed-Random avg rank.")
flags.DEFINE_bool("eval_greedy", True,
                  "In the verdict eval, include main vs fixed priority-Greedy "
                  "avg rank (random fallback outside the heuristic's coverage).")
flags.DEFINE_string(
    "exploit_victim", None,
    "Sequential-exploiter mode: train this run's policy ONLY against the "
    "frozen roster policy id given here (e.g. a snapshot), starting from the "
    "current main weights (or the victim's if none), then report the win-rate "
    "vs the victim.")
flags.DEFINE_bool("exploit_promote", False,
                  "In exploiter mode, fold the trained policy into the roster "
                  "as an exploiter entry when it beats the victim.")
flags.DEFINE_float("exploit_lr", 1e-3,
                   "Learning rate used in exploiter mode (higher than main).")


class EclipsePPOAgent(nn.Module):
  """MLP actor-critic for Eclipse's flat observation vector.

  The critic is a win/rank value head: it outputs 4 logits (P(rank 1..4)) and
  ``get_value`` returns the expected rank-utility (1st=1.0, 2nd=0.5, 3rd=0.0,
  4th=-0.5). Auxiliary heads (``final_vp``) regress terminal quantities from
  the shared trunk, giving the network a dense, learned signal about what leads
  to VP without hand-tuned shaping weights.
  """

  # Rank-utility table (1st..4th), matching ppo.rank_utility's default.
  RANK_UTILITY = (1.0, 0.5, 0.0, -0.5)

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
        layer_init(nn.Linear(width, 4), std=1.0),
    )
    self.actor = nn.Sequential(
        self.shared,
        layer_init(nn.Linear(width, num_actions), std=0.01),
    )
    # Auxiliary heads: predict terminal quantities from the shared trunk.
    self.aux_heads = nn.ModuleDict({
        "final_vp": layer_init(nn.Linear(width, 1), std=1.0),
    })
    self.num_actions = num_actions
    self.device = device
    self.register_buffer("mask_value", torch.tensor(-1e6))

  def get_value(self, x):
    return self.rank_value(self.critic(x))

  def rank_value(self, rank_logits):
    """Expected rank-utility from (..., 4) rank logits."""
    probs = rank_logits.softmax(dim=-1)
    utility = torch.tensor(self.RANK_UTILITY, dtype=rank_logits.dtype,
                           device=rank_logits.device)
    return (probs * utility).sum(dim=-1)

  def value_from_features(self, features):
    """Scalar win value from shared features (sparse learn path)."""
    return self.rank_value(self.critic[-1](features))

  def get_aux(self, features):
    """Auxiliary-head raw outputs keyed by task name."""
    return {name: head(features) for name, head in self.aux_heads.items()}

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    if legal_actions_mask is None:
      legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()
    logits = self.actor(x)
    from open_spiel.python.pytorch.ppo import CategoricalMasked
    probs = CategoricalMasked(logits=logits, masks=legal_actions_mask,
                              mask_value=self.mask_value)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.get_value(
        x), probs.probs


def make_agent_fn(width, depth):
  def agent_fn(num_actions, observation_shape, device):
    return EclipsePPOAgent(num_actions, observation_shape, device,
                           width=width, depth=depth)

  return agent_fn


# Win-mode potential squash: raw VP-unit potentials are mapped onto the
# rank-utility scale ([1..4] -> ~[-0.5, 1]) so shaped rewards and terminal
# rank-utility targets stay comparable. The tensor already normalizes VP by
# /200, so u(vp) = clip(vp/200, -0.5, 1).
def _squash_win(vp):
  return float(np.clip(np.array(vp) / 200.0, -0.5, 1.0))


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


# Eclipse flat action-id layout (see eclipse.cc anonymous namespace):
#   0=PASS, 1=RESEARCH start, 3-26 standard techs, 27-74 rare techs,
#   75=BUILD start, 83=EXPLORE start, 84/85 place/discard, 86-91 rotations,
#   92/93 claim yes/no, 94/95 discovery/2VP, 96-99 keep-ish, 100=stop,
#   101-325 explore zone == galaxy cell, 326-331 TRADE, 332-5731 COLONY_SHIP,
#   5732=INFLUENCE start, 7539=UPGRADE start, 8502=MOVE start.
ACTION_EXPLORE_START = 83
ACTION_EXPLORE_ZONE_START = 101
ACTION_EXPLORE_ZONE_END = 325
ACTION_BUILD_START = 75
ACTION_UPGRADE_START = 7539
ACTION_RESEARCH_START = 1
ACTION_INFLUENCE_START = 5732
ACTION_MOVE_START = 8502
ACTION_COLONY_START = 332
ACTION_COLONY_END = 5731
ACTION_PASS = 0

GALAXY_BASE = 45 + 135 + 125  # obs block C starts after A+B0+5 opponent blocks.
CELL_STRIDE = 6


def _greedy_pick(obs, legal, rng):
  """Priority-heuristic move for the fixed Greedy baseline.

  Only the action-phase macro starts and the explore sub-pipeline get typed
  preferences; any state the heuristic has no rule for (chance, combat,
  upkeep, bankruptcy, diplomacy, reaction, trade, build/upgrade/move choice
  internals, etc.) falls back to a uniformly random legal action.
  """
  s = set(int(a) for a in legal)
  zone_in = lambda a: ACTION_EXPLORE_ZONE_START <= a <= ACTION_EXPLORE_ZONE_END

  if ACTION_EXPLORE_START in s:
    return ACTION_EXPLORE_START
  if 84 in s and 85 in s:
    return 84  # place over discard
  rot = next((a for a in range(86, 92) if a in s), None)
  if rot is not None:
    return rot
  if 92 in s and 93 in s:
    return 92  # take control over decline
  if 94 in s and 95 in s:
    return 95  # immediate 2 banked VP over a discovery draw
  keep = next((a for a in range(96, 100) if a in s), None)
  if keep is not None:
    return keep
  zones = [int(a) for a in legal if zone_in(a)]
  if zones:
    return _best_expand_zone(obs, zones)
  for pid in (ACTION_BUILD_START, ACTION_UPGRADE_START, ACTION_RESEARCH_START,
              ACTION_INFLUENCE_START, ACTION_MOVE_START):
    if pid in s:
      return pid
  colony = [int(a) for a in legal
            if ACTION_COLONY_START <= a <= ACTION_COLONY_END]
  if colony:
    return int(rng.choice(np.asarray(colony)))
  if ACTION_PASS in s:
    return ACTION_PASS
  return int(rng.choice(np.asarray(legal)))


def _best_expand_zone(obs, zones):
  """Among legal explore-zone cells, prefer empty + uncontested, else lowest.

  No galaxy geometry is decoded here, so this is an approximation: fewer
  enemy units, tie-break lowest cell index (deterministic).
  """
  best = None
  best_key = None
  for a in zones:
    c = a - ACTION_EXPLORE_ZONE_START
    o = CELL_STRIDE * c
    enemy = obs[GALAXY_BASE + o + 4] if GALAXY_BASE + o + 4 < len(obs) else 0.0
    key = (enemy, c)
    if best_key is None or key < best_key:
      best_key = key
      best = a
  return best


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


def potential_self(obs, win_squash=False):
  """Potential of the acting seat from its own observation.

  Returns VP units for banked/soft; with ``win_squash`` those are mapped onto
  the rank-utility scale (so shaped rewards stay comparable to terminal
  rank-utility targets). 'learned' is handled separately by the caller (it
  needs the network).
  """
  mode, _, _ = _phi_cached()
  if mode == "banked":
    v = phi_from_obs_slot(obs, SCORE_SELF_SLOT)
  elif mode == "soft":
    v = phi_soft_self(obs)
  else:
    v = 0.0
  return _squash_win(v) if (win_squash and mode in ("banked", "soft")) else v


def potential_opponent(obs_viewer, block, win_squash=False):
  """Potential of a non-acting seat read from the viewer's observation.

  VP units for banked/soft; ``win_squash`` maps onto the rank-utility scale.
  """
  mode, _, _ = _phi_cached()
  if mode == "banked":
    v = phi_from_obs_slot(obs_viewer, block)
  elif mode == "soft":
    v = phi_soft_opponent(obs_viewer, block)
  else:
    v = 0.0
  return _squash_win(v) if (win_squash and mode in ("banked", "soft")) else v


def potential_self_vec(obs_batch, win_squash=False):
  """Vectorized ``potential_self`` over a (num_envs, obs) batch.

  Column reads only, no per-env Python loop (hot shaping path).
  """
  mode, (w_col, w_disk, w_struct, w_amb), _ = _phi_cached()
  if mode == "banked":
    v = obs_batch[:, SCORE_SELF_SLOT] * SCORE_DIVISOR
  elif mode == "soft":
    v = (
        obs_batch[:, SCORE_SELF_SLOT] * SCORE_DIVISOR + w_col * obs_batch[:, 66] * 12.0
        + w_disk * obs_batch[:, 67] * 16.0
        + w_struct * (obs_batch[:, 72] * 10.0 + obs_batch[:, 73] * 6.0)
        + w_amb * obs_batch[:, 74] * 3.0)
  else:
    v = np.zeros(obs_batch.shape[0], dtype=np.float32)
  if win_squash and mode in ("banked", "soft"):
    return np.clip(v / SCORE_DIVISOR, -0.5, 1.0)
  return v


def potential_opponent_vec(obs_batch, blocks, win_squash=False):
  """Vectorized ``potential_opponent``; ``blocks`` is (num_envs,) int of the
  opponent's obs block start per row."""
  n = obs_batch.shape[0]
  ar = np.arange(n)
  mode, (w_col, w_disk, w_struct, w_amb), _ = _phi_cached()
  if mode == "banked":
    v = obs_batch[ar, blocks] * SCORE_DIVISOR
  elif mode == "soft":
    v = (
        obs_batch[ar, blocks] * SCORE_DIVISOR + w_col * obs_batch[ar, blocks + 12] * 12.0
        + w_disk * obs_batch[ar, blocks + 13] * 16.0
        + w_struct * (obs_batch[ar, blocks + 14] * 10.0
                      + obs_batch[ar, blocks + 15] * 6.0)
        + w_amb * obs_batch[ar, blocks + 11] * 3.0)
  else:
    v = np.zeros(n, dtype=np.float32)
  if win_squash and mode in ("banked", "soft"):
    return np.clip(v / SCORE_DIVISOR, -0.5, 1.0)
  return v


def _phi_wins(agent, obs_np, device):
  """Win-value (expected rank-utility) of the mover for each row's own obs."""
  with torch.no_grad():
    x = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)
    return agent.get_value(x).cpu().numpy()


# ── League (population self-play) helpers ───────────────────────────────────

def _league_setup(agent, roster, matchmaker, agent_fn, num_actions,
                  input_shape, device):
  """Initializes league mode: networks + lineups for all envs."""
  lineup = matchmaker.lineups()
  need = set(lineup.reshape(-1).tolist())
  networks = {"main": agent.network}
  for pid in need - {"main"}:
    networks[pid] = roster.load_net(pid, agent_fn, num_actions, input_shape,
                                    device)
    if networks[pid] is None:
      raise ValueError(f"roster has no weights for opponent {pid}")
  agent.setup_league(networks, lineup, "main")
  return lineup


def _refresh_lineups(agent, matchmaker, roster, agent_fn, num_actions,
                     input_shape, device, done_flags):
  """Re-samples lineups for (re)spawned envs and loads any new policies.

  Lineups are fixed per env until that env's episode ends; on reset we give it
  a fresh lineup so newly added snapshots/exploiters enter play.
  """
  for i, done in enumerate(done_flags):
    if not done:
      continue
    agent.lineup[i, :] = np.asarray(matchmaker.sample_lineup(), dtype=object)
  need = set(agent.lineup.reshape(-1).tolist())
  for pid in need:
    if pid in agent.networks:
      continue
    networks = roster.load_net(pid, agent_fn, num_actions, input_shape,
                               device)
    if networks is None:
      raise ValueError(f"roster has no weights for opponent {pid}")
    agent.networks[pid] = networks


def _maybe_snapshot(agent, roster, update, force=False):
  """Captures the main policy into the roster on the snapshot cadence."""
  if force or (update > 0 and update % FLAGS.snapshot_every == 0):
    roster.record_main(agent.network, update)
    roster.add_snapshot(agent.network, update)
    roster.prune(keep_recent=4, keep_spaced=4)


def _eval_squad(agent, roster, agent_fn, num_actions, input_shape, device,
                game_str, num_players, num_games, rng_seed, main_seats=(0, 1)):
  """Plays main (argmax) against a snapshots-only squad.

  ``main_seats`` are driven by the main policy; the remaining seats each draw a
  snapshot from the roster. Returns (main_wins, games, main_ranks) where main
  rank is 1..4 placement, or None if the roster has no opponents. Runs on the
  network's raw policy (argmax over legal actions) so no PPO buffers are
  touched.
  """
  opponents = roster.opponent_ids(exclude_main=True)
  if not opponents:
    return None
  rng = np.random.RandomState(rng_seed)
  nets = {"main": agent.network}
  pool = sorted(opponents[:3])
  for pid in pool:
    nets[pid] = roster.load_net(pid, agent_fn, num_actions, input_shape,
                                device)
  other_seats = [s for s in range(num_players) if s not in main_seats]
  main_wins = 0
  ranks = []
  for g in range(num_games):
    env = rl_environment.Environment(
        game=pyspiel.load_game(game_str),
        chance_event_sampler=rl_environment.ChanceEventSampler(
            seed=rng_seed + g),
        observation_type=rl_environment.ObservationType.OBSERVATION,
        observations_as_numpy=True)
    policy_for = {s: str(rng.choice(pool)) for s in other_seats}
    time_step = env.reset(players="current")
    while not time_step.last():
      seat = int(time_step.observations["current_player"])
      pid = "main" if seat in main_seats else policy_for[seat]
      net = nets[pid]
      obs = time_step.observations["info_state"][seat]
      legal = time_step.observations["legal_actions"][seat]
      with torch.no_grad():
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32))[None].to(
            device)
        logits = net.actor(x)
        mask = torch.full((1, logits.size(1)), -1e6).to(device)
        mask[0, np.asarray(legal, dtype=np.int64)] = logits[
            0, np.asarray(legal, dtype=np.int64)]
        action = int(mask.argmax().item())
      time_step = env.step([action])
    rewards = np.asarray(time_step.rewards, dtype=np.float32)
    main_best = max(rewards[s] for s in main_seats)
    rank = 1 + int(np.sum(rewards > main_best))
    main_wins += int(rank == 1)
    ranks.append(rank)
  return main_wins, num_games, ranks


def _eval_fixed_opponent(agent, bot_pick, game_str, num_players, num_games,
                         rng_seed, main_seats=(0, 1)):
  """Main (argmax) on ``main_seats`` vs one fixed bot policy on every other
  seat. ``bot_pick(obs, legal) -> int``. Returns (main_wins, num_games,
  main_ranks); run on the raw argmax policy, no PPO buffers touched."""
  rng = np.random.RandomState(rng_seed)
  main_wins = 0
  ranks = []
  other = [s for s in range(num_players) if s not in main_seats]
  for g in range(num_games):
    env = rl_environment.Environment(
        game=pyspiel.load_game(game_str),
        chance_event_sampler=rl_environment.ChanceEventSampler(seed=rng_seed + g),
        observation_type=rl_environment.ObservationType.OBSERVATION,
        observations_as_numpy=True)
    time_step = env.reset(players="current")
    while not time_step.last():
      seat = int(time_step.observations["current_player"])
      obs = time_step.observations["info_state"][seat]
      legal = time_step.observations["legal_actions"][seat]
      if seat in main_seats:
        with torch.no_grad():
          x = torch.from_numpy(np.asarray(obs, dtype=np.float32))[None].to(
              agent.device)
          logits = agent.network.actor(x)
          mask = torch.full((1, logits.size(1)), -1e6, device=agent.device)
          la = np.asarray(legal, dtype=np.int64)
          mask[0, la] = logits[0, la]
          action = int(mask.argmax().item())
      else:
        action = bot_pick(obs, legal)
      time_step = env.step([action])
    rewards = np.asarray(time_step.rewards, dtype=np.float32)
    main_best = max(rewards[s] for s in main_seats)
    rank = 1 + int(np.sum(rewards > main_best))
    main_wins += int(rank == 1)
    ranks.append(rank)
  return main_wins, num_games, ranks


def _run_verdict(agent, roster, agent_fn, num_actions, input_shape, device,
                 game_str, num_players, writer, step):
  """Full fail-fast verdict: main {0,1} vs fixed Random, fixed Greedy and the
  snapshot squad. Emits one line per baseline and writes scalars to ``writer``.
  """
  rng = np.random.RandomState(seed=None)
  bot_rng = np.random.RandomState(12345)
  out = {}
  if FLAGS.eval_random:
    rand_bot = lambda _o, legal: int(
        bot_rng.choice(np.asarray(legal, dtype=np.int32)))
    wins, games, ranks = _eval_fixed_opponent(agent, rand_bot, game_str,
                                              num_players, num_games=8,
                                              rng_seed=FLAGS.seed + step)
    avg = float(np.mean(ranks))
    out["random"] = (wins, games, avg)
    _emit(f"  [verdict] vs Random   win {wins}/{games}  avg_rank={avg:.2f}")
    writer.add_scalar("verdict/random_avg_rank", avg, step)
  if FLAGS.eval_greedy:
    greedy_bot = lambda obs, legal: _greedy_pick(
        np.asarray(obs, dtype=np.float32), legal, bot_rng)
    wins, games, ranks = _eval_fixed_opponent(agent, greedy_bot, game_str,
                                              num_players, num_games=8,
                                              rng_seed=FLAGS.seed + step)
    avg = float(np.mean(ranks))
    out["greedy"] = (wins, games, avg)
    _emit(f"  [verdict] vs Greedy   win {wins}/{games}  avg_rank={avg:.2f}")
    writer.add_scalar("verdict/greedy_avg_rank", avg, step)
  if roster is not None:
    res = _eval_squad(agent, roster, agent_fn, num_actions, input_shape,
                      device, game_str, num_players, num_games=8,
                      rng_seed=FLAGS.seed + step)
    if res is not None:
      wins, games, ranks = res
      avg = float(np.mean(ranks))
      out["squad"] = (wins, games, avg)
      _emit(f"  [verdict] vs Squad    win {wins}/{games}  avg_rank={avg:.2f}")
      writer.add_scalar("verdict/squad_win_rate", wins / games, step)
      writer.add_scalar("verdict/squad_avg_rank", avg, step)
  return out


def _eval_head2head(agent, opponent_net, agent_fn, num_actions, input_shape,
                    device, game_str, num_players, num_games, rng_seed):
  """Win-rate of the main policy (argmax) against a single opponent net.

  Main drives seat 0; the opponent drives every other seat. Returns
  (main_wins, num_games, main_ranks).
  """
  nets = {"main": agent.network, "opp": opponent_net}
  other_seats = list(range(1, num_players))
  main_wins = 0
  ranks = []
  for g in range(num_games):
    env = rl_environment.Environment(
        game=pyspiel.load_game(game_str),
        chance_event_sampler=rl_environment.ChanceEventSampler(
            seed=rng_seed + g),
        observation_type=rl_environment.ObservationType.OBSERVATION,
        observations_as_numpy=True)
    time_step = env.reset(players="current")
    while not time_step.last():
      seat = int(time_step.observations["current_player"])
      pid = "main" if seat == 0 else "opp"
      net = nets[pid]
      obs = time_step.observations["info_state"][seat]
      legal = time_step.observations["legal_actions"][seat]
      with torch.no_grad():
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32))[None].to(
            device)
        logits = net.actor(x)
        mask = torch.full((1, logits.size(1)), -1e6).to(device)
        mask[0, np.asarray(legal, dtype=np.int64)] = logits[
            0, np.asarray(legal, dtype=np.int64)]
        action = int(mask.argmax().item())
      time_step = env.step([action])
    rewards = np.asarray(time_step.rewards, dtype=np.float32)
    rank = 1 + int(np.sum(rewards > rewards[0]))
    main_wins += int(rank == 1)
    ranks.append(rank)
  return main_wins, num_games, ranks


def _log_update(agent, episode_returns, recent_returns, writer, update, eval_every=None):
  n_completed = sum(len(r) for r in episode_returns.values())
  recent = recent_returns[-200:]
  nonzero = sum(1 for r in recent if r > 0.5)
  summary = ""
  if recent:
    summary = (f"  mean_term_return={np.mean(recent):.2f}  "
               f"nonzero_episodes={nonzero}/{len(recent)}")
  losses = ""
  metrics = getattr(agent, "last_metrics", None) or {}
  if metrics:
    parts = [
        f"policy_loss={metrics['policy_loss']:.4f}",
        f"value_loss={metrics['value_loss']:.4f}",
        f"entropy={metrics['entropy']:.4f}",
    ]
    if metrics.get("aux_loss") is not None:
      parts.append(f"aux_loss={metrics['aux_loss']:.4f}")
    parts.append(f"approx_kl={metrics['kl']:.4f}")
    parts.append(f"clipfrac={metrics['clipfrac']:.3f}")
    parts.append(f"explained_var={metrics['explained_variance']:.3f}")
    losses = "  " + "  ".join(parts)
  _emit(f"[update {update}] steps={agent.total_steps_done}"
        f"  total_episodes={n_completed}{summary}{losses}")
  if writer is not None:
    writer.add_scalar("charts/num_episodes", n_completed,
                      agent.total_steps_done)
    if recent:
      writer.add_scalar("charts/mean_episode_return", np.mean(recent),
                        agent.total_steps_done)
      writer.add_scalar("charts/nonzero_episodes", nonzero,
                        agent.total_steps_done)


def _parse_game_string(game_str):
  """Splits 'short_name(param1=v1,...)' into (name, params_dict)."""
  if "(" not in game_str:
    return game_str, {}
  name, rest = game_str.split("(", 1)
  if not rest.endswith(")"):
    raise ValueError(f"malformed game string: {game_str}")
  params = {}
  for piece in rest[:-1].split(","):
    if not piece:
      continue
    key, _, val = piece.partition("=")
    params[key.strip()] = val.strip()
  return name.strip(), params


def _render_game_string(name, params):
  if not params:
    return name
  return name + "(" + ",".join(
      f"{k}={v}" for k, v in params.items()) + ")"


def _float_str(value):
  return f"{value:.6f}".rstrip("0").rstrip(".")


def _randomized_game_string(base_game_str, rng_seed):
  """Base game string + per-env rng_seed + opt-in setup randomization."""
  name, params = _parse_game_string(base_game_str)
  params["rng_seed"] = str(int(rng_seed))
  if FLAGS.randomize_races:
    params["randomize_races"] = "true"
    params["race_alien_prob"] = _float_str(FLAGS.race_alien_prob)
  if FLAGS.randomize_npc_difficulty:
    params["randomize_npc_difficulty"] = "true"
  if FLAGS.randomize_warped:
    params["randomize_warped"] = "true"
    params["warped_prob"] = _float_str(FLAGS.warped_prob)
  return _render_game_string(name, params)


def main(_):
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)
  torch.backends.cudnn.deterministic = FLAGS.torch_deterministic

  device = torch.device(
      "cuda" if torch.cuda.is_available() and FLAGS.cuda else "cpu")

  run_name = f"{FLAGS.game}__{FLAGS.seed}__{datetime.now().strftime('%Y%m%d%H%M%S')}"
  if SummaryWriter is None or FLAGS.no_tb:
    writer = NullWriter()
  elif FLAGS.track:
    writer = SummaryWriter(os.path.join(FLAGS.run_dir, FLAGS.track))
  else:
    writer = SummaryWriter(os.path.join(FLAGS.run_dir, run_name))
  # Clean scalar hparam table (absl flag descriptors are not the values).
  cfg = {k: v for k, v in sorted(FLAGS.flag_values_dict().items())
         if not k.startswith("_")}
  writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" %
      ("\n".join([f"|{key}|{value}|" for key, value in cfg.items()])),
  )

  # Each env gets its own seeded game instance (distinct rng_seed) so setup
  # draws, starting tech/discovery markets, tiles, and (when enabled) per-episode
  # race/difficulty/module randomization all differ across environments.
  env_game_strs = [
      _randomized_game_string(FLAGS.game, FLAGS.seed + i)
      for i in range(FLAGS.num_envs)
  ]
  game = pyspiel.load_game(FLAGS.game)
  envs_list = [
      rl_environment.Environment(
          game=pyspiel.load_game(env_game_strs[i]),
          chance_event_sampler=rl_environment.ChanceEventSampler(
              seed=FLAGS.seed + i),
          observation_type=rl_environment.ObservationType.OBSERVATION,
          observations_as_numpy=True)
      for i in range(FLAGS.num_envs)
  ]
  use_async = FLAGS.num_workers > 0 and AsyncVectorEnv is not None
  if use_async:
    envs = AsyncVectorEnv(
        envs_list,
        num_workers=FLAGS.num_workers,
        sampler_seeds=[FLAGS.seed + i for i in range(FLAGS.num_envs)],
        game_strs=env_game_strs,
    )
    game = envs_list[0]._game  # pylint: disable=protected-access
  else:
    envs = SyncVectorEnv(envs_list)
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
      value_mode=FLAGS.value_mode,
      aux_tasks=["final_vp"] if (FLAGS.aux_coef > 0 and
                                 FLAGS.aux_target_mode != "none") else None,
      aux_target_fn=(None if FLAGS.aux_target_mode == "none" else
                     (lambda rvec, m=FLAGS.aux_target_mode:
                      (np.asarray(rvec, dtype=np.float32).reshape(-1, 1)
                       if m == "vp" else
                       np.asarray([
                           rank_utility(rvec, s) for s in range(len(rvec))
                       ], dtype=np.float32).reshape(-1, 1)))),
      aux_coef=FLAGS.aux_coef,
  )

  # Device + resume telemetry before any training starts.
  _emit(f"device={device}  game={FLAGS.game}  num_envs={FLAGS.num_envs}"
        f"  num_workers={FLAGS.num_workers}")
  if FLAGS.resume:
    agent_fn_r = make_agent_fn(FLAGS.nn_width, FLAGS.nn_depth)
    resume_src = FLAGS.resume
    sd = None
    if FLAGS.league or FLAGS.exploit_victim:
      roster_r = PolicyRoster(FLAGS.roster_dir)
      net_r = roster_r.load_net(resume_src, agent_fn_r, game.num_distinct_actions(),
                                input_shape, device)
      if net_r is not None:
        sd = net_r.state_dict()
    if sd is None and os.path.exists(resume_src):
      sd = torch.load(resume_src, map_location=device, weights_only=True)
    if sd is None:
      raise ValueError(
          f"--resume={resume_src}: not a roster id in {FLAGS.roster_dir} and "
          f"not an existing .pt path")
    agent.network.load_state_dict(sd)
    _emit(f"resumed network weights from {resume_src}")

  batch_size = FLAGS.num_envs * FLAGS.num_steps
  num_updates = FLAGS.total_timesteps // batch_size

  # tqdm-style progress over total env steps; it/s = env steps per second.
  global _ACTIVE_PBAR
  pbar = None
  if FLAGS.progress and tqdm is not None:
    pbar = tqdm(
        total=FLAGS.total_timesteps,
        unit="envstep",
        desc=run_name,
        ncols=110,
        dynamic_ncols=True,
    )
    _ACTIVE_PBAR = pbar

  def _pbar_postfix():
    """Per-update diagnostic string for the tqdm bar."""
    if pbar is None:
      return
    metrics = getattr(agent, "last_metrics", None) or {}
    parts = []
    for key in ("policy_loss", "value_loss", "entropy", "kl"):
      if key in metrics:
        parts.append(f"{key}={metrics[key]:.3g}")
    recent = recent_returns[-20:]
    if recent:
      parts.append(f"ret={np.mean(recent):.2g}")
      parts.append(f"nz={sum(1 for r in recent if r > 0.5)}/{len(recent)}")
    pbar.set_postfix_str("  ".join(parts))


  # League (population self-play) setup: roster + matchmaker + lineups.
  agent_fn = make_agent_fn(FLAGS.nn_width, FLAGS.nn_depth)
  num_actions = game.num_distinct_actions()
  roster = None
  matchmaker = None
  exploit_victim_net = None
  if FLAGS.league or FLAGS.exploit_victim:
    roster = PolicyRoster(FLAGS.roster_dir)
  if FLAGS.league:
    matchmaker = Matchmaker(
        roster, FLAGS.num_envs, num_players,
        selfplay_fraction=FLAGS.selfplay_fraction,
        old_fraction=FLAGS.old_fraction, seed=FLAGS.seed + 12345)
    _league_setup(agent, roster, matchmaker, agent_fn, num_actions,
                  input_shape, device)
  elif FLAGS.exploit_victim:
    # Sequential-exploiter mode: one trainable policy (this run) vs a frozen
    # victim filling every other seat. Fixed lineup, no matchmaking/refresh.
    victim_id = FLAGS.exploit_victim
    exploit_victim_net = roster.load_net(victim_id, agent_fn, num_actions,
                                         input_shape, device)
    if exploit_victim_net is None:
      raise ValueError(f"exploit victim {victim_id} not in roster {FLAGS.roster_dir}")
    starter = roster.load_net("main", agent_fn, num_actions, input_shape,
                              device)
    if starter is None:
      starter = exploit_victim_net
    agent.network.load_state_dict(starter.state_dict())
    if FLAGS.exploit_lr > 0:
      for group in agent.optimizer.param_groups:
        group["lr"] = FLAGS.exploit_lr
    lineup = np.tile(
        np.asarray(["main"] + [victim_id] * (num_players - 1), dtype=object),
        (FLAGS.num_envs, 1))
    agent.setup_league({"main": agent.network, victim_id:
                        exploit_victim_net}, lineup, "main")

  # Shaping configuration resolved once per run.
  win_squash = FLAGS.value_mode == "win"
  phi_learned = FLAGS.phi == "learned"
  if phi_learned and not win_squash:
    raise ValueError("--phi=learned requires --value_mode=win (the learned "
                     "potential is the network's win value).")
  phi_mode = FLAGS.phi

  # Per-player episode return logging.
  episode_returns = {i: [] for i in range(FLAGS.num_envs)}
  recent_returns = []

  if use_async:
    _ = envs.reset(players="current")
    step_arrays = envs.reset_np()
    _tm = (np.zeros(5, dtype=np.float64) if FLAGS.timing else None)
    _tm_scale = 1.0 / max(1, FLAGS.num_steps)
    _run_t0 = time.time()
    _last_verdict_ts = [_run_t0]
    _deadline = (_run_t0 + FLAGS.max_seconds
                 if FLAGS.max_seconds and FLAGS.max_seconds > 0 else None)
    for update in range(num_updates):
      if _deadline is not None and time.time() >= _deadline:
        _emit(f"[gate] {FLAGS.max_seconds}s hard deadline reached "
              f"(update {update}, steps={agent.total_steps_done})")
        if FLAGS.league:
          _maybe_snapshot(agent, roster, update, force=True)
        _run_verdict(agent, roster, agent_fn, num_actions, input_shape,
                     device, _randomized_game_string(FLAGS.game,
                                                     FLAGS.seed + update * 7),
                     num_players, writer, agent.total_steps_done)
        writer.flush()
        break
      if _tm is not None:
        _tm[:] = 0.0
      for step in range(FLAGS.num_steps):
        t0 = time.perf_counter() if _tm is not None else None
        acts = agent.step_np(step_arrays)
        t1 = time.perf_counter() if _tm is not None else None
        obs_batch = agent.last_obs_batch
        if FLAGS.shaping and phi_learned:
          phi_prev = _phi_wins(agent, obs_batch, device)
        else:
          phi_prev = potential_self_vec(obs_batch, win_squash)
        t1b = time.perf_counter() if _tm is not None else None
        step_arrays = envs.step_np(acts, reset_if_done=True)
        t2 = time.perf_counter() if _tm is not None else None
        if FLAGS.league:
          _refresh_lineups(agent, matchmaker, roster, agent_fn, num_actions,
                           input_shape, device, step_arrays.dones)
        shaped = np.zeros(FLAGS.num_envs, dtype=np.float32)
        if FLAGS.shaping and phi_mode != "none":
          seats = np.asarray(agent.last_seats)
          new_seats = step_arrays.seats.astype(np.int64)
          not_done = ~step_arrays.dones
          if phi_learned:
            # Learned potential: win-value of whoever is to act next. Computed
            # from each row's own observation at s' (same-scale as the mover's
            # own obs at s), so the telescope is an approximation.
            phi_next = _phi_wins(agent, step_arrays.obs, device)
            shaped[not_done] = FLAGS.gamma * phi_next[not_done] - phi_prev[not_done]
          else:
            # Vectorized: one obs gather for all envs' next potential instead
            # of a per-env Python loop.
            blocks = OPP_BASE + np.where(seats < new_seats, seats, seats - 1) * 25
            phi_next = potential_opponent_vec(step_arrays.obs, blocks, win_squash)
            shaped[not_done] = FLAGS.gamma * phi_next[not_done] - phi_prev[not_done]
        t2b = time.perf_counter() if _tm is not None else None
        agent.post_step_np(step_arrays.rewards, step_arrays.dones,
                           shaped_reward=shaped)
        donor_idx = np.flatnonzero(step_arrays.dones)
        for i in donor_idx:
          ret = float(step_arrays.rewards[i][0])
          episode_returns[i].append(ret)
          recent_returns.append(ret)
        t3 = time.perf_counter() if _tm is not None else None
        if _tm is not None:
          _tm[0] += t1 - t0
          _tm[1] += t1b - t0
          _tm[2] += t2 - t1b
          _tm[3] += t2b - t2
          _tm[4] += t3 - t2b
      agent.learn_np(step_arrays.obs, step_arrays.seats)
      if FLAGS.anneal_lr:
        agent.anneal_learning_rate(update, num_updates)
      if _tm is not None and update % FLAGS.timing_every == 0:
        _emit(f"[timing u{update}] act={_tm[0]*1e3*_tm_scale:.2f}ms/env"
              f"  act+phi={_tm[1]*1e3*_tm_scale:.2f}  env={_tm[2]*1e3*_tm_scale:.2f}"
              f"  shape+refresh={_tm[3]*1e3*_tm_scale:.2f}"
              f"  post={_tm[4]*1e3*_tm_scale:.2f}"
              f"  total={_tm.sum()*1e3*_tm_scale:.2f}")
      if pbar is not None:
        pbar.update(FLAGS.num_envs * FLAGS.num_steps)
        _pbar_postfix()
      if FLAGS.league:
        _maybe_snapshot(agent, roster, update)
      if update % FLAGS.eval_every == 0:
        _log_update(agent, episode_returns, recent_returns, writer, update)
        if FLAGS.verdict_every_sec and time.time() - _last_verdict_ts[0] >= \
            FLAGS.verdict_every_sec:
          _last_verdict_ts[0] = time.time()
          _emit(f"[verdict] gate at update {update} "
                f"(steps={agent.total_steps_done}, "
                f"elapsed={time.time() - _run_t0:.0f}s)")
          _run_verdict(agent, roster, agent_fn, num_actions, input_shape,
                       device,
                       _randomized_game_string(FLAGS.game,
                                               FLAGS.seed + update * 7),
                       num_players, writer, agent.total_steps_done)
        if FLAGS.eval_squad and roster is not None:
          res = _eval_squad(
              agent, roster, agent_fn, num_actions, input_shape, device,
              _randomized_game_string(FLAGS.game, FLAGS.seed + update * 7),
              num_players, num_games=8, rng_seed=FLAGS.seed + update * 7)
          if res is not None:
            wins, games, ranks = res
            avg = float(np.mean(ranks))
            _emit(f"  [squad] main win-rate {wins}/{games}  avg_rank={avg:.2f}")
            writer.add_scalar("squad/main_win_rate", wins / games,
                              agent.total_steps_done)
            writer.add_scalar("squad/avg_rank", avg, agent.total_steps_done)
    if pbar is not None:
      pbar.close()
    envs.close()
  else:
    time_step = envs.reset(players="current")
    for update in range(num_updates):
      for step in range(FLAGS.num_steps):
        agent_output = agent.step(time_step)
        # phi(s) for the acting seat from its own obs (this row).
        # Uses the CPU numpy obs batch already gathered by agent.step (no
        # second GPU->CPU round trip).
        obs_batch = agent.last_obs_batch
        if FLAGS.shaping and phi_learned:
          phi_prev = _phi_wins(agent, obs_batch, device)
        else:
          phi_prev = np.fromiter(
              (potential_self(obs_batch[i], win_squash)
               for i in range(FLAGS.num_envs)),
              dtype=np.float32, count=FLAGS.num_envs)

        time_step, reward, done, unreset = envs.step(
            agent_output, reset_if_done=True, players="current")
        if FLAGS.league:
          _refresh_lineups(agent, matchmaker, roster, agent_fn, num_actions,
                           input_shape, device, done)

        shaped = np.zeros(FLAGS.num_envs, dtype=np.float32)
        if FLAGS.shaping and phi_mode != "none":
          seats = agent.last_seats
          if phi_learned:
            new_obs = np.stack([
                ts.observations["info_state"][ts.observations["current_player"]]
                for ts in time_step
            ], axis=0)
            phi_next = _phi_wins(agent, new_obs, device)
            for i in range(FLAGS.num_envs):
              if done[i]:
                continue
              shaped[i] = FLAGS.gamma * phi_next[i] - phi_prev[i]
          else:
            for i, ts in enumerate(time_step):
              if done[i]:
                continue
              viewer = ts.observations["current_player"]
              seat = seats[i]
              k = opponent_block_index(seat, viewer)
              phi_next = potential_opponent(
                  ts.observations["info_state"][viewer], OPP_BASE + k * 25,
                  win_squash)
              shaped[i] = FLAGS.gamma * phi_next - phi_prev[i]

        agent.post_step(reward, done, shaped_reward=shaped.tolist())

        # Episode return logging.
        for i, ts in enumerate(unreset):
          if ts.last():
            ret = float(ts.rewards[0])
            episode_returns[i].append(ret)
            recent_returns.append(ret)

      agent.learn(time_step)

      if FLAGS.anneal_lr:
        agent.anneal_learning_rate(update, num_updates)

      if pbar is not None:
        pbar.update(FLAGS.num_envs * FLAGS.num_steps)
        _pbar_postfix()

      if FLAGS.league:
        _maybe_snapshot(agent, roster, update)

      if update % FLAGS.eval_every == 0:
        _log_update(agent, episode_returns, recent_returns, writer, update)
        if FLAGS.eval_squad and roster is not None:
          res = _eval_squad(
              agent, roster, agent_fn, num_actions, input_shape, device,
              _randomized_game_string(FLAGS.game, FLAGS.seed + update * 7),
              num_players, num_games=8, rng_seed=FLAGS.seed + update * 7)
          if res is not None:
            wins, games, ranks = res
            avg = float(np.mean(ranks))
            _emit(f"  [squad] main win-rate {wins}/{games}  avg_rank={avg:.2f}")
            writer.add_scalar("squad/main_win_rate", wins / games,
                              agent.total_steps_done)
            writer.add_scalar("squad/avg_rank", avg, agent.total_steps_done)

  # Sequential-exploiter closeout: report the win-rate vs the frozen victim and
  # optionally fold the trained policy into the roster.
  if FLAGS.exploit_victim and exploit_victim_net is not None:
    wins, games, _ = _eval_head2head(
        agent, exploit_victim_net, agent_fn, num_actions, input_shape, device,
        _randomized_game_string(FLAGS.game, FLAGS.seed + 777), num_players,
        num_games=16, rng_seed=FLAGS.seed + 777)
    _emit(f"[exploiter] vs victim {FLAGS.exploit_victim}: "
          f"win-rate {wins}/{games}")
    writer.add_scalar("exploiter/victim_win_rate", wins / games,
                      agent.total_steps_done)
    if FLAGS.exploit_promote and wins / games >= 0.5 and roster is not None:
      roster.add_exploiter(agent.network, agent.updates_done,
                           FLAGS.exploit_victim, win_rate=wins / games)
      _emit(f"[exploiter] promoted to roster")

  if pbar is not None:
    pbar.close()
  writer.close()
  _emit("pilot done")


if __name__ == "__main__":
  app.run(main)
