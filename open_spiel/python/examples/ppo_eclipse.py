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

Reward shaping (default on, --phi=soft) adds banked VP plus in-progress presence
terms (colony ships, disks on sectors, orbitals/monoliths, ambassadors), read
straight out of the observation. Shaping is skipped on the terminal transition,
where the true payoff is used.

`soft` is the Sprint-B3 grid's pick: highest VP in both seeds and the best mean
utility against the stronger (Greedy) baseline. It is *not* policy-invariant, and
that seems to be the point -- the invariant variant (--phi=telescope, which
differences a potential across each seat's own consecutive decisions and so is the
only one that truly telescopes against the per-own-decision gamma in the
self-play GAE) measured mid-pack, because invariance by construction cannot
supply inductive bias. See the --phi flag for the full grid and, importantly, for
why 2 seeds resolve much less than the eval intervals suggest.

This addresses the sparse terminal reward problem without access to
expert/human demonstration data.
"""

import collections
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
from open_spiel.python.pytorch.ppo import rank_of
from open_spiel.python.pytorch.ppo import DEFAULT_RANK_UTILITY as RANK_UTILITY_TABLE
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
flags.DEFINE_enum("phi", "soft",
                  ["banked", "soft", "none", "learned", "telescope"],
                  "Potential definition. Adjudicated by the Sprint-B3 grid "
                  "(5 shapings x 18 min, 2 seeds, judged on batched held-out "
                  "evals). Read the caveat below before trusting any ranking.\n"
                  "'soft' (default) = banked VP plus in-progress presence terms "
                  "(colony ships, disks on sectors, orbitals/monoliths, "
                  "ambassadors). Highest vp_all in *both* seeds (13.00, 9.09) "
                  "and the best mean utility vs Greedy (0.503). Not "
                  "policy-invariant -- it biases toward expansion -- which "
                  "appears to be useful inductive bias here.\n"
                  "'telescope' = banked-VP potential differenced across a seat's "
                  "own consecutive decisions. The only variant that actually "
                  "telescopes against the per-own-decision gamma in the self-play "
                  "GAE, so the only policy-invariant one. It was briefly the "
                  "default on that theoretical basis, but invariance means it "
                  "cannot supply inductive bias, and it measured mid-pack with "
                  "high variance (Greedy 0.279 / 0.543 across seeds).\n"
                  "'banked' = current VP if the game ended now, differenced "
                  "across env steps (does not telescope). Mid-pack but by far the "
                  "most seed-stable (0.430 / 0.434), which makes it the right "
                  "control for architecture A/Bs.\n"
                  "'learned' = the network's own win-value at the *next* acting "
                  "seat's state minus the mover's own -- two different players' "
                  "values, so not a potential difference at all. REFUTED: worst "
                  "vs Greedy and lowest vp_all in all 3 runs it appeared in, "
                  "while having the *highest* survival, i.e. it learns not to die "
                  "without learning to score. It was the pre-Sprint-A default, "
                  "chosen on a metric that could not see VP.\n"
                  "'none' disables shaping. Mid-pack.\n"
                  "CAVEAT: within-run eval intervals are ~+-0.06 but run-to-run "
                  "variance is +-0.17..0.34, so 2 seeds separate only 'learned is "
                  "worst' and 'soft scores most'. Resolving the rest needs ~8-10 "
                  "seeds per cell.")
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
flags.DEFINE_float(
    "rank_vp_beta", 0.002,
    "Slope (utility per VP) of the VP escape bonus added to the terminal "
    "rank utility in --value_mode=win. Tie-averaged rank utility removes the "
    "reward for mutual bankruptcy but leaves the objective flat while every "
    "game still ends all-tied at 0 VP (zero return variance -> zero "
    "advantage); this term supplies a gradient inside that dead zone. It is "
    "clamped so it can never reorder two placements. 0 disables it.")
flags.DEFINE_float(
    "rank_vp_beta_anneal_to", -1.0,
    "If >= 0, linearly anneal --rank_vp_beta to this value over the run, "
    "recovering the pure constant-sum 'finish first' objective once real "
    "outcomes differ. -1 keeps beta constant.")
flags.DEFINE_float(
    "rank_ce_coef", 0.0,
    "Weight of a cross-entropy loss on the *realized* placement, supervising "
    "the 4-output rank critic as the distributional head it is. Without it those "
    "4 logits are trained only through the MSE of the scalar expected utility "
    "they collapse to, which discards the placement distribution. 0 = off; the "
    "C3 A/B turns it on.")
flags.DEFINE_float(
    "aux_coef", 0.1,
    "Weight of auxiliary-head losses. Measured at this value on a minibatch of "
    "rows carrying aux targets, the gradient-norm share is aux 82% / policy 11% "
    "/ value 6%, i.e. the shared trunk is trained mostly as a rank predictor. "
    "That looked like a problem, but an A/B at 0.01 (aux share 0.31) left "
    "approx_kl and clipfrac unchanged and was slightly *behind* on VP at matched "
    "steps, so the dominance is not in fact suppressing policy learning and the "
    "value is left as-is pending the Sprint-B sweep. Two things worth knowing if "
    "you tune it: the mechanism would be direct gradient dominance on the shared "
    "trunk, not grad-norm clipping (the combined norm ~0.34 stays under "
    "--max_grad_norm 0.5, so clipping never engages); and the aux loss is a mean "
    "over *masked* rows, so its effective weight scales with how many episodes "
    "closed in the batch.")
flags.DEFINE_enum(
    "aux_target_mode", "rank", ["vp", "rank", "both", "none"],
    "Aux-head regression target. 'rank' (default) = per-seat tie-aware rank "
    "utility: bounded in [-0.5, 1] by construction, so it can neither vanish "
    "nor dominate. 'vp' = final VP / --aux_vp_scale. 'both' supervises a "
    "normalized-VP head and a rank head. 'none' disables aux heads. Note the "
    "target must be O(1): raw VP targets gave aux_loss ~8-11 against pg_loss "
    "~1e-3, and because the gradient is clipped globally that rescaled the "
    "policy gradient toward zero.")
flags.DEFINE_float(
    "aux_vp_scale", 30.0,
    "Divisor for the 'vp' aux target. A contested Eclipse game scores ~20-40 "
    "VP, so /30 keeps the target O(1); the old /200 (and, after a regression, "
    "no divisor at all) made it either invisible or overwhelming.")
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
flags.DEFINE_bool(
    "nn_norm", False,
    "LayerNorm after each hidden layer. Note running *observation* "
    "normalization is deliberately absent: the Eclipse observation tensor was "
    "measured to lie entirely within [0,1] (one-hots and write_frac ratios), so "
    "there is no input-scale problem for it to fix.")
flags.DEFINE_enum("nn_activation", "tanh", ["tanh", "gelu"],
                  "Hidden activation. Tanh saturates; gelu is the C1 default "
                  "for the wider trunk.")
flags.DEFINE_bool(
    "factored_actions", False,
    "Replace the flat Linear(width, 11117) actor head with a sum of factor "
    "embeddings recovered from the engine's own action layout. The flat head is "
    "80-86% of all parameters and shares nothing between, say, the 5400 "
    "colony-ship actions; the factored head uses 1420 rows (12.8%) and lets "
    "knowledge about a galaxy cell transfer to every action targeting it. The "
    "decode is injective, so no two actions become indistinguishable.")
flags.DEFINE_bool(
    "separate_critic", False,
    "Give the value/aux heads their own trunk. With one shared 64-wide trunk "
    "the measured gradient split on aux-bearing rows was aux 82% / policy 11% / "
    "value 6%, so the representation the policy reads was shaped mostly by the "
    "regression heads.")
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
flags.DEFINE_bool(
    "eval_batched", True,
    "Run verdict evals through AsyncVectorEnv instead of one fresh single-game "
    "environment per game. The old path did one 1-sample forward per decision, "
    "which is why --eval_games stayed at 8 (+-0.18 on a win rate, unable to "
    "separate any two configurations).")
flags.DEFINE_integer(
    "eval_envs", 64,
    "Parallel environments used by the batched evaluator.")
flags.DEFINE_integer(
    "eval_games", 32,
    "Games per baseline in the verdict eval. 8 (the old value) gives a +-0.18 "
    "standard error on a win rate, which cannot separate configurations; the "
    "reported bootstrap interval makes the remaining noise explicit.")
flags.DEFINE_integer(
    "eval_seed_offset", 7777,
    "Eval boards are drawn from FLAGS.seed + this fixed offset, so evals at "
    "different points in training are paired on the same held-out boards "
    "rather than re-rolling the galaxy each time.")
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


class FactoredActorHead(nn.Module):
  """Actor head whose per-action weight is a sum of factor embeddings.

  ``W[a] = sum_slot E[decode[a, slot]]``, so the 5,400 colony-ship actions share
  225 cell rows, 8 slot rows and 3 track rows instead of carrying 5,400
  independent weight vectors. See eclipse/action_factors.py for how the decode
  table is recovered from the engine.

  Interface-compatible with the ``nn.Linear`` it replaces: ``forward`` yields all
  logits (needed for dense/eval paths) and ``rows_for`` yields just the rows the
  sparse path asks for, which is the point -- it never materializes the full
  (num_actions, width) matrix during training.
  """

  def __init__(self, decode, num_rows, num_actions, width):
    super().__init__()
    self.num_actions = num_actions
    self.out_features = num_actions
    self.width = width
    # Same scale as the flat head's layer_init(std=0.01), divided over the slots
    # that sum into each action's weight so the resulting logits keep that scale.
    slots = decode.shape[1]
    self.embedding = nn.Parameter(
        torch.randn(num_rows, width) * (0.01 / np.sqrt(slots)))
    self.bias = nn.Parameter(torch.zeros(num_actions))
    self.register_buffer("decode", torch.from_numpy(decode.astype(np.int64)))

  def rows_for(self, idx):
    """(len(idx), width) weight rows for the given action ids."""
    return self.embedding[self.decode[idx]].sum(dim=1)

  def full_weight(self):
    return self.embedding[self.decode].sum(dim=1)

  def forward(self, features):
    return features @ self.full_weight().t() + self.bias


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

  @staticmethod
  def _trunk(in_features, width, depth, norm, activation):
    """MLP trunk. LayerNorm + GELU by default.

    Tanh saturates and, at width 64, the original trunk had to serve the policy,
    a 4-way rank critic and the aux heads simultaneously. Note that running
    observation normalization is deliberately *not* added: the Eclipse
    observation tensor was measured to lie entirely within [0, 1] (every slot is
    a one-hot or a write_frac ratio), so there is no input-scale problem to fix,
    and LayerNorm handles the rest.
    """
    layers = []
    for _ in range(depth):
      layers.append(layer_init(nn.Linear(in_features, width)))
      if norm:
        layers.append(nn.LayerNorm(width))
      layers.append(nn.GELU() if activation == "gelu" else nn.Tanh())
      in_features = width
    return nn.Sequential(*layers)

  def __init__(self, num_actions, observation_shape, device, width=64,
               depth=2, aux_tasks=("final_vp",), norm=False,
               activation="tanh", separate_critic=False,
               factored_actions=None):
    super().__init__()
    in_features = int(np.array(observation_shape).prod())
    self.shared = self._trunk(in_features, width, depth, norm, activation)
    self.separate_critic = separate_critic

    if separate_critic:
      # An independent trunk for the value/aux objectives. With a shared trunk
      # the measured gradient split on aux-bearing rows was aux 82% / policy 11%
      # / value 6%, i.e. the representation the policy reads was being shaped
      # mostly by the regression heads.
      self.critic_trunk = self._trunk(in_features, width, depth, norm,
                                      activation)
    else:
      self.critic_trunk = self.shared

    self.critic = nn.Sequential(
        self.critic_trunk,
        layer_init(nn.Linear(width, 4), std=1.0),
    )
    if factored_actions is None:
      actor_head = layer_init(nn.Linear(width, num_actions), std=0.01)
    else:
      actor_head = FactoredActorHead(
          factored_actions.decode, factored_actions.num_rows, num_actions,
          width)
    self.actor = nn.Sequential(self.shared, actor_head)
    # Auxiliary heads hang off the critic trunk (they are terminal-outcome
    # regressions, same job as the value head).
    self.aux_heads = nn.ModuleDict({
        name: layer_init(nn.Linear(width, 1), std=1.0)
        for name in (aux_tasks or ())
    })
    self.num_actions = num_actions
    self.device = device
    self.register_buffer("mask_value", torch.tensor(-1e6))

  def get_value(self, x):
    return self.rank_value(self.critic(x))

  def rank_logits_from_obs(self, x):
    """(B, 4) rank logits -- the distributional critic's raw output."""
    return self.critic(x)

  def value_from_obs(self, x):
    """Scalar value straight from observations.

    Required when the critic has its own trunk: the sparse paths compute actor
    features once and would otherwise feed those to the value head.
    """
    return self.rank_value(self.critic(x))

  def aux_from_obs(self, x):
    feats = self.critic_trunk(x)
    return {name: head(feats) for name, head in self.aux_heads.items()}

  def rank_value(self, rank_logits):
    """Expected rank-utility from (..., 4) rank logits."""
    probs = rank_logits.softmax(dim=-1)
    utility = torch.tensor(self.RANK_UTILITY, dtype=rank_logits.dtype,
                           device=rank_logits.device)
    return (probs * utility).sum(dim=-1)

  def value_from_features(self, features):
    """Scalar win value from shared features (sparse learn path)."""
    return self.rank_value(self.critic[-1](features))

  def value_bounds(self):
    """(min, max) representable value.

    ``rank_value`` is a convex combination of RANK_UTILITY, so the critic is
    *hard bounded* to [-0.5, 1.0]. Any value target outside that band is
    unfittable no matter how long training runs, which caps explained variance
    and corrupts every advantage derived from it. PPO reports the out-of-band
    fraction so a shaping choice that pushes returns out of range is visible.
    """
    return min(self.RANK_UTILITY), max(self.RANK_UTILITY)

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


def make_agent_fn(width, depth, aux_tasks=("final_vp",), norm=False,
                  activation="tanh", separate_critic=False,
                  factored_actions=None):
  def agent_fn(num_actions, observation_shape, device):
    return EclipsePPOAgent(num_actions, observation_shape, device,
                           width=width, depth=depth, aux_tasks=aux_tasks,
                           norm=norm, activation=activation,
                           separate_critic=separate_critic,
                           factored_actions=factored_actions)

  return agent_fn


# Aux-head names by --aux_target_mode, and the per-seat target each produces.
_AUX_TASKS_BY_MODE = {
    "vp": ("final_vp",),
    "rank": ("final_rank",),
    "both": ("final_vp", "final_rank"),
    "none": (),
}


def build_aux_targets(mode, vp_scale):
  """(task names, target fn) for ``--aux_target_mode``.

  The target fn maps a terminal per-seat payoff vector to a
  (num_players, num_tasks) matrix. Targets must be O(1): the gradient is clipped
  globally, so an aux term far larger than the policy term rescales the policy
  gradient toward zero. Raw-VP targets (the state this replaces) reached
  aux_loss ~8-11 against pg_loss ~1e-3.
  """
  tasks = _AUX_TASKS_BY_MODE[mode]
  if not tasks:
    return None, None

  def target_fn(rvec):
    arr = np.asarray(rvec, dtype=np.float32)
    cols = []
    for task in tasks:
      if task == "final_vp":
        cols.append(arr / float(vp_scale))
      else:  # final_rank: bounded in [-0.5, 1] by construction.
        cols.append(np.asarray(
            [rank_utility(arr, s) for s in range(arr.shape[0])],
            dtype=np.float32))
    return np.stack(cols, axis=1)

  return list(tasks), target_fn


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
  """Block index of `seat` within `viewer`'s opponent blocks, or None.

  Returns None when ``seat == viewer``: that seat occupies the *self* block, not
  an opponent block. Previously this returned ``seat - 1`` in that case, reading
  a different player's slots -- and for seat 0, index -1, i.e. ``OPP_BASE - 25``,
  which is unrelated observation memory. The same-seat case is common in Eclipse
  because macro actions (explore -> place -> rotate, build/upgrade/move
  internals) keep one seat acting for several consecutive steps.
  """
  if seat == viewer:
    return None
  return seat if seat < viewer else seat - 1


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
ACTION_MOVE_START = 9142  # verified via action_to_string; 8502 is an UPGRADE part
ACTION_COLONY_START = 332
ACTION_COLONY_END = 5731
ACTION_PASS = 0

GALAXY_BASE = 45 + 135 + 125  # obs block C starts after A+B0+5 opponent blocks.
CELL_STRIDE = 6


# Observation slots used by the episode diagnostics (see eclipse.cc layout):
#   A + 0        round / 8
#   B0 + 7       self "eliminated"       -> index 52
#   Bn + 2       opponent n "eliminated" -> OPP_BASE + block*25 + 2
ROUND_SLOT = 0
SELF_ELIM_SLOT = 45 + 7
OPP_ELIM_OFFSET = 2
MAX_ROUNDS = 8


class EpisodeDiagnostics:
  """Why episodes end, reconstructed from the acting seat's observation.

  The single most important fact about Eclipse self-play was invisible in the
  old telemetry: unskilled play does not merely score 0, it goes *bankrupt* --
  every seat eliminated in round 1-2, after which the remaining rounds are
  empty. `mean_episode_return` (seat-0 VP) could not distinguish that from a
  contested game, and after the A0 engine fix it no longer even shows up as a
  zero return, because eliminated seats now score what they had banked.

  Everything here comes from the observation the trainer already gathers: every
  seat's `eliminated` bit is visible in the acting seat's own view (its own slot
  plus one per opponent block), and the round counter is in the global block. So
  no engine change and no extra env traffic is needed.

  Tracked per episode: the round at which each seat was eliminated (elimination
  is monotone, so first-seen is the answer), and the furthest round reached.
  """

  def __init__(self, num_envs, num_players, history=400):
    self.num_envs = num_envs
    self.num_players = num_players
    self.elim_round = np.zeros((num_envs, num_players), dtype=np.int16)
    self.elim_round.fill(-1)          # -1 = still alive
    self.max_round = np.zeros(num_envs, dtype=np.int16)
    self.survivors = collections.deque(maxlen=history)
    self.elim_rounds = collections.deque(maxlen=history)
    self.rounds_reached = collections.deque(maxlen=history)
    self.all_seat_vp = collections.deque(maxlen=history)
    self.wipeouts = collections.deque(maxlen=history)
    self._cols = None

  def _elim_columns(self, seats):
    """(num_players, num_envs) obs column holding each seat's eliminated bit."""
    seat_ids = np.arange(self.num_players, dtype=np.int64)[:, None]
    viewers = np.asarray(seats, dtype=np.int64)[None, :]
    block = seat_ids - (seat_ids > viewers).astype(np.int64)
    return np.where(seat_ids == viewers, SELF_ELIM_SLOT,
                    OPP_BASE + block * 25 + OPP_ELIM_OFFSET)

  def observe(self, obs_batch, seats):
    """Folds one step's observations into the per-episode trackers."""
    obs = np.asarray(obs_batch)
    rounds = np.rint(obs[:, ROUND_SLOT] * MAX_ROUNDS).astype(np.int16)
    np.maximum(self.max_round, rounds, out=self.max_round)
    cols = self._elim_columns(seats)
    rows = np.arange(obs.shape[0], dtype=np.int64)[None, :]
    flags = obs[rows, cols] > 0.5          # (num_players, num_envs)
    newly = flags.T & (self.elim_round < 0)
    if newly.any():
      self.elim_round[newly] = np.broadcast_to(
          rounds[:, None], self.elim_round.shape)[newly]

  def close_episodes(self, done_idx, rewards):
    """Records finished episodes and resets their trackers."""
    for i in done_idx:
      i = int(i)
      elim = self.elim_round[i]
      alive = int(np.sum(elim < 0))
      self.survivors.append(alive)
      self.wipeouts.append(1 if alive == 0 else 0)
      self.elim_rounds.append(
          float(np.mean(np.where(elim < 0, MAX_ROUNDS, elim))))
      self.rounds_reached.append(int(self.max_round[i]))
      self.all_seat_vp.append(np.asarray(rewards[i], dtype=np.float32).copy())
      self.elim_round[i] = -1
      self.max_round[i] = 0

  def summary(self):
    if not self.survivors:
      return None
    vp = np.stack(self.all_seat_vp)
    return {
        "wipeout_rate": float(np.mean(self.wipeouts)),
        "survivors": float(np.mean(self.survivors)),
        "mean_elim_round": float(np.mean(self.elim_rounds)),
        "rounds_reached": float(np.mean(self.rounds_reached)),
        "vp_all_seats_mean": float(vp.mean()),
        "vp_all_seats_max": float(vp.max(axis=1).mean()),
        "episodes": len(self.survivors),
    }


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


def _train_state_path(roster_dir):
  return os.path.join(roster_dir, "train_state.pt")


def _save_train_state(agent, roster_dir):
  """Persists optimizer state and counters alongside the weights.

  ``record_main`` stores only ``net.state_dict()``, so resuming restarted Adam
  from scratch and reset the step counter -- which also restarted the LR
  annealing schedule at full LR. The Sprint-1 grid resumed twice, so every cell
  in it ran on a sawtooth LR schedule.
  """
  torch.save({
      "optimizer": agent.optimizer.state_dict(),
      "total_steps_done": agent.total_steps_done,
      "updates_done": agent.updates_done,
      "rank_vp_beta": agent.rank_vp_beta,
      "learning_rate": agent.learning_rate,
  }, _train_state_path(roster_dir))


def _load_train_state(agent, roster_dir):
  """Restores optimizer/counters if a train_state.pt is present."""
  path = _train_state_path(roster_dir)
  if not os.path.exists(path):
    return False
  state = torch.load(path, map_location=agent.device, weights_only=False)
  agent.optimizer.load_state_dict(state["optimizer"])
  agent.total_steps_done = int(state.get("total_steps_done", 0))
  agent.updates_done = int(state.get("updates_done", 0))
  if "rank_vp_beta" in state:
    agent.rank_vp_beta = float(state["rank_vp_beta"])
    agent.rank_vp_beta_initial = agent.rank_vp_beta
  return True


def _maybe_snapshot(agent, roster, update, force=False):
  """Captures the main policy into the roster on the snapshot cadence.

  Reachable without --league too: a plain self-play run previously trained for
  hours and wrote no weights at all, because this was only ever called from the
  league branch.
  """
  if roster is None or FLAGS.snapshot_every <= 0:
    return
  if force or (update > 0 and update % FLAGS.snapshot_every == 0):
    roster.record_main(agent.network, update)
    roster.add_snapshot(agent.network, update)
    roster.prune(keep_recent=4, keep_spaced=4)
    _save_train_state(agent, str(roster.save_dir))


# Result of an evaluation match set. ``utils`` is the per-game mean tie-aware
# rank utility over main's seats -- the quantity the training objective actually
# optimizes, and the one to judge progress on. ``ranks`` (main's best placement)
# is kept for continuity with earlier logs but is a much blunter instrument: it
# reports 1 for any game nobody strictly beat main in, which for Eclipse means a
# 0-0-0-0 mutual-bankruptcy game scores as a win.
EvalResult = collections.namedtuple(
    "EvalResult", ["wins", "games", "ranks", "utils"])


def main_outcome(rewards, main_seats):
  """(mean rank utility over main's seats, main's best placement) for one game.

  Uses the shared tie-aware ``rank_utility`` with no VP escape bonus: evaluation
  measures the true constant-sum objective, not the training-time nudge.
  """
  utility = float(np.mean(
      [rank_utility(rewards, s) for s in main_seats]))
  best_rank = min(rank_of(rewards, s) for s in main_seats)
  return utility, best_rank


def chance_utility(num_players):
  """Mean rank utility of an *equal-strength* policy.

  Because tie-aware rank utility is constant-sum, every seat averages
  ``sum(table) / num_players`` under symmetry -- independent of how many seats
  main occupies. This is the null hypothesis every strength number must clear.
  """
  table = RANK_UTILITY_TABLE[:num_players]
  return sum(table) / max(1, num_players)


def mean_ci(values, num_boot=2000, seed=0, alpha=0.05):
  """(mean, lo, hi) bootstrap percentile interval for the mean of ``values``."""
  arr = np.asarray(values, dtype=np.float64)
  if arr.size == 0:
    return float("nan"), float("nan"), float("nan")
  if arr.size == 1:
    return float(arr[0]), float("nan"), float("nan")
  rng = np.random.RandomState(seed)
  idx = rng.randint(0, arr.size, size=(num_boot, arr.size))
  means = arr[idx].mean(axis=1)
  return (float(arr.mean()),
          float(np.percentile(means, 100.0 * alpha / 2.0)),
          float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))))


def _fmt_eval(label, res, num_players):
  """One-line report: utility vs the chance level, with an interval."""
  mean, lo, hi = mean_ci(res.utils)
  chance = chance_utility(num_players)
  beats = "" if np.isnan(lo) else ("  BEATS-CHANCE" if lo > chance else
                                   ("  BELOW-CHANCE" if hi < chance else
                                    "  inconclusive"))
  ci = "" if np.isnan(lo) else f" [{lo:+.3f},{hi:+.3f}]"
  return (f"  [verdict] vs {label:<8s} utility={mean:+.3f}{ci} "
          f"(chance {chance:+.3f}, n={res.games})  "
          f"best_rank={np.mean(res.ranks):.2f}  win={res.wins}/{res.games}"
          f"{beats}")


def _argmax_over_legal(net, obs_np, legal_rows, legal_cols, idx, device):
  """Greedy action per row of ``idx``, restricted to that row's legal set.

  One batched forward for the whole group; illegal logits are masked to -inf
  before the argmax. The dense mask is affordable here (eval batches are small
  and this runs at eval cadence, not in the rollout hot loop).
  """
  with torch.no_grad():
    x = torch.from_numpy(obs_np[idx]).to(device)
    logits = net.actor(x)
    mask = torch.full_like(logits, float("-inf"))
    local = np.full(obs_np.shape[0], -1, dtype=np.int64)
    local[idx] = np.arange(len(idx))
    keep = local[legal_rows] >= 0
    rows = torch.from_numpy(local[legal_rows[keep]]).to(device)
    cols = torch.from_numpy(legal_cols[keep].astype(np.int64)).to(device)
    mask[rows, cols] = logits[rows, cols]
    return mask.argmax(dim=1).detach().cpu().numpy()


def evaluate_batched(policies, lineup, game_strs, num_players, num_games,
                     num_workers, device, main_seats, max_legal):
  """Plays ``num_games`` complete games in parallel and scores main's outcomes.

  Replaces the per-game single-env evaluators, which built a fresh
  ``rl_environment`` per game and ran one 1-sample forward per decision. At 8
  games that gave a +-0.18 standard error on a win rate -- unable to separate any
  two configurations -- and it was ~100x slower than it needed to be, which is
  precisely why the number stayed at 8.

  ``policies`` maps a policy id to either an ``nn.Module`` (driven greedily) or a
  ``bot(obs, legal) -> action`` callable. ``lineup`` is (num_envs, num_players)
  of policy ids. ``game_strs`` fixes the boards, so repeated calls at different
  points in training are paired on the same galaxies.

  Returns an EvalResult whose ``utils`` are per-game mean tie-aware rank
  utilities over ``main_seats``, plus an EpisodeDiagnostics for the eval games.
  """
  num_envs = len(game_strs)
  envs = [
      rl_environment.Environment(
          game=pyspiel.load_game(game_strs[i]),
          chance_event_sampler=rl_environment.ChanceEventSampler(seed=1 + i),
          observation_type=rl_environment.ObservationType.OBSERVATION,
          observations_as_numpy=True)
      for i in range(num_envs)
  ]
  vec = AsyncVectorEnv(envs, num_workers=min(num_workers, num_envs),
                       sampler_seeds=[1 + i for i in range(num_envs)],
                       game_strs=game_strs, max_legal=max_legal)
  diag = EpisodeDiagnostics(num_envs, num_players, history=max(num_games, 1))
  utils, ranks, wins = [], [], 0
  try:
    vec.reset(players="current")
    arrays = vec.reset_np()
    # Bounded: a game is ~150 decisions, so this cannot spin forever if some env
    # stalls -- it exits and reports however many games completed.
    max_steps = 400 * (num_games // max(1, num_envs) + 2)
    for _ in range(max_steps):
      if len(utils) >= num_games:
        break
      seats = arrays.seats.astype(np.int64)
      diag.observe(arrays.obs, seats)
      pids = np.array([lineup[i][seats[i]] for i in range(num_envs)],
                      dtype=object)
      actions = np.zeros(num_envs, dtype=np.int32)
      counts = np.bincount(arrays.legal_rows.astype(np.int64),
                           minlength=num_envs)
      offsets = np.zeros(num_envs, dtype=np.int64)
      np.cumsum(counts[:-1], out=offsets[1:])
      for pid in set(pids.tolist()):
        idx = np.flatnonzero(pids == pid)
        policy = policies[pid]
        if isinstance(policy, nn.Module):
          actions[idx] = _argmax_over_legal(
              policy, arrays.obs, arrays.legal_rows, arrays.legal_cols, idx,
              device)
        else:
          for i in idx:
            legal = arrays.legal_cols[offsets[i]:offsets[i] + counts[i]]
            actions[i] = policy(arrays.obs[i], legal)
      arrays = vec.step_np(actions, reset_if_done=True)
      done_idx = np.flatnonzero(arrays.dones)
      if done_idx.size:
        diag.close_episodes(done_idx, arrays.rewards)
        for i in done_idx:
          if len(utils) >= num_games:
            break
          utility, rank = main_outcome(arrays.rewards[i], main_seats)
          utils.append(utility)
          ranks.append(rank)
          wins += int(rank == 1)
  finally:
    vec.close()
  return EvalResult(wins, len(utils), ranks, utils), diag


def _eval_squad(agent, roster, agent_fn, num_actions, input_shape, device,
                game_str, num_players, num_games, rng_seed, main_seats=(0, 1)):
  """Plays main (argmax) against a snapshots-only squad.

  ``main_seats`` are driven by the main policy; the remaining seats each draw a
  snapshot from the roster. Returns an ``EvalResult``, or None if the roster has
  no opponents. Runs on the network's raw policy (argmax over legal actions) so
  no PPO buffers are touched.
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
  utils = []
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
    utility, rank = main_outcome(rewards, main_seats)
    main_wins += int(rank == 1)
    ranks.append(rank)
    utils.append(utility)
  return EvalResult(main_wins, num_games, ranks, utils)


def _eval_fixed_opponent(agent, bot_pick, game_str, num_players, num_games,
                         rng_seed, main_seats=(0, 1)):
  """Main (argmax) on ``main_seats`` vs one fixed bot policy on every other
  seat. ``bot_pick(obs, legal) -> int``. Returns (main_wins, num_games,
  main_ranks); run on the raw argmax policy, no PPO buffers touched."""
  rng = np.random.RandomState(rng_seed)
  main_wins = 0
  ranks = []
  utils = []
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
    utility, rank = main_outcome(rewards, main_seats)
    main_wins += int(rank == 1)
    ranks.append(rank)
    utils.append(utility)
  return EvalResult(main_wins, num_games, ranks, utils)


def _run_verdict(agent, roster, agent_fn, num_actions, input_shape, device,
                 game_str, num_players, writer, step):
  """Full fail-fast verdict: main {0,1} vs fixed Random, fixed Greedy and the
  snapshot squad. Emits one line per baseline and writes scalars to ``writer``.
  """
  bot_rng = np.random.RandomState(12345)
  num_games = FLAGS.eval_games
  out = {}

  def _record(label, key, res):
    if res is None:
      return
    mean, lo, hi = mean_ci(res.utils)
    out[key] = res
    _emit(_fmt_eval(label, res, num_players))
    writer.add_scalar(f"verdict/{key}_utility", mean, step)
    writer.add_scalar(f"verdict/{key}_avg_rank", float(np.mean(res.ranks)),
                      step)
    writer.add_scalar(f"verdict/{key}_win_rate", res.wins / max(1, res.games),
                      step)
    if not np.isnan(lo):
      writer.add_scalar(f"verdict/{key}_utility_lo", lo, step)
      writer.add_scalar(f"verdict/{key}_utility_hi", hi, step)

  # The eval seed set is fixed (independent of `step`) so measurements at
  # different points in training are paired on the same boards; a seed that
  # moved with the step count added board variance to every comparison.
  eval_seed = FLAGS.seed + FLAGS.eval_seed_offset
  writer.add_scalar("verdict/chance_utility", chance_utility(num_players), step)
  rand_bot = lambda _o, legal: int(
      bot_rng.choice(np.asarray(legal, dtype=np.int32)))
  greedy_bot = lambda obs, legal: _greedy_pick(
      np.asarray(obs, dtype=np.float32), legal, bot_rng)

  if FLAGS.eval_batched:
    # Fixed held-out boards, so evals at different points in training are paired.
    eval_strs = [_randomized_game_string(FLAGS.game, eval_seed + j)
                 for j in range(FLAGS.eval_envs)]
    main_seats = (0, 1)
    def _batched(bot):
      lineup = [[("main" if s in main_seats else "bot")
                 for s in range(num_players)] for _ in range(FLAGS.eval_envs)]
      res, diag = evaluate_batched(
          {"main": agent.network, "bot": bot}, lineup, eval_strs, num_players,
          FLAGS.eval_games, max(1, FLAGS.num_workers), device, main_seats,
          num_actions)
      return res, diag
    if FLAGS.eval_random:
      res, edg = _batched(rand_bot)
      _record("Random", "random", res)
      dstats = edg.summary()
      if dstats:
        _emit(f"    eval-game health: wipeout={dstats['wipeout_rate']:.2f} "
              f"elim_round={dstats['mean_elim_round']:.2f}/8 "
              f"vp_all={dstats['vp_all_seats_mean']:.2f}")
        for k, v in dstats.items():
          if k != "episodes":
            writer.add_scalar(f"verdict_health/{k}", v, step)
    if FLAGS.eval_greedy:
      res, _ = _batched(greedy_bot)
      _record("Greedy", "greedy", res)
    return out

  if FLAGS.eval_random:
    _record("Random", "random",
            _eval_fixed_opponent(agent, rand_bot, game_str, num_players,
                                 num_games=num_games, rng_seed=eval_seed))
  if FLAGS.eval_greedy:
    _record("Greedy", "greedy",
            _eval_fixed_opponent(agent, greedy_bot, game_str, num_players,
                                 num_games=num_games, rng_seed=eval_seed))
  if roster is not None:
    _record("Squad", "squad",
            _eval_squad(agent, roster, agent_fn, num_actions, input_shape,
                        device, game_str, num_players, num_games=num_games,
                        rng_seed=eval_seed))
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
  utils = []
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
    utility, rank = main_outcome(rewards, (0,))
    main_wins += int(rank == 1)
    ranks.append(rank)
    utils.append(utility)
  return EvalResult(main_wins, num_games, ranks, utils)


def _log_update(agent, episode_returns, recent_returns, writer, update,
                eval_every=None, diag=None):
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
    if metrics.get("aux_share") is not None:
      # Fraction of the total loss magnitude that is the aux term. The grad-norm
      # clip is global, so a large share silently shrinks the policy gradient.
      parts.append(f"aux_share={metrics['aux_share']:.2f}")
    parts.append(f"approx_kl={metrics['kl']:.4f}")
    parts.append(f"clipfrac={metrics['clipfrac']:.3f}")
    parts.append(f"explained_var={metrics['explained_variance']:.3f}")
    losses = "  " + "  ".join(parts)
  # Headline health line: why episodes are ending. `mean_episode_return`
  # (seat-0 VP) cannot distinguish "everyone went bankrupt in round 2" from a
  # contested game, and after the eliminated-player scoring fix a wipeout no
  # longer even reads as a zero return.
  health = ""
  dstats = diag.summary() if diag is not None else None
  if dstats:
    health = (f"  wipeout={dstats['wipeout_rate']:.2f}"
              f"  survivors={dstats['survivors']:.2f}/{agent.num_players}"
              f"  elim_round={dstats['mean_elim_round']:.2f}/8"
              f"  vp_all={dstats['vp_all_seats_mean']:.2f}"
              f"  vp_best={dstats['vp_all_seats_max']:.2f}")
  _emit(f"[update {update}] steps={agent.total_steps_done}"
        f"  total_episodes={n_completed}{health}{summary}{losses}")
  if writer is not None:
    writer.add_scalar("charts/num_episodes", n_completed,
                      agent.total_steps_done)
    if recent:
      writer.add_scalar("charts/mean_episode_return", np.mean(recent),
                        agent.total_steps_done)
      writer.add_scalar("charts/nonzero_episodes", nonzero,
                        agent.total_steps_done)
    if dstats:
      for key in ("wipeout_rate", "survivors", "mean_elim_round",
                  "rounds_reached", "vp_all_seats_mean", "vp_all_seats_max"):
        writer.add_scalar(f"health/{key}", dstats[key], agent.total_steps_done)


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
        # Full action space: Eclipse decision nodes reach ~130 legal actions
        # mid-game while the initial state has ~13, so any probed/guessed
        # bound silently drops the high-id action blocks (MOVE, UPGRADE).
        max_legal=game.num_distinct_actions(),
    )
    game = envs_list[0]._game  # pylint: disable=protected-access
  else:
    envs = SyncVectorEnv(envs_list)
    game = envs.envs[0]._game  # pylint: disable=protected-access
  input_shape = tuple(game.observation_tensor_shape())
  num_players = game.num_players()

  factored = None
  if FLAGS.factored_actions:
    from open_spiel.python.eclipse.action_factors import factorization_from_game
    factored = factorization_from_game(game)
    _emit(f"factored actor head: {factored.summary()}")

  aux_tasks, aux_target_fn = build_aux_targets(
      FLAGS.aux_target_mode if FLAGS.aux_coef > 0 else "none",
      FLAGS.aux_vp_scale)
  if aux_target_fn is not None:
    # Sanity-check the target scale against a plausibly-high Eclipse result
    # before spending GPU hours: an O(10) target with aux_coef=0.1 crowds the
    # policy gradient out of the global grad-norm clip.
    probe = aux_target_fn(np.array([40.0, 25.0, 10.0, 0.0], dtype=np.float32))
    biggest = float(np.max(np.abs(probe)))
    _emit(f"aux_tasks={aux_tasks} target_mode={FLAGS.aux_target_mode} "
          f"max|target| at 40 VP = {biggest:.3f}")
    if biggest > 3.0:
      raise ValueError(
          f"aux target magnitude {biggest:.2f} is too large to sit next to a "
          f"policy loss of order 1e-2 under a global grad-norm clip; lower "
          f"--aux_coef or raise --aux_vp_scale")

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
      agent_fn=make_agent_fn(
        FLAGS.nn_width, FLAGS.nn_depth, tuple(aux_tasks or ()),
        norm=FLAGS.nn_norm, activation=FLAGS.nn_activation,
        separate_critic=FLAGS.separate_critic, factored_actions=factored),
      value_mode=FLAGS.value_mode,
      aux_tasks=aux_tasks,
      aux_target_fn=aux_target_fn,
      aux_coef=FLAGS.aux_coef,
      rank_vp_beta=(FLAGS.rank_vp_beta if FLAGS.value_mode == "win" else 0.0),
      rank_ce_coef=(FLAGS.rank_ce_coef if FLAGS.value_mode == "win"
                    else 0.0),
  )

  # Device + resume telemetry before any training starts.
  _emit(f"device={device}  game={FLAGS.game}  num_envs={FLAGS.num_envs}"
        f"  num_workers={FLAGS.num_workers}")
  if FLAGS.resume:
    agent_fn_r = make_agent_fn(
        FLAGS.nn_width, FLAGS.nn_depth, tuple(aux_tasks or ()),
        norm=FLAGS.nn_norm, activation=FLAGS.nn_activation,
        separate_critic=FLAGS.separate_critic, factored_actions=factored)
    resume_src = FLAGS.resume
    sd = None
    # Resolve roster ids ("main", "snap_u100", ...) whenever the roster dir
    # exists, not only in league mode -- snapshots are written unconditionally
    # now, so `--resume=main` must work for a plain self-play run too.
    if os.path.isdir(FLAGS.roster_dir):
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
    # Tolerant load: --aux_target_mode determines which aux heads exist, so a
    # checkpoint written under a different mode has different head names. The
    # trunk/actor/critic still transfer; report exactly what did not.
    incompatible = agent.network.load_state_dict(sd, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
      _emit(f"resume: partial load from {resume_src} — "
            f"freshly initialized {list(incompatible.missing_keys)}, "
            f"ignored {list(incompatible.unexpected_keys)}")
    _emit(f"resumed network weights from {resume_src}")
    if _load_train_state(agent, FLAGS.roster_dir):
      _emit(f"resumed optimizer + counters: steps={agent.total_steps_done} "
            f"updates={agent.updates_done} lr_base={agent.learning_rate:.2e} "
            f"rank_vp_beta={agent.rank_vp_beta:.4g}")
    else:
      _emit("no train_state.pt found: Adam moments and step counters start "
            "fresh, so the LR anneal schedule restarts at full LR")

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
  agent_fn = make_agent_fn(
        FLAGS.nn_width, FLAGS.nn_depth, tuple(aux_tasks or ()),
        norm=FLAGS.nn_norm, activation=FLAGS.nn_activation,
        separate_critic=FLAGS.separate_critic, factored_actions=factored)
  num_actions = game.num_distinct_actions()
  roster = None
  matchmaker = None
  exploit_victim_net = None
  # A roster is created whenever there is anything to checkpoint, not only in
  # league mode: without this a plain self-play run wrote no weights at all.
  if FLAGS.league or FLAGS.exploit_victim or FLAGS.snapshot_every > 0:
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
      # Must update the *base* LR: anneal_learning_rate recomputes from it, so
      # writing param_groups directly was reverted after the first update and
      # the exploiter never actually trained at its intended rate.
      agent.set_learning_rate(FLAGS.exploit_lr)
    lineup = np.tile(
        np.asarray(["main"] + [victim_id] * (num_players - 1), dtype=object),
        (FLAGS.num_envs, 1))
    agent.setup_league({"main": agent.network, victim_id:
                        exploit_victim_net}, lineup, "main")

  # Shaping configuration resolved once per run.
  win_squash = FLAGS.value_mode == "win"
  phi_learned = FLAGS.phi == "learned"
  # 'telescope' is handled inside PPO (post_step(phi=...)): the delta spans a
  # seat's own consecutive decisions, so it cannot be computed from a single
  # env step here.
  phi_telescope = FLAGS.phi == "telescope"
  if phi_learned and not win_squash:
    raise ValueError("--phi=learned requires --value_mode=win (the learned "
                     "potential is the network's win value).")
  phi_mode = FLAGS.phi
  if FLAGS.shaping and phi_telescope:
    _emit("shaping: telescope phi (banked VP, differenced across each seat's "
          "own consecutive decisions)")

  # Per-player episode return logging.
  episode_returns = {i: [] for i in range(FLAGS.num_envs)}
  recent_returns = []
  # Why episodes end (bankruptcy vs. a played-out game) -- the signal the old
  # telemetry could not express.
  diag = EpisodeDiagnostics(FLAGS.num_envs, num_players)

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
        diag.observe(obs_batch, agent.last_seats)
        if FLAGS.shaping and phi_learned:
          phi_prev = _phi_wins(agent, obs_batch, device)
        else:
          phi_prev = potential_self_vec(obs_batch, win_squash)
        t1b = time.perf_counter() if _tm is not None else None
        step_arrays = envs.step_np(acts, reset_if_done=True)
        t2 = time.perf_counter() if _tm is not None else None
        shaped = np.zeros(FLAGS.num_envs, dtype=np.float32)
        if FLAGS.shaping and phi_mode not in ("none", "telescope"):
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
            # phi(s') for the acting seat. When a *different* seat is now to
            # move, read the mover's opponent block in that viewer's obs; when
            # the same seat keeps acting (common in Eclipse macro actions:
            # explore -> place -> rotate, build/upgrade/move internals) the
            # mover is the viewer, so read its self slots. The block-relative
            # soft-phi offsets differ between the two layouts, hence two calls
            # rather than one index trick. Previously the same-seat case read
            # seat-1's block -- and for seat 0, unrelated memory at OPP_BASE-25.
            same = seats == new_seats
            blocks = OPP_BASE + np.where(seats < new_seats, seats,
                                         np.maximum(seats - 1, 0)) * 25
            phi_next = potential_opponent_vec(step_arrays.obs, blocks,
                                              win_squash)
            if same.any():
              phi_next = np.where(
                  same, potential_self_vec(step_arrays.obs, win_squash),
                  phi_next)
            shaped[not_done] = FLAGS.gamma * phi_next[not_done] - phi_prev[not_done]
        t2b = time.perf_counter() if _tm is not None else None
        agent.post_step_np(
            step_arrays.rewards, step_arrays.dones, shaped_reward=shaped,
            phi=(phi_prev if (FLAGS.shaping and phi_telescope) else None))
        # After post_step: terminal closeout for the finished episode must see
        # the lineup that generated it, not the one sampled for the next.
        if FLAGS.league:
          _refresh_lineups(agent, matchmaker, roster, agent_fn, num_actions,
                           input_shape, device, step_arrays.dones)
        donor_idx = np.flatnonzero(step_arrays.dones)
        if donor_idx.size:
          diag.close_episodes(donor_idx, step_arrays.rewards)
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
      if FLAGS.rank_vp_beta_anneal_to >= 0.0:
        agent.anneal_rank_vp_beta(update, num_updates,
                                  FLAGS.rank_vp_beta_anneal_to)
      if _tm is not None and update % FLAGS.timing_every == 0:
        _emit(f"[timing u{update}] act={_tm[0]*1e3*_tm_scale:.2f}ms/env"
              f"  act+phi={_tm[1]*1e3*_tm_scale:.2f}  env={_tm[2]*1e3*_tm_scale:.2f}"
              f"  shape+refresh={_tm[3]*1e3*_tm_scale:.2f}"
              f"  post={_tm[4]*1e3*_tm_scale:.2f}"
              f"  total={_tm.sum()*1e3*_tm_scale:.2f}")
      if pbar is not None:
        pbar.update(FLAGS.num_envs * FLAGS.num_steps)
        _pbar_postfix()
      _maybe_snapshot(agent, roster, update)
      if update % FLAGS.eval_every == 0:
        _log_update(agent, episode_returns, recent_returns, writer, update,
                    diag=diag)
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
          eval_seed = FLAGS.seed + FLAGS.eval_seed_offset
          res = _eval_squad(
              agent, roster, agent_fn, num_actions, input_shape, device,
              _randomized_game_string(FLAGS.game, eval_seed),
              num_players, num_games=FLAGS.eval_games, rng_seed=eval_seed)
          if res is not None:
            mean, lo, hi = mean_ci(res.utils)
            _emit(_fmt_eval("Squad", res, num_players))
            writer.add_scalar("squad/main_utility", mean,
                              agent.total_steps_done)
            writer.add_scalar("squad/main_win_rate",
                              res.wins / max(1, res.games),
                              agent.total_steps_done)
            writer.add_scalar("squad/avg_rank", float(np.mean(res.ranks)),
                              agent.total_steps_done)
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
        diag.observe(obs_batch, agent.last_seats)
        if FLAGS.shaping and phi_learned:
          phi_prev = _phi_wins(agent, obs_batch, device)
        else:
          phi_prev = np.fromiter(
              (potential_self(obs_batch[i], win_squash)
               for i in range(FLAGS.num_envs)),
              dtype=np.float32, count=FLAGS.num_envs)

        time_step, reward, done, unreset = envs.step(
            agent_output, reset_if_done=True, players="current")
        shaped = np.zeros(FLAGS.num_envs, dtype=np.float32)
        if FLAGS.shaping and phi_mode not in ("none", "telescope"):
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
              obs_viewer = ts.observations["info_state"][viewer]
              k = opponent_block_index(seat, viewer)
              if k is None:
                # Same seat still to move: it is the viewer, so read its self
                # slots rather than an opponent block.
                phi_next = potential_self(obs_viewer, win_squash)
              else:
                phi_next = potential_opponent(obs_viewer, OPP_BASE + k * 25,
                                              win_squash)
              shaped[i] = FLAGS.gamma * phi_next - phi_prev[i]

        agent.post_step(
            reward, done, shaped_reward=shaped.tolist(),
            phi=(phi_prev if (FLAGS.shaping and phi_telescope) else None))
        # See the async loop: refresh only after terminal bookkeeping.
        if FLAGS.league:
          _refresh_lineups(agent, matchmaker, roster, agent_fn, num_actions,
                           input_shape, device, done)

        # Episode return logging.
        finished = [i for i, ts in enumerate(unreset) if ts.last()]
        if finished:
          diag.close_episodes(
              finished, {i: unreset[i].rewards for i in finished})
        for i in finished:
          ret = float(unreset[i].rewards[0])
          episode_returns[i].append(ret)
          recent_returns.append(ret)

      agent.learn(time_step)

      if FLAGS.anneal_lr:
        agent.anneal_learning_rate(update, num_updates)
      if FLAGS.rank_vp_beta_anneal_to >= 0.0:
        agent.anneal_rank_vp_beta(update, num_updates,
                                  FLAGS.rank_vp_beta_anneal_to)

      if pbar is not None:
        pbar.update(FLAGS.num_envs * FLAGS.num_steps)
        _pbar_postfix()

      _maybe_snapshot(agent, roster, update)

      if update % FLAGS.eval_every == 0:
        _log_update(agent, episode_returns, recent_returns, writer, update,
                    diag=diag)
        if FLAGS.eval_squad and roster is not None:
          eval_seed = FLAGS.seed + FLAGS.eval_seed_offset
          res = _eval_squad(
              agent, roster, agent_fn, num_actions, input_shape, device,
              _randomized_game_string(FLAGS.game, eval_seed),
              num_players, num_games=FLAGS.eval_games, rng_seed=eval_seed)
          if res is not None:
            mean, lo, hi = mean_ci(res.utils)
            _emit(_fmt_eval("Squad", res, num_players))
            writer.add_scalar("squad/main_utility", mean,
                              agent.total_steps_done)
            writer.add_scalar("squad/main_win_rate",
                              res.wins / max(1, res.games),
                              agent.total_steps_done)
            writer.add_scalar("squad/avg_rank", float(np.mean(res.ranks)),
                              agent.total_steps_done)

  # Sequential-exploiter closeout: report the win-rate vs the frozen victim and
  # optionally fold the trained policy into the roster.
  if FLAGS.exploit_victim and exploit_victim_net is not None:
    h2h = _eval_head2head(
        agent, exploit_victim_net, agent_fn, num_actions, input_shape, device,
        _randomized_game_string(FLAGS.game, FLAGS.seed + 777), num_players,
        num_games=FLAGS.eval_games, rng_seed=FLAGS.seed + 777)
    win_rate = h2h.wins / max(1, h2h.games)
    mean, lo, hi = mean_ci(h2h.utils)
    chance = chance_utility(num_players)
    _emit(f"[exploiter] vs victim {FLAGS.exploit_victim}: "
          f"utility={mean:+.3f} [{lo:+.3f},{hi:+.3f}] (chance {chance:+.3f})  "
          f"win-rate {h2h.wins}/{h2h.games}")
    writer.add_scalar("exploiter/victim_win_rate", win_rate,
                      agent.total_steps_done)
    writer.add_scalar("exploiter/victim_utility", mean, agent.total_steps_done)
    # Promote on beating the chance level with a non-overlapping interval, not
    # on a raw win-rate threshold: main holds 1 of num_players seats here, so
    # an equal-strength policy already wins 1/num_players of the time.
    promote = (not np.isnan(lo)) and lo > chance
    if FLAGS.exploit_promote and promote and roster is not None:
      roster.add_exploiter(agent.network, agent.updates_done,
                           FLAGS.exploit_victim, win_rate=win_rate)
      _emit("[exploiter] promoted to roster")
    elif FLAGS.exploit_promote:
      _emit("[exploiter] not promoted (did not beat chance utility)")

  if pbar is not None:
    pbar.close()
  writer.close()
  _emit("pilot done")


if __name__ == "__main__":
  app.run(main)
