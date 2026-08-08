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

This is the Stage 0.5/1 training path (see docs/eclipse_rl_todo.md). A single
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
import json
import os
import random
import time
import warnings
from absl import app
from absl import flags
import numpy as np
import torch
from torch import nn

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.eclipse import obs_layout
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
                  "Legacy alias used only when --lr_schedule is unset: True maps "
                  "to 'fixed', False maps to 'fixed'. Kept for back-compat with "
                  "old command lines; --lr_schedule wins when set explicitly. "
                  "NOTE: this used to map True -> 'kl', which silently enabled a "
                  "never-validated controller on every run that did not pass "
                  "--lr_schedule. See the safety note on --lr_schedule.")
flags.DEFINE_enum(
    "lr_schedule", None, ["fixed", "anneal", "kl"],
    "How the learning rate is mutated after each update. 'fixed' = never "
    "touch it. 'anneal' = the historical linear decay to zero over the run "
    "(driven off agent.updates_done, resume-continuing). 'kl' = closed-loop "
    "KL-targeted controller (cls. Item 4): react to the realized per-update KL "
    "every step with a small multiplicative step and clamp within "
    "[--lr_min, --lr_max]. A fixed anneal shuts off learning past convergence, "
    "so 'kl' is the default when this flag is unset (falls back to --anneal_lr "
    "if that is set False).")
flags.DEFINE_float("kl_target", 0.02,
                   "Target per-update KL for the 'kl' LR controller.")
flags.DEFINE_float("kl_lr_tau", 0.05,
                   "EMA weight for the kl controller's KL estimate (small = "
                   "smooth; reacts over many updates).")
flags.DEFINE_float("lr_min", 1e-6, "Floor for the 'kl' LR controller.")
flags.DEFINE_float("lr_max", 1e-2, "Ceiling for the 'kl' LR controller.")

flags.DEFINE_float("ent_lo", None,
                   "Lower entropy bound for the entropy-band controller "
                   "(None disables).")
flags.DEFINE_float("ent_hi", None,
                   "Upper entropy bound for the entropy-band controller "
                   "(None disables).")
flags.DEFINE_float("ent_step", 0.1,
                   "Fractional per-fire change to --ent_coef for the "
                   "entropy-band controller.")
flags.DEFINE_integer("ent_control_every", 15,
                     "Rate-limit the entropy-band controller to fire this "
                     "often (updates), giving it a longer time constant than "
                     "the KL/LR loop.")

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
flags.DEFINE_float(
    "gamma", 0.998,
    "Discount factor (used for potential-based shaping too). Raised from 0.99 "
    "on 2026-08-08 for horizon reasons: Eclipse pays its real reward only at "
    "the end, and GAE follows each seat's OWN decision subsequence, which runs "
    "~80 decisions per player. At 0.99 the terminal rank utility is discounted "
    "to 0.99^80 ~ 0.45 by mid-game -- over half of the only true signal is "
    "thrown away. At 0.998 it is 0.998^80 ~ 0.85. 0.998 is also what the "
    "hide-and-seek PPO setup used (Baker et al. 2019). UNTESTED on the ladder: "
    "the 1.2204 baseline was trained at 0.99, so any run at this default "
    "differs from that baseline in this respect too -- keep it in mind when "
    "attributing a result.")
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
     "aux_target_mode", "rank", ["vp", "rank", "both", "breakdown", "none"],
     "Aux-head regression target. 'rank' (default) = per-seat tie-aware rank "
     "utility: bounded in [-0.5, 1] by construction, so it can neither vanish "
     "nor dominate. 'vp' = final VP / --aux_vp_scale. 'both' supervises a "
     "normalized-VP head and a rank head. 'breakdown' supervises 9 heads, one "
     "per final-VP category (reputation/ambassador/sector/monolith/"
     "discovery/tech/traitor/species/minor), each read back-filled from the "
     "terminal observation. 'none' disables aux heads. Note the "
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
flags.DEFINE_enum(
    "encoder", "spatial", ["flat", "spatial"],
    "Observation encoder. 'spatial' slices the flat tensor into a galaxy conv "
    "tower + viewer self block + tail blocks + a relational masked mean+max "
    "seat pool (the flat Linear(24714, width) first layer was a ~5.1x "
    "throughput collapse). 'flat' preserves the old dense trunk as a control.")
flags.DEFINE_bool(
    "spatial_pointer", False,
    "Add a pointer/attention term from the spatial encoder's per-cell conv "
    "features into FactoredActorHead's cell-indexed logits ("
    "'colonize cell 42' should depend on cell 42's actual state, which a "
    "global-average-pooled trunk feature otherwise never sees). Requires "
    "--encoder=spatial and --factored_actions.")
flags.DEFINE_bool(
    "amp", False,
    "bf16 autocast (torch.autocast) around the PPO learn-path minibatch "
    "forward+loss only -- never the rollout/act path. See PPO.__init__'s "
    "`amp` docstring for why. No-op (bit-identical) when off.")
flags.DEFINE_bool(
    "channels_last", False,
    "Run SpatialEclipseEncoder's conv tower in channels_last memory format. "
    "obs_layout.galaxy_view returns a permuted, non-contiguous view that "
    "cudnn otherwise silently copies on every call (measured 20.24ms vs "
    "14.44ms channels_last, fwd only, 4096 rows); this makes that copy "
    "explicit and cheaper. No-op (bit-identical) when off.")
flags.DEFINE_bool(
    "compile_encoder", False,
    "torch.compile SpatialEclipseEncoder._encode. Falls back to eager with a "
    "warning if compilation fails. No-op when off or --encoder=flat.")

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


class SpatialEclipseEncoder(nn.Module):
  """Spatial + relational encoder for the flat Eclipse observation tensor.

  The flat ``Linear(24714, width)`` trunk measured a 5.1x collapse (2,655 vs
  13,635 envstep/s) because the first layers held ~21M params. This encoder
  instead slices the flat tensor into semantically meaningful pieces inside its
  own ``forward``, so ``ppo.py`` keeps working unchanged -- it only ever calls
  ``net.shared(x)`` and reads ``(B, width)``:

  - galaxy block -> a small residual conv tower over ``(88, 15, 15)``,
  - viewer's own seat block (slot 0) -> MLP,
  - the tail blocks (tech market / combat / upkeep / action states) -> MLP,
  - the 6 seat blocks -> a SHARED per-seat MLP with permutation-invariant
    masked mean+max pooling, validity from the ``P_OCCUPIED`` bit (never from
    block content), so one net serves 4/5/6 players.

  All branch latents are fused to a single ``(B, width)`` vector that the rank
  critic, aux heads and FactoredActorHead consume exactly as before.

  GroupNorm/LayerNorm are used instead of BatchNorm2d: rollout ``act()`` sees
  small temporally-correlated batches while ``learn()`` reshuffles, so running
  batch stats would be wrong in one path or the other.
  """

  # Conv tower output channels -- a class attribute (not a bare literal) so
  # SpatialFactoredActorHead's pointer query can size itself off it.
  CELL_FEATURE_CHANNELS = 64

  def __init__(self, width, depth=1, activation="gelu", device=None,
               channels_last=False, compile_encoder=False):
    super().__init__()
    self.device = device
    act = nn.GELU if activation == "gelu" else nn.Tanh
    c = self.CELL_FEATURE_CHANNELS

    # ── Galaxy branch: (B, 88, 15, 15) conv tower ───────────────────────────
    self.conv = nn.Sequential(
        layer_init(nn.Conv2d(obs_layout.CELL_CHANNELS, c, 3, padding=1)),
        nn.GroupNorm(8, c),
        act(),
    )
    self.conv_res = nn.Sequential(
        layer_init(nn.Conv2d(c, c, 3, padding=1)),
        nn.GroupNorm(8, c),
        act(),
    )
    self.galaxy_fc = layer_init(nn.Linear(c, width))

    # ── Viewer self block: (547,) MLP ───────────────────────────────────────
    self.self_mlp = self._mlp(obs_layout.PLAYER_SIZE, width, depth, act)

    # ── Tail blocks (tech market + combat + upkeep + action states) ─────────
    tail_size = (obs_layout.TECH_MARKET_SIZE + obs_layout.COMBAT_SIZE
                 + obs_layout.UPKEEP_SIZE + obs_layout.ACTION_STATES_SIZE)
    self.tail_mlp = self._mlp(tail_size, width, depth, act)

    # ── Relational: shared per-seat MLP -> masked mean+max pool ─────────────
    self.seat_mlp = self._mlp(obs_layout.PLAYER_SIZE, width, depth, act)
    self.rel_fc = layer_init(nn.Linear(2 * width, width))

    # ── Fuse the four branch latents to (B, width) ──────────────────────────
    self.fuse = layer_init(nn.Linear(4 * width, width))

    # --channels_last: obs_layout.galaxy_view returns a permuted (non-
    # contiguous) view, which cudnn otherwise silently re-copies to contiguous
    # NCHW on every call. Putting the conv tower's weights in channels_last
    # memory format and making the input contiguous in that same format up
    # front (see _encode_impl) turns that hidden copy into one explicit,
    # cheaper one. Purely a memory-layout change -- values are unaffected.
    self.channels_last = channels_last
    if channels_last:
      self.conv = self.conv.to(memory_format=torch.channels_last)
      self.conv_res = self.conv_res.to(memory_format=torch.channels_last)

    # --compile_encoder: see _encode's dispatch for the graceful-fallback
    # mechanics.
    self._compiled_encode = None
    if compile_encoder:
      # dynamic=True: the learn path always calls with the same fixed batch
      # (num_envs*num_steps/num_minibatches), but the sparse act path's batch
      # size is the per-step legal-action count, which changes every call.
      # Compiling per-shape would mean a recompile per distinct batch size on
      # the act path (a recompile storm); one dynamic-shape compile serves
      # both call sites.
      self._compiled_encode = torch.compile(self._encode_impl, dynamic=True)

  def _mlp(self, in_features, width, depth, act):
    layers = []
    cur = in_features
    for _ in range(depth):
      layers.append(layer_init(nn.Linear(cur, width)))
      layers.append(act())
      cur = width
    return nn.Sequential(*layers)

  def forward(self, x):
    return self._encode(x)[0]

  def forward_with_cells(self, x):
    """Like ``forward``, but also returns the pre-pool per-cell conv features.

    Returns:
      fused: (B, width) -- byte-identical to ``forward(x)``.
      h_cells: (B, CELL_FEATURE_CHANNELS, 225) conv features per board cell,
        indexed by ``obs_layout.hex_to_index(q, r)`` (see the flatten check in
        ``forward``/module docstring: galaxy_view's last two dims are
        (q_idx, r_idx), and a plain reshape preserves that as the linear cell
        id -- no permute needed).
    """
    return self._encode(x)

  def _encode(self, x):
    """Dispatches to the (optionally torch.compile'd) encoder body.

    --compile_encoder failures surface at the first real call, not at
    construction time (torch.compile is lazy), so the try/except lives here:
    one bad call falls back to eager for the rest of the run instead of
    crashing training.
    """
    if self._compiled_encode is None:
      return self._encode_impl(x)
    try:
      return self._compiled_encode(x)
    except Exception as e:  # pylint: disable=broad-except
      warnings.warn(
          f"torch.compile(SpatialEclipseEncoder) failed at runtime ({e!r}); "
          "falling back to eager for the rest of this run.")
      self._compiled_encode = None
      return self._encode_impl(x)

  def _encode_impl(self, x):
    b = x.shape[0]
    x = x.reshape(b, -1)

    # Galaxy: (B, CELL_CHANNELS, 15, 15).
    gal = obs_layout.galaxy_view(x)
    if self.channels_last:
      gal = gal.contiguous(memory_format=torch.channels_last)
    h = self.conv(gal)
    h = h + self.conv_res(h)
    h_cells = h.reshape(b, self.CELL_FEATURE_CHANNELS, obs_layout.GALAXY_CELLS)
    gal_lat = self.galaxy_fc(h_cells.mean(dim=-1))  # global avg pool -> (B,width)

    # Viewer self block (slot 0 is always the viewer).
    self_start = obs_layout.PLAYERS_START
    self_lat = self.self_mlp(x[:, self_start:self_start +
                               obs_layout.PLAYER_SIZE])

    # Tail blocks.
    tail = torch.cat([
        x[:, obs_layout.TECH_MARKET_START:
          obs_layout.TECH_MARKET_START + obs_layout.TECH_MARKET_SIZE],
        x[:, obs_layout.COMBAT_START:
          obs_layout.COMBAT_START + obs_layout.COMBAT_SIZE],
        x[:, obs_layout.UPKEEP_START:
          obs_layout.UPKEEP_START + obs_layout.UPKEEP_SIZE],
        x[:, obs_layout.ACTION_STATES_START:
          obs_layout.ACTION_STATES_START + obs_layout.ACTION_STATES_SIZE],
    ], dim=1)
    tail_lat = self.tail_mlp(tail)

    # Relational pool over the 6 seat blocks. Occupied bits gate validity --
    # for a 4-player game slots 4-5 are marked empty and masked out.
    seats = x[:, obs_layout.PLAYERS_START:
              obs_layout.PLAYERS_START +
              obs_layout.SEAT_SLOTS * obs_layout.PLAYER_SIZE
              ].reshape(b, obs_layout.SEAT_SLOTS, obs_layout.PLAYER_SIZE)
    seat_h = self.seat_mlp(seats)                # (B, 6, width)
    occ = seats[:, :, obs_layout.P_OCCUPIED] >= 0.5   # (B, 6)
    mask = occ.unsqueeze(-1)
    denom = occ.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (seat_h * mask).sum(dim=1) / denom    # (B, width)
    neg = -1e9
    masked_max = torch.where(
        mask, seat_h, torch.full_like(seat_h, neg)).max(dim=1).values
    rel_lat = self.rel_fc(torch.cat([mean, masked_max], dim=1))

    fused = self.fuse(torch.cat(
        [gal_lat, self_lat, tail_lat, rel_lat], dim=1))
    return fused, h_cells


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


class SpatialFactoredActorHead(FactoredActorHead):
  """FactoredActorHead + a pointer term into the per-cell conv features.

  The base class's ``W[a] = sum_slot E[decode[a, slot]]`` weight is a pure
  function of the action id: the state of board cell 42 never influences the
  logit for "colonize cell 42" because nothing in that weight depends on the
  observation. This adds, for every action with a real board coordinate
  (``cell_id[a] >= 0``), a joint term between the observation and the action:

      logit[a] = base[a] + <query(features), h_cells[:, :, cell_id[a]]>

  where ``h_cells`` is ``SpatialEclipseEncoder``'s pre-pool per-cell conv
  output (see ``forward_with_cells``). ``query`` is ``layer_init(..., std=0.01)``
  so the pointer term starts near zero and cannot destabilise early training.

  TRAP: this class inherits ``rows_for``/``full_weight`` from FactoredActorHead
  UNCHANGED, and those return the BASE term ONLY -- they have no ``cells``
  argument to add the pointer term through. Any code path that does
  ``hasattr(head, 'rows_for')`` and calls it directly (instead of going through
  ``logits_for`` or ``forward(features, cells)``) will silently get the
  pre-fix, cell-blind policy. Never call ``rows_for``/``full_weight`` on this
  head expecting complete logits.
  """

  def __init__(self, decode, num_rows, num_actions, width, cell_id,
               cell_channels):
    super().__init__(decode, num_rows, num_actions, width)
    cell_id = np.asarray(cell_id, dtype=np.int64)
    self.register_buffer("cell_id", torch.from_numpy(cell_id))
    self.register_buffer("is_cell", torch.from_numpy(cell_id >= 0))
    self.query = layer_init(nn.Linear(width, cell_channels), std=0.01)

  def _pointer_term(self, q, cell_feat, is_cell):
    """<q, cell_feat> gated to 0 where the action has no board cell."""
    pointer = (q * cell_feat).sum(-1)
    return torch.where(is_cell, pointer, torch.zeros_like(pointer))

  def logits_for(self, features, cells, rows, cols):
    """(M,) logits for the (rows[i], cols[i]) pairs (sparse path).

    ``features``: (B, width) shared features. ``cells``: (B, cell_channels,
    225) per-cell conv features, or None (base-only, e.g. --spatial_pointer
    off or the flat encoder). ``rows``/``cols``: (M,) torch tensors.
    """
    base = ((features[rows] * self.rows_for(cols)).sum(-1)
            + self.bias[cols])
    if cells is None:
      return base
    is_cell = self.is_cell[cols]
    q = self.query(features[rows])                      # (M, cell_channels)
    cid = self.cell_id[cols].clamp(min=0)                # safe dummy index
    cell_feat = cells[rows].gather(
        2, cid.view(-1, 1, 1).expand(-1, cells.shape[1], 1)).squeeze(-1)
    return base + self._pointer_term(q, cell_feat, is_cell)

  def forward(self, features, cells=None):
    """(B, num_actions) dense logits. ``cells`` as in ``logits_for``."""
    base = features @ self.full_weight().t() + self.bias
    if cells is None:
      return base
    q = self.query(features)                             # (B, cell_channels)
    cell_logits = torch.einsum("bc,bcn->bn", q, cells)    # (B, 225)
    cid = self.cell_id.clamp(min=0)                       # (num_actions,)
    pointer = cell_logits[:, cid] * self.is_cell.to(cell_logits.dtype)
    return base + pointer


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
               factored_actions=None, encoder="flat", spatial_pointer=False,
               channels_last=False, compile_encoder=False):
    super().__init__()
    in_features = int(np.array(observation_shape).prod())
    self.encoder = encoder
    if spatial_pointer and encoder != "spatial":
      raise ValueError(
          "--spatial_pointer requires --encoder=spatial: a flat trunk has no "
          "per-cell features for the pointer term to read, and silently "
          "degrading to the base-only head would defeat the point of the flag.")
    if spatial_pointer and factored_actions is None:
      raise ValueError(
          "--spatial_pointer requires --factored_actions: the pointer term "
          "needs the per-action cell_id table that only the factorization "
          "carries.")

    if encoder == "spatial":
      self.shared = SpatialEclipseEncoder(
          width, depth=max(1, depth), activation=activation, device=device,
          channels_last=channels_last, compile_encoder=compile_encoder)
    else:
      self.shared = self._trunk(in_features, width, depth, norm, activation)
    self.separate_critic = separate_critic

    if separate_critic:
      # An independent trunk for the value/aux objectives. With a shared trunk
      # the measured gradient split on aux-bearing rows was aux 82% / policy 11%
      # / value 6%, i.e. the representation the policy reads was being shaped
      # mostly by the regression heads.
      if encoder == "spatial":
        self.critic_trunk = SpatialEclipseEncoder(
            width, depth=max(1, depth), activation=activation, device=device,
            channels_last=channels_last, compile_encoder=compile_encoder)
      else:
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
    elif spatial_pointer:
      actor_head = SpatialFactoredActorHead(
          factored_actions.decode, factored_actions.num_rows, num_actions,
          width, factored_actions.cell_id,
          SpatialEclipseEncoder.CELL_FEATURE_CHANNELS)
    else:
      actor_head = FactoredActorHead(
          factored_actions.decode, factored_actions.num_rows, num_actions,
          width)
    self.spatial_pointer = spatial_pointer
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

  def dense_logits(self, x):
    """(B, num_actions) dense logits -- the eval/argmax entry point.

    ``self.actor`` stays a literal ``nn.Sequential(self.shared, actor_head)``
    (ppo.py's ``_sparse_supported`` checks that shape), but ``nn.Sequential``
    only ever chains one tensor, so a spatial pointer term's per-cell features
    cannot flow through it. This bypasses the Sequential to route ``cells``
    into the head when both the trunk and the head support it, so eval scores
    the actual policy being trained rather than a base-only approximation of
    it; it falls back to plain ``self.actor(x)`` otherwise (flat encoder, no
    factored head, or --spatial_pointer off).
    """
    forward_with_cells = getattr(self.shared, "forward_with_cells", None)
    head = self.actor[-1]
    if forward_with_cells is not None and hasattr(head, "cell_id"):
      features, cells = forward_with_cells(x)
      return head.forward(features, cells)
    return self.actor(x)

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    if legal_actions_mask is None:
      legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()
    logits = self.dense_logits(x)
    from open_spiel.python.pytorch.ppo import CategoricalMasked
    probs = CategoricalMasked(logits=logits, masks=legal_actions_mask,
                              mask_value=self.mask_value)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.get_value(
        x), probs.probs


def make_agent_fn(width, depth, aux_tasks=("final_vp",), norm=False,
                  activation="tanh", separate_critic=False,
                  factored_actions=None, encoder="flat",
                  spatial_pointer=False, channels_last=False,
                  compile_encoder=False):
  def agent_fn(num_actions, observation_shape, device):
    return EclipsePPOAgent(num_actions, observation_shape, device,
                           width=width, depth=depth, aux_tasks=aux_tasks,
                           norm=norm, activation=activation,
                           separate_critic=separate_critic,
                           factored_actions=factored_actions,
                           encoder=encoder, spatial_pointer=spatial_pointer,
                           channels_last=channels_last,
                           compile_encoder=compile_encoder)

  return agent_fn


# Aux-head names by --aux_target_mode, and the per-seat target each produces.
_AUX_TASKS_BY_MODE = {
    "vp": ("final_vp",),
    "rank": ("final_rank",),
    "both": ("final_vp", "final_rank"),
    "breakdown": (
        "bd_reputation", "bd_ambassador", "bd_sector", "bd_monolith",
        "bd_discovery", "bd_tech_track", "bd_traitor", "bd_species",
        "bd_minor_species"),
    "none": (),
}

def build_aux_targets(mode, vp_scale):
  """(task names, target fn) for ``--aux_target_mode``.

  The target fn maps a terminal per-seat payoff vector to a
  (num_players, num_tasks) matrix. Targets must be O(1): the gradient is clipped
  globally, so an aux term far larger than the policy term rescales the policy
  gradient toward zero. Raw-VP targets (the state this replaces) reached
  aux_loss ~8-11 against pg_loss ~1e-3.

  For ``breakdown`` the targets are the 9 per-category VPs read from the
  terminal *observation* (``terminal_obs``, canonicalised to the acting seat),
  each already normalized to [0,1] in the tensor -- see ``P_VP_BREAKDOWN``.
  ``terminal_obs=None`` (used by the pre-run magnitude probe) returns ``None``
  so the caller leaves those rows unmasked.
  """
  tasks = _AUX_TASKS_BY_MODE[mode]
  if not tasks:
    return None, None

  def target_fn(rvec, terminal_obs=None, acting_seat=None):
    arr = np.asarray(rvec, dtype=np.float32)
    num_players = arr.shape[0]
    if mode == "breakdown":
      if terminal_obs is None or acting_seat is None:
        return None
      cols = []
      for s in range(num_players):
        slot = obs_layout.slot_for_seat(s, int(acting_seat), num_players)
        base = obs_layout.player_block_start(slot) + obs_layout.P_VP_BREAKDOWN
        cols.append(np.asarray(terminal_obs[base:base + len(tasks)],
                               dtype=np.float32))
      return np.stack(cols, axis=0)
    cols = []
    for task in tasks:
      if task == "final_vp":
        cols.append(arr / float(vp_scale))
      else:  # final_rank: bounded in [-0.5, 1] by construction.
        cols.append(np.asarray(
            [rank_utility(arr, s) for s in range(num_players)],
            dtype=np.float32))
    return np.stack(cols, axis=1)

  if mode == "breakdown":
    # Tag so ``ppo.py`` knows to route the terminal observation through -- a
    # pre-existing one-arg extractor (e.g. in tests) is otherwise left alone.
    target_fn.needs_terminal_obs = True
  return list(tasks), target_fn


# Win-mode potential squash: raw VP-unit potentials are mapped onto the
# rank-utility scale ([1..4] -> ~[-0.5, 1]) so shaped rewards and terminal
# rank-utility targets stay comparable. The tensor already normalizes VP by
# /200, so u(vp) = clip(vp/200, -0.5, 1).
def _squash_win(vp):
  return float(np.clip(np.array(vp) / 200.0, -0.5, 1.0))


# Every observation offset now comes from one place -- see obs_layout.py, which
# mirrors open_spiel/games/eclipse/observation.h and asserts its TOTAL against
# the engine's real observation size. The previous hand-written constants here
# (SCORE_SELF_SLOT, OPP_BASE, GALAXY_BASE, CELL_STRIDE and the offsets buried
# inside phi_soft_*) all silently addressed the wrong floats the moment the C++
# layout moved, which it now has.
SCORE_DIVISOR = 60.0  # total_vp is normalized by /60 in the tensor.


def player_slot(seat, viewer, num_players):
  """Block slot holding ``seat`` in ``viewer``'s observation.

  The tensor is canonicalised to the viewer: slot 0 is *always* the viewer and
  slots 1..n-1 are the other seats in wrapping order. There is no longer a
  separate self/opponent schema, so unlike the old ``opponent_block_index`` this
  is total -- every seat, including the viewer, has a slot.
  """
  return obs_layout.slot_for_seat(seat, viewer, num_players)


def score_slot(seat, viewer, num_players):
  """Index of ``seat``'s live total-VP float in ``viewer``'s observation."""
  return (obs_layout.player_block_start(player_slot(seat, viewer, num_players))
          + obs_layout.P_VP_TOTAL)


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

GALAXY_BASE = obs_layout.GALAXY_START
CELL_STRIDE = obs_layout.CELL_CHANNELS

ROUND_SLOT = obs_layout.GLOBAL_START  # round / MAX_ROUNDS
MAX_ROUNDS = 8


def elim_slot(seat, viewer, num_players):
  """Index of ``seat``'s `eliminated` flag in ``viewer``'s observation."""
  return (obs_layout.player_block_start(player_slot(seat, viewer, num_players))
          + obs_layout.P_ELIMINATED)


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
    # Per-VP-category breakdown (the 9 scoring sources, in P_VP_BREAKDOWN
    # order), harvested from terminal observations at episode closeout. Each
    # value is already normalized to [0,1] in the tensor. Tracked separately for
    # the actor's own seat and the all-seats mean, so "how the agent does per
    # VP source" is observable without an aux head.
    self.vp_categories = (
        "reputation", "ambassador", "sector", "monolith", "discovery",
        "tech_track", "traitor", "species", "minor_species")
    self.num_vp_categories = len(self.vp_categories)
    self.actor_cat_vp = collections.deque(maxlen=history)
    self.all_cat_vp = collections.deque(maxlen=history)
    self._cat_cols_cache = {}

  def _elim_columns(self, seats):
    """(num_players, num_envs) obs column holding each seat's eliminated bit.

    Every seat now has an identical block and the tensor is canonicalised to the
    viewer, so this is one wrapping subtraction rather than a self/opponent
    special case.
    """
    seat_ids = np.arange(self.num_players, dtype=np.int64)[:, None]
    viewers = np.asarray(seats, dtype=np.int64)[None, :]
    slot = (seat_ids - viewers) % self.num_players
    return (obs_layout.PLAYERS_START + slot * obs_layout.PLAYER_SIZE
            + obs_layout.P_ELIMINATED)

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

  def record_breakdown(self, obs_batch, seats, donor_idx):
    """Harvest the 9 VP category values for finished episodes.

    ``obs_batch[i]`` is the *terminal* decision-obs of env ``i`` (the acting
    seat's view, canonicalised so slot 0 is that seat) and ``seats[i]`` is that
    acting seat's absolute id. Each seat's block is read via ``slot_for_seat``
    and ``P_VP_BREAKDOWN`` (the identical read ``build_aux_targets`` uses for
    breakdown mode) -- values are already in [0,1].
    """
    for i in donor_idx:
      i = int(i)
      viewer = int(seats[i])
      # The acting seat is slot 0 in its own canonicalised view.
      actor_base = obs_layout.PLAYERS_START + obs_layout.P_VP_BREAKDOWN
      self.actor_cat_vp.append(
          np.asarray(obs_batch[i][actor_base:actor_base
                                  + self.num_vp_categories], dtype=np.float32))
      cats = np.zeros(self.num_vp_categories, dtype=np.float32)
      for s in range(self.num_players):
        slot = obs_layout.slot_for_seat(s, viewer, self.num_players)
        base = (obs_layout.PLAYERS_START
                + slot * obs_layout.PLAYER_SIZE
                + obs_layout.P_VP_BREAKDOWN)
        cats += np.asarray(
            obs_batch[i][base:base + self.num_vp_categories], dtype=np.float32)
      self.all_cat_vp.append(cats / self.num_players)

  def breakdown_summary(self):
    """Per-VP-category means: (actor_seat_means, all_seats_means) or (None,
    None) when nothing recorded."""
    if not self.actor_cat_vp:
      return None, None
    actor = np.stack(self.actor_cat_vp)
    alls = np.stack(self.all_cat_vp)
    return actor.mean(axis=0), alls.mean(axis=0)

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
    base = GALAXY_BASE + CELL_STRIDE * c
    if base + obs_layout.C_ENEMY_SHIPS + obs_layout.PLAYER_SHIP_TYPES > len(obs):
      enemy = 0.0
    else:
      lo = base + obs_layout.C_ENEMY_SHIPS
      enemy = float(sum(obs[lo:lo + obs_layout.PLAYER_SHIP_TYPES]))
    key = (enemy, c)
    if best_key is None or key < best_key:
      best_key = key
      best = a
  return best


class _GreedyPickV2:
  """Observation-aware reference heuristic (stronger than ``_greedy_pick``).

  ``_greedy_pick`` types only the explore macro-starts and the explore
  sub-pipeline; everything else (build/colony/move/combat/upkeep/diplomacy/
  trade internals) falls back to a uniform random pick, so the "+0.70 vs
  Greedy" verdict means "beats near-random" and saturates early. This bot
  reads the (now complete) observation tensor and the engine's own action-id
  arithmetic to make *informed* choices among the legal cell-targeted
  families, so it is a real reference an RL agent must beat.

  Decoder (mirrors eclipse.cc, all cell-targeted ids use ``hex_to_index``,
  which is exactly ``obs_layout.hex_to_index``):
    explore zone: cell = id - 101
    colony ship : cell = (id - 332) / 24  ;  street = (id - 332) % 24
    build choice: enc = id - 6188 ; cell = enc % 225, type = enc / 225

  Deliberately out of scope (deferred, matches B3 plan): diplomacy and trade
  internals still fall back to random, exactly as today. A live reference for
  those would need the ambassador/trade tables, not the raw tensor.
  """

  ACTION_BUILD_START = 6188
  ACTION_BUILD_STOP = ACTION_BUILD_START - 1   # 6187, eclipse.cc
  GALAXY_CELLS = obs_layout.GALAXY_CELLS
  COLONY_CODES = 24

  BUILD_TYPES = ("INTERCEPTOR", "CRUISER", "DREADNOUGHT", "STARBASE",
                 "ORBITAL", "MONOLITH")

  def __call__(self, obs, legal, rng=None):
    if rng is None:
      rng = np.random
    obs = np.asarray(obs, dtype=np.float32)
    s = set(int(a) for a in legal)

    # Explore sub-pipeline: keep exact behaviour of v1 (it is already typed).
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

    # Explore-zone choice: prefer a present, planet-bearing, uncontested cell.
    zones = [int(a) for a in legal
             if ACTION_EXPLORE_ZONE_START <= a <= ACTION_EXPLORE_ZONE_END]
    if zones:
      return self._pick_explore_zone(obs, zones)

    # Build choice: prefer a defensive hull on our planet, else pass.
    builds = [int(a) for a in legal if self.ACTION_BUILD_START <= a]
    if builds:
      c = self._pick_build(obs, builds)
      if c is not None:
        return c

    # Colony placement: prefer the highest-value reachable cell.
    colony = [int(a) for a in legal
              if ACTION_COLONY_START <= a <= ACTION_COLONY_END]
    if colony:
      return self._pick_colony(obs, colony, rng)

    # Macro-starts and the pass, as v1.
    for pid in (ACTION_BUILD_START, ACTION_UPGRADE_START, ACTION_RESEARCH_START,
                ACTION_INFLUENCE_START, ACTION_MOVE_START):
      if pid in s:
        return pid
    if ACTION_PASS in s:
      return ACTION_PASS
    return int(rng.choice(np.asarray(legal)))

  # -- per-family scoring --------------------------------------------------

  def _cell_score(self, obs, cell):
    """Higher = a better cell to expand/colonize/build on (0..~3)."""
    base = GALAXY_BASE + CELL_STRIDE * cell
    if not (base + obs_layout.C_PLANET_PRINTED + obs_layout.PLANET_TYPE_COUNT
            <= len(obs)):
      return -1.0, 0.0
    has_planet = bool(
        obs[base + obs_layout.C_PLANET_PRINTED:
            base + obs_layout.C_PLANET_PRINTED
            + obs_layout.PLANET_TYPE_COUNT].sum())
    vp = float(obs[base + obs_layout.C_POINTS])
    lo = base + obs_layout.C_ENEMY_SHIPS
    enemy = float(sum(obs[lo:lo + obs_layout.PLAYER_SHIP_TYPES]))
    score = vp * 2.0 + (1.0 if has_planet else 0.0) - enemy * 1.5
    return score, vp

  def _pick_explore_zone(self, obs, zones):
    best, best_key = None, None
    for a in zones:
      cell = a - ACTION_EXPLORE_ZONE_START
      score, vp = self._cell_score(obs, cell)
      key = (score, -vp, cell)
      if best_key is None or key > best_key:
        best_key = key
        best = a
    return best

  def _pick_colony(self, obs, colony, rng):
    by_cell = {}
    for a in colony:
      cell = (a - ACTION_COLONY_START) // self.COLONY_CODES
      by_cell.setdefault(cell, []).append(a)
    best, best_key = None, None
    for cell, acts in by_cell.items():
      score, vp = self._cell_score(obs, cell)
      # Colony street (slot*3+track): prefer the first (money/slot 0) entry.
      acts = sorted(acts)
      key = (score, -vp, acts[0])
      if best_key is None or key > best_key:
        best_key = key
        best = acts[0]
    if best is not None:
      return best
    return int(rng.choice(np.asarray(colony)))

  def _pick_build(self, obs, builds):
    """Build defender/economic hull on the best owned cell; else BUILD_STOP.

    A cheap ENEMY-free colony is what an RL economy needs, but a naive 'build
    anything' just burns cash. Pick INTERCEPTOR on the strongest *own* cell
    when it would be cheap; otherwise stop.  Keep it weak-but-sane: the point
    is a floor a trained policy must clear, not a tuned strategy.
    """
    # BUILD_STOP is the lowest legal build-range id in most states; only act
    # when at least one non-stop build is legal.
    bstop = self.ACTION_BUILD_STOP
    choices = [a for a in builds if a != bstop]
    if not choices:
      return (bstop if bstop in builds else None)
    # Prefer the cheapest ship (INTERCEPTOR=0) on our best planet cell.
    best, best_key = None, None
    for a in choices:
      enc = a - self.ACTION_BUILD_START
      cell = enc % self.GALAXY_CELLS
      btype = enc // self.GALAXY_CELLS
      if btype != 0:          # only interceptors here (cheap, valid in most)
        continue
      score, vp = self._cell_score(obs, cell)
      key = (score, -vp, a)
      if best_key is None or key > best_key:
        best_key = key
        best = a
    return best if best is not None else (
        bstop if bstop in builds else None)


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


# Denominators the C++ writer used for each field, so the reads below recover
# real units rather than normalised ones.
_PHI_SCALES = (
    (obs_layout.P_COLONY_TOTAL, 12.0),
    (obs_layout.P_DISKS_ON_SECTORS, 16.0),
    (obs_layout.P_ORBITALS, 10.0),
    (obs_layout.P_MONOLITHS, 6.0),
    (obs_layout.P_AMBASSADOR_HELD, 5.0),
)


def phi_soft_block(obs, base):
  """Soft potential for the seat whose block starts at ``base``.

  Self and opponent now share one identical block schema, so this replaces the
  old ``phi_soft_self``/``phi_soft_opponent`` pair -- which had to duplicate the
  same formula against two different sets of offsets.
  """
  _, (w_colony, w_disk, w_struct, w_amb), _ = _phi_cached()
  vp = float(obs[base + obs_layout.P_VP_TOTAL]) * SCORE_DIVISOR
  colony = float(obs[base + obs_layout.P_COLONY_TOTAL]) * 12.0
  disks = float(obs[base + obs_layout.P_DISKS_ON_SECTORS]) * 16.0
  orbitals = float(obs[base + obs_layout.P_ORBITALS]) * 10.0
  monoliths = float(obs[base + obs_layout.P_MONOLITHS]) * 6.0
  amb = float(obs[base + obs_layout.P_AMBASSADOR_HELD]) * 5.0
  return vp + (w_colony * colony + w_disk * disks +
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
  # The viewer is always block slot 0 now.
  return potential_block(obs, obs_layout.player_block_start(0), win_squash)


def potential_block(obs, base, win_squash=False):
  """Potential of the seat whose block starts at ``base``.

  VP units for banked/soft; ``win_squash`` maps onto the rank-utility scale.
  """
  mode, _, _ = _phi_cached()
  if mode == "banked":
    v = float(obs[base + obs_layout.P_VP_TOTAL]) * SCORE_DIVISOR
  elif mode == "soft":
    v = phi_soft_block(obs, base)
  else:
    v = 0.0
  return _squash_win(v) if (win_squash and mode in ("banked", "soft")) else v


def potential_opponent(obs_viewer, block, win_squash=False):
  """Backwards-compatible alias -- ``block`` is now a full block start index."""
  return potential_block(obs_viewer, block, win_squash)


def potential_self_vec(obs_batch, win_squash=False):
  """Vectorized ``potential_self`` over a (num_envs, obs) batch.

  Column reads only, no per-env Python loop (hot shaping path).
  """
  mode, (w_col, w_disk, w_struct, w_amb), _ = _phi_cached()
  # The viewer is always block slot 0.
  return potential_block_vec(
      obs_batch,
      np.full(obs_batch.shape[0], obs_layout.player_block_start(0),
              dtype=np.int64),
      win_squash)


def potential_block_vec(obs_batch, blocks, win_squash=False):
  """Vectorized potential; ``blocks`` is (num_envs,) block-start index per row.

  Self and opponent share one block schema, so a single formula now serves both
  -- the two vectorized variants used to duplicate it against different offsets.
  """
  n = obs_batch.shape[0]
  ar = np.arange(n)
  mode, (w_col, w_disk, w_struct, w_amb), _ = _phi_cached()
  blocks = np.asarray(blocks, dtype=np.int64)
  if mode == "banked":
    v = obs_batch[ar, blocks + obs_layout.P_VP_TOTAL] * SCORE_DIVISOR
  elif mode == "soft":
    v = (obs_batch[ar, blocks + obs_layout.P_VP_TOTAL] * SCORE_DIVISOR
         + w_col * obs_batch[ar, blocks + obs_layout.P_COLONY_TOTAL] * 12.0
         + w_disk * obs_batch[ar, blocks + obs_layout.P_DISKS_ON_SECTORS] * 16.0
         + w_struct * (obs_batch[ar, blocks + obs_layout.P_ORBITALS] * 10.0
                       + obs_batch[ar, blocks + obs_layout.P_MONOLITHS] * 6.0)
         + w_amb * obs_batch[ar, blocks + obs_layout.P_AMBASSADOR_HELD] * 5.0)
  else:
    v = np.zeros(n, dtype=np.float32)
  if win_squash and mode in ("banked", "soft"):
    return np.clip(v / SCORE_DIVISOR, -0.5, 1.0)
  return v


def potential_opponent_vec(obs_batch, blocks, win_squash=False):
  """Backwards-compatible alias for ``potential_block_vec``."""
  return potential_block_vec(obs_batch, blocks, win_squash)


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
      "entropy_coef": agent.entropy_coef,
      "kl_ema": getattr(agent, "_kl_ema", None),
      "ent_ema": getattr(agent, "_ent_ema", None),
      "ent_control_count": getattr(agent, "_ent_control_count", None),
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
  if "learning_rate" in state:
    # Restore the *base* LR so the anneal continues from where it stopped
    # instead of restarting at the full base rate every resume.
    agent.set_learning_rate(float(state["learning_rate"]))
  if "entropy_coef" in state:
    agent.entropy_coef = float(state["entropy_coef"])
  for key in ("_kl_ema", "_ent_ema", "_ent_control_count"):
    if state.get(key) is not None:
      setattr(agent, key, state[key])
  return True


def _write_arch(roster_dir, num_actions, input_shape, aux_tasks):
  """Persists the network architecture beside the roster.

  ``roster.json`` stores only weights and indices, not how the network was
  built, so an off-line consumer (roster_ladder.py) could not reload
  checkpoints without re-deriving the arch from the run's command line.
  ``arch.json`` makes rosters self-describing going forward; consumers fall
  back to the caller's arch flags when it is absent (pre-roster-ladder runs).
  """
  arch = {
      "width": FLAGS.nn_width,
      "depth": FLAGS.nn_depth,
      "norm": FLAGS.nn_norm,
      "activation": FLAGS.nn_activation,
      "separate_critic": FLAGS.separate_critic,
      "factored_actions": bool(FLAGS.factored_actions),
      "aux_tasks": list(aux_tasks or ()),
      "num_actions": int(num_actions),
      "input_shape": list(input_shape),
      "encoder": FLAGS.encoder,
      "spatial_pointer": bool(FLAGS.spatial_pointer),
  }
  with open(os.path.join(str(roster_dir), "arch.json"), "w") as f:
    json.dump(arch, f, indent=2)


def _step_controllers(agent, writer, update, num_updates):
  """Mutates LR / ent_coef once per update, per --lr_schedule / ent bounds.

  Shared by the async and sync loops so the two can never drift apart. Returns
  a dict of control scalars for the writer (empty when nothing fired).

  LR is mutually exclusive by schedule: 'kl' uses the closed-loop controller,
  'anneal' the historical linear decay, 'fixed' leaves it alone. All LR writes
  go through set_learning_rate so base/param-groups stay aligned. The entropy
  band is independent and rate-limited to a far longer time constant.
  """
  ctrl = {}
  schedule = FLAGS.lr_schedule
  if schedule is None:
    # Default to 'fixed'. This used to default to the closed-loop 'kl'
    # controller, which meant every run that did not explicitly pass
    # --lr_schedule silently enabled a controller that has never been validated
    # on a real run. With the observed approx_kl ~0.003 against --kl_target=0.02
    # it computes mult = exp(0.85) ~ 2.34 every update and, because
    # --kl_lr_tau=0.05 barely moves the EMA, saturates --lr_max=1e-2 (40x base)
    # within ~5 updates. Every run that produced a good policy so far used a
    # fixed/near-fixed LR, so that is the safe default. Opt in with
    # --lr_schedule=kl, and watch control/lr for the first 50 updates.
    schedule = "fixed"

  if schedule == "anneal":
    agent.anneal_learning_rate(
        min(agent.updates_done, num_updates - 1), num_updates)
    ctrl["lr"] = agent.optimizer.param_groups[0]["lr"]
  elif schedule == "kl":
    kl = (agent.last_metrics or {}).get("kl")
    if kl is not None:
      mult = agent.kl_step_lr(
          float(kl), FLAGS.kl_target, FLAGS.kl_lr_tau, FLAGS.lr_min,
          FLAGS.lr_max)
      ctrl["kl_ema"] = agent._kl_ema
      ctrl["lr_multiplier"] = mult
      ctrl["lr"] = agent.optimizer.param_groups[0]["lr"]

  if FLAGS.ent_lo is not None and FLAGS.ent_hi is not None:
    ent = (agent.last_metrics or {}).get("entropy")
    if ent is not None:
      new_coef = agent.entropy_band_step(
          float(ent), FLAGS.ent_lo, FLAGS.ent_hi, FLAGS.ent_step,
          FLAGS.ent_control_every)
      ctrl["ent_coef"] = new_coef
      ctrl["ent_ema"] = agent._ent_ema

  if writer is not None and ctrl:
    for name, value in ctrl.items():
      writer.add_scalar(f"control/{name}", value, agent.total_steps_done)
  return ctrl


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
    logits = net.dense_logits(x)
    mask = torch.full_like(logits, float("-inf"))
    local = np.full(obs_np.shape[0], -1, dtype=np.int64)
    local[idx] = np.arange(len(idx))
    keep = local[legal_rows] >= 0
    rows = torch.from_numpy(local[legal_rows[keep]]).to(device)
    cols = torch.from_numpy(legal_cols[keep].astype(np.int64)).to(device)
    mask[rows, cols] = logits[rows, cols]
    return mask.argmax(dim=1).detach().cpu().numpy()


def evaluate_batched(policies, lineup, game_strs, num_players, num_games,
                     num_workers, device, main_seats, max_legal,
                     return_seat_utils=False):
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
  seat_utils = []  # (per-game, num_players) rank utilities when requested.
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
          if return_seat_utils:
            seat_utils.append([rank_utility(arrays.rewards[i], s)
                               for s in range(num_players)])
  finally:
    vec.close()
  res = EvalResult(wins, len(utils), ranks, utils)
  if return_seat_utils:
    return res, diag, np.asarray(seat_utils, dtype=np.float64)
  return res, diag


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
        logits = net.dense_logits(x)
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
          logits = agent.network.dense_logits(x)
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
        logits = net.dense_logits(x)
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
              f"  rounds={dstats['rounds_reached']:.1f}/9"
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
    if diag is not None:
      actor_means, all_means = diag.breakdown_summary()
      if actor_means is not None:
        for name, am, al in zip(diag.vp_categories, actor_means, all_means):
          writer.add_scalar(f"vp_breakdown/actor_{name}", float(am),
                            agent.total_steps_done)
          writer.add_scalar(f"vp_breakdown/allseats_{name}", float(al),
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
    # Breakdown targets are normalized to [0,1] in the tensor, so the probe
    # (which has no terminal obs) is skipped for that mode.
    if probe is not None:
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
        separate_critic=FLAGS.separate_critic, factored_actions=factored,
        encoder=FLAGS.encoder, spatial_pointer=FLAGS.spatial_pointer,
        channels_last=FLAGS.channels_last,
        compile_encoder=FLAGS.compile_encoder),
      value_mode=FLAGS.value_mode,
      aux_tasks=aux_tasks,
      aux_target_fn=aux_target_fn,
      aux_coef=FLAGS.aux_coef,
      rank_vp_beta=(FLAGS.rank_vp_beta if FLAGS.value_mode == "win" else 0.0),
      rank_ce_coef=(FLAGS.rank_ce_coef if FLAGS.value_mode == "win"
                    else 0.0),
      amp=FLAGS.amp,
  )

  # Device + resume telemetry before any training starts.
  _emit(f"device={device}  game={FLAGS.game}  num_envs={FLAGS.num_envs}"
        f"  num_workers={FLAGS.num_workers}")
  if FLAGS.resume:
    agent_fn_r = make_agent_fn(
        FLAGS.nn_width, FLAGS.nn_depth, tuple(aux_tasks or ()),
        norm=FLAGS.nn_norm, activation=FLAGS.nn_activation,
        separate_critic=FLAGS.separate_critic, factored_actions=factored,
        encoder=FLAGS.encoder, spatial_pointer=FLAGS.spatial_pointer,
        channels_last=FLAGS.channels_last,
        compile_encoder=FLAGS.compile_encoder)
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
        separate_critic=FLAGS.separate_critic, factored_actions=factored,
        encoder=FLAGS.encoder, spatial_pointer=FLAGS.spatial_pointer,
        channels_last=FLAGS.channels_last,
        compile_encoder=FLAGS.compile_encoder)
  num_actions = game.num_distinct_actions()
  roster = None
  matchmaker = None
  exploit_victim_net = None
  # A roster is created whenever there is anything to checkpoint, not only in
  # league mode: without this a plain self-play run wrote no weights at all.
  if FLAGS.league or FLAGS.exploit_victim or FLAGS.snapshot_every > 0:
    roster = PolicyRoster(FLAGS.roster_dir)
    _write_arch(roster.save_dir, num_actions, input_shape, aux_tasks)
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
            # phi(s') for the seat that just acted, read out of whichever seat
            # is now to move. Every seat has an identical block and the tensor
            # is canonicalised to the viewer, so this is one wrapping
            # subtraction -- the old self-vs-opponent branch (and its two
            # different offset layouts) is gone.
            blocks = (obs_layout.PLAYERS_START
                      + ((seats - new_seats) % num_players)
                      * obs_layout.PLAYER_SIZE)
            phi_next = potential_block_vec(step_arrays.obs, blocks, win_squash)
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
          diag.record_breakdown(obs_batch, agent.last_seats, donor_idx)
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
      _step_controllers(agent, writer, update, num_updates)
      if FLAGS.rank_vp_beta_anneal_to >= 0.0:
        agent.anneal_rank_vp_beta(agent.updates_done, num_updates,
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
          # The async loop otherwise flushes the writer only at a hard exit;
          # a mid-run flush makes verdict utility observable live in TFEvents
          # instead of appearing only when the process ends.
          writer.flush()
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
              # One identical block per seat, canonicalised to the viewer, so
              # the same-seat and different-seat cases are the same read.
              base = obs_layout.player_block_start(
                  obs_layout.slot_for_seat(seat, viewer, num_players))
              phi_next = potential_block(obs_viewer, base, win_squash)
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
          diag.record_breakdown(obs_batch, agent.last_seats, finished)
        for i in finished:
          ret = float(unreset[i].rewards[0])
          episode_returns[i].append(ret)
          recent_returns.append(ret)

      agent.learn(time_step)

      _step_controllers(agent, writer, update, num_updates)
      if FLAGS.rank_vp_beta_anneal_to >= 0.0:
        agent.anneal_rank_vp_beta(agent.updates_done, num_updates,
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
