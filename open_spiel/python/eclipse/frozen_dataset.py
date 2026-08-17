# Copyright 2026 DeepMind Technologies Limited
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
"""Frozen diagnostic dataset schema for the Eclipse 4-player RL project.

This module is the **data contract** between the collector (which writes
episodes) and the offline critic work (which reads them). It is deliberately
dependency-light: only numpy + the standard library (json/os) at the boundary,
plus read-only reuse of ``obs_layout`` (offsets) and ``ppo`` (rank targets).

One record = one acting step of one seat in one episode. Each episode is
written to its own ``.npz``; a ``manifest.json`` beside them carries the
metadata and the (seeded 80/20, whole-episode) train/val split.

Schema of a per-episode dict (all arrays shaped ``(n_steps, ...)``):

  obs                  (n, obs_len) f32   acting seat's observation
                                             (``info_state[seat]``)
  seat                 (n,) i32           acting/viewing seat 0..3
  episode_id           (n,) i32           whole-episode key
  round_idx            (n,) i32           round, derived from obs global block
  phase                (n,) i32           phase, derived from obs global block
  rank_target          (n, 4) f32         soft tie target over 4 ranks
  rank_utility_terminal(n,) f32           undiscounted terminal rank utility
  vp_components        (n, 9) f32         acting seat's 9 VP components
  vp_all_seats         (n, 4, 9) f32      all 4 seats' 9 components,
                                            viewer-relative slot order
  return_              (n,) f32           discounted return under intended
                                            shaping+gamma (see discounted_return)

Field computations (mirror of open_spiel/games/eclipse/observation.h block A):

  round  : argmax of the round one-hot at obs[1..9] (width 9 == kMaxRounds).
           If the one-hot is degenerate the round is ambiguous -> store -1.
  phase  : argmax of the phase one-hot at obs[10..13] (width 4 ==
           kRoundPhaseCount; order ACTION, COMBAT, UPKEEP, CLEANUP). Degenerate
           -> store -1.
  vp     : obs[player_block_start(slot) + P_VP_BREAKDOWN : +9]. In the acting
           seat's own obs the viewer is slot 0.

``obs_len`` is never hardcoded here: it comes from the manifest (the real
training value is 24714 per arch.json; 37596 is the raw engine TOTAL). The
module only touches the global (0..13) and player (146..146+3*547) sub-blocks,
so any obs_len that covers them works.

``return_`` follows next-work.md: "store the exact discounted return produced
by the intended shaping and gamma". It is the discounted sum of the shaped
reward stream over the acting seat's own decision chain **plus** the
undiscounted terminal utility discounted to the first step:

    return_ = sum_k g^k * r_shaped[t+k]  +  g^n * rank_utility_terminal

where n is the length of the seat's own decision chain. ``discounted_return``
computes the shaped-chain part; the collector composes the final term. Tests
treat ``return_`` as an opaque persisted field.
"""

import json
import os
import time

import numpy as np

from open_spiel.python.eclipse import obs_layout
from open_spiel.python.pytorch import ppo

# ── global-block offsets (observation.h block A) ──────────────────────────
# Block A: 1 (round/kMaxRounds scalar) + kMaxRounds one-hot + kRoundPhaseCount
# one-hot + ...  Global block starts at obs_layout.GLOBAL_START (0).
_K_MAX_ROUNDS = 9
_K_ROUND_PHASE_COUNT = 4
GLOBAL_ROUND_SCALAR = 0
GLOBAL_ROUND_ONEHOT = 1
GLOBAL_ROUND_WIDTH = _K_MAX_ROUNDS
GLOBAL_PHASE_ONEHOT = GLOBAL_ROUND_ONEHOT + GLOBAL_ROUND_WIDTH
GLOBAL_PHASE_WIDTH = _K_ROUND_PHASE_COUNT

_NUM_COMPONENTS = 9

# ── split ─────────────────────────────────────────────────────────────────
_SPLIT_TRAIN_FRAC = 0.8


def _episode_split(episode_id, split_seed):
  """Deterministic train/val label for a whole episode (order-independent).

  Seeded on ``(episode_id, split_seed)`` so episodes get a stable side no
  matter what order the collector streams them in.
  """
  rng = np.random.RandomState(int(episode_id) * 1000003 + int(split_seed))
  return "train" if rng.rand() < _SPLIT_TRAIN_FRAC else "val"


# ── rank helpers (thin wrappers over ppo) ─────────────────────────────────
def rank_target(terminal_vp, seat):
  """Soft 4-way tie-aware placement target for ``seat``."""
  return ppo.rank_distribution(np.asarray(terminal_vp, dtype=np.float32), seat)


def rank_utility_terminal(terminal_vp, seat):
  """Undiscounted terminal rank utility for ``seat`` (vp_beta=0)."""
  return ppo.rank_utility(np.asarray(terminal_vp, dtype=np.float32), seat,
                          vp_beta=0.0)


# ── VP components ──────────────────────────────────────────────────────────
def vp_components_viewer_relative(terminal_obs_row, seat, num_players):
  """The 9 VP components of ``seat``, viewer-relative, from one obs row.

  ``terminal_obs_row`` is the acting seat's observation, so the viewer is the
  acting seat itself. ``seat`` is that acting seat (or any seat in an episode
  owned by it); the seat's own block is always slot 0 of the viewer's obs.
  """
  slot = obs_layout.slot_for_seat(seat, seat, num_players)
  base = obs_layout.player_block_start(slot) + obs_layout.P_VP_BREAKDOWN
  return np.asarray(
      terminal_obs_row[base:base + _NUM_COMPONENTS], dtype=np.float32)


def _vp_all_seats_viewer_relative(terminal_obs_row, num_players):
  """(num_players, 9) components in viewer-relative slot order (viewer=slot 0)."""
  out = np.zeros((num_players, _NUM_COMPONENTS), dtype=np.float32)
  for slot in range(num_players):
    base = obs_layout.player_block_start(slot) + obs_layout.P_VP_BREAKDOWN
    out[slot] = np.asarray(terminal_obs_row[base:base + _NUM_COMPONENTS],
                           dtype=np.float32)
  return out


# ── obs-derived fields ─────────────────────────────────────────────────────
def derive_round_phase(obs):
  """Vectorized (n,) round_idx and (n,) phase from a stack of obs rows."""
  obs = np.asarray(obs)
  round_seg = obs[:, GLOBAL_ROUND_ONEHOT:GLOBAL_ROUND_ONEHOT + GLOBAL_ROUND_WIDTH]
  phase_seg = obs[:, GLOBAL_PHASE_ONEHOT:GLOBAL_PHASE_ONEHOT + GLOBAL_PHASE_WIDTH]
  round_valid = np.allclose(round_seg.sum(axis=1), 1.0, atol=1e-4)
  phase_valid = np.allclose(phase_seg.sum(axis=1), 1.0, atol=1e-4)
  round_idx = np.where(round_valid, round_seg.argmax(axis=1), -1).astype(np.int32)
  phase = np.where(phase_valid, phase_seg.argmax(axis=1), -1).astype(np.int32)
  return round_idx, phase


# ── discounted return ──────────────────────────────────────────────────────
def discounted_return(shaped_rewards_per_seat, seat, gamma):
  """Discounted sum of the shaped reward stream over ``seat``'s own chain.

  ``shaped_rewards_per_seat`` is either a dict ``{seat: 1d array}`` or a 2D
  sequence whose row ``seat`` is that seat's shaped rewards in chronological
  order. Returns ``sum_k g^k * r[k]`` for that chain (the terminal-utility term
  is added separately by the collector; see the module docstring).
  """
  if isinstance(shaped_rewards_per_seat, dict):
    chain = np.asarray(shaped_rewards_per_seat[seat], dtype=np.float32)
  else:
    chain = np.asarray(shaped_rewards_per_seat[seat], dtype=np.float32)
  powers = gamma ** np.arange(len(chain))
  return float(np.dot(chain, powers))


# ── IO ─────────────────────────────────────────────────────────────────────
def default_manifest(num_players, obs_len, gamma, vp_scale, rank_vp_beta,
                     split_seed=0, **extra):
  """Seed metadata; caller fills policy hashes, seeds, config, cli, revision."""
  manifest = {
      "schema_version": 1,
      "num_players": num_players,
      "obs_len": obs_len,
      "gamma": gamma,
      "vp_scale": vp_scale,
      "rank_vp_beta": rank_vp_beta,
      "split_seed": split_seed,
      "split": {"train": [], "val": []},
      "npz": [],
      # populated in write_episodes if not supplied:
      "code_revision": None,
      "collector_cli": None,
      "collection_ts": None,
  }
  manifest.update(extra)
  return manifest


def write_episodes(out_dir, manifest_dict, batches):
  """Write one .npz per episode plus a manifest.json; returns the manifest.

  ``batches`` is an iterable of per-episode schema dicts. The train/val split
  is whole-episode, seeded from ``manifest_dict["split_seed"]``, and the
  resulting episode-id lists are stored in the manifest.
  """
  os.makedirs(out_dir, exist_ok=True)
  manifest = dict(manifest_dict)
  manifest.setdefault("npz", [])
  manifest.setdefault("split", {"train": [], "val": []})
  manifest.setdefault("code_revision", _git_revision())
  manifest.setdefault("collection_ts", time.time())
  split_seed = manifest.get("split_seed", 0)

  train_ids = set(manifest["split"]["train"])
  val_ids = set(manifest["split"]["val"])
  for episode in batches:
    if "round_idx" not in episode or "phase" not in episode:
      round_idx, phase = derive_round_phase(episode["obs"])
      episode = dict(episode)
      episode.setdefault("round_idx", round_idx)
      episode.setdefault("phase", phase)
    ep_id = int(episode["episode_id"][0])
    side = _episode_split(ep_id, split_seed)
    (train_ids if side == "train" else val_ids).add(ep_id)
    fname = f"episode_{ep_id:08d}.npz"
    np.savez(os.path.join(out_dir, fname), **episode)
    manifest["npz"].append(fname)

  manifest["split"] = {
      "train": sorted(train_ids),
      "val": sorted(val_ids),
  }
  with open(os.path.join(out_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
  return manifest


def read_rows(out_dir, manifest):
  """Load every npz in ``manifest`` and concatenate into one dict of arrays."""
  result = None
  for fname in manifest["npz"]:
    with np.load(os.path.join(out_dir, fname), allow_pickle=False) as data:
      arrays = {k: data[k] for k in data.files}
    if result is None:
      result = {k: [arrays[k]] for k in arrays}
    else:
      for k, v in arrays.items():
        result[k].append(v)
  if result is None:
    return {}
  return {k: np.concatenate(v, axis=0) if len(v) > 1 else v[0]
          for k, v in result.items()}


def concat_episodes(episodes):
  """Concatenate a list of per-episode schema dicts into one dict of arrays."""
  keys = set()
  for ep in episodes:
    keys.update(ep.keys())
  return {k: np.concatenate([ep[k] for ep in episodes], axis=0)
          for k in keys}


def _git_revision():
  """Best-effort git rev-parse short SHA; None if unavailable."""
  import subprocess
  try:
    out = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stderr=subprocess.DEVNULL)
    return out.decode().strip()
  except (OSError, subprocess.SubprocessError):
    return None
