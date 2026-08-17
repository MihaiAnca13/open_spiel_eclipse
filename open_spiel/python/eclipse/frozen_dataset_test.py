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
"""Tests for the frozen diagnostic dataset schema (TDD: RED -> GREEN)."""

import os
import tempfile

import numpy as np

from open_spiel.python.eclipse import frozen_dataset, obs_layout
from open_spiel.python.pytorch import ppo

NUM_PLAYERS = 4
OBS_LEN = 2000  # synthetic (>= global + 4 player blocks); real caller passes 24714.
GAMMA = 0.995
VP_SCALE = 200.0


def _make_obs(steps, rounds, phases):
  """Synthetic obs rows with valid round/phase one-hots in the global block."""
  obs = np.zeros((steps, OBS_LEN), dtype=np.float32)
  onehot = np.zeros((steps, obs_layout.SEAT_SLOTS), dtype=np.float32)
  onehot[np.arange(steps), (rounds + np.arange(steps)) % 4] = 1.0
  obs[:, obs_layout.GLOBAL_START + frozen_dataset.GLOBAL_ROUND_ONEHOT:
         obs_layout.GLOBAL_START + frozen_dataset.GLOBAL_ROUND_ONEHOT +
         frozen_dataset.GLOBAL_ROUND_WIDTH] = 0.0
  obs[np.arange(steps), obs_layout.GLOBAL_START +
      frozen_dataset.GLOBAL_ROUND_ONEHOT + (rounds % 9)] = 1.0
  obs[:, obs_layout.GLOBAL_START + frozen_dataset.GLOBAL_PHASE_ONEHOT:
         obs_layout.GLOBAL_START + frozen_dataset.GLOBAL_PHASE_ONEHOT +
         frozen_dataset.GLOBAL_PHASE_WIDTH] = 0.0
  obs[np.arange(steps), obs_layout.GLOBAL_START +
      frozen_dataset.GLOBAL_PHASE_ONEHOT + phases] = 1.0
  return obs


def _make_episode(episode_id, seat, n_steps):
  """A synthetic full-schema episode dict of `n_steps` acting steps for `seat`."""
  rounds = np.arange(n_steps) % 9
  phases = np.arange(n_steps) % 4
  obs = _make_obs(n_steps, rounds, phases)
  np.random.seed(episode_id)
  rank_target = np.random.rand(n_steps, 4).astype(np.float32)
  rank_target /= rank_target.sum(axis=1, keepdims=True)
  return {
      "obs": obs,
      "seat": np.full(n_steps, seat, dtype=np.int32),
      "episode_id": np.full(n_steps, episode_id, dtype=np.int32),
      "round_idx": np.zeros(n_steps, dtype=np.int32),
      "phase": np.zeros(n_steps, dtype=np.int32),
      "rank_target": rank_target.astype(np.float32),
      "rank_utility_terminal": np.random.rand(n_steps).astype(np.float32),
      "vp_components": np.random.rand(n_steps, 9).astype(np.float32),
      "vp_all_seats": np.random.rand(n_steps, NUM_PLAYERS, 9).astype(np.float32),
      "return_": np.random.rand(n_steps).astype(np.float32),
  }


def test_round_trip_exact_equality():
  """Write a synthetic full-schema batch, read it back, assert exact equality."""
  manifest = frozen_dataset.default_manifest(
      num_players=NUM_PLAYERS, obs_len=OBS_LEN, gamma=GAMMA, vp_scale=VP_SCALE,
      rank_vp_beta=0.0)
  episodes = [_make_episode(11, 0, 5), _make_episode(12, 2, 7),
              _make_episode(13, 3, 4)]

  with tempfile.TemporaryDirectory() as tmp:
    written = frozen_dataset.write_episodes(tmp, manifest, episodes)
    with open(os.path.join(tmp, "manifest.json")) as f:
      import json
      back = json.load(f)
    assert set(back) == set(written)

    read = frozen_dataset.read_rows(tmp, written)
    for arr in read.values():
      assert isinstance(arr, np.ndarray)

    orig = frozen_dataset.concat_episodes(episodes)
    for key in ("obs", "seat", "episode_id", "rank_target", "vp_components",
                "vp_all_seats", "return_"):
      np.testing.assert_array_equal(read[key], orig[key])
    # round_idx/phase are derived from obs by the module, so match those too.
    np.testing.assert_array_equal(read["round_idx"], orig["round_idx"])
    np.testing.assert_array_equal(read["phase"], orig["phase"])


def test_manifest_metadata_populated_on_write():
  """default_manifest's None placeholders are filled by write_episodes.

  The schema docstring promises code_revision / collector_cli / collection_ts
  are "populated in write_episodes if not supplied" -- but default_manifest
  pre-fills them with None, which defeats a setdefault (the key exists). This
  pins that the on-disk manifest gets a real code revision, the cli, and a
  timestamp so a collected dataset is traceable to the exact code revision.
  """
  manifest = frozen_dataset.default_manifest(
      num_players=NUM_PLAYERS, obs_len=OBS_LEN, gamma=GAMMA, vp_scale=VP_SCALE,
      rank_vp_beta=0.0, collector_cli="pytest test_manifest_metadata")
  episodes = [_make_episode(7, 0, 4)]

  with tempfile.TemporaryDirectory() as tmp:
    frozen_dataset.write_episodes(tmp, manifest, episodes)
    with open(os.path.join(tmp, "manifest.json")) as f:
      import json
      back = json.load(f)
    # code_revision must not be the None placeholder (real git SHAs are truthy).
    assert back.get("code_revision"), "code_revision not populated on disk"
    assert back.get("collector_cli") == "pytest test_manifest_metadata"
    assert back.get("collection_ts"), "collection_ts not populated on disk"


def test_split_disjointness():
  """train/val episode sets are disjoint; a step's episode_id decides its side."""
  manifest = frozen_dataset.default_manifest(
      num_players=NUM_PLAYERS, obs_len=OBS_LEN, gamma=GAMMA, vp_scale=VP_SCALE,
      rank_vp_beta=0.0, split_seed=42)
  episodes = [_make_episode(100 + i, i % NUM_PLAYERS, 3) for i in range(20)]

  with tempfile.TemporaryDirectory() as tmp:
    written = frozen_dataset.write_episodes(tmp, manifest, episodes)
    with open(os.path.join(tmp, "manifest.json")) as f:
      import json
      back = json.load(f)
    train_ids = set(back["split"]["train"])
    val_ids = set(back["split"]["val"])
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) + len(val_ids) == 20

    read = frozen_dataset.read_rows(tmp, written)
    for train_ep in train_ids:
      rows = (read["episode_id"] == train_ep)
      assert rows.any()
      # every row of a train episode is labelled train
    for val_ep in val_ids:
      rows = read["episode_id"] == val_ep
      assert rows.any()
    # cross-check: no episode id appears on both sides
    ids = set(np.unique(read["episode_id"]).tolist())
    assert ids == train_ids | val_ids
    assert ids.isdisjoint(train_ids & val_ids)


def test_viewer_relative_reorder_is_its_own_inverse():
  """slot_for_seat round-tripped is the identity for every seat."""
  for viewer in range(NUM_PLAYERS):
    for seat in range(NUM_PLAYERS):
      slot = obs_layout.slot_for_seat(seat, viewer, NUM_PLAYERS)
      back = obs_layout.seat_for_slot(slot, viewer, NUM_PLAYERS)
      assert back == seat, (viewer, seat, slot)
      assert 0 <= slot < NUM_PLAYERS


def test_rank_target_sums_to_one_and_ties_uniform():
  """rank_target sums to 1; an all-tied outcome is uniform over 4 ranks."""
  rng = np.random.RandomState(0)
  for _ in range(20):
    vp = rng.rand(NUM_PLAYERS).astype(np.float32)
    for seat in range(NUM_PLAYERS):
      t = frozen_dataset.rank_target(vp, seat)
      np.testing.assert_allclose(t.sum(), 1.0, atol=1e-6)
      assert t.shape == (4,)
  # all-tied outcome -> uniform over the 4 ranks.
  tied = np.full(NUM_PLAYERS, 3.0, dtype=np.float32)
  for seat in range(NUM_PLAYERS):
    t = frozen_dataset.rank_target(tied, seat)
    np.testing.assert_allclose(t, np.full(4, 0.25), atol=1e-6)


def test_rank_utility_terminal_constant_sum():
  """sum over seats of rank_utility_terminal == sum(DEFAULT_RANK_UTILITY)."""
  rng = np.random.RandomState(1)
  expected = float(sum(ppo.DEFAULT_RANK_UTILITY))
  for _ in range(20):
    vp = rng.rand(NUM_PLAYERS).astype(np.float32)
    total = sum(frozen_dataset.rank_utility_terminal(vp, s)
                for s in range(NUM_PLAYERS))
    np.testing.assert_allclose(total, expected, atol=1e-6)


def test_discounted_return_shaping_chain():
  """discounted_return matches hand-computed per-seat chain sum."""
  chain = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  rewards = {0: chain, 1: np.array([0.5, 0.5], dtype=np.float32)}
  got = frozen_dataset.discounted_return(rewards, 0, 0.5)
  want = 1.0 + 0.5 * 2.0 + 0.5 ** 2 * 3.0
  np.testing.assert_allclose(got, want, rtol=1e-6)
