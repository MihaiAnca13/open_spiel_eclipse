#!/usr/bin/env python3
"""Tests for the frozen-checkpoint collector (TDD: RED -> GREEN).

The four designated frozen checkpoints (runs/long_v2/{snap_u2500,main}.pt,
runs/long8h/main.pt, runs/_judge/baseline/main.pt) were trained with the PRE-V2
``SpatialEclipseEncoder`` (fuse = Linear(4*width, width) -> [64, 256]). The
current repo encoder is V2 (reads the appended keyed-entity block, fuse ->
[64, 320]) and is INCOMPATIBLE with those weights. ``collect_frozen.py``
therefore vendors a standalone pre-V2 encoder (``PreV2Encoder``/``PreV2Net``)
that consumes exactly the pre-V2 24714-length obs prefix while the DATASET still
records the FULL 37596-length engine observation row.

This suite covers:
  * the stub-policy smoke path (no weights) exercises the FULL pipeline and must
    record the FULL 37596-wide obs rows (not the 24714 prefix),
  * the schema fields (rank_target / vp_components / vp_all_seats / return_ /
    episode_id) and the round-trip through ``frozen_dataset`` readers,
  * the REAL pre-V2 checkpoint load (gated on the checkpoint files being
    present): the vendored ``PreV2Net`` must load with no missing actor head,
    the encoder ``fuse`` must be the pre-V2 ``[64, 256]`` geometry, and a
    forward on the 24714-length pre-V2 obs prefix must yield finite logits.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch
import pyspiel
from absl import flags as absl_flags

from open_spiel.python.eclipse import frozen_dataset, obs_layout

import collect_frozen

# The collector's seeded board draws go through ``pe._randomized_game_string``,
# which reads absl flags (randomize_races, ...). ``main()`` parses them; the
# tests drive the collector in-process, so parse them here first.
absl_flags.FLAGS(["collect_frozen_test"])

_GAME_STR = collect_frozen._GAME_STR
_NUM_PLAYERS = 4
_FULL_OBS_LEN = obs_layout.TOTAL          # 37596, asserted by obs_layout._self_check
_PRE_V2_LEN = obs_layout.V2_KEYED_START   # 24714

_OWNDIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_OWNDIR)
_REAL_CKPT = os.path.join(_REPO, "runs", "long_v2", "main.pt")


def _collect_stub(tmp, stub=True):
  """Run the collector in-process (workers=0) for one stub-policy episode."""
  game = pyspiel.load_game(_GAME_STR)
  num_actions = game.num_distinct_actions()
  arch = {
      "width": 64, "depth": 2, "norm": False, "activation": "tanh",
      "separate_critic": False, "factored_actions": False,
      "aux_tasks": ["final_rank"], "num_actions": int(num_actions),
      "input_shape": [_PRE_V2_LEN], "encoder": "spatial",
  }
  with open(os.path.join(tmp, "arch.json"), "w") as f:
    json.dump(arch, f, indent=2)
  ckpt = os.path.join(tmp, "main.pt")
  args = (ckpt, tmp, "cpu", 0.99, 1000, 1, _GAME_STR, stub)
  _, eps = collect_frozen._collect_worker(args)
  return eps


class CollectStubTest(unittest.TestCase):

  def _collect_n(self, n_episodes, seeds=None):
    """Collect `n_episodes` stub episodes and fill episode_id like main()."""
    with tempfile.TemporaryDirectory() as tmp:
      eps = []
      for i in range(n_episodes):
        eps.extend(_collect_stub(tmp, stub=True))
      ep_id = 0
      for ep in eps:
        n = int(ep["seat"].shape[0])
        ep["episode_id"] = np.full(n, ep_id, dtype=np.int32)
        ep_id += 1
      return tmp, eps

  def test_full_obs_rows_are_37596_wide(self):
    """Recorded obs is the FULL engine row (37596), not the 24714 prefix."""
    _, eps = self._collect_n(2)
    for ep in eps:
      self.assertEqual(ep["obs"].shape[1], _FULL_OBS_LEN)
      self.assertEqual(ep["obs"].shape[1], 37596)

  def test_schema_field_shapes(self):
    """rank_target (n,4), vp_components (n,9), vp_all_seats (n,4,9), scalars (n,)."""
    _, eps = self._collect_n(1)
    for ep in eps:
      n = int(ep["seat"].shape[0])
      self.assertGreater(n, 0)
      self.assertEqual(ep["seat"].shape, (n,))
      self.assertEqual(ep["round_idx"].shape, (n,))
      self.assertEqual(ep["phase"].shape, (n,))
      self.assertEqual(ep["rank_target"].shape, (n, 4))
      self.assertEqual(ep["rank_utility_terminal"].shape, (n,))
      self.assertEqual(ep["vp_components"].shape, (n, 9))
      self.assertEqual(ep["vp_all_seats"].shape, (n, _NUM_PLAYERS, 9))
      self.assertEqual(ep["return_"].shape, (n,))

  def test_return_is_finite(self):
    """Every row's discounted return_ is finite."""
    _, eps = self._collect_n(2)
    for ep in eps:
      self.assertTrue(np.isfinite(ep["return_"]).all(),
                      "non-finite return_ found")

  def test_round_trip_schema_unchanged_and_episode_id_filled(self):
    """write_episodes -> read_rows preserves obs width and the whole schema."""
    tmp, eps = self._collect_n(3)
    manifest = frozen_dataset.default_manifest(
        _NUM_PLAYERS, _FULL_OBS_LEN, 0.99, vp_scale=1.0, rank_vp_beta=0.0,
        split_seed=0)
    written = frozen_dataset.write_episodes(tmp, manifest, eps)
    # episode_id filled for every row by the write path.
    for ep in eps:
      self.assertIn("episode_id", ep)
      self.assertGreater(int(ep["episode_id"].max()), -1)
      self.assertEqual(ep["episode_id"].shape, ep["seat"].shape)
    read = frozen_dataset.read_rows(tmp, written)
    self.assertEqual(read["obs"].shape[1], _FULL_OBS_LEN)
    self.assertEqual(read["rank_target"].shape[1], 4)
    self.assertEqual(read["vp_components"].shape[1], 9)
    self.assertEqual(read["vp_all_seats"].shape[1:], (_NUM_PLAYERS, 9))
    self.assertIn("return_", read)
    self.assertIn("episode_id", read)


@unittest.skipUnless(os.path.exists(_REAL_CKPT),
                     f"real pre-V2 checkpoint absent: {_REAL_CKPT}")
class RealCheckpointLoadTest(unittest.TestCase):

  def setUp(self):
    with open(os.path.join(os.path.dirname(_REAL_CKPT), "arch.json")) as f:
      self.arch = json.load(f)
    self.width = int(self.arch["width"])
    self.depth = int(self.arch["depth"])
    self.activation = str(self.arch["activation"])
    self.num_actions = int(self.arch["num_actions"])

  def test_preV2Net_loads_with_no_missing_actor(self):
    """The checkpoint's actor head fully loads (no missing key starts with actor)."""
    net = collect_frozen.PreV2Net(
        self.width, self.depth, self.activation, self.num_actions)
    sd = torch.load(_REAL_CKPT, map_location="cpu", weights_only=True)
    missing, _ = net.load_state_dict(sd, strict=False)
    missing_actor = [k for k in missing if k.startswith("actor")]
    self.assertEqual(missing_actor, [],
                     f"actor weights missing after load: {missing_actor}")

  def test_fuse_is_prev2_256_geometry(self):
    """Pre-V2 encoder fuse is Linear(4*width, width) -> [64,256] (not 320)."""
    net = collect_frozen.PreV2Net(
        self.width, self.depth, self.activation, self.num_actions)
    self.assertEqual(tuple(net.shared.fuse.weight.shape),
                     (self.width, 4 * self.width))
    self.assertEqual(tuple(net.shared.fuse.weight.shape), (64, 256))

  def test_forward_on_prev2_prefix_is_finite(self):
    """net.actor on the 24714-prefix produces finite full-width logits."""
    net = collect_frozen.PreV2Net(
        self.width, self.depth, self.activation, self.num_actions)
    sd = torch.load(_REAL_CKPT, map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=False)
    net.eval()
    x = torch.zeros(2, _PRE_V2_LEN)
    with torch.no_grad():
      logits = net.actor(x)
    self.assertEqual(logits.shape, (2, self.num_actions))
    self.assertTrue(torch.isfinite(logits).all())

  def test_real_checkpoint_one_episode_full_obs(self):
    """One real-load episode records 37596-wide obs and reaches terminal."""
    with tempfile.TemporaryDirectory() as tmp:
      with open(os.path.join(os.path.dirname(_REAL_CKPT), "arch.json")) as f:
        json.dump(json.load(f), open(os.path.join(tmp, "arch.json"), "w"))
      sd = torch.load(_REAL_CKPT, map_location="cpu", weights_only=True)
      torch.save(sd, os.path.join(tmp, "main.pt"))
      net, _ = collect_frozen._build_net(
          tmp, os.path.join(tmp, "main.pt"), "cpu",
          pyspiel.load_game(_GAME_STR).num_distinct_actions(),
          tuple(self.arch["input_shape"]),
          pyspiel.load_game(_GAME_STR), stub_policy=False)
      ep, _ = collect_frozen._play_one_episode(
          net, "cpu", _GAME_STR, 0.99, 4242)
      self.assertGreater(int(ep["seat"].shape[0]), 0)
      self.assertEqual(ep["obs"].shape[1], _FULL_OBS_LEN)
      self.assertEqual(ep["obs"].shape[1], 37596)


if __name__ == "__main__":
  unittest.main()
