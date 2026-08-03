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
"""Tests for the league roster + matchmaker (open_spiel.python.examples.league)."""

import os
import tempfile
from absl.testing import absltest
import numpy as np
import torch
from torch import nn

from open_spiel.python.examples.league import Matchmaker
from open_spiel.python.examples.league import PolicyRoster


class _TinyNet(nn.Module):

  def __init__(self, num_actions, observation_shape=None, device="cpu"):
    super().__init__()
    del observation_shape
    self.actor = nn.Linear(8, num_actions)

  def forward(self, x):
    return self.actor(x)


class PolicyRosterTest(absltest.TestCase):

  def _make_net(self):
    net = _TinyNet(5)
    with torch.no_grad():
      net.actor.weight.normal_(0, 1)
    return net

  def test_save_load_roundtrip(self):
    with tempfile.TemporaryDirectory() as d:
      roster = PolicyRoster(d)
      net = self._make_net()
      before = net.actor.weight.detach().clone()
      roster.record_main(net, update=10)
      roster.add_snapshot(self._make_net(), update=20)
      roster.add_exploiter(self._make_net(), update=30, against_victim="snap_u20",
                           win_rate=0.6)

      # Fresh roster object reads the same index + weights.
      roster2 = PolicyRoster(d)
      self.assertEqual(len(roster2), 3)
      self.assertEqual(roster2.main_id, "main")
      loaded = roster2.load_net("main", _TinyNet, 5, (8,), "cpu")
      torch.testing.assert_close(loaded.actor.weight, before)
      self.assertIsNotNone(roster2.get("snap_u20"))
      self.assertEqual(roster2.get("expl_u30_vsnap_u20").win_rate, 0.6)

  def test_load_missing_returns_none(self):
    with tempfile.TemporaryDirectory() as d:
      roster = PolicyRoster(d)
      self.assertIsNone(roster.load_net("main", _TinyNet, 5, (8,), "cpu"))

  def test_prune_bounds_and_keeps_file_spread(self):
    with tempfile.TemporaryDirectory() as d:
      roster = PolicyRoster(d)
      roster.record_main(self._make_net(), update=0)
      for u in range(1, 21):
        roster.add_snapshot(self._make_net(), update=u)
      self.assertEqual(len(roster), 21)
      before_files = len([f for f in os.listdir(d) if f.endswith(".pt")])
      roster.prune(keep_recent=4, keep_spaced=4)
      after_files = len([f for f in os.listdir(d) if f.endswith(".pt")])
      # Main + 8 kept snapshots.
      self.assertEqual(after_files, 9)
      self.assertEqual(len(roster), 9)
      self.assertLess(after_files, before_files)


class MatchmakerTest(absltest.TestCase):

  def test_lineup_shapes_and_selfplay_seats(self):
    with tempfile.TemporaryDirectory() as d:
      roster = PolicyRoster(d)
      roster.record_main(_TinyNet(5), update=1)
      roster.add_snapshot(_TinyNet(5), update=2)
      mm = Matchmaker(roster, num_envs=8, num_players=4, train_pid="main",
                      selfplay_fraction=1.0, seed=0)
      lineups = mm.lineups()
      self.assertEqual(lineups.shape, (8, 4))
      self.assertTrue(np.all(lineups == "main"))

  def test_mixed_lineups_include_main_and_opponents(self):
    with tempfile.TemporaryDirectory() as d:
      roster = PolicyRoster(d)
      roster.record_main(_TinyNet(5), update=1)
      roster.add_snapshot(_TinyNet(5), update=2)
      mm = Matchmaker(roster, num_envs=64, num_players=4, train_pid="main",
                      selfplay_fraction=0.0, old_fraction=0.0, seed=1)
      lineups = mm.lineups()
      # With no self-play, opponents always come from the roster; but the
      # exploit-victim fallback keeps main on seat 0 is enforced in sample_lineup.
      self.assertEqual(lineups.shape, (64, 4))
      ids = set(lineups.reshape(-1).tolist())
      self.assertIn("main", ids)
      self.assertIn("snap_u2", ids)


if __name__ == "__main__":
  absltest.main()
