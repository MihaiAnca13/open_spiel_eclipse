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

  def test_repeated_prune_keeps_a_genuine_mid_run_snapshot(self):
    """Pruning after EVERY snapshot, as _maybe_snapshot does, must keep a spread.

    The test above prunes once at the end, which any spacing rule passes. This
    one prunes on every snapshot, which is what breaks an index-based rule.
    """
    with tempfile.TemporaryDirectory() as d:
      roster = PolicyRoster(d)
      roster.record_main(self._make_net(), update=0)
      last = 100 * 22
      for u in range(100, last + 1, 100):          # 22 snapshots, as a long run
        roster.add_snapshot(self._make_net(), update=u)
        roster.prune(keep_recent=4, keep_spaced=4)

      births = sorted(e.birth_update for e in roster.entries.values()
                      if e.role != "main")
      self.assertLessEqual(len(births), 8, f"prune did not bound: {births}")
      self.assertGreaterEqual(len(births), 5, f"pruned too hard: {births}")

      # The point of keep_spaced: something must survive from the MIDDLE of the
      # run, not just the head and the tail.
      lo, hi = births[0], births[-1]
      mid_lo, mid_hi = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
      in_middle = [b for b in births if mid_lo <= b <= mid_hi]
      self.assertTrue(
          in_middle,
          f"no snapshot survived in the middle half of [{lo}, {hi}]: {births}. "
          f"Spacing must be by birth_update, not by index in the surviving "
          f"list, or repeated pruning collapses to the two ends.")

      # And the largest gap must not swallow most of the run.
      gaps = [b - a for a, b in zip(births, births[1:])]
      self.assertLess(
          max(gaps), 0.6 * (hi - lo),
          f"largest gap {max(gaps)} spans most of [{lo}, {hi}]: {births}")


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

  def test_live_opponent_set_is_bounded_but_still_covers_the_roster(self):
    """A batch may hold few distinct opponents; the run must still see them all.

    ``PPO._act_sparse`` runs one encoder forward per distinct policy in the
    batch, so per-batch policy count is a throughput term, not a free parameter:
    256 rows measured 1.12x at 4 distinct policies, 2.05x at 8 and 8.32x at 32
    versus a single group. Left unbounded it grows with the roster and the run
    decays -- a league smoke run went from 22.8 to 41.7 ms/step act over 45
    updates, still climbing.

    Both halves matter. Bounding without refresh would train against a fixed
    handful of snapshots forever, which is a silent *quality* regression rather
    than a speed one, so assert coverage too.

    The bound is 2x the cap, not the cap: around a refresh the outgoing and
    incoming live sets briefly coexist. That is not a test artifact -- in real
    training an env keeps its lineup until its episode ends, so opponents from
    the previous live set stay in play for some steps after the switch. 2x8 =
    the 2.05x point on the cost curve, transiently, which the measured act
    plateau (~21 ms flat from update 25 to 55) confirms is fine.
    """
    with tempfile.TemporaryDirectory() as d:
      roster = PolicyRoster(d)
      roster.record_main(_TinyNet(5), update=1)
      for u in range(2, 32):
        roster.add_snapshot(_TinyNet(5), update=u)
      self.assertGreaterEqual(len(roster.opponent_ids(exclude_main=True)), 30)

      mm = Matchmaker(roster, num_envs=64, num_players=4, train_pid="main",
                      selfplay_fraction=0.0, old_fraction=0.0, seed=3,
                      max_live_opponents=4, live_refresh=200)

      # Any single batch stays inside the cap, allowing for one refresh
      # straddling it. Without the bound this would be ~30.
      for _ in range(20):
        batch = set(mm.lineups().reshape(-1).tolist()) - {"main"}
        self.assertLessEqual(
            len(batch), 8,
            f"{len(batch)} distinct opponents in one batch exceeds 2x the cap")

      # ...but over many refreshes the run reaches most of the roster.
      seen = set()
      for _ in range(400):
        seen.update(set(mm.lineups().reshape(-1).tolist()) - {"main"})
      self.assertGreater(
          len(seen), 20,
          f"only {len(seen)} of 30 snapshots ever entered play; the live set is "
          f"not refreshing and the league has quietly become fixed-opponent")

  def test_max_live_opponents_zero_restores_unbounded_sampling(self):
    with tempfile.TemporaryDirectory() as d:
      roster = PolicyRoster(d)
      roster.record_main(_TinyNet(5), update=1)
      for u in range(2, 20):
        roster.add_snapshot(_TinyNet(5), update=u)
      mm = Matchmaker(roster, num_envs=256, num_players=4, train_pid="main",
                      selfplay_fraction=0.0, old_fraction=0.0, seed=4,
                      max_live_opponents=0)
      batch = set(mm.lineups().reshape(-1).tolist()) - {"main"}
      self.assertGreater(len(batch), 4)


if __name__ == "__main__":
  absltest.main()
