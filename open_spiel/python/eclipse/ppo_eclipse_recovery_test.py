"""Regression tests for Eclipse PPO run-recoverability fixes.

Covers the two directly-testable pieces of "make runs recoverable": stale
league-opponent eviction and forcing a final checkpoint off the snapshot
cadence. The --resume optimizer-state gating lives inline in main() and is a
one-line id comparison, not exercised here.
"""

import os
import shutil
import tempfile

from absl.testing import absltest
import numpy as np
import torch

from open_spiel.python.examples.league import PolicyRoster
from open_spiel.python.examples.ppo_eclipse import FLAGS
from open_spiel.python.examples.ppo_eclipse import _maybe_snapshot
from open_spiel.python.examples.ppo_eclipse import _refresh_lineups
from open_spiel.python.examples.ppo_eclipse import _train_state_path

# _maybe_snapshot reads --snapshot_every/--roster_keep_* off the global FLAGS.
# Under a plain `pytest` invocation (unlike absltest.main()) nothing parses
# them first, so every flag access raises UnparsedFlagAccessError.
FLAGS(["ppo_eclipse_recovery_test"])


def _agent_fn(num_actions, input_shape, device):
  del num_actions, input_shape
  return torch.nn.Linear(2, 2).to(device)


class _FakeAgent(object):

  def __init__(self, network, lineup):
    self.network = network
    self.networks = {"main": network}
    self.lineup = lineup
    self.optimizer = torch.optim.Adam(network.parameters())
    self.total_steps_done = 123
    self.updates_done = 7
    self.rank_vp_beta = 0.5
    self.learning_rate = 1e-4
    self.entropy_coef = 0.01


class _FakeMatchmaker(object):

  def __init__(self, next_lineup):
    self.next_lineup = next_lineup

  def sample_lineup(self):
    return self.next_lineup


class RefreshLineupsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._tmp)

  def test_evicts_opponents_no_longer_in_any_lineup(self):
    roster = PolicyRoster(self._tmp)
    roster.save_weights("old_opp", torch.nn.Linear(2, 2))
    roster.save_weights("new_opp", torch.nn.Linear(2, 2))

    lineup = np.array([["main", "old_opp"], ["main", "old_opp"]],
                      dtype=object)
    agent = _FakeAgent(torch.nn.Linear(2, 2), lineup)
    agent.networks["old_opp"] = roster.load_net(
        "old_opp", _agent_fn, 4, (4,), "cpu")

    # Only env 0 resets. Env 1 keeps "old_opp" live, so it must survive.
    matchmaker = _FakeMatchmaker(["main", "new_opp"])
    _refresh_lineups(agent, matchmaker, roster, _agent_fn, 4, (4,), "cpu",
                     done_flags=[True, False])
    self.assertIn("new_opp", agent.networks)
    self.assertIn("old_opp", agent.networks)

    # Now every env has rotated off "old_opp": it must be evicted, even
    # though roster pruning was never called.
    matchmaker.next_lineup = ["main", "new_opp"]
    _refresh_lineups(agent, matchmaker, roster, _agent_fn, 4, (4,), "cpu",
                     done_flags=[False, True])
    self.assertNotIn("old_opp", agent.networks)
    self.assertIn("main", agent.networks)
    self.assertIn("new_opp", agent.networks)


class MaybeSnapshotForceTest(absltest.TestCase):

  def test_force_saves_even_off_the_snapshot_cadence(self):
    tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp)
    roster = PolicyRoster(tmp)
    agent = _FakeAgent(torch.nn.Linear(2, 2), np.zeros((1, 1), dtype=object))
    agent.updates_done = 7  # not a multiple of --snapshot_every's default 25

    _maybe_snapshot(agent, roster, update=0, force=False)
    self.assertFalse(os.path.exists(_train_state_path(str(roster.save_dir))))

    _maybe_snapshot(agent, roster, update=0, force=True)
    self.assertTrue(os.path.exists(_train_state_path(str(roster.save_dir))))
    self.assertEqual(roster.entries["main"].birth_update, 7)


if __name__ == "__main__":
  absltest.main()
