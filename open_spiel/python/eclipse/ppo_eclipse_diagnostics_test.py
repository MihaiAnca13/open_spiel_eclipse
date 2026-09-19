"""Unit tests for Eclipse PPO pilot-health telemetry."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.examples.ppo_eclipse import EpisodeDiagnostics
from open_spiel.python.examples.ppo_eclipse import _NORMAL_TERMINAL_ROUND
from open_spiel.python.examples.ppo_eclipse import apply_stalled_game_penalty
from open_spiel.python.examples.ppo_eclipse import apply_universal_bankruptcy_penalty
from open_spiel.python.pytorch.ppo import rank_utility


class EpisodeDiagnosticsTest(absltest.TestCase):

  def test_universal_bankruptcy_loses_to_any_normal_rank_outcome(self):
    """A score edge must not make eliminating everyone a PPO attractor."""
    rewards = np.asarray([[4.0, 3.0, 2.0, 1.0],
                          [8.0, 7.0, 6.0, 5.0]], dtype=np.float32)
    eliminated = np.asarray([[True, True, True, True],
                           [True, True, False, True]])

    adjusted = apply_universal_bankruptcy_penalty(
        rewards, eliminated, penalty=2.0)

    np.testing.assert_allclose(adjusted[0], [-1.0, -1.5, -2.0, -2.5])
    np.testing.assert_allclose(adjusted[1], [1.0, 0.5, 0.0, -0.5])
    self.assertLess(float(adjusted[0].max()), -0.5)

  def test_stalled_game_loses_to_any_normal_rank_outcome(self):
    """Leading when the move cap lands must not beat playing the game out."""
    targets = np.asarray([[1.0, 0.5, 0.0, -0.5],
                          [1.0, 0.5, 0.0, -0.5]], dtype=np.float32)
    rounds = np.asarray([4, _NORMAL_TERMINAL_ROUND])

    adjusted = apply_stalled_game_penalty(targets.copy(), rounds, penalty=2.0)

    # The stalled game sinks for every seat...
    np.testing.assert_allclose(adjusted[0], [-1.0, -1.5, -2.0, -2.5])
    # ...and a game that actually finished is untouched.
    np.testing.assert_allclose(adjusted[1], [1.0, 0.5, 0.0, -0.5])
    # Even the best stall is worse than the worst honest result, so no seat
    # can gain by running the clock out.
    self.assertLess(float(adjusted[0].max()), -0.5)

  def test_zeroing_payoffs_would_pay_a_losing_seat_to_stall(self):
    """Why the guard shifts the targets instead of flattening the payoffs.

    An all-equal payoff vector is an all-TIE, which fractional ranking scores
    at the mean of the table. That is well above last place, so "nobody scores"
    would be a promotion for the seat that was losing.
    """
    self.assertAlmostEqual(rank_utility([0.0, 0.0, 0.0, 0.0], 0), 0.25)
    self.assertAlmostEqual(rank_utility([30.0, 20.0, 10.0, 5.0], 3), -0.5)
    self.assertGreater(rank_utility([0.0, 0.0, 0.0, 0.0], 0),
                       rank_utility([30.0, 20.0, 10.0, 5.0], 3))

  def test_stall_penalty_disabled_is_a_no_op(self):
    targets = np.asarray([[1.0, 0.5, 0.0, -0.5]], dtype=np.float32)
    out = apply_stalled_game_penalty(targets.copy(), np.asarray([4]), 0.0)
    np.testing.assert_allclose(out, targets)

  def test_terminal_summary_separates_normal_and_safety_endings(self):
    diag = EpisodeDiagnostics(
        num_envs=2, num_players=2, history=4,
        terminal_target_fn=lambda payoff, seat: payoff[seat] / 10.0)
    # Env 0 reaches round eight with one eliminated player; env 1 ends through
    # the move-count backstop before normal round closeout.
    diag.elim_round[0, 1] = 3
    safety = diag.close_episodes(
        [0, 1], np.asarray([[10.0, 5.0], [4.0, 4.0]], dtype=np.float32),
        np.asarray([_NORMAL_TERMINAL_ROUND, 6], dtype=np.int16))

    self.assertEqual(safety, [(1, 6)])
    summary = diag.summary()
    self.assertEqual(summary["normal_ending_rate"], 0.5)
    self.assertEqual(summary["safety_cap_endings"], 1.0)
    self.assertEqual(summary["all_tied_rate"], 0.5)
    self.assertEqual(summary["mean_eliminations"], 0.5)
    self.assertAlmostEqual(summary["terminal_target_variance"], 0.061875)

  def test_terminal_elimination_captures_last_action_wipeout(self):
    diag = EpisodeDiagnostics(num_envs=1, num_players=2, history=4)
    diag.elim_round[0, 0] = 3

    diag.close_episodes(
        [0], np.asarray([[2.0, 1.0]], dtype=np.float32),
        terminal_eliminated=np.asarray([[True, True]]))

    self.assertEqual(list(diag.survivors), [0])
    self.assertEqual(list(diag.wipeouts), [1])
    self.assertEqual(list(diag.eliminations), [2])


if __name__ == "__main__":
  absltest.main()
