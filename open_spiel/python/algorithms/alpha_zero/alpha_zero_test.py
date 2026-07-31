# Copyright 2019 DeepMind Technologies Limited
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

"""Tests for the N-player training-loop logic, without spinning up actors."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import utils


class BuildValueTargetTest(absltest.TestCase):

  def test_every_state_gets_the_full_normalized_returns_vector(self):
    # Deliberately non-zero-sum, 3-player returns -- if the old
    # `game_outcome = trajectory.returns[0]` bug were still present, every
    # state's value target would collapse to a single scalar derived only
    # from player 0's return, discarding players 1 and 2 entirely.
    returns = [10.0, 200.0, 45.0]
    min_utility, max_utility = 0.0, 255.0

    target = alpha_zero.build_value_target(returns, min_utility, max_utility)

    self.assertEqual(target.shape, (3,))
    expected = utils.normalize_value(
        np.asarray(returns), min_utility, max_utility
    )
    np.testing.assert_allclose(target, expected)
    # The three players' normalized targets must be distinct, i.e. the
    # per-player structure survived the normalization step.
    self.assertLen(set(target.tolist()), 3)

  def test_identity_for_symmetric_zero_sum_range(self):
    # For games like tic_tac_toe (min_utility=-1, max_utility=1), the
    # normalization must be the identity -- this is what makes the change
    # behavior-preserving for existing 2-player zero-sum games.
    returns = [1.0, -1.0]
    target = alpha_zero.build_value_target(returns, -1.0, 1.0)
    np.testing.assert_allclose(target, returns)


class WinnerBucketTest(absltest.TestCase):

  def test_single_winner(self):
    self.assertEqual(alpha_zero.winner_bucket([10, 200, 45], 3), 1)
    self.assertEqual(alpha_zero.winner_bucket([1.0, -1.0], 2), 0)

  def test_tie_goes_to_draw_bucket(self):
    self.assertEqual(alpha_zero.winner_bucket([5, 5, 5], 3), 3)

  def test_two_way_tie_among_more_players_goes_to_draw_bucket(self):
    self.assertEqual(alpha_zero.winner_bucket([5, 5, 1, 0], 4), 4)


if __name__ == "__main__":
  absltest.main()
