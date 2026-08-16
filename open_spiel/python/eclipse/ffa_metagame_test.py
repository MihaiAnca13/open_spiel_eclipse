# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Focused tests for four-player empirical-game evaluation."""

import numpy as np
import torch

from absl.testing import absltest

import pyspiel
from open_spiel.python.egt import alpharank
from open_spiel.python.eclipse import ffa_metagame
from open_spiel.python.examples import ppo_eclipse as pe


def _lowest_legal_action(unused_obs, legal_actions):
  del unused_obs
  return int(legal_actions[0])


def _highest_legal_action(unused_obs, legal_actions):
  del unused_obs
  return int(legal_actions[-1])


class FfaMetagameTest(absltest.TestCase):

  def test_alpharank_numeric_compute_does_not_require_visualizer(self):
    payoff_tables = [np.zeros((2, 2, 2, 2), dtype=np.float64) for _ in range(4)]
    _, _, pi, _, shape = alpharank.compute(payoff_tables, use_inf_alpha=True)

    np.testing.assert_array_equal(shape, [2, 2, 2, 2])
    self.assertEqual(pi.shape, (16,))
    self.assertAlmostEqual(float(pi.sum()), 1.0)

  def test_payoff_tables_preserve_seat_and_profile_axes(self):
    samples = np.empty((2, 2, 2, 2, 3, 4), dtype=np.float64)
    for seat in range(4):
      for replicate in range(3):
        samples[..., replicate, seat] = 100 * seat + replicate

    tables = ffa_metagame.payoff_tables_from_samples(samples)

    self.assertLen(tables, 4)
    for seat, table in enumerate(tables):
      self.assertEqual(table.shape, (2, 2, 2, 2))
      np.testing.assert_allclose(table, 100 * seat + 1)

  def test_one_shot_evaluation_is_input_ordered_and_reproducible(self):
    game_strings = [
        "eclipse(players=4,rng_seed=7)",
        "eclipse(players=4,rng_seed=9)",
    ]
    game = pyspiel.load_game(game_strings[0])
    policies = {"low": _lowest_legal_action, "high": _highest_legal_action}
    lineups = [
        ["low", "high", "low", "high"],
        ["high", "low", "high", "low"],
    ]
    seeds = [11, 12]

    first = self._evaluate(policies, lineups, game_strings, seeds,
                           game.num_distinct_actions())
    repeated = self._evaluate(policies, lineups, game_strings, seeds,
                              game.num_distinct_actions())
    reordered = self._evaluate(
        policies, list(reversed(lineups)), list(reversed(game_strings)),
        list(reversed(seeds)), game.num_distinct_actions())

    self.assertEqual(first[0].games, 2)
    np.testing.assert_array_equal(first[0].utils, repeated[0].utils)
    np.testing.assert_array_equal(first[2], repeated[2])
    np.testing.assert_array_equal(reordered[0].utils, first[0].utils[::-1])
    np.testing.assert_array_equal(reordered[2], first[2][::-1])

  def _evaluate(self, policies, lineups, game_strings, seeds, max_legal):
    return pe.evaluate_batched(
        policies, lineups, game_strings, 4, len(game_strings), 1,
        torch.device("cpu"), (0,), max_legal, return_seat_utils=True,
        sampler_seeds=seeds, one_episode_per_env=True)


if __name__ == "__main__":
  absltest.main()
