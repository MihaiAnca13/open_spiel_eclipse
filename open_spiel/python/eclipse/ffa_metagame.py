# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Collect and analyze a four-policy, four-player Eclipse empirical game.

Each policy profile is evaluated once per supplied scenario seed. The raw
per-seat utilities are saved so the AlphaRank report can be reproduced without
replaying games.
"""

import hashlib
import itertools
import json
import os

from absl import app
from absl import flags
import numpy as np
import torch

import pyspiel
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import utils as egt_utils
from open_spiel.python.eclipse.action_factors import factorization_from_game
from open_spiel.python.eclipse import roster_ladder
from open_spiel.python.examples import ppo_eclipse as pe
from open_spiel.python.examples.league import PolicyRoster

FLAGS = flags.FLAGS

flags.DEFINE_string("metagame_roster_dir", None,
                    "Roster containing the four policies to evaluate.")
flags.DEFINE_string("metagame_policy_ids", None,
                    "Exactly four comma-separated roster policy ids.")
flags.DEFINE_integer("metagame_replicates", 32,
                     "Matched setup/chance scenarios per policy profile.")
flags.DEFINE_integer("metagame_envs", 64,
                     "One-shot evaluation environments per batch.")
flags.DEFINE_integer("metagame_workers", 16,
                     "Async evaluation workers.")
flags.DEFINE_integer("metagame_setup_seed", 100000,
                     "First deterministic Eclipse setup seed.")
flags.DEFINE_integer("metagame_chance_seed", 200000,
                     "First external chance-sampler seed.")
flags.DEFINE_integer("metagame_boot", 300,
                     "Global-replicate bootstrap samples for AlphaRank masses.")
flags.DEFINE_integer("metagame_boot_seed", 0,
                     "Bootstrap random seed.")
flags.DEFINE_string("metagame_out", None,
                    "Output report JSON path; raw utilities use the same .npz base.")


def _sha256(path):
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def payoff_tables_from_samples(samples):
  """Converts ``(K,K,K,K,R,4)`` utilities into four payoff tensors."""
  samples = np.asarray(samples, dtype=np.float64)
  if samples.ndim != 6 or samples.shape[-1] != 4:
    raise ValueError(f"expected (K,K,K,K,R,4) utilities, got {samples.shape}")
  if len(set(samples.shape[:4])) != 1:
    raise ValueError(f"policy axes must have equal length, got {samples.shape}")
  if not np.isfinite(samples).all():
    raise ValueError("utilities contain incomplete evaluation results")
  return [samples[..., seat].mean(axis=4) for seat in range(4)]


def _marginals(payoff_tables, pi):
  shape = np.asarray(payoff_tables[0].shape, dtype=np.int32)
  marginals = np.zeros((len(shape), shape[0]), dtype=np.float64)
  for profile_id, mass in enumerate(pi):
    profile = egt_utils.get_strat_profile_from_id(shape, profile_id)
    for seat, policy_index in enumerate(profile):
      marginals[seat, policy_index] += mass
  return marginals


def _alpharank_report(samples, boot, boot_seed, policy_ids):
  payoff_tables = payoff_tables_from_samples(samples)
  pi, epsilon = alpharank.sweep_pi_vs_epsilon(
      payoff_tables, return_epsilon=True)
  marginals = _marginals(payoff_tables, pi)
  profile_shape = np.asarray(payoff_tables[0].shape, dtype=np.int32)
  top = np.argsort(pi)[::-1][:min(8, len(pi))]
  report = {
      "epsilon": float(epsilon),
      "seat_marginals": marginals.tolist(),
      "seat_average": marginals.mean(axis=0).tolist(),
      "top_profiles": [{
          "policies": [policy_ids[i] for i in egt_utils.get_strat_profile_from_id(
              profile_shape, int(profile_id))],
          "mass": float(pi[profile_id]),
      } for profile_id in top],
  }
  if boot <= 0:
    return report

  replicates = samples.shape[4]
  rng = np.random.RandomState(boot_seed)
  masses = np.empty((boot, 4, len(policy_ids)), dtype=np.float64)
  for b in range(boot):
    indices = rng.randint(0, replicates, size=replicates)
    tables = [samples[:, :, :, :, indices, seat].mean(axis=4)
              for seat in range(4)]
    _, _, boot_pi, _, _ = alpharank.compute(
        tables, use_inf_alpha=True, inf_alpha_eps=epsilon)
    masses[b] = _marginals(tables, boot_pi)
  average = masses.mean(axis=1)
  report["seat_average_ci"] = np.percentile(
      average, [2.5, 97.5], axis=0).T.tolist()
  return report


def _load_policies(roster_dir, policy_ids, game, device):
  roster = PolicyRoster(roster_dir)
  input_shape = tuple(game.observation_tensor_shape())
  num_actions = game.num_distinct_actions()
  arch = roster_ladder._resolve_arch(roster_dir, num_actions, input_shape)
  factored = None
  if arch.get("factored_actions"):
    factored = factorization_from_game(game)
  agent_fn = pe.make_agent_fn(
      int(arch["width"]), int(arch["depth"]),
      tuple(arch.get("aux_tasks") or ()), norm=bool(arch["norm"]),
      activation=arch["activation"],
      separate_critic=bool(arch["separate_critic"]),
      factored_actions=factored, encoder=str(arch.get("encoder", "flat")))
  policies = {}
  metadata = []
  for policy_id in policy_ids:
    entry = roster.get(policy_id)
    if entry is None or not entry.path or not os.path.exists(entry.path):
      raise ValueError(f"roster has no weights for {policy_id!r}")
    policies[policy_id] = roster_ladder._load_net_tolerant(
        agent_fn, entry.path, num_actions, input_shape, device, policy_id).to(device)
    metadata.append({
        "id": policy_id,
        "birth_update": int(entry.birth_update),
        "path": entry.path,
        "sha256": _sha256(entry.path),
    })
  return policies, metadata, num_actions


def _collect_samples(policies, policy_ids, game, device):
  replicates = FLAGS.metagame_replicates
  if replicates <= 0 or FLAGS.metagame_envs <= 0:
    raise ValueError("metagame_replicates and metagame_envs must be positive")
  profiles = list(itertools.product(range(4), repeat=4))
  tasks = [(profile, replicate) for profile in profiles
           for replicate in range(replicates)]
  samples = np.full((4, 4, 4, 4, replicates, 4), np.nan, dtype=np.float32)
  setup_seeds = [FLAGS.metagame_setup_seed + r for r in range(replicates)]
  chance_seeds = [FLAGS.metagame_chance_seed + r for r in range(replicates)]
  max_legal = game.num_distinct_actions()

  for start in range(0, len(tasks), FLAGS.metagame_envs):
    batch = tasks[start:start + FLAGS.metagame_envs]
    lineups = [[policy_ids[i] for i in profile] for profile, _ in batch]
    game_strs = [pe._randomized_game_string(FLAGS.game, setup_seeds[replicate])
                 for _, replicate in batch]
    sampler_seeds = [chance_seeds[replicate] for _, replicate in batch]
    _, _, utilities = pe.evaluate_batched(
        policies, lineups, game_strs, game.num_players(), len(batch),
        FLAGS.metagame_workers, device, (0,), max_legal,
        return_seat_utils=True, sampler_seeds=sampler_seeds,
        one_episode_per_env=True)
    for row, (profile, replicate) in enumerate(batch):
      samples[profile + (replicate, slice(None))] = utilities[row]
    print(f"evaluated {min(start + len(batch), len(tasks))}/{len(tasks)} profiles")
  return samples, setup_seeds, chance_seeds


def main(_):
  if not FLAGS.metagame_roster_dir or not FLAGS.metagame_policy_ids:
    raise ValueError("--metagame_roster_dir and --metagame_policy_ids are required")
  if not FLAGS.metagame_out:
    raise ValueError("--metagame_out is required")
  if FLAGS.metagame_out.endswith(".npz"):
    raise ValueError("--metagame_out must name the JSON report, not the raw .npz")
  if os.path.exists(FLAGS.metagame_out):
    raise FileExistsError(f"refusing to overwrite {FLAGS.metagame_out}")
  policy_ids = [x.strip() for x in FLAGS.metagame_policy_ids.split(",") if x.strip()]
  if len(policy_ids) != 4 or len(set(policy_ids)) != 4:
    raise ValueError("--metagame_policy_ids must contain exactly four unique ids")

  device = torch.device("cuda" if torch.cuda.is_available() and FLAGS.cuda else "cpu")
  game = pyspiel.load_game(FLAGS.game)
  if game.num_players() != 4:
    raise ValueError(
        "FFA collector requires a four-player game, got "
        f"{game.num_players()}")
  policies, metadata, _ = _load_policies(
      FLAGS.metagame_roster_dir, policy_ids, game, device)
  samples, setup_seeds, chance_seeds = _collect_samples(
      policies, policy_ids, game, device)
  analysis = _alpharank_report(
      samples, FLAGS.metagame_boot, FLAGS.metagame_boot_seed, policy_ids)
  raw_path = os.path.splitext(FLAGS.metagame_out)[0] + ".npz"
  if os.path.exists(raw_path):
    raise FileExistsError(f"refusing to overwrite {raw_path}")
  np.savez_compressed(raw_path, utilities=samples)
  report = {
      "schema_version": 1,
      "game": FLAGS.game,
      "action_selection": "greedy_argmax",
      "roster_dir": FLAGS.metagame_roster_dir,
      "policies": metadata,
      "setup_seeds": setup_seeds,
      "chance_seeds": chance_seeds,
      "replicates": FLAGS.metagame_replicates,
      "raw_utilities": raw_path,
      "raw_utilities_axes": [
          "seat_0_policy", "seat_1_policy", "seat_2_policy",
          "seat_3_policy", "replicate", "utility_seat"],
      "alpharank": analysis,
  }
  with open(FLAGS.metagame_out, "w") as f:
    json.dump(report, f, indent=2)
  print(f"wrote {raw_path} and {FLAGS.metagame_out}")


if __name__ == "__main__":
  app.run(main)
