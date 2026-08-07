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
"""Elo/rating ladder over a policy roster (Sprint B1 item).

Rates the snapshots of a **single** training run against each other in a
round-robin, on fixed held-out boards, using tie-aware rank utility -- so
training improvement is measurable once the fixed Random/Greedy baselines are
saturated. Policies of one roster are played in one tournament; ratings from
different rosters land on a common scale only because Random is pinned at 0
in every tournament (metrics are identified only up to an additive constant).

Reuses the training evaluator end-to-end:
  * ``ppo_eclipse.evaluate_batched`` (async, N games in parallel on paired
    boards) -- with ``return_seat_utils=True`` so every seat is scored, not
    just the "main" side.
  * ``ppo_eclipse.make_agent_fn`` / ``factorization_from_game`` to rebuild each
    run's network architecture (from ``arch.json`` when present, else from the
    caller's arch flags, which match the run's launch).
  * ``ppo.rank_utility`` (tie-aware) for scoring; never a raw win rate.
  * ``pyspiel.elo.compute_ratings_from_matrices`` as a binarized cross-check.

Rating model
  No win/loss binarization for the headline number. In a 2-policy game the two
  sides' mean rank utilities are constant-sum (sum of the utility table is
  1.0), so the per-game margin ``m = u_i - u_j`` carries all the information.
  We fit per-policy ratings ``r`` to ``m = r_i - r_j`` by least squares
  (Random anchored at 0) -- a margin-aware rating on a readable scale (a rating
  of ~0.10 = a persistent +0.10 utility edge over the field). The binarized
  Bradley-Terry Elo is reported alongside; disagreement between the two
  orderings is treated as a signal, not smoothed over.

Monotonicity
  Judgements are made on **well-separated** snapshots (adjacent 200-update
  snapshots inside an 18-minute run are nearly indistinguishable; their
  inversions are noise). We report Kendall tau + exact permutation p over
  coarse age representatives, plus which cross-bucket regressions clear the
  bootstrap CI. A regression that clears the CI is a real finding.
"""

import json
import os
from absl import app
from absl import flags
import numpy as np
import torch

import pyspiel
from open_spiel.python.eclipse.action_factors import factorization_from_game
from open_spiel.python.examples import ppo_eclipse as pe
from open_spiel.python.examples.league import PolicyRoster

FLAGS = flags.FLAGS

# Ladder-specific flags. The arch/eval flags (game, seed, nn_*, separate_critic,
# factored_actions, aux_target_mode, eval_envs, num_workers, cuda) are the ones
# ppo_eclipse.py already defines and are reused unchanged.
flags.DEFINE_string("ladder_roster_dir", None,
                    "Roster directory to rate (snapshots + main, optionally "
                    "+ Random/Greedy anchors). Policies are rated only against "
                    "each other within this roster.")
flags.DEFINE_integer("ladder_games_per_dir", 128,
                     "Games per pair per seat direction (both directions are "
                     "played on the same boards, so a full pair sees 2x).")
flags.DEFINE_bool("ladder_include_bots", True,
                  "Add fixed Random and Greedy bots to the tournament. Random "
                  "is pinned at rating 0, which is what makes ratings across "
                  "different rosters comparable.")
flags.DEFINE_bool("ladder_include_heuristic", True,
                  "Also add the observation-aware reference heuristic "
                  "(_GreedyPickV2) as a separate anchor row, so the saturated "
                  "'vs Greedy' column is not the only strength probe.")
flags.DEFINE_integer("ladder_seed_offset", 7777,
                     "Held-out boards are drawn from seed + this offset, fixed "
                     "for the whole tournament (paired comparisons).")
flags.DEFINE_string("ladder_out", None,
                    "JSON artifact path (default <roster_dir>/ladder.json).")
flags.DEFINE_integer("ladder_bins", 3,
                     "Coarse age buckets for the monotonicity test; the newest "
                     "snapshot of each bucket is the representative.")
flags.DEFINE_integer("ladder_min_sep", 200,
                     "Minimum birth-update gap for a pair of snapshots to count "
                     "in the regression scan (adjacent sub-cadence snapshots are "
                     "near-duplicates; their flips are noise).")
flags.DEFINE_float("ladder_alpha", 0.05,
                   "Permutation significance threshold for Kendall tau.")
flags.DEFINE_integer("ladder_boot", 300,
                     "Bootstrap resamples for rating CIs (0 disables).")
flags.DEFINE_integer("ladder_boot_seed", 0,
                     "RNG seed for the bootstrap.")


def _resolve_arch(num_actions, input_shape):
  """Network architecture to rebuild checkpoints with.

  Prefer the persisted ``arch.json`` (written by ppo_eclipse since the ladder
  was introduced); fall back to the caller's arch flags, which reproduce the
  run's own launch flags for pre-existing rosters.
  """
  arch_path = os.path.join(FLAGS.ladder_roster_dir, "arch.json")
  if os.path.exists(arch_path):
    with open(arch_path) as f:
      return json.load(f)
  tasks, _ = pe.build_aux_targets(
      FLAGS.aux_target_mode if FLAGS.aux_coef > 0 else "none",
      FLAGS.aux_vp_scale)
  return {
      "width": FLAGS.nn_width,
      "depth": FLAGS.nn_depth,
      "norm": FLAGS.nn_norm,
      "activation": FLAGS.nn_activation,
      "separate_critic": FLAGS.separate_critic,
      "factored_actions": bool(FLAGS.factored_actions),
      "aux_tasks": list(tasks or ()),
      "num_actions": int(num_actions),
      "input_shape": [int(s) for s in input_shape],
      "encoder": FLAGS.encoder,
  }


def _load_net_tolerant(agent_fn, path, num_actions, input_shape, device,
                       policy_id):
  """Load a checkpoint into a fresh net; only the actor head is required.

  Aux heads can differ across runs (either because of --aux_target_mode or a
  stale state_dict) and are irrelevant for evaluation, which reads only
  ``net.actor``. Mirrors the tolerant load ``ppo_eclipse --resume`` uses.
  """
  net = agent_fn(num_actions, input_shape, device)
  sd = torch.load(path, map_location=device, weights_only=True)
  missing, unexpected = net.load_state_dict(sd, strict=False)
  missing_actor = [k for k in missing if k.startswith("actor")]
  if missing_actor:
    raise RuntimeError(
        f"{policy_id}: actor weights missing after load ({missing_actor}); "
        f"the arch (arch.json/flags) does not match this checkpoint")
  net.eval()
  return net


def _kendall_tau(seq):
  """Kendall tau of the natural order against the value order of ``seq``."""
  n = len(seq)
  if n < 2:
    return 0.0
  c, d = 0, 0
  for i in range(n):
    for j in range(i + 1, n):
      if seq[j] > seq[i]:
        c += 1
      elif seq[j] < seq[i]:
        d += 1
  denom = n * (n - 1) / 2.0
  return (c - d) / denom


def _permutation_p(seq, rng):
  """Two-sided exact permutation p for Kendall tau of ``seq`` (age order)."""
  import itertools
  n = len(seq)
  obs = abs(_kendall_tau(seq))
  count = total = 0
  for perm in itertools.permutations(range(n)):
    perm_seq = [seq[i] for i in perm]
    if abs(_kendall_tau(perm_seq)) >= obs:
      count += 1
    total += 1
  return count / float(total)


def _fit_margin_ratings(policy_ids, anchor_idx, pair_margins, rng=None,
                        n_boot=300):
  """Least-squares margin ratings ``m = r_i - r_j`` with ``anchor`` pinned at 0.

  Returns (ratings, ratings_ci) over ``policy_ids``. ``ratings_ci`` is
  (lo, hi) bootstrap percentile per policy, or (nan,nan) when n_boot==0.
  """
  p = len(policy_ids)
  free = [i for i in range(p) if i != anchor_idx]
  rows, y = [], []
  for (i, j), margins in pair_margins.items():
    for m in margins:
      row = np.zeros(p)
      if i != anchor_idx:
        row[i] += 1.0
      if j != anchor_idx:
        row[j] -= 1.0
      rows.append(row[free])
      y.append(m)
  x = np.asarray(rows, dtype=np.float64)
  yy = np.asarray(y, dtype=np.float64)
  if x.shape[0] == 0:
    return [0.0] * p, [(float("nan"), float("nan"))] * p

  def _solve(xx, yy_):
    coef, _, _, _ = np.linalg.lstsq(xx, yy_, rcond=None)
    out = np.zeros(p)
    out[anchor_idx] = 0.0
    out[free] = coef
    return out - out[anchor_idx]

  ratings = _solve(x, yy)
  lo = np.full(p, np.nan)
  hi = np.full(p, np.nan)
  if n_boot and x.shape[0] >= 2:
    rng = rng or np.random.RandomState(0)
    boot = np.zeros((n_boot, p))
    idx = np.arange(x.shape[0])
    for b in range(n_boot):
      pick = rng.choice(idx, size=x.shape[0], replace=True)
      boot[b] = _solve(x[pick], yy[pick])
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
  return list(ratings), list(zip(lo, hi))


def main(_):
  if not FLAGS.ladder_roster_dir:
    raise ValueError("--ladder_roster_dir is required")
  device = torch.device(
      "cuda" if torch.cuda.is_available() and FLAGS.cuda else "cpu")
  game = pyspiel.load_game(FLAGS.game)
  num_actions = game.num_distinct_actions()
  num_players = game.num_players()
  input_shape = tuple(game.observation_tensor_shape())

  arch = _resolve_arch(num_actions, input_shape)
  factored = None
  if arch.get("factored_actions"):
    factored = factorization_from_game(game)
  agent_fn = pe.make_agent_fn(
      int(arch["width"]), int(arch["depth"]), tuple(arch.get("aux_tasks") or ()),
      norm=bool(arch["norm"]), activation=arch["activation"],
      separate_critic=bool(arch["separate_critic"]),
      factored_actions=factored,
      encoder=str(arch.get("encoder", "flat")))
  print(f"rating roster      : {FLAGS.ladder_roster_dir}")
  print(f"arch (arch.json?)  : width={arch['width']} depth={arch['depth']} "
        f"norm={arch['norm']} act={arch['activation']} "
        f"sep_critic={arch['separate_critic']} "
        f"factored={arch['factored_actions']} aux={arch.get('aux_tasks')} "
        f"encoder={arch.get('encoder')}")
  print(f"device={device}  games/dir={FLAGS.ladder_games_per_dir}  "
        f"eval_envs={FLAGS.eval_envs}  workers={FLAGS.num_workers}")

  roster = PolicyRoster(FLAGS.ladder_roster_dir)
  # Nets: snapshots in birth order, then main. Bots appended last (Random
  # first so it is the natural rating anchor).
  snapshots = sorted([e for e in roster.entries.values()
                      if e.role == "snapshot"], key=lambda e: e.birth_update)
  main_entry = roster.get("main")
  net_policies = [{
      "id": e.policy_id, "kind": "net",
      "birth": int(e.birth_update),
      "net": _load_net_tolerant(agent_fn, e.path, num_actions, input_shape,
                                device, e.policy_id),
  } for e in snapshots]
  if main_entry is not None:
    net_policies.append({
        "id": main_entry.policy_id, "kind": "net",
        "birth": int(main_entry.birth_update),
        "net": _load_net_tolerant(agent_fn, main_entry.path, num_actions,
                                  input_shape, device, main_entry.policy_id),
    })
  policies = list(net_policies)
  for net in policies:
    net["net"].to(device)
  if FLAGS.ladder_include_bots:
    bot_rng = np.random.RandomState(FLAGS.seed + 12345)
    policies.append({
        "id": "Random", "kind": "bot", "birth": None,
        "bot": lambda o, legal: int(bot_rng.choice(np.asarray(legal,
                                                              dtype=np.int32))),
    })
    policies.append({
        "id": "Greedy", "kind": "bot", "birth": None,
        "bot": lambda obs, legal: pe._greedy_pick(
            np.asarray(obs, dtype=np.float32), legal, bot_rng),
    })
  if FLAGS.ladder_include_heuristic:
    policies.append({
        "id": "Heuristic", "kind": "bot", "birth": None,
        "bot": pe._GreedyPickV2(),
    })

  p = len(policies)
  ids = [pol["id"] for pol in policies]
  index = {pol["id"]: i for i, pol in enumerate(policies)}
  print(f"policies            : {p} -> {ids}")

  # Fixed held-out boards for the whole tournament (paired comparisons).
  ladder_seed = FLAGS.seed + FLAGS.ladder_seed_offset
  eval_strs = [pe._randomized_game_string(FLAGS.game, ladder_seed + j)
               for j in range(FLAGS.eval_envs)]
  n_pairs = p * (p - 1) // 2
  print(f"round-robin pairs   : {n_pairs} x {FLAGS.ladder_games_per_dir * 2} games")

  pair_margins = {}
  sums = [0.0] * p
  counts = [0] * p
  win = np.zeros((p, p), dtype=np.int64)
  draw = np.zeros((p, p), dtype=np.int64)

  def _handler(pol):
    return pol["net"] if pol["kind"] == "net" else pol["bot"]

  pair_list = [(i, j) for i in range(p) for j in range(i + 1, p)]
  for n_pair, (i, j) in enumerate(pair_list):
    a, b = policies[i], policies[j]
    dir_margins = []
    for direction in (0, 1):
      lineup = [[a["id"], a["id"], b["id"], b["id"]] for _ in range(len(eval_strs))]
      if direction:
        lineup = [row[::-1] for row in lineup]
      _, _, seat_utils = pe.evaluate_batched(
          {a["id"]: _handler(a), b["id"]: _handler(b)}, lineup, eval_strs,
          num_players, FLAGS.ladder_games_per_dir, FLAGS.num_workers,
          device, (0, 1), num_actions, return_seat_utils=True)
      g = len(seat_utils)
      if g == 0:
        continue
      if direction == 0:  # a seats {0,1}, b seats {2,3}
        ua = seat_utils[:, :2].mean(axis=1)
        ub = seat_utils[:, 2:].mean(axis=1)
      else:               # a seats {2,3}, b seats {0,1}
        ua = seat_utils[:, 2:].mean(axis=1)
        ub = seat_utils[:, :2].mean(axis=1)
      dir_margins.append(ua - ub)
      sums[i] += ua.sum()
      sums[j] += ub.sum()
      counts[i] += g
      counts[j] += g
    if not dir_margins:
      print(f"  pair {a['id']} vs {b['id']}: no completed games -- aborting")
      raise RuntimeError("no games completed; check eval num_workers/devices")
    margins = np.concatenate(dir_margins)
    pair_margins[(i, j)] = margins

    # Identical-policy gate: main and a snapshot that share birth_update carry
    # the same weights, so on the *same boards* direction 2 must be the exact
    # mirror of direction 1 (m2 == -m1), and the pooled margin averages to 0.
    # Per-game margins are NOT zero -- identical policies play a symmetric but
    # random game, so their two seat-pairs see different galaxies game to game
    # -- but the two directions cancel exactly. Any deviation is a bug.
    if a["kind"] == "net" and b["kind"] == "net" and a["birth"] == b["birth"]:
      if len(dir_margins) == 2:
        mirror = np.abs(dir_margins[0] + dir_margins[1]).max()
        if mirror != 0.0:
          raise AssertionError(
              f"{a['id']} vs {b['id']} share birth_update {a['birth']} (same "
              f"weights) but the two seat directions do not mirror "
              f"(|m1+m2|_max={mirror:.3e}); identical policies on identical "
              f"boards must play mirror-identically")
      if np.abs(np.mean(margins)) != 0.0:
        raise AssertionError(
            f"{a['id']} vs {b['id']} share birth_update {a['birth']} (same "
            f"weights) but the pooled pair margin is {np.mean(margins):.3e}; "
            f"identical policies must rate identically on balanced seats")

    for m in margins:
      if m > 0:
        win[i, j] += 1
      elif m < 0:
        win[j, i] += 1
      else:
        draw[i, j] += 1
        draw[j, i] += 1
    if (n_pair + 1) % 100 == 0 or n_pair + 1 == n_pairs:
      print(f"  played {n_pair + 1}/{n_pairs} pairs")

  mean_u = [sums[k] / max(1, counts[k]) for k in range(p)]

  # Anchor: Random pinned at 0 (or the mean net rating if bots are off).
  anchor_idx = index["Random"] if "Random" in index else None

  # 1) Margin-based rating (primary).
  ratings, cis = _fit_margin_ratings(ids, anchor_idx, pair_margins,
                                     n_boot=FLAGS.ladder_boot,
                                     rng=np.random.RandomState(
                                         FLAGS.ladder_boot_seed))

  # 2) Binarized Bradley-Terry Elo (cross-check, discards margin by design).
  elo = None
  try:
    elo = np.asarray(pyspiel.elo.compute_ratings_from_matrices(
        win.tolist(), draw.tolist()), dtype=np.float64)
    if anchor_idx is not None:
      elo = elo - elo[anchor_idx]
  except Exception as exc:  # pylint: disable=broad-except
    print(f"  [warn] pyspiel Elo unavailable ({exc}); reporting margin rating only")

  # Pairwise raw evidence table (mean_ci of the margin).
  for (i, j), margins in pair_margins.items():
    mean, lo, hi = pe.mean_ci(margins)
    sig = "" if np.isnan(lo) else ("  *" if lo > 0 or hi < 0 else "")
    print(f"    {ids[i] :<12s} vs {ids[j] :<12s}  dmargin={mean:+.3f}"
          f" [{lo:+.3f},{hi:+.3f}]  n={len(margins)}{sig}")

  # Rating table.
  print("\n  ratings (Random pinned at 0; margin model is authoritative)")
  order = sorted(range(p), key=lambda k: -ratings[k])
  header = (f"    {'policy':<12s} {'kind':<4s} {'upd':>6s} {'games':>6s} "
            f"{'mean_u':>6s} {'rating':>7s} {'[ci]':>16s} {'elo':>7s}")
  print(header)
  for k in order:
    lo, hi = cis[k]
    ci = "" if np.isnan(lo) else f"[{lo:+.3f},{hi:+.3f}]"
    elo_s = "" if elo is None else f"{elo[k]:+.1f}"
    print(f"    {ids[k]:<12s} {policies[k]['kind']:<4s} "
          f"{policies[k]['birth'] if policies[k]['birth'] is not None else '-':>6}"
          f" {counts[k]:>6d} {mean_u[k]:+.3f} "
          f"{ratings[k]:+.3f} {ci:>16s} {elo_s:>7s}")

  # Monotonicity over well-separated snapshots.
  nets = sorted([(k, policies[k]["birth"], ratings[k]) for k in range(p)
                 if policies[k]["kind"] == "net"], key=lambda t: t[1])
  # main is a re-save of the last forced snapshot, so collapse same-birth nets
  # (keep the last) before judging age monotonicity -- otherwise the duplicate
  # counts twice.
  seen = {}
  for t in nets:
    seen[t[1]] = t
  nets = [seen[b] for b in sorted(seen)]
  mono = {"ok": False, "detail": "not enough distinct net policies"}
  if len(nets) >= 2:
    births = [t[1] for t in nets]
    rat_seq = [t[2] for t in nets]
    full_tau = _kendall_tau(rat_seq)
    full_p = _permutation_p(rat_seq, np.random.RandomState(0))
    # Representatives: newest per coarse age bucket, plus the earliest net --
    # an early peak must never hide behind the newest-of-bucket pick (e.g. an
    # early-overfit snapshot that outrates its immediate successor).
    split = np.array_split(np.arange(len(nets)), FLAGS.ladder_bins)
    reps = list(dict.fromkeys([0] + [grp[-1] for grp in split if len(grp)]))
    rep_tau = None if len(reps) < 2 else _kendall_tau([rat_seq[r] for r in reps])
    rep_p = None
    if rep_tau is not None:
      rep_p = _permutation_p([rat_seq[r] for r in reps],
                             np.random.RandomState(0))
    # (a) Regressions: any well-separated older->newer pair whose joint
    # bootstrap CI reverses orientation is a real finding -- a later snapshot
    # rated strictly below an earlier one. (b) Monotone evidence: every
    # adjacent pair rises clear of the joint CI.
    regressions = []
    for ai in range(len(nets)):
      for bi in range(ai + 1, len(nets)):
        a, b = nets[ai], nets[bi]
        if b[1] - a[1] < FLAGS.ladder_min_sep:
          continue
        la, ha = cis[a[0]]
        lb, hb = cis[b[0]]
        if (not np.isnan(la)) and (not np.isnan(hb)) and la > hb:
          regressions.append((a[1], b[1], ratings[a[0]], ratings[b[0]]))
    strict_up = True
    for ai in range(len(nets) - 1):
      a, b = nets[ai], nets[ai + 1]
      la, ha = cis[a[0]]
      lb, hb = cis[b[0]]
      if not (np.isnan(la) or np.isnan(hb)):
        strict_up = strict_up and ha < lb
    alpha = FLAGS.ladder_alpha
    if regressions:
      verdict = "NON-MONOTONE"
      why = (f"{len(regressions)} well-separated snapshot(s) rated strictly "
             f"below an earlier one (joint CI)")
    elif strict_up:
      verdict = "MONOTONE"
      why = "every adjacent snapshot pair rises clear of the joint CI"
      if full_p >= alpha:
        why += "; tau significance limited by n"
    elif full_tau > 0 and full_p < alpha:
      verdict = "MONOTONE"
      why = f"full_tau={full_tau:+.2f} significant (p={full_p:.3f})"
    elif full_tau < 0 and full_p < alpha:
      verdict = "NON-MONOTONE"
      why = f"full_tau={full_tau:+.2f} significantly negative (p={full_p:.3f})"
    else:
      verdict = "inconclusive"
      why = "flat or underpowered (tau sign not significant, no CI-clear regression)"
    mono = {
        "ok": verdict == "MONOTONE",
        "verdict": verdict,
        "why": why,
        "full_tau": float(full_tau),
        "full_p": float(full_p),
        "reps_births": [int(births[r]) for r in reps],
        "rep_tau": None if rep_tau is None else float(rep_tau),
        "rep_p": None if rep_p is None else float(rep_p),
        "regressions": [[int(x[0]), int(x[1]), float(x[2]), float(x[3])]
                        for x in regressions],
    }
    print(f"\n  monotonicity (age -> rating): full_tau={full_tau:+.2f} "
          f"(p={full_p:.3f}) over {len(nets)} nets; "
          f"reps={[int(births[r]) for r in reps]}"
          + (f" tau={rep_tau:+.2f} p={rep_p:.3f}" if rep_tau is not None else ""))
    print(f"  VERDICT: {verdict} -- {why}")
    for r in regressions:
      print(f"    regression: birth {r[0]} ({r[2]:+.3f}) > birth {r[1]} "
            f"({r[3]:+.3f})")

  # Cross-check: do margin vs binarized Elo orderings disagree on the nets?
  if elo is not None and len(nets) >= 2:
    m_seq = [ratings[k] for k, _, _ in sorted(nets, key=lambda t: t[0])]
    e_seq = [elo[k] for k, _, _ in sorted(nets, key=lambda t: t[0])]
    if _kendall_tau(m_seq) and _kendall_tau(e_seq) and \
       np.sign(_kendall_tau(m_seq)) != np.sign(_kendall_tau(e_seq)):
      print("  [signal] margin rating and binarized Elo order the nets in"
            " opposite directions -- treat both with caution")

  out_path = FLAGS.ladder_out or os.path.join(FLAGS.ladder_roster_dir,
                                              "ladder.json")
  payload = {
      "roster_dir": FLAGS.ladder_roster_dir,
      "anchor": "Random" if anchor_idx is not None else None,
      "games_per_dir": FLAGS.ladder_games_per_dir,
      "seed": FLAGS.seed,
      "ladder_seed_offset": FLAGS.ladder_seed_offset,
      "eval_strs": eval_strs,
      "policies": [{
          "id": ids[k], "kind": policies[k]["kind"],
          "birth_update": policies[k]["birth"],
          "games": counts[k], "mean_utility": float(mean_u[k]),
          "rating": float(ratings[k]),
          "rating_ci": [float(cis[k][0]), float(cis[k][1])],
          "elo": None if elo is None else float(elo[k]),
      } for k in range(p)],
      "pairs": [{
          "a": ids[i], "b": ids[j], "n": int(len(pair_margins[(i, j)])),
          "mean_margin": float(np.mean(pair_margins[(i, j)])),
          "win": int(win[i, j]), "lose": int(win[j, i]),
          "draw": int(draw[i, j]),
      } for (i, j) in pair_list],
      "monotonicity": mono,
  }
  with open(out_path, "w") as f:
    json.dump(payload, f, indent=2, default=float)
  print(f"\nwrote {out_path}")


if __name__ == "__main__":
  app.run(main)
