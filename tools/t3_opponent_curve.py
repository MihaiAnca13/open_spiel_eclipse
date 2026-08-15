#!/usr/bin/env python3
"""T3 -- price the act path against K distinct league policies, on BIG.

WHY
  ``PPO._act_sparse`` groups the rollout batch by policy id and runs one encoder
  forward per distinct policy (ppo.py:1592-1602). The row count is unchanged; the
  launches just get smaller and more numerous, and the cost is steeply
  superlinear. ``--max_live_opponents`` bounds how many distinct opponents can be
  live, and its default of 4 was picked off the 12 GB box's curve:

      K distinct policies :   1     2     4     8    16    32
      cost vs K=1 (12 GB) : 1.00x 1.00x 1.12x 2.05x 4.26x 8.32x

  BIG's card is far wider, so the knee is probably further right and a larger
  live set is likely free -- which is strictly better league play. This measures
  the curve there so the long run's flag is set off BIG's own numbers.

WHAT IT MEASURES
  The real ``_act_sparse`` on a real PPO agent with the production encoder, real
  Eclipse observations and real legal-action masks -- not a hand-rolled encoder
  loop. Everything downstream of the per-group split (the pointer head's
  ``_pack_logits``, the Gumbel sample, the segment LSE) is per-group too, so a
  benchmark that timed only the encoder forward would understate the slope.

  Rows are held at --num_rows across every K and split as evenly as the batch
  allows, which is the production shape: K groups of roughly batch/K rows.

  Both eager and --compile_encoder are measured. This is not redundant: league
  groups have *varying* sizes, so a compiled encoder can pay recompilation or
  dynamic-shape costs that eager does not, and the committed config compiles.
  If the compiled curve is worse, that is a finding for the long run, not noise.

NOTE ON WHAT K MEANS
  ``max_live_opponents`` bounds the opponent set EXCLUDING main
  (league.py:254-256), so a batch holds at most max_live_opponents + 1 distinct
  policies. This script reports both, so the flag can be set without guessing.
"""
import argparse
import time

import numpy as np
import torch
from absl import flags as absl_flags

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.async_vector_env import AsyncVectorEnv
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.examples.ppo_eclipse import (
    make_agent_fn, _randomized_game_string)
from open_spiel.python.eclipse.action_factors import factorization_from_game


def build(num_rows, num_workers, compile_encoder, seed, device):
  """A PPO agent, its agent_fn, and one real batch of step arrays."""
  game_strs = [_randomized_game_string("eclipse(players=4)", seed + i)
               for i in range(num_rows)]
  envs_list = [
      rl_environment.Environment(
          game=pyspiel.load_game(game_strs[i]),
          chance_event_sampler=rl_environment.ChanceEventSampler(seed=seed + i),
          observation_type=rl_environment.ObservationType.OBSERVATION,
          # Without this every reset pays observation_tensor() at ~24x
          # observation_tensor_into() -- see docs/eclipse_rl_todo.md Operational.
          observations_as_numpy=True)
      for i in range(num_rows)
  ]
  game = pyspiel.load_game("eclipse(players=4)")
  envs = AsyncVectorEnv(
      envs_list, num_workers=num_workers,
      sampler_seeds=[seed + i for i in range(num_rows)],
      game_strs=game_strs, max_legal=game.num_distinct_actions())
  game = envs_list[0]._game  # pylint: disable=protected-access
  input_shape = tuple(game.observation_tensor_shape())
  num_actions = game.num_distinct_actions()

  agent_fn = make_agent_fn(
      64, 2, ("final_rank",), activation="tanh",
      factored_actions=factorization_from_game(game),
      encoder="spatial", compile_encoder=compile_encoder)

  agent = PPO(
      input_shape=input_shape,
      num_actions=game.num_distinct_actions(),
      num_players=game.num_players(),
      player_id=0,
      num_envs=num_rows,
      # 4, not the production 128: _act_sparse never touches the rollout buffer,
      # and a full-size one would burn 2.5 GB for nothing.
      steps_per_batch=4,
      num_minibatches=4,
      update_epochs=4,
      learning_rate=2.5e-4,
      device=device,
      agent_fn=agent_fn,
      value_mode="win",
      aux_tasks=["final_rank"],
      aux_target_fn=(lambda rvec: np.asarray(rvec, dtype=np.float32).reshape(
          -1, 1)),
      aux_coef=0.1,
      amp=True,
  )
  envs.reset(players="current")
  sa = envs.reset_np()
  # Walk the envs into mid-game before timing anything. The initial state has
  # ~13 legal actions where mid-game reaches ~130, and the pointer head's
  # per-group cost scales with the legal-action count, so timing the opening
  # would understate every K. Actions are drawn uniformly from each row's legal
  # set rather than taking the first legal id -- "always the lowest action id"
  # walks one narrow, unrepresentative line through the game.
  rng = np.random.RandomState(seed)
  for _ in range(60):
    acts = np.empty(num_rows, dtype=np.int32)
    for i in range(num_rows):
      legal = sa.legal_cols[sa.legal_rows == i]
      acts[i] = legal[rng.randint(len(legal))]
    sa = envs.step_np(acts, reset_if_done=True)
  return agent, agent_fn, num_actions, input_shape, envs, sa


def time_k(agent, snaps, k, sa, repeats, device):
  """Median seconds for one _act_sparse over `sa`'s batch split into k groups."""
  num_rows = sa.obs.shape[0]
  nets = {"main": agent.network}
  for j in range(k - 1):
    nets[f"snap{j}"] = snaps[j]
  names = ["main"] + [f"snap{j}" for j in range(k - 1)]

  # pids must come out of lineup[i, seats[i]], so set every seat of row i to the
  # policy that row should use. Round-robin gives k groups of ~num_rows/k.
  lineup = np.empty((num_rows, agent.num_players), dtype=object)
  for i in range(num_rows):
    lineup[i, :] = names[i % k]
  agent.setup_league(nets, lineup, "main")

  obs = torch.from_numpy(sa.obs).to(device)
  seats = np.asarray(sa.seats)
  rows = np.asarray(sa.legal_rows)
  cols = np.asarray(sa.legal_cols)

  # Assert the split is what we think it is before timing it.
  pids = np.asarray([lineup[i, int(seats[i])] for i in range(num_rows)])
  assert len(np.unique(pids)) == k, (k, len(np.unique(pids)))

  with torch.no_grad():
    for _ in range(5):                       # warmup: allocator + any compile
      agent._act_sparse(obs, rows, cols, seats)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
      t0 = time.perf_counter()
      agent._act_sparse(obs, rows, cols, seats)
      torch.cuda.synchronize()
      times.append(time.perf_counter() - t0)
  return float(np.median(times)), int(np.unique(pids, return_counts=True)[1].min())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--num_rows", type=int, default=256)
  ap.add_argument("--num_workers", type=int, default=16)
  ap.add_argument("--ks", type=str, default="1,2,4,6,8,12,16,24,32")
  ap.add_argument("--repeats", type=int, default=25)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--knee", type=float, default=1.15,
                  help="ratio rule from the plan; reported, not decisive")
  ap.add_argument("--num_steps", type=int, default=128,
                  help="rollout steps per update, to convert ms/step to s/update")
  ap.add_argument("--baseline_update", type=float, default=5.11,
                  help="measured update seconds at K=1 for this env count "
                       "(T0: 5.11 at 256 envs, 18.87 at 1024)")
  args = ap.parse_args()
  ks = [int(x) for x in args.ks.split(",")]
  device = "cuda"

  # ppo_eclipse._randomized_game_string reads absl FLAGS (randomize_races,
  # race_alien_prob, randomize_npc_difficulty, randomize_warped, warped_prob).
  # Importing the module defines those flags but never parses them, so touching
  # one raises UnparsedFlagAccessError. Parse with an empty argv to take every
  # default -- which is exactly the production setup, since run_v2.sh overrides
  # none of them. This script owns its own CLI via argparse, so absl must not
  # see sys.argv.
  absl_flags.FLAGS(["t3_opponent_curve"])

  print(f"=== T3 act-path cost vs K distinct policies ===")
  print(f"rows={args.num_rows}  repeats={args.repeats}  "
        f"gpu={torch.cuda.get_device_name(0)}")

  results = {}
  for compile_encoder in (False, True):
    tag = "compiled" if compile_encoder else "eager"
    print(f"\n--- {tag} encoder ---")
    agent, agent_fn, num_actions, input_shape, envs, sa = build(
        args.num_rows, args.num_workers, compile_encoder, args.seed, device)
    # Snapshots are built through agent_fn + load_state_dict, exactly as
    # PolicyRoster loads them from disk -- NOT with copy.deepcopy. Deep-copying a
    # torch.compile-wrapped module is not the object the league actually holds,
    # and under --compile_encoder it is the difference between measuring the
    # production act path and measuring an artefact of the benchmark.
    max_k = max(k for k in [int(x) for x in args.ks.split(",")]
                if k <= args.num_rows)
    snaps = []
    sd = agent.network.state_dict()
    for _ in range(max_k - 1):
      snap = agent_fn(num_actions, input_shape, device)
      snap.load_state_dict(sd)
      for p in snap.parameters():
        p.requires_grad_(False)
      snap.eval()
      snap.to(device)
      snaps.append(snap)
    # Report the shape actually measured. The 12 GB box's 1.12x-at-K=4 figure was
    # taken at some particular legal-action density, and a curve measured at a
    # different one is not comparable to it.
    per_row = np.bincount(np.asarray(sa.legal_rows),
                          minlength=args.num_rows)
    print(f"  batch shape: {args.num_rows} rows, "
          f"{len(sa.legal_cols)} legal pairs, "
          f"mean {per_row.mean():.1f} / min {per_row.min()} / "
          f"max {per_row.max()} legal actions per row")
    base = None
    rows_out = []
    for k in ks:
      if k > args.num_rows:
        continue
      t, smallest = time_k(agent, snaps, k, sa, args.repeats, device)
      if base is None:
        base = t
      ratio = t / base
      rows_out.append((k, t, ratio, smallest))
      print(f"  K={k:<3} {t * 1e3:7.3f} ms   {ratio:5.2f}x   "
            f"(smallest group {smallest} rows)")
    results[tag] = rows_out
    if hasattr(envs, "close"):
      envs.close()
    del agent, envs, sa, snaps
    torch.cuda.empty_cache()

  # The "largest K under 1.15x" rule was written for a curve
  # whose knee sat at K=4. On BIG the curve is ~linear from K=2, so that rule
  # returns K=1 -- and "K=1" is NOT expressible as a flag value: 0 means
  # UNBOUNDED (league.py:255), which is the worst possible setting, and 1 already
  # admits 2 distinct policies. So the ratio rule is reported but does not decide.
  #
  # The decision is made on the only denominator that matters: what the extra act
  # cost does to the UPDATE. _act_sparse is a slice of the `act` phase, which is a
  # slice of the rollout, which is a slice of the update -- so a 3.6x act-path
  # ratio is nowhere near a 3.6x slowdown.
  print(f"\n=== ratio rule (knee at {args.knee}x), reported not decisive ===")
  for tag, rows_out in results.items():
    ok = [k for (k, _, r, _) in rows_out if r <= args.knee]
    best = max(ok) if ok else 1
    note = ("  <- the whole curve is above the knee; the ratio rule cannot pick "
            "a flag value here" if best == 1 else "")
    print(f"  {tag:9s}: largest K under {args.knee}x is K={best}{note}")

  print(f"\n=== throughput cost of each K, at num_steps={args.num_steps} and a "
        f"baseline update of {args.baseline_update:.2f}s ===")
  print("  (extra _act_sparse time per rollout step x num_steps, against the "
        "measured update)")
  ref = results.get("compiled") or results.get("eager")
  base_t = ref[0][1]
  print(f"  {'K':>3} {'opponents':>10} {'extra s/update':>15} {'update':>8} "
        f"{'sps change':>11}")
  for (k, t, _, _) in ref:
    # time_k returns SECONDS. num_steps rollout steps per update, so the extra
    # per-update cost is a plain multiply -- an earlier version divided by 1e3
    # here as if t were milliseconds and printed a table of zeros.
    extra = (t - base_t) * args.num_steps
    upd = args.baseline_update + extra
    print(f"  {k:>3} {max(k - 1, 0):>10} {extra:>14.2f}s {upd:>7.2f}s "
          f"{100 * (args.baseline_update / upd - 1):>10.1f}%")
  print("\n  --max_live_opponents = K - 1 (the cap excludes main, league.py:254).")
  print("  NEVER set it to 0: 0 disables the bound entirely and restores the")
  print("  unbounded roster whose act cost climbed all run with nothing in the")
  print("  loss series or the ratings to show for it.")
  print("  Pick the largest opponent count whose sps cost you are willing to pay")
  print("  for the league-quality it buys -- that trade is a QUALITY question and")
  print("  this table only prices one side of it.")


if __name__ == "__main__":
  main()
