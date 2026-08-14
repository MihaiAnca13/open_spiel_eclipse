#!/usr/bin/env python3
"""Where does the `act` phase actually go? Decompose step_np at a given env count.

WHY
  T0 measured the training loop's `act` slot (which is exactly one `step_np` call)
  at 43.1 ms/step at 1,024 envs. T3 measured `_act_sparse` -- the network forward,
  the thing `act` is assumed to be -- at 5.97 ms on the same batch size. So ~37
  ms/step, about a quarter of the whole 18.87 s update, is spent somewhere other
  than the network, and no doc accounts for it.

  next_work.md does list "the per-step `_last_decision` copies" under
  "Deliberately not doing", but the measurement behind that entry was taken at
  256 envs (12.40 ms against a pooled 11.88 ms -- a wash, which is a statement
  about the FIX, not about the size of the cost). Both the copy volume and the
  Python loop scale linearly in num_envs, so at 1,024 envs the same code is a
  materially bigger share than the note implies.

WHAT IT MEASURES
  Each piece of step_np, on real observations and real legal masks, at the env
  count you ask for -- then checks the sum against a real end-to-end step_np call.
  If the sum does not reconstruct the total, the decomposition is wrong and should
  not be believed.
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


def build(num_envs, num_workers, seed, device, steps_per_batch):
  game_strs = [_randomized_game_string("eclipse(players=4)", seed + i)
               for i in range(num_envs)]
  envs_list = [
      rl_environment.Environment(
          game=pyspiel.load_game(game_strs[i]),
          chance_event_sampler=rl_environment.ChanceEventSampler(seed=seed + i),
          observation_type=rl_environment.ObservationType.OBSERVATION,
          observations_as_numpy=True)
      for i in range(num_envs)
  ]
  game = pyspiel.load_game("eclipse(players=4)")
  envs = AsyncVectorEnv(
      envs_list, num_workers=num_workers,
      sampler_seeds=[seed + i for i in range(num_envs)],
      game_strs=game_strs, max_legal=game.num_distinct_actions())
  game = envs_list[0]._game  # pylint: disable=protected-access
  agent = PPO(
      input_shape=tuple(game.observation_tensor_shape()),
      num_actions=game.num_distinct_actions(),
      num_players=game.num_players(), player_id=0,
      num_envs=num_envs, steps_per_batch=steps_per_batch,
      num_minibatches=16, update_epochs=4, learning_rate=2.5e-4,
      device=device,
      agent_fn=make_agent_fn(
          64, 2, ("final_rank",), activation="tanh",
          factored_actions=factorization_from_game(game),
          encoder="spatial", compile_encoder=True),
      value_mode="win", aux_tasks=["final_rank"],
      aux_target_fn=(lambda r: np.asarray(r, dtype=np.float32).reshape(-1, 1)),
      aux_coef=0.1, amp=True,
  )
  envs.reset(players="current")
  sa = envs.reset_np()
  rng = np.random.RandomState(seed)
  for _ in range(40):
    acts = np.empty(num_envs, dtype=np.int32)
    for i in range(num_envs):
      legal = sa.legal_cols[sa.legal_rows == i]
      acts[i] = legal[rng.randint(len(legal))]
    sa = envs.step_np(acts, reset_if_done=True)
  return agent, envs, sa


def med(fn, reps, sync=True):
  for _ in range(3):
    fn()
  if sync:
    torch.cuda.synchronize()
  ts = []
  for _ in range(reps):
    t0 = time.perf_counter()
    fn()
    if sync:
      torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
  return float(np.median(ts)) * 1e3


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--num_envs", type=int, default=1024)
  ap.add_argument("--num_workers", type=int, default=16)
  ap.add_argument("--reps", type=int, default=20)
  ap.add_argument("--seed", type=int, default=1)
  args = ap.parse_args()
  absl_flags.FLAGS(["act_decompose"])
  device = "cuda"

  agent, envs, sa = build(args.num_envs, args.num_workers, args.seed, device,
                          steps_per_batch=4)
  n = args.num_envs
  obs_cpu, seats = sa.obs, np.asarray(sa.seats)
  mask_rows, mask_cols = np.asarray(sa.legal_rows), np.asarray(sa.legal_cols)
  print(f"=== step_np decomposition, {n} envs, {args.num_workers} workers ===")
  print(f"obs bytes/step = {obs_cpu.nbytes / 1e6:.1f} MB  "
        f"dtype={obs_cpu.dtype}  legal pairs={len(mask_cols)}")

  # 1. the H2D upload the forward pass needs.
  def h2d():
    return torch.from_numpy(obs_cpu).to(device)
  t_h2d = med(h2d, args.reps)

  obs_dev = torch.from_numpy(obs_cpu).to(device)

  # 2. the network forward -- what `act` is usually assumed to BE.
  with torch.no_grad():
    t_net = med(lambda: agent._act_sparse(obs_dev, mask_rows, mask_cols, seats),
                args.reps)
    action, logprob, _, value = agent._act_sparse(
        obs_dev, mask_rows, mask_cols, seats)

  # 3. the rollout-buffer row writes (players/trainable/obs/actions/...).
  def rowwrites():
    row = 0
    agent.players[row] = torch.from_numpy(seats.astype(np.int64)).to(device)
    agent.players_cpu[row] = torch.from_numpy(seats.astype(np.int64))
    tr = torch.tensor([agent._acts_trainable(i, int(s))
                       for i, s in enumerate(seats)], dtype=torch.bool)
    agent.trainable_cpu[row] = tr
    agent.trainable[row] = tr.to(device)
    agent.obs[row] = obs_dev if agent.obs_on_device else torch.from_numpy(obs_cpu)
    agent.actions[row] = action
    agent.logprobs[row] = logprob
    agent.values[row] = value.flatten()
    agent.legal_rows_packed[row] = mask_rows.astype(np.int64)
    agent.legal_cols_packed[row] = mask_cols.astype(np.int64)
  t_rows = med(rowwrites, args.reps)

  # 3b. just the _acts_trainable python loop, split out of the above.
  def trainable_loop():
    return torch.tensor([agent._acts_trainable(i, int(s))
                         for i, s in enumerate(seats)], dtype=torch.bool)
  t_trainable = med(trainable_loop, args.reps, sync=False)

  # 4. the selfplay _last_decision bookkeeping (ppo.py:956-975), verbatim.
  def last_decision():
    action_np = action.detach().cpu().numpy()
    logprob_np = logprob.detach().cpu().numpy().ravel()
    value_np = value.detach().cpu().numpy().ravel()
    lens = np.bincount(mask_rows.astype(np.int64), minlength=n)
    offsets = np.zeros(n, dtype=np.int64)
    np.cumsum(lens[:-1], out=offsets[1:])
    for i, s in enumerate(seats):
      if not agent._acts_trainable(i, int(s)):
        continue
      k = int(lens[i])
      cols = (mask_cols[offsets[i]:offsets[i] + k].astype(np.int64)
              if k else np.zeros(0, dtype=np.int64))
      agent._last_decision[i][int(s)] = (
          obs_cpu[i].copy(), cols, int(action_np[i]),
          float(logprob_np[i]), float(value_np[i]))
  t_last = med(last_decision, args.reps, sync=False)

  # 4b. of that, how much is the 150 KB-per-env obs copy alone?
  def obs_copies():
    for i in range(n):
      _ = obs_cpu[i].copy()
  t_copies = med(obs_copies, args.reps, sync=False)

  # 5. ground truth: the real thing, end to end.
  t_total = med(lambda: agent.step_np(sa), args.reps)

  parts = [
      ("H2D obs upload", t_h2d),
      ("_act_sparse (the network)", t_net),
      ("rollout row writes", t_rows),
      ("   of which _acts_trainable loop", t_trainable),
      ("_last_decision bookkeeping", t_last),
      ("   of which per-env obs .copy()", t_copies),
  ]
  print()
  for name, v in parts:
    lead = "  " if name.startswith("   ") else ""
    print(f"{lead}{name:<38} {v:8.2f} ms   {100 * v / t_total:5.1f}% of step_np")
  summed = t_h2d + t_net + t_rows + t_last
  print(f"\n  {'sum of the four top-level pieces':<38} {summed:8.2f} ms")
  print(f"  {'measured step_np end to end':<38} {t_total:8.2f} ms   "
        f"(unexplained {t_total - summed:+.2f} ms)")
  print(f"\n  per update at 128 steps: step_np = {t_total * 128 / 1e3:.2f} s, "
        f"of which _last_decision = {t_last * 128 / 1e3:.2f} s")
  if hasattr(envs, "close"):
    envs.close()


if __name__ == "__main__":
  main()
