#!/usr/bin/env python3
"""T4 -- where does `learn` actually go? It is 50% of the update and unexplained.

`learn` is 9.37 s of BIG's 18.87 s update at 1,024 envs: 64 forward+backward
passes (4 epochs x 16 minibatches) at ~146 ms per 8,192-row minibatch. docs/eclipse_rl_todo.md
says the "memory-bound at a flat 0.96 MB/row" characterisation has no identified
cause, that nothing in the repo profiles memory, and that the one existing 9.08 GB
figure came from a scratch script profiling the ENCODER ONLY -- no pointer head, no
optimizer state, no rollout buffer -- so the real peak is higher than recorded.

Runs at 256 envs / num_minibatches=4, which gives the SAME 8,192 rows per minibatch
as the production 1,024/16 while needing a quarter of the passes -- the per-pass
breakdown is what is being asked for, and this reaches it 4x sooner.

Reports CUDA time and memory by operator, and calls out the four candidates T4
names: the (B,64,225) conv/GroupNorm chain, unit_attn, the tail MLP, and the
pointer head's _pairs intermediates. Also reports which SDPA backend unit_attn
actually gets -- `need_weights=False` is set, but a bool key_padding_mask rules out
the flash backend, and the math fallback materializes B x heads x 128 x 128 probs.
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


def build(num_envs, num_workers, steps, mb, seed, device, compile_encoder):
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
      num_envs=num_envs, steps_per_batch=steps,
      num_minibatches=mb, update_epochs=4, learning_rate=2.5e-4,
      device=device,
      agent_fn=make_agent_fn(
          64, 2, ("final_rank",), activation="tanh",
          factored_actions=factorization_from_game(game),
          encoder="spatial", compile_encoder=compile_encoder),
      value_mode="win", aux_tasks=["final_rank"],
      aux_target_fn=(lambda r: np.asarray(r, dtype=np.float32).reshape(-1, 1)),
      aux_coef=0.1, amp=True,
  )
  envs.reset(players="current")
  sa = envs.reset_np()
  return agent, envs, sa


def fill_batch(agent, envs, sa, steps):
  """One real rollout, so learn() sees a real batch rather than noise."""
  for _ in range(steps):
    acts = agent.step_np(sa)
    sa = envs.step_np(acts, reset_if_done=True)
    agent.post_step_np(sa.rewards, sa.dones,
                       shaped_reward=np.zeros(agent.num_envs, dtype=np.float32))
  return sa


def which_sdpa_backend():
  """Which SDPA backend a bool-masked 8-head 128-token attention actually gets."""
  q = torch.randn(64, 8, 128, 8, device="cuda", dtype=torch.bfloat16)
  mask = torch.zeros(64, 1, 1, 128, device="cuda", dtype=torch.bool)
  out = {}
  from torch.nn.attention import sdpa_kernel, SDPBackend
  for name, backend in (("flash", SDPBackend.FLASH_ATTENTION),
                        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                        ("math", SDPBackend.MATH)):
    try:
      with sdpa_kernel(backend):
        torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask)
      out[name] = "available"
    except Exception as e:  # pylint: disable=broad-except
      out[name] = f"UNAVAILABLE ({type(e).__name__})"
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--num_envs", type=int, default=256)
  ap.add_argument("--num_workers", type=int, default=16)
  ap.add_argument("--num_steps", type=int, default=128)
  ap.add_argument("--num_minibatches", type=int, default=4)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--compile_encoder", action="store_true")
  ap.add_argument("--topk", type=int, default=22)
  args = ap.parse_args()
  absl_flags.FLAGS(["learn_profile"])
  device = "cuda"

  rows_per_mb = args.num_envs * args.num_steps // args.num_minibatches
  print(f"=== T4 learn profile ===")
  print(f"{args.num_envs} envs x {args.num_steps} steps / {args.num_minibatches}"
        f" minibatches = {rows_per_mb} rows per minibatch "
        f"(production 1024/16 is also {1024 * 128 // 16})")
  print(f"compile_encoder={args.compile_encoder}  gpu="
        f"{torch.cuda.get_device_name(0)}")

  print("\n--- SDPA backends for a bool-masked 8x128 attention ---")
  for k, v in which_sdpa_backend().items():
    print(f"  {k:<14} {v}")
  print("  (unit_attn passes key_padding_mask, so flash is ruled out; if math is")
  print("   the fallback it materializes heads x 128 x 128 probs per row.)")

  agent, envs, sa = build(args.num_envs, args.num_workers, args.num_steps,
                          args.num_minibatches, args.seed, device,
                          args.compile_encoder)
  print(f"\nfilling one real rollout ({args.num_steps} steps)...")
  sa = fill_batch(agent, envs, sa, args.num_steps)

  # Untimed warm learn so allocator/compile settle, then a timed one.
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  agent.learn_np(sa.obs, sa.seats)
  torch.cuda.synchronize()
  warm = time.perf_counter() - t0
  print(f"learn_np warm pass: {warm:.2f}s")

  sa = fill_batch(agent, envs, sa, args.num_steps)
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  with torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA],
      profile_memory=True, record_shapes=True, with_stack=False) as prof:
    agent.learn_np(sa.obs, sa.seats)
  torch.cuda.synchronize()
  timed = time.perf_counter() - t0
  passes = 4 * args.num_minibatches
  print(f"learn_np profiled pass: {timed:.2f}s over {passes} fwd+bwd passes "
        f"= {1e3 * timed / passes:.0f} ms/pass (profiler adds overhead)")
  print(f"peak CUDA allocated during learn: "
        f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB  "
        f"reserved {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
  print(f"  = {torch.cuda.max_memory_allocated() / rows_per_mb / 1e6:.3f} MB/row "
        f"(the recorded unexplained figure is 'a flat 0.96 MB/row')")

  # Shapes first: a synthetic reproduction of "index backward" built from guessed
  # shapes came out 40x cheaper than the profile, so the shapes are the finding.
  print(f"\n--- indexing ops BY INPUT SHAPE (the actual tensors) ---")
  kshape = prof.key_averages(group_by_input_shape=True)
  idx_ops = [e for e in kshape
             if any(p in e.key.lower() for p in
                    ("index_put", "index_backward", "embedding_dense",
                     "sum_and_scat", "aten::index", "scatter"))]
  idx_ops.sort(key=lambda e: -float(getattr(e, "self_device_time_total", 0) or
                                    getattr(e, "self_cuda_time_total", 0) or 0))
  for e in idx_ops[:args.topk]:
    cu = float(getattr(e, "self_device_time_total", 0) or
               getattr(e, "self_cuda_time_total", 0) or 0)
    print(f"  {e.key[:46]:<46} {cu / 1e3:8.1f} ms  n={e.count:<5} "
          f"shapes={str(e.input_shapes)[:88]}")

  ka = prof.key_averages()
  print(f"\n--- top {args.topk} by self CUDA time ---")
  print(ka.table(sort_by="self_cuda_time_total", row_limit=args.topk,
                 max_name_column_width=55))
  print(f"\n--- top {args.topk} by self CUDA memory ---")
  print(ka.table(sort_by="self_cuda_memory_usage", row_limit=args.topk,
                 max_name_column_width=55))

  # The four candidates T4 names, matched by operator name.
  buckets = {
      "conv / GroupNorm chain": ("conv", "cudnn", "group_norm", "native_group"),
      "unit_attn (SDPA / bmm / softmax)": ("scaled_dot_product", "attention",
                                           "bmm", "softmax", "baddbmm"),
      "MLP / addmm / linear": ("addmm", "linear", "mm", "matmul"),
      "gather / index (pointer _pairs)": ("gather", "index", "take", "scatter"),
      "elementwise / copy": ("copy_", "mul", "add", "where", "clamp", "cat"),
  }
  totals = {k: 0.0 for k in buckets}
  grand = 0.0
  for e in ka:
    cu = float(getattr(e, "self_device_time_total", 0) or
               getattr(e, "self_cuda_time_total", 0) or 0)
    grand += cu
    low = e.key.lower()
    for label, pats in buckets.items():
      if any(p in low for p in pats):
        totals[label] += cu
        break
  print("\n--- T4's named candidates, by self CUDA time ---")
  for label, v in sorted(totals.items(), key=lambda kv: -kv[1]):
    print(f"  {label:<34} {v / 1e3:8.1f} ms  {100 * v / max(grand, 1):5.1f}%")
  print(f"  {'(all ops)':<34} {grand / 1e3:8.1f} ms")
  if hasattr(envs, "close"):
    envs.close()


if __name__ == "__main__":
  main()
