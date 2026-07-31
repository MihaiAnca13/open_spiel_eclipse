# Copyright 2026 The OpenSpiel Authors. All rights reserved.
#
# Stage 0 profiling: measure SyncVectorEnv env-stepping throughput on Eclipse
# alone (no NN involved), at increasing num_envs, to find where the sequential
# Python env-stepping loop becomes the bottleneck relative to GPU forward-pass
# time (benchmark_forward.py).
#
# Eclipse is turn-based (SEQUENTIAL) with SAMPLED_STOCHASTIC chance nodes.
# rl_environment.Environment internally resolves chance nodes via
# _sample_external_events() before returning control, so the benchmark only
# needs to pick a legal action for each env's current player each step.

import os
import random
import sys
import time

sys.path.insert(0, "/home/mihai/personal/open_spiel_eclipse/build/open_spiel/python")
sys.path.insert(0, "/home/mihai/personal/open_spiel_eclipse")

from open_spiel.python import rl_environment
from open_spiel.python.vector_env import SyncVectorEnv

GAME = "eclipse"
NUM_STEPS = 120  # decisions advanced per env (wall-seconds bounded)
NUM_RUNS = 2  # reps for stability


def make_envs(num_envs):
  envs = [
      rl_environment.Environment(GAME, include_full_state=False)
      for _ in range(num_envs)
  ]
  return SyncVectorEnv(envs)


def run(num_envs, total_steps):
  vec = make_envs(num_envs)
  time_steps = vec.reset()
  # step_outputs mirror what ppo_example-style loops pass to vec.step
  step_outputs = [None] * num_envs
  steps = 0
  t0 = time.perf_counter()
  while steps < total_steps:
    for i in range(num_envs):
      cur = time_steps[i].observations["current_player"]
      legal = time_steps[i].observations["legal_actions"][cur]
      step_outputs[i] = type("SO", (), {"action": random.choice(legal)})()
    time_steps, reward, done, _ = vec.step(step_outputs, reset_if_done=True)
    steps += 1
  dt = time.perf_counter() - t0
  return steps / dt, dt


def main():
  print(f"game={GAME} steps_per_run={NUM_STEPS} runs={NUM_RUNS}")
  print(f"{'num_envs':>8} {'env_steps/s':>12} {'decisions/s':>12} "
        f"{'ms/batch':>10} {'matches_forward_at_ms':>22}")
  print("-" * 72)

  # GPU forward batch times from benchmark_forward.py for comparison:
  gpu_forward_ms = {64: 0.685, 128: 0.844, 256: 1.313}
  out_path = os.path.dirname(os.path.abspath(__file__)) + "/stage0_env_results.txt"
  with open(out_path, "w") as f:
    f.write("num_envs ms_per_batch\n")
  for num_envs in [64, 128, 256, 512]:
    best = 0.0
    best_dt = 0.0
    for _ in range(NUM_RUNS):
      rate, dt = run(num_envs, NUM_STEPS)
      if rate > best:
        best = rate
        best_dt = dt
    ms_batch = best_dt / NUM_STEPS * 1e3
    with open(out_path, "a") as f:
      f.write(f"{num_envs} {ms_batch}\n")
    match = "n/a"
    if num_envs in gpu_forward_ms:
      ratio = ms_batch / gpu_forward_ms[num_envs]
      match = f"{ratio:.2f}x gpu-fwd"
    print(f"{num_envs:>8} {best:>12.0f} {best * num_envs:>12.0f} "
          f"{ms_batch:>10.1f} {match:>22}", flush=True)


if __name__ == "__main__":
  main()
