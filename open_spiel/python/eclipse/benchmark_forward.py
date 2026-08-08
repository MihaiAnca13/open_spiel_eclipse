# Copyright 2026 The OpenSpiel Authors. All rights reserved.
#
# Stage 0 profiling: microbenchmark of NN forward-pass latency for the
# Eclipse-sized network (obs=24714, actions=11117), CPU vs GPU, batched over
# num_envs parallel game environments.
#
# Mirrors the PPOAgent MLP structure (open_spiel/python/pytorch/ppo.py) but at
# the realistic width/depth AlphaZero used for Eclipse (nn_width=2**10, depth 4).
# Times the full PPO training-path forward: actor forward + CategoricalMasked
# sample/log_prob/entropy + critic forward.

import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from open_spiel.python.eclipse import obs_layout

INVALID_ACTION_PENALTY = -1e6

OBS = obs_layout.TOTAL  # Eclipse observation_tensor size
NUM_ACTIONS = 11117  # Eclipse num_distinct_actions (confirmed)
N_PLAYERS = 4


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class CategoricalMasked(Categorical):
  """A masked categorical."""

  def __init__(self,
               probs=None,
               logits=None,
               validate_args=None,
               masks=None,
               mask_value=None):
    logits = torch.where(masks.bool(), logits, mask_value)
    super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class EclipseNet(nn.Module):
  """MLP sized to Eclipse (obs=OBS), 2**10-wide, actor out=11117."""

  def __init__(self, obs=OBS, width=2**10, depth=4, num_actions=NUM_ACTIONS):
    super().__init__()
    layers = [layer_init(nn.Linear(obs, width)), nn.Tanh()]
    for _ in range(depth - 1):
      layers += [layer_init(nn.Linear(width, width)), nn.Tanh()]
    self.body = nn.Sequential(*layers)
    self.actor = layer_init(nn.Linear(width, num_actions), std=0.01)
    self.critic = layer_init(nn.Linear(width, 1), std=1.0)
    self.num_actions = num_actions
    self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

  def forward_train(self, x, mask):
    hidden = self.body(x)
    logits = self.actor(hidden)
    probs = CategoricalMasked(
        logits=logits, masks=mask, mask_value=self.mask_value)
    action = probs.sample()
    log_prob = probs.log_prob(action)
    entropy = probs.entropy()
    value = self.critic(hidden)
    return action, log_prob, entropy, value


@torch.no_grad()
def bench(net, x, mask, device, reps=50, warmup=20):
  for _ in range(warmup):
    net.forward_train(x, mask)
  if device.type == "cuda":
    torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(reps):
    net.forward_train(x, mask)
  if device.type == "cuda":
    torch.cuda.synchronize()
  dt = (time.perf_counter() - t0) / reps
  return dt


def main():
  num_envs_list = [1, 64, 128, 256]
  device_cpu = torch.device("cpu")
  device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(f"obs={OBS} actions={NUM_ACTIONS} width=1024 depth=4")
  print(f"cuda available: {torch.cuda.is_available()}"
        + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
  print(f"{'num_envs':>8} {'cpu_ms':>10} {'gpu_ms':>10} {'gpu_speedup':>12} "
        f"{'per_env_gpu_us':>15}")
  print("-" * 65)

  rows = {}
  network_size_cpu = 0
  for num_envs in num_envs_list:
    x = torch.randn(num_envs, OBS)
    mask = torch.zeros(num_envs, NUM_ACTIONS, dtype=torch.bool)
    # First real Eclipse player decision has ~13 legal actions; use a tiny,
    # realistic mask rather than all-ones to represent the masked cost.
    for i in range(num_envs):
      rng = np.random.default_rng(i)
      idx = rng.choice(NUM_ACTIONS, size=13, replace=False)
      mask[i, idx] = 1

    net_cpu = EclipseNet().to(device_cpu).eval()
    network_size_cpu = sum(p.numel() for p in net_cpu.parameters())
    t_cpu = bench(net_cpu, x.to(device_cpu), mask.to(device_cpu), device_cpu)

    gpu_ms = float("nan")
    if torch.cuda.is_available():
      net_gpu = EclipseNet().to(device_gpu).eval()
      t_gpu = bench(net_gpu, x.to(device_gpu), mask.to(device_gpu), device_gpu)
      gpu_ms = t_gpu * 1e3
      speedup = t_cpu / t_gpu
    else:
      speedup = float("nan")

    cpu_ms = t_cpu * 1e3
    per_env_us = gpu_ms * 1e3 / num_envs
    rows[num_envs] = (cpu_ms, gpu_ms, speedup, per_env_us)
    print(f"{num_envs:>8} {cpu_ms:>10.3f} {gpu_ms:>10.3f} {speedup:>12.2f}x "
          f"{per_env_us:>15.2f}")

  print(f"\nnetwork params: {network_size_cpu / 1e6:.2f} M")

  # Save machine-readable results alongside this script (the old
  # STAGE0_RESULTS.md summary was deleted -- its numbers were for the
  # 1,785-float observation and the obsolete dense MLP).
  out_dir = os.path.dirname(os.path.abspath(__file__))
  with open(out_dir + "/stage0_forward_results.txt", "w") as f:
    for num_envs, (cpu_ms, gpu_ms, speedup, per_env_us) in rows.items():
      f.write(f"{num_envs} {cpu_ms} {gpu_ms} {speedup} {per_env_us}\n")


if __name__ == "__main__":
  main()
