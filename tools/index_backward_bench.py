#!/usr/bin/env python3
"""How much of learn's 43.6% indexing-backward cost is recoverable?

T4's profile says aten::_index_put_impl_ / indexing_backward_kernel is 43.6% of
learn's CUDA time, and the index family together is 50.9%. It comes from
EclipseTypedPointerHead._pairs, which performs ~7 gradient-carrying advanced-index
gathers keyed on `rows` -- one entry per legal (state, action) pair, so ~161k
indices scattering back into 8,192 rows, ~20 duplicates each. PyTorch's backward
for duplicated advanced indexing is sort-based and is the known slow path.

The critical structural fact the current code does not exploit: `rows` comes from
async_vector_env._legal_indices as np.repeat(arange(num_envs), lens), so it is
SORTED with contiguous per-row segments. Sorted segments admit much cheaper
formulations than a general scatter.

This prices the alternatives at production shapes before anything in the model is
touched. It measures the FORWARD+BACKWARD of one gather, since the backward is the
expensive half.
"""
import argparse
import time

import numpy as np
import torch


def timed(fn, reps=30, warmup=5):
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  ts = []
  for _ in range(reps):
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
  return float(np.median(ts)) * 1e3


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--rows", type=int, default=8192, help="minibatch states")
  ap.add_argument("--per_row", type=float, default=19.7,
                  help="mean legal actions per row (measured 19.7 at 1024 envs)")
  ap.add_argument("--width", type=int, default=64)
  ap.add_argument("--reps", type=int, default=30)
  args = ap.parse_args()
  dev = "cuda"
  torch.manual_seed(0)

  B, W = args.rows, args.width
  counts = torch.full((B,), int(round(args.per_row)), device=dev,
                      dtype=torch.long)
  M = int(counts.sum())
  # Sorted, contiguous segments -- exactly what _legal_indices produces.
  rows = torch.repeat_interleave(torch.arange(B, device=dev), counts)
  assert bool((rows.diff() >= 0).all()), "rows must be sorted"
  print(f"=== indexing backward at production shape ===")
  print(f"rows(states)={B}  pairs={M}  width={W}  "
        f"duplicates/row={M / B:.1f}  gpu={torch.cuda.get_device_name(0)}")
  print(f"upstream grad tensor is ({M}, {W}) = "
        f"{M * W * 4 / 1e6:.1f} MB fp32\n")

  feats = torch.randn(B, W, device=dev, requires_grad=True)
  gout = torch.randn(M, W, device=dev)

  def bwd(make):
    def run():
      if feats.grad is not None:
        feats.grad = None
      out = make()
      out.backward(gout)
    return run

  variants = {
      # What _pairs does today: features[rows].
      "advanced index  features[rows]":
          lambda: feats[rows],
      # Same semantics, different kernel: index_select's backward is index_add_.
      "index_select(0, rows)":
          lambda: torch.index_select(feats, 0, rows),
      # gather needs an expanded index but its backward is scatter_add.
      "gather(0, rows.expand)":
          lambda: feats.gather(0, rows.unsqueeze(-1).expand(-1, W)),
      # Exploits the sorted-contiguous-segment structure. Backward of
      # repeat_interleave along dim 0 is a segment sum, not a general scatter.
      "repeat_interleave(counts)":
          lambda: torch.repeat_interleave(feats, counts, dim=0, output_size=M),
  }
  base = None
  results = {}
  for name, make in variants.items():
    t = timed(bwd(make), args.reps)
    if base is None:
      base = t
    results[name] = t
    print(f"  {name:<34} {t:7.3f} ms   {base / t:5.2f}x vs current")

  # Correctness: every variant must produce the same gradient.
  print("\n  gradient equality vs advanced indexing:")
  feats.grad = None
  feats[rows].backward(gout)
  ref = feats.grad.clone()
  for name, make in variants.items():
    feats.grad = None
    make().backward(gout)
    same = torch.allclose(feats.grad, ref, rtol=0, atol=0)
    close = torch.allclose(feats.grad, ref, rtol=1e-5, atol=1e-5)
    print(f"    {name:<34} bitwise={same}  allclose={close}")

  # The other half of the win: hoisting a linear ABOVE the gather. `_pairs`
  # applies cell_query/unit_query/slot_query/seat_query to features[rows], i.e.
  # to an (M, W) tensor. Those are linear, so Linear(features)[rows] ==
  # Linear(features[rows]) -- computing them on (B, W) first is ~20x less matmul
  # AND leaves the gather to carry W columns either way.
  print(f"\n=== hoisting a Linear above the gather ({M} vs {B} rows) ===")
  lin = torch.nn.Linear(W, W).to(dev)
  g_small = torch.randn(M, W, device=dev)

  def after():          # what _pairs does: gather, then project
    if feats.grad is not None:
      feats.grad = None
    lin(feats[rows]).backward(g_small)

  def before():         # project on (B, W), then gather
    if feats.grad is not None:
      feats.grad = None
    lin(feats)[rows].backward(g_small)

  t_after, t_before = timed(after, args.reps), timed(before, args.reps)
  print(f"  project AFTER gather (current)     {t_after:7.3f} ms")
  print(f"  project BEFORE gather              {t_before:7.3f} ms   "
        f"{t_after / t_before:5.2f}x")
  print("\n  Note: 4 of _pairs' query projections are per-pair today, so this")
  print("  hoist applies 4 times over -- and the profile already shows the")
  print("  matmuls are only 2.8%, so the win here is the smaller backward.")


if __name__ == "__main__":
  main()
