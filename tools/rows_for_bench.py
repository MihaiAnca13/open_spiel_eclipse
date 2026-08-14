#!/usr/bin/env python3
"""Price the single hottest op in training: FactoredActorHead.rows_for's backward.

T4's profile, grouped by input shape, puts aten::_index_put_impl_ scattering
(153520, 4, 64) values into a (1420, 64) target at ~70 ms per call and ~1.2 s of a
4.4 s learn -- 43.6% of learn, the largest single cost in the whole training loop.
That target is `actor.1.embedding`, the factored-action embedding, and the op is

    rows_for(idx) = self.embedding[self.decode[idx]].sum(dim=1)

which (a) materializes an (M, 4, width) intermediate and (b) backpropagates through
generic advanced indexing -- a sort-based scatter of M*4 = 614k gradient rows into
1,420 embedding rows, ~430 duplicates each, which is the pathological case.

"Sum `slots` embedding rows per item" is exactly what embedding_bag computes, as one
fused op with a backward specialised for it. This measures the swap at production
shapes and checks the numerics.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F


def timed(fn, reps=20, warmup=5):
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
  ap.add_argument("--pairs", type=int, default=153520, help="legal (state,action) pairs")
  ap.add_argument("--num_rows", type=int, default=1420, help="factor embedding rows")
  ap.add_argument("--num_actions", type=int, default=11117)
  ap.add_argument("--slots", type=int, default=4)
  ap.add_argument("--width", type=int, default=64)
  ap.add_argument("--reps", type=int, default=20)
  ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  args = ap.parse_args()
  dev = "cuda"
  dt = getattr(torch, args.dtype)
  torch.manual_seed(0)

  M, R, S, W = args.pairs, args.num_rows, args.slots, args.width
  decode = torch.randint(0, R, (args.num_actions, S), device=dev)
  cols = torch.randint(0, args.num_actions, (M,), device=dev)
  idx = decode[cols]                                    # (M, S)
  emb = torch.randn(R, W, device=dev, dtype=dt, requires_grad=True)
  gout = torch.randn(M, W, device=dev, dtype=dt)

  print("=== rows_for backward: advanced indexing vs embedding_bag ===")
  print(f"pairs={M}  factor rows={R}  slots={S}  width={W}  {args.dtype}")
  print(f"duplicates per embedding row = {M * S / R:.0f}")
  print(f"(M, slots, width) intermediate the current form materializes: "
        f"{M * S * W * torch.empty((), dtype=dt).element_size() / 1e6:.0f} MB\n")

  def current():
    emb.grad = None
    (emb[idx].sum(dim=1)).backward(gout)

  def bag():
    emb.grad = None
    F.embedding_bag(idx, emb, mode="sum").backward(gout)

  t_cur = timed(current, args.reps)
  t_bag = timed(bag, args.reps)
  print(f"  current  embedding[decode[idx]].sum(1)   {t_cur:8.3f} ms")
  print(f"  embedding_bag(idx, emb, mode='sum')      {t_bag:8.3f} ms   "
          f"{t_cur / t_bag:5.2f}x")

  # Forward-only, to separate the intermediate from the scatter.
  with torch.no_grad():
    e = emb.detach()
    tf_cur = timed(lambda: e[idx].sum(dim=1), args.reps)
    tf_bag = timed(lambda: F.embedding_bag(idx, e, mode="sum"), args.reps)
  print(f"\n  forward only, current                   {tf_cur:8.3f} ms")
  print(f"  forward only, embedding_bag             {tf_bag:8.3f} ms   "
        f"{tf_cur / tf_bag:5.2f}x")
  print(f"  => backward share: current "
        f"{100 * (t_cur - tf_cur) / t_cur:.0f}%, bag "
        f"{100 * (t_bag - tf_bag) / t_bag:.0f}%")

  # Numerics. Sum order differs, so bitwise equality is not expected; what matters
  # is that both the value and the gradient agree to well inside the 4.5e-3 error
  # that --amp's bf16 autocast already accepts.
  e32 = emb.detach().float().requires_grad_(True)
  g32 = gout.float()
  out_a = e32[idx].sum(dim=1)
  out_a.backward(g32)
  ga = e32.grad.clone()
  e32.grad = None
  out_b = F.embedding_bag(idx, e32, mode="sum")
  out_b.backward(g32)
  gb = e32.grad.clone()
  fwd_err = (out_a - out_b).abs().max().item()
  bwd_err = (ga - gb).abs().max().item()
  denom = max(ga.abs().max().item(), 1e-12)
  print(f"\n  forward  max|diff| = {fwd_err:.3e}")
  print(f"  backward max|diff| = {bwd_err:.3e}  "
        f"(relative to max|grad| {denom:.3e}: {bwd_err / denom:.2e})")
  print(f"  --amp's bf16 autocast already accepts 4.5e-3, per "
        f"eclipse_rl_todo.md's obs-dtype note")

  saved = t_cur - t_bag
  print(f"\n  saving {saved:.2f} ms per rows_for call.")
  print(f"  learn calls it once per fwd+bwd pass; 64 passes per update at "
        f"1024 envs/mb=16")
  print(f"  => {64 * saved / 1e3:.2f} s of an 18.87 s update "
        f"({100 * 64 * saved / 1e3 / 18.87:.1f}%)")


if __name__ == "__main__":
  main()
