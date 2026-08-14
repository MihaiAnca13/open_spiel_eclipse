#!/usr/bin/env python3
"""Price the real hot spot: _pick's backward into the (B, 225, 64) cell table.

T4's profile puts aten::_index_put_impl_ at 43.6% of learn, 7.55 ms per call. An
isolated features[rows] gather is only 0.278 ms, so the cost is not there -- it is
in EclipseTypedPointerHead._pick, which advanced-indexes a (B, N, D) entity table.
The backward of `table[rows, idx]` must allocate and ZERO a full zeros_like(table)
and scatter into it, and for the cell table that is

    (8192, 225, 64) fp32 = 472 MB, per call, per pass, 64 passes per update.

_pairs calls _pick on the cell table THREE times (cell, destination, slot_cell),
so it pays that allocate-and-zero three times over for one logical operation.

Measures:
  1. the current per-call cost, to confirm the profile
  2. one COMBINED gather with concatenated indices -- three picks become one, so
     one gradient buffer is allocated and zeroed instead of three
  3. a flat 2D view formulation, in case reshaping changes the kernel chosen
"""
import argparse
import time

import numpy as np
import torch


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
  ap.add_argument("--rows", type=int, default=8192)
  ap.add_argument("--per_row", type=int, default=20)
  ap.add_argument("--cells", type=int, default=225)
  ap.add_argument("--chan", type=int, default=64)
  ap.add_argument("--reps", type=int, default=20)
  ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  args = ap.parse_args()
  dev = "cuda"
  dt = getattr(torch, args.dtype)
  torch.manual_seed(0)

  B, N, D = args.rows, args.cells, args.chan
  M = B * args.per_row
  rows = torch.repeat_interleave(
      torch.arange(B, device=dev),
      torch.full((B,), args.per_row, device=dev, dtype=torch.long))
  idx = torch.randint(0, N, (M,), device=dev)
  idx2 = torch.randint(0, N, (M,), device=dev)
  idx3 = torch.randint(0, N, (M,), device=dev)

  tbl_bytes = B * N * D * torch.empty((), dtype=dt).element_size()
  print("=== _pick backward into the cell table ===")
  print(f"table=({B}, {N}, {D}) {args.dtype} = {tbl_bytes / 1e6:.0f} MB   "
        f"pairs={M}   gpu={torch.cuda.get_device_name(0)}")
  print(f"gradient buffer allocated+zeroed per _pick call: "
        f"{tbl_bytes / 1e6:.0f} MB\n")

  def fresh():
    return torch.randn(B, N, D, device=dev, dtype=dt, requires_grad=True)

  tbl = fresh()
  g1 = torch.randn(M, D, device=dev, dtype=dt)

  def one_pick():
    tbl.grad = None
    tbl[rows, idx].backward(g1)
  t1 = timed(one_pick, args.reps)
  print(f"  1 pick  (current, x1)              {t1:7.3f} ms")

  def three_picks():
    tbl.grad = None
    out = (tbl[rows, idx].sum() + tbl[rows, idx2].sum()
           + tbl[rows, idx3].sum())
    out.backward()
  t3 = timed(three_picks, args.reps)
  print(f"  3 picks (what _pairs does today)   {t3:7.3f} ms")

  # THE FIX: concatenate the three index vectors into one advanced-index call, so
  # one gradient buffer is allocated and zeroed rather than three.
  rows3 = rows.repeat(3)
  idx_all = torch.cat([idx, idx2, idx3])

  def combined():
    tbl.grad = None
    tbl[rows3, idx_all].sum().backward()
  tc = timed(combined, args.reps)
  print(f"  1 COMBINED pick (3 concatenated)   {tc:7.3f} ms   "
        f"{t3 / tc:5.2f}x vs 3 separate")

  # Flat 2D view: does reshaping pick a cheaper kernel?
  flat_idx = rows3 * N + idx_all

  def flat():
    tbl.grad = None
    tbl.reshape(B * N, D)[flat_idx].sum().backward()
  tf = timed(flat, args.reps)
  print(f"  1 combined via flat (B*N, D) view  {tf:7.3f} ms   "
        f"{t3 / tf:5.2f}x vs 3 separate")

  def flat_index_select():
    tbl.grad = None
    torch.index_select(tbl.reshape(B * N, D), 0, flat_idx).sum().backward()
  ts = timed(flat_index_select, args.reps)
  print(f"  1 combined flat + index_select     {ts:7.3f} ms   "
        f"{t3 / ts:5.2f}x vs 3 separate")

  # Correctness of the combined form against three separate picks.
  tbl.grad = None
  (tbl[rows, idx].sum() + tbl[rows, idx2].sum() + tbl[rows, idx3].sum()).backward()
  ref = tbl.grad.clone()
  tbl.grad = None
  tbl[rows3, idx_all].sum().backward()
  print(f"\n  combined gradient == 3-separate gradient: "
        f"bitwise={torch.equal(tbl.grad, ref)}  "
        f"allclose={torch.allclose(tbl.grad, ref, rtol=1e-5, atol=1e-5)}")

  best = min(tc, tf, ts)
  print(f"\n  per learn pass, cell-table picks: {t3:.2f} -> {best:.2f} ms "
        f"({t3 - best:.2f} ms saved)")
  print(f"  x64 passes per update at 1024 envs/mb=16: "
        f"{64 * (t3 - best) / 1e3:.2f} s of an 18.87 s update "
        f"({100 * 64 * (t3 - best) / 1e3 / 18.87:.1f}%)")


if __name__ == "__main__":
  main()
