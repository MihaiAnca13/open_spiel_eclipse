# Auto-research notes — search state (agent edits this — NOT immutable)

Read this first. It is the running memory carried between fresh sessions so the
search does not restart from scratch each time. The immutable truth is the
results TSV; this file is the agent's own reasoning about where to look next.

## Baseline (fresh run on this tree)

- score_steps = 125952 (updates=82, steps_per_sec=1968) — the number to beat.

## Search so far

- [ea6beb02] Removed the redundant per-step `.astype(np.int64)` copy of
  mask_rows/mask_cols in PPO.step_np (they already arrive as freshly allocated
  int64 from _collect/_legal_indices). Pure lossless micro-churn cut in the
  ~10.5k-step hot path; ~0 measurable at 12 envs but clean. Result: TBD (driver).
- KEY FINDING for the search: bench.sh hard-pins num_minibatches=4 AND
  update_epochs=4 on the ppo_eclipse invocation, so the NOTES "hyperparameter
  sweep" idea #1 is NOT actionable for the scored config — changing the
  ppo_eclipse defaults does nothing. Drop that item; a real sweep would need
  to touch the immutable bench (forbidden).

## Where to look next (next session explores high on this list; delete or mark
## done items as you go so you do not repeat an exact idea)

- [x] ~~Hyperparameter sweep around the config (num_minibatches, update_epochs,
      lr, entropy_coef)~~ — NOT actionable: bench.sh pins these flags.
- [ ] Async vector env efficiency: fewer sync points, better worker overlap
      (open_spiel/python/async_vector_env.py + overlap_record in ppo_eclipse).
- [ ] Observation/action pipe: cheaper feature extraction, smaller obs, faster
      factorization (open_spiel/python/eclipse/*).
- [ ] C++ game/observation speedups (needs `cmake --build build` — note the
      build here so the driver/user does it before the next bench).
- [ ] Buffer/memory: obs buffer device/dtype, avoid copies in learn().

## Rules this file follows

- Each session appends 2-3 lines: idea tried, result, next guess.
- The driver appends KEEP/DISCARD verdicts with measured steps.
- Never let this file grow into a transcript — it is a compact index only.
- [18:55] KEEP steps=456192 (audit pass)

## [19:40 session] — PROFILING finding + act-path copy/transfer reduction

- PROFILED the real config (12 envs, 4x4, amp, GPU): workers are ~99% idle
  (ASYNC_PROF: step=0.19ms vs wait=37ms) — the bottleneck is the MAIN THREAD,
  not the env. Per-update split: act (network forward) ~0.26s, learn ~0.18s,
  env-residual ~0.05s, post ~0.02s, overlap=1.00x (bookkeeping gap is shorter
  than the env step, so overlap can't hide it). Both act and learn are
  small-batch GPU kernel-Launch-LATENCY bound (12-row forwards on a 4080 are
  ~2ms of launch; enving in workers is microseconds). => Real jumps need a
  STRUCTURAL change, not micro-churn: (a) group-based pipelining (act group A
  while env group B steps — the only way to hide the act forward, which is 5x
  the env step), or (b) fusing _gumbel_sample/_segment_lse_entropy's ~10
  scatter kernels per step. Small-batch act is the single biggest lever and it
  is NOT addressable by cutting transfers.
- TRIED: removed redundant per-step host work in act+learn — seats.astype once
  (was twice), coalesced legal rows+cols into one H2D/launch, dropped the
  redundant .astype on already-int64 cnt_mb/sparse_counts. All bit-identical
  (verified by construction), so learning is unchanged. Expected win small
  (<5%) because the phase is launch-bound, not transfer-bound.
- NEXT GUESS: pursue group-pipelining of act vs env (biggest real lever,
  hides the ~0.26s/update act phase against idle workers) OR fuse the 10
  per-step scatter kernels. The C++ obs writer is NOT the lever — workers are
  99% idle, so their 0.19ms step can't move main-thread-bound throughput.
