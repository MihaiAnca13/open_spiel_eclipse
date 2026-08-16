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
