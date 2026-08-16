# Auto-research notes — search state (agent edits this — NOT immutable)

Read this first. It is the running memory carried between fresh sessions so the
search does not restart from scratch each time. The immutable truth is the
results TSV; this file is the agent's own reasoning about where to look next.

## Baseline (fresh run on this tree)

- score_steps = 125952 (updates=82, steps_per_sec=1968) — the number to beat.

## Search so far

- (none yet — fresh driver loop)

## Where to look next (next session explores high on this list; delete or mark
## done items as you go so you do not repeat an exact idea)

- [ ] Hyperparameter sweep around the config (num_minibatches, update_epochs,
      lr, entropy_coef) — cheapest win, low risk.
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
