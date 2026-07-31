# Stage 0 — Setup + Profiling Results

Date: 2026-07-31. Machine: NVIDIA GeForce RTX 4080 Laptop GPU (12 GB), driver
CUDA 13.2, PyTorch 2.11.0+cu128, Python 3.12.

## What was done

1. Installed CUDA-enabled PyTorch into `.venv`
   (`pip install torch --index-url .../whl/cu128`, verified
   `torch.cuda.is_available() == True`, device = RTX 4080 Laptop GPU).
2. Microbenchmarked NN forward-pass latency (CPU vs GPU) at num_envs
   1 / 64 / 128 / 256 on an Eclipse-sized network.
3. Measured `SyncVectorEnv` Eclipse env-stepping throughput (no NN) at
   num_envs 64 / 128 / 256 / 512.

Scripts (in `open_spiel/python/eclipse/`): `benchmark_forward.py`,
`benchmark_env.py`. Raw data: `stage0_forward_results.txt`,
`stage0_env_results.txt`.

## Confirmed game facts

- `eclipse` game: num_players = 4, num_distinct_actions = 11117 (~11K),
  observation_tensor size = 1785 (flat). First real player decision has ~13
  legal actions.
- `Dynamics.SEQUENTIAL` (turn-based), `ChanceMode.SAMPLED_STOCHASTIC`,
  `Utility.GENERAL_SUM`, `RewardModel.TERMINAL`.
- `rl_environment.Environment` resolves chance nodes internally
  (`_sample_external_events`) before control returns — confirmed no special
  handling needed by training loop.
- Random play ends in terminal states with `rewards=[0,0,0,0]` when nobody
  scores (consistent with kTerminal reward model).

## 1. Forward-pass latency (obs=1785 → width 1024, depth 4 → actor 11117)

Batch = (num_envs, 1785); timed full PPO training-path forward: actor forward +
`CategoricalMasked` sample/log_prob/entropy + critic forward. 16.37 M params.

| num_envs | CPU ms/batch | GPU ms/batch | GPU speedup | per-env GPU µs |
|----------|-------------|-------------|-------------|----------------|
| 1        | 1.53        | 0.507       | 3.0x        | 507            |
| 64       | 27.8        | 0.786       | 35.4x       | 12.3           |
| 128      | 44.6        | 0.888       | 50.3x       | 6.9            |
| 256      | 82.4        | 1.143       | 72.1x       | 4.5            |

Clear GPU win, growing with batch: at 256 envs the GPU is ~72x faster than CPU
and the marginal cost is ~4.5 µs per env (vs ~0.5 ms serialized per env). GPU
forward of a full batch is ~1 ms — the network itself is NOT a bottleneck.

## 2. Env-stepping throughput (Eclipse, no NN)

Random legal actions through `SyncVectorEnv.step(reset_if_done=True)`.

| num_envs | env-steps/s | decisions/s | ms/batch | vs GPU forward batch |
|----------|-------------|-------------|----------|----------------------|
| 64       | 68          | 4362        | 14.7     | 21.4x slower         |
| 128      | 24          | 3075        | 41.6     | 49.3x slower         |
| 256      | 11          | 2711        | 94.4     | 71.9x slower         |
| 512      | 4           | 2106        | 243.2    | n/a                  |

## Conclusions

- **Env-stepping is the dominant bottleneck**, exactly as the plan hypothesized:
  at every num_envs the sequential Python env loop is ~21–72x slower than the
  corresponding GPU forward pass. At num_envs=256, env stepping = 94.4 ms vs GPU
  forward = 1.14 ms.
- The bottleneck **worsens with num_envs**: decisions/s *drops* from 4362 (64
  envs) to 2106 (512 envs) because the sequential loop cost scales with num_envs
  while benefits don't. So plain `SyncVectorEnv` caps useful throughput; running
  more envs just wastes wall time.
- GPU forward is effectively free by comparison; never the constraint.
- **Implication for Stage 1+:** a working first version can run on `SyncVectorEnv`
  (e.g. num_envs ~64–256, where env steps/s is still OK), but the plan's noted
  "async/multi-process vector_env variant" is not just a nice-to-have — it is the
  main lever for real throughput. `vector_env.py` currently has **no** async /
  multiprocessing variant (only `SyncVectorEnv`). Building one is a clear
  follow-up, not a blocker for the first working version.

## Notes

- Forward `CategoricalMasked` cost over the full 11117 actions is included and
  acceptable at these batch sizes (~1 ms total on GPU), confirming the plan's
  "fixed compute/memory cost, not a correctness issue" note.
- No core source changes were made this stage (measurement only). All artifacts
  live under `open_spiel/python/eclipse/`.
