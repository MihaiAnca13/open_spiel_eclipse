# Next work: throughput and observation size

Written 2026-08-13, after the session that took Eclipse PPO from 2,225 to
8,863 env-steps/s on one RTX PRO 6000 (98 GB, 12 physical cores shared with two
resident vLLM workers).

Everything below is measured on that machine unless labelled *projected*.

## Where we ended up

| config | SPS | batch (timesteps) |
|---|---|---|
| previously recommended (128 envs, sync) | 2,225 | 16,384 |
| repo's committed 256/16 config, as found | 2,800 | 32,768 |
| + device-resident obs buffer, no duplicate encoder pass | 4,130 | 32,768 |
| + `--compile_encoder` retargeted at the real path | 5,157 | 32,768 |
| + lazy row source, fp16 buffer (1024 envs) | 5,735 | 131,072 |
| **+ `--update_epochs=1`** | **8,863** | **131,072** |

Env count is now free up to at least 4,096 envs — throughput falls only 3%
across a 4x range, because the obs buffer stopped being the constraint:

| envs | `num_minibatches` | buffer (fp16) | SPS | batch |
|---|---|---|---|---|
| 1024 | 16 | 9.9 GB device | 5,735 | 131,072 |
| 2048 | 32 | 19.7 GB device | 5,664 | 262,144 |
| 4096 | 64 | 39.4 GB device | 5,542 | 524,288 |

Hold the minibatch near 8,192 rows. It is the most efficient size measured
(47,800 rows/s vs 43,100 at 16,384 and 41,280 at 32,768), and the activation
peak scales with it — 1024 envs at `num_minibatches=4` OOMs on a 7.1 GiB
backward allocation while the same run at `num_minibatches=16` is fine.

Current per-update split at 1024 envs, `update_epochs=1` (14.85 s total):

| phase | seconds | share |
|---|---|---|
| act + shaping | 6.03 | 41% |
| env stepping | 5.66 | 38% |
| learn | 2.81 | 19% |
| other | 0.35 | 2% |

## 1. Overlap acting with env stepping

**Worth ~25% of the update. Semantically free.**

Act and env are strictly serialized: the GPU idles for 5.66 s while the workers
step, then 20 worker processes idle for 6.03 s while the GPU acts. Overlapping
them makes rollout cost `max(act, env)` instead of their sum — 12.05 s becomes
about 6.4 s, and the update 14.85 s → ~9.2 s, i.e. **~14,300 SPS** (*projected*).

It is free because policy parameters do not change during a rollout. Whether env
*i* and env *j* were stepped at the same wall-clock instant is not observable in
the data; each env's trajectory is independent and both halves act under
identical weights. This is a pure reordering, not an approximation.

**Do not** extend this to overlapping envs with `learn()`. That one is worth more
(learn is 19% now, was 47% at `update_epochs=4`) but it is a different thing: it
requires actors to run on stale parameters, which is what OpenAI Five needed 512
dedicated forward-pass GPUs — separate from its 512 optimizer GPUs — to do
without contention. On one card it buys less and costs correctness reasoning.
Deliberately out of scope.

### Sketch

`AsyncVectorEnv` already separates "release go" from "acquire done" inside
`_run_step`, so split it:

```python
def send(self, actions):   # write action_buf, release every worker's `go`
def wait(self):            # acquire every `done`, return self._collect()
```

Then split the envs into two groups and alternate, so one group's CPU step
overlaps the other group's GPU forward:

```
send(A); loop:  act(B) on GPU  ||  A steps on CPU
         wait(A); send(B); act(A) || B steps
```

### The one real obstacle

`post_step_np` writes `self.rewards[row]` and `self.dones[row]` for a whole row
of *all* envs. With two groups it needs to accept an env subset. The per-env
state it touches (`_last_decision`, `_episode_start_row`, `_pending_phi`) is
already indexed per env, so only the row-level tensor writes need the subset
argument. Budget the care here, not in the env layer — this is the terminal
attribution path.

Row bookkeeping is otherwise fine: both groups perform the same 128 steps, so
they never desynchronise and both write the same `row` index for their own env
columns.

## 2. `update_epochs=1`

**Measured 1.55x: 5,735 → 8,863 SPS at 1024 envs.** `learn` drops from 10.84 s to
2.81 s because it runs `update_epochs × num_minibatches` forward+backward passes,
so one epoch is exactly a quarter of the iterations.

This is the right regime now that collection is fast. It is also OpenAI Five's
regime — they ran sample reuse ≈ 1.0–1.1 for the "Rerun" experiment (0.8–2.7 at
peak). With a 131k–524k timestep batch there is little reason to revisit each
sample four times.

What to watch when switching:

- **`approx_kl` and `clipfrac` will drop** — a quarter as many gradient steps per
  batch means less policy movement per update. The learning rate almost certainly
  wants to go up. Do not read the lower KL as "more stable"; it is less
  progress per update, and the throughput has to pay for it.
- **`explained_variance` may fall** at first: the value head also gets a quarter
  of the updates. Consider whether the critic wants its own epoch count.
- Four epochs at 5,735 SPS and one epoch at 8,863 SPS see the same number of
  samples per wall-clock hour only if one epoch learns as much per sample. That
  is the actual experiment; the throughput number alone does not settle it.

## 3. Shrink the observation

This is the highest-leverage item on the list, because the 37,596-float
observation is simultaneously:

- the env-step cost — a full `rl_environment.step` is 46 µs of which the action
  itself is ~4 µs; the rest is writing 150 KB of observation. **Env stepping is
  38% of the update**, and this is most of it. (Measured indirectly — confirm
  directly before acting.)
- the H2D cost on the act path — 38.5 MB per step at 256 envs.
- the buffer size, hence the env ceiling.
- the learn memory traffic. Profiling the encoder at minibatch 8,192: matmuls are
  only **19 ms of 127 ms**; the rest is `copy_` (23.4 ms) and `cat` (15.7 ms) on
  4 GB intermediates, 20.8 GB peak. Learn is memory-bound, not compute-bound.

For scale: OpenAI Five used ~16,000 inputs per hero for a far more complex game.
AlphaStar used 8×128×128 world planes + **512 entities × 43 features** + scalars.
Their entity-list encoding is why their cost tracks what is on the board rather
than the board's capacity.

Where our floats go:

| block | floats | share |
|---|---|---|
| galaxy (225 cells × 88 channels) | 19,800 | 53% |
| V2 planet slots (1800 × 4) | 7,200 | 19% |
| player blocks (6 × 547) | 3,282 | 9% |
| V2 units (128 × 24) | 3,072 | 8% |
| everything else | ~4,242 | 11% |

### 3a. Distribute the slot branch's first linear layer

**Do this first: encoder-only, lossless, no game changes, keeps rosters valid.**

`SpatialEclipseEncoder._encode_context_impl` builds

```python
slot_cells = h_cells.transpose(1, 2).repeat_interleave(PLANET_SLOTS_PER_CELL, dim=1)
slot_h = self.slot_mlp(torch.cat([slots, slot_cells], dim=-1))
```

which materializes `(B, 1800, 64)` and then `(B, 1800, 68)` — 4 GB each at
minibatch 8,192, the largest intermediates in the whole learn phase, and the
`cat` the profile caught at 15.7 ms.

A linear layer distributes over concatenation:

```
Linear([slots, cells]) = W_s · slots + W_c · cells + b
```

and `cells` is a `repeat_interleave` of 225 cell features into 1800 slots, so

```
W_c · repeat_interleave(h, 8) == repeat_interleave(W_c · h, 8)
```

Apply `W_c` to the 225 cells *before* the 8x replication. This deletes both big
tensors and cuts that layer's matmul ~5.7x (1800×68 → 1800×4 + 225×64).

Checkpoints migrate mechanically: `slot_mlp[0]`'s `(64, 68)` weight splits into
`(64, 4)` and `(64, 64)` by slicing — same parameters, regrouped.

Validate it the way the aux-recompute removal was validated: compare losses and
gradients against the current implementation in fp32, then confirm the residual
collapses in float64 (that is the signature of accumulation order rather than a
semantic change).

### 3b. Entity-list the planet slots

1800 slots exist because 225 cells × 8 capacity, but most cells hold 0–3 planets.
Emit an actual list with a validity bit — exactly what the V2 unit rows already do
with `U_VALID` — and 7,200 floats become roughly 800. Same information, indexed
instead of padded.

### 3c. Entity-list the galaxy

The biggest block, and mostly unexplored or empty cells. A fixed-capacity list of
~96 occupied cells × 88 channels takes 19,800 → ~8,400 with nothing lost.

### Cost of 3b and 3c

Both change the C++ observation writer and `obs_layout.py`, which **invalidates
every existing snapshot and roster** and means retraining from scratch. Do them at
a point where the league is being reset anyway. 3a has none of that cost, which is
why it goes first.

## Deliberately not doing

- **Overlapping envs with `learn()`** — see §1.
- **Eliminating the per-step `_last_decision` observation copies.** Measured
  4.43 ms/step at 256 envs (0.57 s/update, ~9%). The self-play closeout path
  stores a full 150 KB observation per env per step to serve a case that only
  fires when a seat's last decision predates the batch. The clean fix stores a
  `(row, env)` reference and snapshots only at the batch boundary — 32x fewer
  copies — but it rewrites terminal attribution, where a silent bug already cost
  this project a 408M-step run. Not worth 9% without a test that exercises the
  cross-batch branch.
- **More env-stepping micro-optimization.** At 482 env-steps/s/core against
  OpenAI Five's ~4.9, the per-core work is not the problem; core count is. The
  exception is §3, which reduces the step cost by reducing what it writes.

## Operational notes

- **A crashed run holds the GPU indefinitely.** When the parent dies, the async
  workers block forever on their semaphores and nothing releases VRAM. A 91 GB
  zombie made a later clean config look like a fresh OOM. Check `nvidia-smi`
  before believing an out-of-memory result.
- **`--max_seconds` is only honoured on the async path** (`num_workers > 0`). It
  is silently ignored in sync mode.
- **Compiling the wrong function is invisible.** `--compile_encoder` spent its
  life wrapping `_encode_impl`, which no hot path calls; nothing errored, the flag
  just did progressively less. If the encoder entry points are refactored again,
  re-check that `_compiled_context` is what the act and learn paths actually
  reach — compile on vs off measured 3.99 s vs 4.04 s of learn when it was
  misdirected, and 2.74 s once corrected.

## Current recommended config

```
--num_envs=1024 --num_steps=128 --num_workers=20 --num_minibatches=16 \
--update_epochs=1 --amp --compile_encoder --lr_schedule=fixed
```

~8,863 SPS. Raise `num_envs` and `num_minibatches` together (keeping the
minibatch near 8,192 rows) if a larger batch is wanted; 4,096 envs is validated.
