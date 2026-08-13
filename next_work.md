# Next work: throughput and observation size

> **UPDATE 2026-08-13 (branch `perf/obs-and-overlap`).** Five items below are now
> done and three of the measurements that drove the ordering are superseded. Read
> "What changed on 2026-08-13" first; the rest of this file is the pre-change
> analysis and its item numbering still applies to what is left.

## What changed on 2026-08-13

All measured on **12GB**. Every number here is from a *paired* A/B (arms run
alternately in one session) because the box picked up a load-average-45
background workload mid-session and unpaired absolute numbers from that window
are worthless.

| change | effect | verified by |
|---|---|---|
| Observation writer skips empty capacity (`observation.cpp`) | `observation_tensor_into` **13.0 -> 5.0 us** (2.6x) | 2,442-tensor bitwise A/B vs master's writer: **identical** |
| Duplicate whole-tensor `std::fill` removed (`eclipse.cc:1719`) | folded into the above | same A/B |
| `_collect()` double-buffers instead of allocating 38.5 MB/step | env phase **13.5 -> 7.9 ms** (1.71x), **+14% SPS** | paired A/B, 20 steady updates/arm |
| Timing line reports **wall clock**, not a sum of phases | overlap work becomes measurable at all | new `overlap Nx` field reads 1.00x while serialized |
| `--max_seconds` honoured on the sync path | was silently ignored at `num_workers=0` | — |
| `obs_layout.validate(game)` called in training | C++/Python layout skew now fails loudly | — |

Net on this box, 256 envs / 128 steps / 16 workers / mb 16:
**~6,400 -> ~7,300 SPS** paired, and ~7,900 SPS uncontended.

### The observation writer was scanning capacity, not content

The tensor is **97.5% zeros** (measured: 877-1,250 nonzero of 37,596). The writer
walked all 225 galaxy cells, all 1,800 planet-slot rows and wrote ~12 unit
channels per cell regardless of occupancy, into a span it had *already* zeroed --
twice, because `eclipse.cc` and `observation.cpp` both filled it. Real occupancy
is 7-11 cells, 24-38 slot rows, 6-10 unit rows.

Three skips (empty cells write no unit channels; unplaced cells write no slot
rows; slot rows above the orbital row are not walked) plus deleting the duplicate
fill gave 2.6x with **bitwise identical output** across 4 game configs x 6 seeds
x every seat.

**This matters for item 3.** A large part of what 3b/3c were going to buy in
env-step cost has now been bought for free, with no tensor change and no retrain.
What remains for item 3 is buffer bytes, H2D bytes, and the learn-input term.

### V4 is done, and the answer is "fp16 is free"

New test `open_spiel/python/pytorch/ppo_obs_dtype_test.py`. Comparing two
training runs cannot answer this -- the loss series is bit-identical for two
updates then diverges chaotically from GPU reduction order alone (verified: three
master runs at one seed agree exactly on `policy_loss`/`entropy` at u0-u2 and
disagree by 0.0022 on `aux_loss`). So the test measures the claim directly, at
the point of consumption:

```
fp16 STORAGE error only : 6.302e-13
bf16 COMPUTE error only : 4.505e-03   <- already accepted by --amp
both together           : 4.505e-03
```

Seven orders of magnitude apart, and combining them does not compound. **Do not
pin `--obs_buffer_dtype=float32` for the long run.**

### The league had a throughput cliff that would have eaten the long run

Found by running the thing rather than reasoning about it. A `--league` smoke run
showed `act` climbing monotonically **22.8 -> 41.7 ms/step over 45 updates and
still rising**, while `env` and `learn` stayed flat.

Cause: `PPO._act_sparse` (`ppo.py:1594-1602`) groups the batch by policy id and
runs **one encoder forward per distinct policy**. Total rows are unchanged, but
the launches get smaller and more numerous. Measured, 256 rows through the
production encoder:

| K distinct policies | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| cost vs K=1 | 1.00x | 1.00x | **1.12x** | 2.05x | 4.26x | 8.32x |

`Matchmaker.sample_lineup` drew from `roster.opponent_ids()` -- **the entire
roster, unbounded**. So K grew all run. At `--snapshot_every=100` an 8h run
reaches ~35 snapshots; the run would have decayed toward the 4-8x column while
looking perfectly healthy in every loss and rating metric.

Fixed by bounding the *live* opponent set (`--max_live_opponents`, default 4;
`--live_opponent_refresh`, default 2000 lineup samples; 0 restores the old
behaviour). Every snapshot still enters play -- clustered in time instead of
interleaved -- which is the usual shape of large-scale league play. Same smoke
run after the fix:

| update | 10 | 25 | 35 | 45 | 55 |
|---|---|---|---|---|---|
| act, unbounded | 22.8 | 28.9 | 36.7 | 41.7 | (still rising) |
| act, bounded | 15.6 | 21.3 | 21.2 | **21.1** | **22.1** (flat) |

`act` now **plateaus** once the live set fills, instead of growing without bound.
At update 45 that is 2x, and the gap widens with run length. `league_test`
asserts both halves: a batch stays within 2x the cap (the transient around a
refresh, since an env holds its lineup until its episode ends), *and* >20 of 30
snapshots still enter play over many refreshes -- because bounding without
refreshing would be a silent quality regression instead of a speed one.

### Corrections to this file's own numbers

- **"env-step is 68% of a 19 us step" is no longer the right frame.** The env
  *phase* is not dominated by the engine at all. At 256 envs/16 workers it split
  `_run_step` 4.82 ms / `_legal_indices` 0.32 ms / **buffer copies 7.32 ms** --
  i.e. 59% of the env phase was `_collect` memcpy, and 65% of *that* was
  mmap/munmap of a fresh 38.5 MB array every step, not the copy itself.
- **Pinning the H2D staging buffer was tried and reverted.** The isolated
  microbenchmark is compelling (7.9 -> 3.5 ms for 38.5 MB) and it does not
  survive contact with the real loop: paired A/B measured act 19.4 -> 23.0
  ms/step, a net loss. Allocating pinned buffers *pre-fork* additionally poisons
  every worker's inherited CUDA context (7,900 -> 3,000 SPS, with the env phase
  tripling -- the tell that it is a fork problem, not a transfer problem).
- **`_last_decision`'s per-env copies cannot be fixed by pooling.** Reusing
  preallocated destinations measured 12.40 ms against `.copy()`'s 11.88 ms --
  glibc already recycles those chunks. The doc's original analysis stands: the
  only real fix is the `(row, env)` reference design, and it rewrites terminal
  attribution.
- **Item 1's payoff shrank as a side effect.** With env at 7.9 ms against act's
  ~15 ms, overlap now buys ~22% of the update here, not the ~46% it would have
  bought before `_collect` was fixed. Still the largest single item left.

Two measurement sessions, on two machines, and they disagree in places that
matter. Keep the labels straight:

- **BIG** — RTX PRO 6000, 98 GB, 12 physical cores shared with two resident vLLM
  workers. Source of the throughput ladder below. Written 2026-08-13 on branch
  `perf/ppo-throughput`, i.e. *before* the merge.
- **12GB** — RTX 4080 Laptop, 11.6 GB usable, 32 cores, 31 GB RAM (~15 GB free).
  Source of everything under "Corrections" and "What the 12 GB box can verify".
  Measured 2026-08-13 *after* the merge.

Anything not labelled and not marked *projected* is BIG.

## Merge status

`perf/ppo-throughput` is merged into master as `25938bdd`. The branch forked
from `365a059a`, before master's V2 keyed-entity encoder work, so both lineages
independently rewrote the same two hot spots. What survived:

| piece | kept from | why |
|---|---|---|
| `_ObsRows` (device-resident + fp16 obs buffer) | branch | master's `_select_obs_rows` had the no-`cat` half only; the device/dtype half is where the SPS came from |
| `TypedPointerActorHead._pick` + `slot_target` | **master** | the branch's `_entity_tables` needs the `(B,1800,width)` `slot_mlp` output master deleted — a hard incompatibility, not a preference |
| `--compile_encoder` retarget | branch (naming) | both fixed the same misdirected compile |
| aux/rank shared-trunk recompute removal | branch | master lacked it entirely |
| FCM adapter behind `OPEN_SPIEL_BUILD_WITH_FCM` | branch | master's unconditional path needs a local FCM checkout |

One bug fixed during the merge: the branch's `rank_logits_from_features` reuse
had no `features is not None` guard, and the dense path leaves `features` None.

## Corrections to the pre-merge measurements

Three claims in the original write-up are now wrong. They were the basis for the
priority ordering, so re-read the ordering below rather than the old one.

### Item 3a is already done, and over-delivered

3a was "distribute the slot branch's first linear layer" to kill the
`(B, 1800, 64)` and `(B, 1800, 68)` intermediates. Master's encoder rewrite
deleted that branch outright: planet slots are now passed through **raw**
`(B, 1800, 4)`, and only the rows an action actually points at get embedded, by
`slot_target` inside the head. 98.2% of those 1,800 rows are padding, so this is
strictly better than the algebraic refactor 3a proposed.

Encoder forward+backward peak, **12GB**, at the real `nn_width=64 nn_depth=2`:

| minibatch | peak | of which input | activations | per row |
|---|---|---|---|---|
| 1,024 | 1.15 GB | 0.15 | 1.00 | 0.98 MB |
| 2,048 | 2.29 GB | 0.31 | 1.98 | 0.97 MB |
| 4,096 | 4.56 GB | 0.62 | 3.94 | 0.96 MB |
| 8,192 | 9.08 GB | 1.23 | 7.85 | 0.96 MB |
| 16,384 | OOM | — | — | — |

Against the **20.8 GB** BIG profiled at minibatch 8,192 pre-merge: the peak is
down ~2.3x. Scaling is dead linear at 0.96 MB/row, so learn is still
memory-bound — but the cause has moved off the slot branch, and "matmuls are
only 19 ms of 127 ms" needs re-profiling before anything is built on it.

### The env-step breakdown was wrong in both directions

The claim was "a full `rl_environment.step` is 46 µs of which the action itself
is ~4 µs; the rest is writing 150 KB of observation", flagged *measured
indirectly — confirm directly before acting*. Confirmed directly, **12GB**, from
a mid-game state (move 133), median of 50-call batches:

| call | µs |
|---|---|
| `legal_actions()` | 0.8 |
| `apply_action()` | 5.2 |
| `observation_tensor_into(0, fp32_buf)` — **the path training uses** | 13.0 |
| `observation_tensor(0)` — boxed Python list | 308.0 |

So per decision it is ~19 µs, not 46, and the observation write is 68% of it —
not the ~91% the indirect estimate implied. `apply_action` at 5.2 µs matches the
old ~4 µs.

The 24x gap between the two observation calls is a live landmine. `rl_environment`
only takes the in-place path when constructed with `observations_as_numpy=True`;
every call site in `ppo_eclipse.py` passes it today (five of them), and any new
one that forgets pays 308 µs per observation. If a future eval or ladder path
looks inexplicably slow, check that flag first.

### Env count is *not* free on 12 GB

BIG's "env count is free up to 4,096" holds only where the buffer fits. The
`auto` budget is 50% of free VRAM = **6.1 GB** here, so:

| envs × steps | buffer (fp16) | `auto` places it |
|---|---|---|
| 128 × 128 | 1.23 GB | device |
| 256 × 128 | 2.46 GB | device |
| 512 × 128 | 4.93 GB | device |
| 1,024 × 128 | 9.86 GB | **CPU fallback** |

Combined with the activation table: 512 envs (4.93 GB buffer) plus minibatch
4,096 (4.56 GB peak) is 9.5 GB of 11.6 — it fits, with nothing to spare. The
comfortable ceiling here is **256–512 envs at minibatch 2,048–4,096**.

## What the 12 GB box can verify, and what it cannot

| | verifiable here | needs BIG |
|---|---|---|
| correctness / numerical parity of any change | yes | — |
| the 2,800 → 5,157 SPS steps (256 envs, device buffer, compile) | yes, same regime | — |
| 5,735 → 8,863 (1,024 envs fp16, `update_epochs=1`) | no — buffer falls back to CPU | yes |
| minibatch efficiency curve above 8,192 rows | no — OOM | yes |
| does `update_epochs=1` *learn* as well per sample | no — needs long runs | yes |
| env-step cost and obs-write share | **better here** (32 cores vs 12 shared) | — |

## Work items, in order

### 1. Overlap acting with env stepping

Still the top item: worth ~25% of the update, needs no retraining, and is a pure
reordering rather than an approximation — policy parameters do not change during
a rollout, so whether env *i* and env *j* stepped at the same wall-clock instant
is not observable in the data. Act and env are strictly serialized today (BIG:
6.03 s and 5.66 s of a 14.85 s update), so rollout cost becomes `max` instead of
`sum`: ~14,300 SPS *projected*.

**Do not** extend this to overlapping envs with `learn()`. That needs actors on
stale parameters — what OpenAI Five used 512 dedicated forward-pass GPUs to do
without contention. Out of scope on one card.

The sketch stands: split `AsyncVectorEnv._run_step` (`open_spiel/python/async_vector_env.py:222`)
into `send` (write `action_buf`, release every `_go`) and `wait` (acquire every
`_done`, `_collect`) — the release and acquire loops are already separate
statements. Then alternate two env groups so one group's CPU step overlaps the
other's GPU forward.

The original write-up said only `post_step_np`'s row-level tensor writes need an
env-subset argument. That understates it. The real list, all in
`open_spiel/python/pytorch/ppo.py`:

- `post_step_np` (`:1247`) owns `self.cur_batch_idx += 1` and
  `self.total_steps_done += self.num_envs` (`:1301`). Called once per group, both
  double-count. Row advancement has to move out of it.
- Its internals are written over all envs: `np.arange(self.num_envs)`,
  `for i in range(self.num_envs)`, `np.zeros(self.num_envs)`. All need to be over
  the subset.
- The act path writes whole rows too — `self.obs[row]`, `self.actions[row]`,
  `self.logprobs[row]`, `self.values[row]`, `self.trainable[row]` (around `:940`).
- `_collect()` copies the full `(N, obs_size)` array per call (38.5 MB at 256
  envs). Two groups means two calls per logical step, so it needs a row subset or
  the overlap gives back part of what it wins.

Per-env state (`_last_decision`, `_episode_start_row`, `_pending_phi`) is already
per-env and needs nothing. Row bookkeeping is safe: both groups run the same 128
steps and write the same `row` for their own env columns, so they cannot
desynchronise.

This is the terminal-attribution path, where a silent bug already cost this
project a 408M-step run. Budget the care here, not in the env layer, and do not
land it without V4 below.

### 2. `update_epochs=1`

No code needed — merged and available. BIG measured 1.55x (5,735 → 8,863 at
1,024 envs); `learn` drops 10.84 s → 2.81 s because it runs
`update_epochs × num_minibatches` forward+backward passes. This is also OpenAI
Five's regime (sample reuse ≈ 1.0–1.1 for "Rerun").

What to watch, and none of it is settled by throughput:

- `approx_kl` and `clipfrac` **will** drop — a quarter as many gradient steps per
  batch is less policy movement per update, not more stability. The LR almost
  certainly wants to go up.
- `explained_variance` may fall: the value head also gets a quarter of the
  updates. Consider a separate epoch count for the critic.
- Same samples/hour only holds if one epoch learns as much per sample. That is
  the actual experiment, and it needs BIG.

### 3. Shrink the observation

Revised expected value, given the corrections above. The 37,596-float
observation is still the common cause behind four costs, but the sizes changed:

| cost | pre-merge belief | now |
|---|---|---|
| env-step | ~91% of a 46 µs step | 68% of a 19 µs step (**12GB**) |
| H2D on act | 38.5 MB/step at 256 envs | unchanged |
| buffer size / env ceiling | binding | binding, and the 12 GB ceiling above |
| learn memory traffic | 20.8 GB peak, slot branch | 9.08 GB peak, cause unidentified |

Where the floats are:

| block | floats | share |
|---|---|---|
| galaxy (225 cells × 88 channels) | 19,800 | 53% |
| V2 planet slots (1,800 × 4) | 7,200 | 19% |
| player blocks (6 × 547) | 3,282 | 9% |
| V2 units (128 × 24) | 3,072 | 8% |
| everything else | ~4,242 | 11% |

For scale: OpenAI Five used ~16,000 inputs per hero for a more complex game;
AlphaStar used 8×128×128 world planes + **512 entities × 43 features** + scalars.
Entity-list encoding is why their cost tracks what is on the board rather than
the board's capacity.

**3b. Entity-list the planet slots.** 1,800 slots exist because 225 cells × 8
capacity, but most cells hold 0–3 planets (measured: 32 valid rows). Emit a real
list with a validity bit — exactly what the V2 unit rows already do with
`U_VALID` — and 7,200 floats become ~800. Buys buffer size and env-step cost.
Note it buys *little* learn-phase memory: `slot_target` is already per-pointer,
so the head's cost does not scale with 1,800.

**3c. Entity-list the galaxy.** The biggest block, mostly unexplored or empty
cells. A fixed-capacity list of ~96 occupied cells × 88 channels takes 19,800 →
~8,400. This is the one that should also cut learn activations, since it shrinks
the conv tower's input from 225 cells to ~96 — but that is a *hypothesis*, and V6
below is how to check it before paying the cost.

**Cost of both:** they change the C++ observation writer and `obs_layout.py`,
which invalidates every existing snapshot and roster and means retraining from
scratch. Do them at a point where the league is being reset anyway.

Current roster state, in case that gate is already open: `runs/roster/main.pt`
and its snapshots are from 2026-08-07 at observation size **1,785**, against
today's 37,596. They are already incompatible — there is no live roster
investment for 3b/3c to destroy. (`runs/roster/arch.json` was rewritten
2026-08-13 by post-merge smoke runs, which take `--roster_dir=runs/roster` by
default; no `.pt` was touched.)

## Verification plan on the 12 GB box

In order. V1–V3 are done; V4–V6 gate the work items above.

**V1. Merge parity — DONE.** `open_spiel/python/pytorch/ppo_obs_rows_test.py`,
14 cases: `_ObsRows` against the `torch.cat` reference it replaced, across
{cpu, cuda} × {fp32, fp16}, including the `select(bool mask)` →
`select(slice)` composition `_learn_core` performs for `--league`. Plus
`ppo_league`, `ppo_sparse_act`, `ppo_win`, `ppo_selfplay`, `action_factors` all
green.

**V2. Merge smoke — DONE.** Live runs, all clean: auto-placed `cuda/float16`
buffer; forced `--obs_buffer_device=cpu --obs_buffer_dtype=float32`; `--league`;
`--compile_encoder`. Both encoder entry points confirmed to reach the compiled
function (`forward_with_context` and `forward`), and `ctx.slots` confirmed
`(B, 1800, 4)` — i.e. master's raw-slot encoder, not the branch's.

**V3. Cost baselines — DONE.** The activation and env-step tables above.

**V4. fp16 buffer does not move the loss.** The one merge risk no test covers:
the obs buffer now *stores* fp16. Two runs, same `--seed`, identical otherwise,
`--obs_buffer_dtype=float32` vs `float16`, ~50 updates, `--no_tb --timing`.
Compare the `policy_loss / value_loss / entropy / approx_kl / explained_var`
series. Pass: differences stay at autocast-rounding scale and do not trend. A
trend means fp16 storage is losing something the bf16 autocast was not already
discarding — in which case pin `float32` before any long run.

**V5. Act/env overlap equivalence — the gate for item 1.** Fixed seed, two
groups vs one group, same number of steps. Pass: `self.obs`, `self.actions`,
`self.logprobs`, `self.values`, `self.rewards`, `self.dones` and
`self.trainable` are **bitwise identical**, and `total_steps_done` and
`cur_batch_idx` match exactly. Include at least one episode boundary inside the
batch so terminal closeout is exercised, and assert `_extra_samples` matches in
count and content. This is cheap here and it is the only thing standing between
item 1 and another silent-attribution run.

**V6. Where the 7.85 GB of activations actually is — the gate for 3c.** Before
paying a from-scratch retrain for the galaxy entity-list, confirm the conv tower
is the thing to shrink. Profile the merged encoder at minibatch 4,096 (fits
here) per-op, and record the conv-tower share against the head's and the tail
MLP's. If the tower is not dominant, 3c buys buffer size only and 3b is the
better first move.

**V7. Throughput regime check.** Reproduce the BIG ladder's first three rungs at
256 envs — committed config, then device buffer, then retargeted compile — using
`--timing --timing_every` and reading `sps` / `learn_share` off the new timing
line. Pass: the *ordering and rough proportions* match BIG. Absolute SPS will
not, and should not be compared: this box has 32 cores against BIG's 12 shared,
so the env phase is relatively cheaper here and will understate item 1's win.

Bound every run: this box is shared and has crashed twice. `systemd-run --user
--scope -p MemoryMax=14G -p CPUQuota=800%` in front of the command, and check
`nvidia-smi` before believing an OOM — a crashed run's async workers block
forever on their semaphores and nothing releases VRAM.

## Recommended config

**BIG** (~8,863 SPS):

```
--num_envs=1024 --num_steps=128 --num_workers=20 --num_minibatches=16 \
--update_epochs=1 --amp --compile_encoder --lr_schedule=fixed
```

**12GB** (verification, not throughput):

```
--num_envs=256 --num_steps=128 --num_workers=16 --num_minibatches=16 \
--update_epochs=1 --amp --compile_encoder --lr_schedule=fixed
```

Raise `num_envs` and `num_minibatches` together, keeping the minibatch near
8,192 rows on BIG (most efficient measured: 47,800 rows/s vs 43,100 at 16,384
and 41,280 at 32,768) and near 2,048–4,096 here.

## Deliberately not doing

- **Overlapping envs with `learn()`** — see item 1.
- **Eliminating the per-step `_last_decision` observation copies.** 4.43 ms/step
  at 256 envs (0.57 s/update, ~9%). The self-play closeout path stores a full
  150 KB observation per env per step to serve a case that only fires when a
  seat's last decision predates the batch. The clean fix stores a `(row, env)`
  reference and snapshots only at the batch boundary — 32x fewer copies — but it
  rewrites terminal attribution. Not worth 9% without a test that exercises the
  cross-batch branch. (V5 would build most of that test; revisit after item 1.)
- **More env-stepping micro-optimization.** At 482 env-steps/s/core against
  OpenAI Five's ~4.9, per-core work is not the problem; core count is. The
  exception is item 3, which reduces the step cost by reducing what it writes.

## Operational notes

- **A crashed run holds the GPU indefinitely.** When the parent dies the async
  workers block forever on their semaphores and nothing releases VRAM. A 91 GB
  zombie made a later clean config look like a fresh OOM. Check `nvidia-smi`
  before believing an out-of-memory result.
- **`--max_seconds` is only honoured on the async path** (`num_workers > 0`).
  Silently ignored in sync mode. Two-line fix, not yet done.
- **Compiling the wrong function is invisible.** `--compile_encoder` spent its
  life wrapping a body no hot path called; nothing errored, the flag just did
  less and less. Compile on vs off measured 3.99 s vs 4.04 s of learn while
  misdirected, and 2.74 s once corrected. If the encoder entry points are
  refactored again, re-check that `_compiled_context` is what the act and learn
  paths actually reach — the spy check in V2 is four lines.
- **`observation_tensor()` costs 24x `observation_tensor_into()`.** Any
  `rl_environment` built without `observations_as_numpy=True` pays 308 µs per
  observation instead of 13.
- **pyspiel import on this box:**
  `PYTHONPATH=$PWD/build/open_spiel/python:$PWD .venv/bin/python`. The
  `build/` tree's `.so` is current; `cmake-build-debug-clang/`'s is from June.
