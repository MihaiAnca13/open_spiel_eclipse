# Eclipse 4P RL — state, findings, and queued work

**Goal:** an Eclipse agent strong enough to play against competitively in 4-player games.

**This file is the single source of truth.** It absorbed `next_work.md`,
`docs/eclipse_rl_plan.md` and `docs/eclipse_observation_v2_mcts_notes.md` on
2026-08-15. Session history before 2026-08-13 (waves 1-3, the 12 GB box's
architecture sprint, the Items 0-5 log) was pruned in the same pass — it is
superseded by the BIG re-baseline below and recoverable from git at `7cafe2d0`.
The observation-writer half lives in `docs/eclipse_observation_v2.md`.

---

## Current state — read this first

**Throughput work is done.** 18.87 → 12.44 s/update at 1,024 envs (1.52x, sps
6,945 → 10,540; 10,704 real sps over a clean 30-min run). Detail under "BIG
re-baseline" below.

**T1 has RUN. It did not settle the config, and it surfaced a bigger problem: at
both epoch counts the run stops improving early, and at `update_epochs=1` it then
regresses.** Headline:

- ue=4 is **flat after update 100** — its whole 1,622-update rating range is
  narrower than one CI.
- ue=1 **learns then collapses**: +0.967 (u100) → +1.114 (u1700) → **+0.935**
  (final), ending below its own update-100 snapshot.
- The mechanical PASS is a 0.004 margin on a best-of-5 vs best-of-5 comparison
  whose two maxima sit 1,600 updates apart. Head-to-head of the two *finals*
  favours ue=4.

**So the gate is no longer "which epoch count" — it is "why does this stop
learning". Do not start a long run until that is understood**: at either setting
most of the hours buy nothing, and at ue=1 they actively cost rating.

A long run is expensive and its failure modes are quiet. The two that already bit
this project — a terminal-attribution bug that cost 408M steps, and a league
throughput cliff invisible in every metric — were both found by *running the
thing*, not by reasoning about it. That held again: the biggest win this round
(the pointer head's gradient scatter, 43.6% of `learn`) was invisible until
something profiled it.

## Ground rules

1. **Every pre-2026-08-13 SPS number in any doc is phase-sum-derived and wrong
   under overlap.** T0 re-baselined; use the numbers in `eclipse_rl_todo.md`.
2. **Judge strength only via ladder `rating` / `rating_ci`.** Never `elo`,
   `vp_all`, `mean_episode_return`, or vs-Greedy. Ratings are not comparable
   across ladder runs — so both T1 arms must be rated in ONE tournament.
3. **Never A/B a training change by diffing a loss curve.** It is bit-identical
   for two updates and chaotic after; see the determinism note in
   `eclipse_rl_todo.md`.
4. Run arms **sequentially**. Two arms thrashed the GPU to ~45 SPS; three OOM'd.
5. **GPU 3 only.** GPUs 1–2 hold two resident vLLM workers (~96.8 GB each,
   permanent); GPU 0 drives the display. Export `CUDA_VISIBLE_DEVICES=3`.
6. **When a new measurement contradicts a recorded number, suspect the new
   harness first.** Three "the doc is wrong on BIG" findings this round were all
   instrumentation bugs. The three genuine overturns each survived a *paired
   end-to-end run* — that is the distinction that separates them.
7. **Peaking then slipping is a recurring shape, not a one-off.** The 2026-08-09
   `long8h` run peaked at u3900 (+1.2195) and finished lower at u4200 (+1.1431),
   and T1's ue=1 arm did the same far harder. Snapshot densely enough to locate a
   peak, and never assume the final policy is the best one.

## The settled configuration

Every value below was measured on BIG, not inherited:

```
--num_envs=1024 --num_steps=128 --num_workers=16 --num_minibatches=16 \
--update_epochs=<T1> --max_live_opponents=4 \
--amp --compile_encoder --lr_schedule=fixed --league \
--overlap_record --snapshot_every=100 --timing --timing_every=50
```

- `num_workers=16`, **not the 20 this file used to recommend** — that came off the
  12 GB box's 32 cores. 12→16 cuts the env phase 9%; 16/20/24 are a plateau inside
  the noise. 16 leaves cores for the co-resident vLLM workers.
- `num_minibatches=16` keeps 8,192 rows/minibatch, the most efficient point
  measured. Raise `num_envs` and `num_minibatches` together or that ratio drifts.
- `max_live_opponents=4` costs 6.5% of the update and is **not free** — roughly
  1.6% of throughput per extra live opponent. Never 0 (0 means unbounded).

## THE GATE NOW: why does training stop improving?

Both T1 arms plateau or regress well before their 5 hours are up. Until this is
understood, a long run is buying hours of nothing. Ordered by how cheaply each can
be tested, and all judged on ladder `rating` only:

1. **Entropy collapse.** ue=1 fell to 0.68 against ue=4's 0.92, and ue=1 is the arm
   that regressed — a more deterministic policy is a more exploitable one in a
   4-player game. Test: raise `--ent_coef` (currently 0.05) on a 2h ue=1 arm and
   rate against this run's snapshots.
2. **No LR decay.** `--lr_schedule=fixed` throughout, and the doc bans the `kl`
   controller as unvalidated. The regression starting around u1700 is consistent
   with too much late movement. Test: cosine or step decay, same budget.
3. **League overfitting to the bounded live set.** `--max_live_opponents=4` keeps K
   small for throughput, but a policy trained against 4 rotating opponents may
   sharpen against them specifically. Test: raise the cap for a 2h arm — it costs
   ~1.6% throughput per extra opponent, which is affordable for a diagnostic.

Whatever is tested, **snapshot densely** (`--snapshot_every=25`) so the peak is
locatable. The roster now keeps a genuine mid-run spread; before 2026-08-15 it
collapsed to the two ends and hid exactly this shape.

Also worth doing regardless: **the peak policy already beats everything else
measured.** `t1_ue1:snap_u1700` (+1.114) is the strongest net in the tournament. If
a strong model is wanted now rather than an explanation, that snapshot is it.

---

## T1 results, 2026-08-14/15 — and the finding that matters more than T1

Two arms, equal wall clock (18,000 s each), 1,024 envs, `--league`,
`--max_live_opponents=4`, seed 1, LR 2.5e-4. Judged by ONE ladder tournament,
`games_per_dir=64`, 13 policies, 1,536 games per policy.

| | ue=4 | ue=1 |
|---|---|---|
| updates | 1,622 | 2,192 |
| steps | 212,598,784 | 287,309,824 |
| real sps | 11,811 | 15,962 |
| **sample ratio** | 1.00 | **1.35x** |

Note the ratio: T1 was designed when `update_epochs=1` was believed to buy **1.55x**.
After the throughput work `learn` is only ~13% of the ue=1 update, so fewer epochs
save much less. **The premise of the experiment shifted under it**; re-derive this
ratio before quoting any older figure.

### Ratings

| ue=4 | rating | ue=1 | rating |
|---|---|---|---|
| u100 | **+1.054** | u100 | +0.967 |
| u1200 | +1.022 | u1700 | **+1.114** |
| u1400 | +1.053 | u1900 | +1.107 |
| u1600 | +1.038 | u2100 | +1.060 |
| main@1622 | +1.030 | main@2192 | **+0.935** |

(Anchors: Greedy +0.194, Heuristic +0.177, Random 0. Every net is far above the
bots and the arms are within ~0.1 of each other, so all the action is in a narrow
band roughly two CI widths wide.)

### The verdict, and why it must be quoted with its caveats

`tools/t1_verdict.py` prints **PASS**: ue=1's best (u1700, lower bound +1.084)
clears ue=4's best (u100, upper bound +1.080). **The margin is 0.004.** Three
things have to be said alongside it:

1. **It is a best-of-5 vs best-of-5 comparison**, and max-selection over five
   noisy policies inflates both arms. Only `u1700` strictly clears; `u1900`
   (+1.107, lower bound +1.077) does not.
2. **The two maxima sit at completely different ages** — u1700 against u100. That
   is not a controlled comparison of epoch counts.
3. **Head-to-head of the two FINAL policies favours ue=4**: margin +0.057, W/L/D
   41/40/47. The policy you would actually ship from ue=1 is the worst net in the
   tournament.

The pass rule ("ue=1's rating lower bound clears ue=4's upper bound")
**never said which policy**, and best-of-arm vs final-of-arm give opposite answers
here. That is a defect in the rule, not a close call to be resolved by picking the
flattering reading.

### The real finding: neither arm improves late, and one regresses hard

- **ue=4 does not learn after update 100.** Its whole 1,622-update range spans
  +1.022 to +1.054 — narrower than one CI. 212M steps bought nothing measurable.
- **ue=1 learns, then collapses.** u100 → u1700 is +0.967 → +1.114 (CI-clear, a
  real gain). u1700 → main is +1.114 → +0.935 (CI-clear), ending **below its own
  update-100 snapshot**. The ladder flagged it independently: `NON-MONOTONE — 2
  well-separated snapshots rated strictly below an earlier one`.

So ue=1 is the better *learner* and ue=4 is merely stable. **A long run at either
setting spends most of its hours past the point of gain, and at ue=1 it spends them
getting worse.** Diagnose this before committing to a long run. Suspects, in order:
entropy collapse (ue=1 fell to 0.68 against ue=4's 0.92, i.e. a more deterministic
and more exploitable policy), no LR decay (`--lr_schedule=fixed`), and league
overfitting to the bounded live-opponent set.

### `vp_all` rose while the ladder rating fell — the rule earns itself again

Across exactly the window where ue=1's rating dropped +1.114 → +0.935, its in-run
`vp_all` climbed 14.62 → 15.81 and `mean_episode_return` sat at 17 against ue=4's
8.5. Anyone watching the training log would have concluded ue=1 was pulling ahead.
**This is why `vp_all` / `mean_episode_return` / vs-Greedy are on the
never-judge-by list.** Only the ladder saw the regression.


### The pass rule itself is defective

**THE RULE AS WRITTEN IS UNDERSPECIFIED — it never says WHICH policy per arm.**
`t1_verdict.py` chose best-of-arm; comparing finals instead gives the OPPOSITE
answer on the 2026-08-14 data (ue=1 wins on best-of-arm by 0.004; ue=4 wins
head-to-head on finals). Any future use must state which, and pick it *before*
seeing the ratings. Best-of-arm also has a selection bias that widens with the
number of snapshots rated, so it is only honest with the same count on both sides.

If it fails, **re-run once with a raised LR** before discarding it. A null result
at an LR tuned for 4x the gradient steps is not a result.

**Budget the tournament before running it.** It costs **1.387 s/game** and is a
full round-robin, so cost is quadratic in snapshot count: two 5h arms' full rosters
at the default `--ladder_games_per_dir=128` is a **~35-hour job**. Subsample with
`tools/prune_roster.py` (4 snapshots/arm at `games_per_dir=64` ≈ 3.8h). Cut
snapshots first, games/dir last — dropping games widens every CI, which directly
weakens the pass test.

---

## BIG re-baseline, 2026-08-14 (RTX PRO 6000 Blackwell, GPU 3, tree `6f7d9219`)

The T0 pre-flight, run on BIG. **These supersede every
pre-2026-08-13 SPS number for this machine**, which were phase-sum-derived.
Harness: `run_t0_baseline.sh` + `tools/parse_t0.py`.

`overlap` reads **1.00x on all five rungs**, so the wall-clock bracket is sound
and the phase breakdown may still be summed. That was T0's gate.

Every rung: 128 steps, `--amp`, fp16 device obs buffer, `update_epochs=4`, and
**8,192 rows per minibatch** (mb=4 at 256 envs, mb=16 at 1,024) so the env-count
rung is not also a minibatch-size change. `num_workers=16`. Means over updates
4–20; `STEADY` is the last three, which is the number to compare rungs on.

| rung | update | learn | rollout | learn_share | steady sps | act:env |
|---|---|---|---|---|---|---|
| base (cpu buffer, no compile) | 7.40s | 4.46s | 2.94s | 60% | 4,427 | 2.14:1 |
| + device obs buffer | 5.75s | 2.91s | 2.84s | 51% | 5,702 | 1.85:1 |
| + `--compile_encoder` | 5.11s | 2.25s | 2.85s | 45% | 6,419 | 1.82:1 |
| 1,024 envs, mb=16 | 18.87s | 9.37s | 9.50s | 50% | **6,945** | 1.50:1 |
| + `--league` (bookkeeping only) | 5.01s | 2.20s | 2.81s | 45% | 6,541 | 1.82:1 |

- **The device obs buffer is worth 1.29x and `--compile_encoder` a further 1.13x,
  and both land entirely in `learn`.** `act` (13.8 → 13.7 ms) and `env` (7.5 →
  7.5 ms) are flat across all three rungs. Neither flag touches the rollout.
- **`learn` is only 45–50% of the update here, against 65% on the 12 GB box.**
  This single difference is what re-prices T2 below.
- **1,024 envs beats 256 by 8%** at equal rows/minibatch, so the gain is
  rollout-side amortization, not a learn effect.
- `torch.compile` warms up for ~10 updates past update 0 (learn 2.74 → 2.34 →
  2.26 s at u4/u8/u12) and its one-off trace costs ~29 s of update 0. Averaging
  the warmup in makes a compile win read as a regression; `parse_t0.py` reports a
  steady-state window for exactly this reason.
- **Read the `league` rung narrowly.** It ran `--snapshot_every=0`, so the roster
  never grew past `main` and the batch held K=1 distinct policies. It prices
  `_refresh_lineups` bookkeeping — which is free — and **not** the per-policy act
  cost, which is the expensive, superlinear half of league play. Do not quote it
  as "the cost of `--league`".

### T2 (act/env overlap) is worth ~3x more on BIG than the 12 GB numbers implied

T2 was originally sized at ~22% of the rollout but only ~7% of wall clock, and
concluded it "is a minor item and arguably not worth its risk" unless
`update_epochs=1` wins. On BIG, at `update_epochs=4`, measured:

| config | min(act+phi, env) | of rollout | **of wall clock** |
|---|---|---|---|
| 256 envs | 7.50 of 22.29 ms | 34% | 19% |
| 1,024 envs | 28.97 of 74.23 ms | 39% | **20%** |

The reason is not that overlap got better — it is that `learn` no longer
dominates (45–50% vs 65%) while the env phase got relatively *more* expensive, so
the same overlap covers a larger share of a smaller denominator. **T2's value
therefore no longer hinges on T1**: it is a fifth of wall clock at
`update_epochs=4` and would be ~30% at `update_epochs=1`. Re-read T2's
"contingent on T1" framing against these numbers before deprioritizing it.

### `--compile_encoder` did not cover the pointer head

The flag wrapped `SpatialEclipseEncoder._encode_context_impl` only, leaving
`TypedPointerActorHead._pairs` eager. After the `embedding_bag` / combined-pick work
above, `elementwise / copy` became `learn`'s largest bucket at **32.8%** — the
`torch.where` / `cat` / `sum(-1)` chain in `_pairs`, which is exactly what an
inductor fusion is for.

Split into `_pairs` (dispatcher) and `_pairs_impl` (body), compiled under the same
flag with the same lazy fallback-to-eager-on-first-exception pattern the encoder
uses, and `dynamic=True` for the same reason — the pair count M changes on every
call, so a static graph would recompile endlessly.

**Re-check the compile target whenever the head's entry points are refactored.**
This is the second time this trap has fired in this file: `forward_with_context`
once bypassed the compiled body entirely, making a measured 1.57x silently inert.
Both `_encode_context` and `_pairs` fall back to eager on any exception with only a
`warnings.warn`, so an inert compile is quiet by design.


### act/env overlap without env groups: +7.9%, and the ceiling is CPU contention

T2's payoff was reachable without the env-group split. `AsyncVectorEnv.start_step` /
`await_step` split the round trip, and `PPO.flush_selfplay_record` defers the per-env
`_last_decision` bookkeeping, so the loop releases the workers and does the
bookkeeping while they run (`--overlap_record`, default on). Safe by the invariant
`_collect` already documents: workers write only shm, `_collect` copies into one of
two alternating generation buffers, and a held `_StepArrays` points at a generation.

Paired A/B, 1,024 envs, `--league` with `--snapshot_every=0` so K=1 (every row
trainable, the worst case for bookkeeping):

| | nooverlap | overlap |
|---|---|---|
| `act` (step_np alone) | 30.21 ms | **9.50 ms** |
| `act+phi` (incl. the deferred flush) | 30.56 ms | 38.75 ms |
| `env` (residual wait) | 27.31 ms | **11.84 ms** |
| `rollout` | 7.70s | 6.72s |
| `update` | 13.42s | **12.44s** |
| steady sps | 9,765 | **10,540** |

The mechanism is confirmed — `act` fell 30.2 → 9.5 ms as the bookkeeping left
`step_np`, and the env *wait* fell 27.3 → 11.8 ms because it now happens during it.

**But only ~15.5 ms of the 27.3 ms env step got hidden, and the reason is CPU
contention, which caps this whole class of optimisation on this box.** The
bookkeeping is ~29 ms of memcpy-heavy main-thread work competing with 16 env workers
for 12 physical cores that the two resident vLLM workers also use. Overlapping CPU
work with CPU work has limited headroom here; an estimate that treats the main thread
and the workers as independent (this one predicted ~30%) will be roughly 4x
optimistic.

**Consequence: the env-group split is not worth building.** It would need main-thread
CPU for act-side work while workers run — the same wall. `T2 sizing` now reads 12% of
wall clock, down from 26%, so the headroom that remains is small. Same argument
retires the `(row, env)` reference design for `_last_decision`'s copies.

### Result so far: 18.87 → 12.44 s/update at 1,024 envs (1.52x, sps +52%)

| change | update | steady sps |
|---|---|---|
| T0 baseline | 18.87s | 6,945 |
| + `embedding_bag` + combined cell pick (`learn` 1.59x) | 15.30s | 8,569 |
| + pinned collect destination + vectorised `step_np` | 13.96s | 9,390 |
| + compiled pointer head | ~13.42s | 9,765 |
| + `--overlap_record` | **12.44s** | **10,540** |

A 30-minute production run at this config completed cleanly: 147 updates,
19,267,584 steps in 1800 s = **10,704 real sps** end to end, including startup, the
one-off compile trace and snapshotting. No NaN, no errors, `overlap` 1.00x throughout.

That run also validated the league cap end to end. As snapshots accumulated to the
4-opponent cap, `act` rose 25.6 → 34.7 ms (+36%) and `update` 11.34 → 12.24 s
(+7.9%) — against T3's predicted 6.5% for K=5, i.e. the right magnitude. The rise is
bounded, unlike the pre-fix behaviour (22.8 → 41.7 ms over 45 updates and still
climbing), though 147 updates is too short to demonstrate a hard plateau; `league_test`
asserts the cap directly and that is the real guarantee.

Three changes, each measured end to end on the T0 harness with a compiled encoder,
1,024 envs / mb=16. `learn`'s 1.59x came from the pointer head; the rollout's from
the act phase.

| | baseline | +learn fix | +act cleanup +pinned collect |
|---|---|---|---|
| `act` ms/step | 42.51 | 42.51 | **31.07** |
| `env` ms/step | 28.31 | 28.31 | 28.15 |
| `rollout` | 9.50s | 9.40s | 7.93s |
| `learn` | 9.37s | 5.90s | 6.03s |
| **`update`** | **18.87s** | 15.30s | **13.96s** |
| steady sps | 6,945 | 8,569 | **9,390** |

`act:env` is now **1.10:1** (was 1.50:1), i.e. nearly balanced — which is the most
favourable possible shape for T2, whose payoff is `min(act, env)`. T2 is now worth
**46% of the rollout and 26% of wall clock**. Every rollout-side fix makes T2 worth
*more*, and every learn-side fix does too; they must be re-sized after each change,
not sized once.

### The `act` phase is 14% network and 82% bookkeeping — and the note that dismissed it was measured at the wrong env count

**This is the largest throughput item now known, and it was hiding behind a
"deliberately not doing" entry.** `tools/act_decompose.py` breaks `step_np` (which
is exactly what the `act` timing slot brackets) into its pieces on real
observations and real legal masks, and checks the sum against a real end-to-end
call so the decomposition can be believed:

| piece of `step_np` | 1,024 envs | share | 256 envs | scaling |
|---|---|---|---|---|
| `_last_decision` bookkeeping (`PPO._last_decision`) | **19.06 ms** | 47.5% | 4.51 ms | 4.2x |
| — of which per-env `obs_cpu[i].copy()` | 10.19 ms | 25.4% | 1.70 ms | **6.0x** |
| H2D obs upload (154 MB float32) | **14.03 ms** | 35.0% | 3.05 ms | 4.6x |
| `_act_sparse` — the actual network | 5.78 ms | 14.4% | 3.11 ms | **1.86x** |
| rollout row writes | 0.33 ms | 0.8% | 0.09 ms | 3.7x |
| — of which the `_acts_trainable` loop | 0.16 ms | 0.4% | 0.04 ms | |
| **sum / measured end to end** | 39.21 / **40.10 ms** | | 10.77 / 12.38 ms | |

**The network is 14% of `act`.** The other 33 ms — `4.24 s of every 18.87 s
update, 22.5% of the whole update` — is memcpy and Python bookkeeping performing
no arithmetic.

Why this was missed: the planning notes *did* list "Eliminating the per-step
`_last_decision` copies" — but under **Deliberately not doing**, justified by
"pooling is a measured wash" (12.40 ms against a pooled 11.88 ms). That
measurement was taken at **256 envs** and it tested whether *one particular fix*
helped. It says nothing about the magnitude of the cost, and every term in it
scales with `num_envs`: the Python loop linearly, the copies **superlinearly**
(6.0x for 4x envs — the same mmap-threshold signature the 12 GB work already found
in `_collect`, where allocating a fresh 38.5 MB buffer per step cost more than the
copy). A cost dismissed at 256 envs is 4x the slice at the 1,024 the long run
wants.

Note `_last_decision` is 19.06 ms of which only 10.19 ms is the obs copies. The
remaining ~8.9 ms is **three separate `.cpu().numpy()` D→H transfers** plus the
per-env Python loop and its column slicing — and that half needs no semantic
change, so it splits into:

- **~8.9 ms/step (1.14 s/update, 6%) — low risk.** Batch the three D→H transfers
  into one; hoist the per-env column slicing out of the Python loop.
- **10.19 ms/step (1.30 s/update, 6.9%) — needs the rewrite.** The copies exist
  because `obs_cpu` is a view into shared memory that the next env step
  overwrites, and `_collect` deliberately keeps only two live generations, while
  terminal attribution may need a seat's decision from many steps back. The fix is
  the `(row, env)` reference design pointing into the **rollout buffer**, which is
  device-resident and persists for the whole batch — that is why it "rewrites
  terminal attribution", and terminal attribution is where the bug that cost 408M
  steps lived. Gate it exactly as T2 demands: bitwise golden reference first.

#### Pinning, done under the two conditions the negative result asks for

The H2D is **bandwidth-bound**: 154 MB fp32 at 11.1 GB/s = 13.9 ms, and 77 MB fp16
at 11.6 GB/s = 6.7 ms. Pinned host memory reaches 2.8 ms.

The fix is to change the *destination* of the copy `_collect` already does, not to
add a staging copy. `AsyncVectorEnv._alloc_obs_dest` returns a numpy **view of a
pinned torch tensor**, and `torch.from_numpy` on such a view reports
`is_pinned() == True` — so `ppo.step_np`'s existing
`torch.from_numpy(obs_cpu).to(device)` takes the pinned path with **no change in
ppo.py at all**. Both conditions the old entry demands are met by construction:

- **No extra copy.** `np.copyto` into plain 10.86 ms vs into pinned 10.98 ms.
- **Strictly post-fork.** `_collect_bufs` is allocated lazily on the first
  `_collect`, which runs after `__init__` has forked the pool.

Measured in the real loop, paired: **act 42.51 → 31.07 ms**, and critically **env
28.31 → 28.15 ms — flat**. A tripled env phase was the signature of the original
fork-context failure; its absence is what says this is the transfer win and not the
old trap. Falls back to plain host memory when torch or CUDA is absent.

#### Deprioritized (not unsafe): fp16 on the wire

Halving the wire dtype would halve three copies (`_collect`'s memcpy, the H2D, and
`_last_decision`'s obs copies). It is **deprioritized because it is now mostly
redundant**, not because it is unsafe:

- the H2D half is already captured by the pinned destination above (14.2 → 2.8 ms);
- the `_last_decision` half is going away entirely with the `(row, env)` reference
  design, which removes those copies rather than shrinking them.

What remains is `_collect`'s memcpy in the env phase — a real but modest slice,
against a change that touches every observation consumer.

**On safety, for the record, because a first pass got this wrong.** The concern was
that the tensor encodes categorical ids as normalised scalars recovered with
`.round().long()`, and a decode dividing by `S` needs error below `0.5/S` while
fp16 gives 4.9e-4. Enumerating the decodes that **actually read the observation**:

| decode | scale | needs | verdict |
|---|---|---|---|
| `rotation` | `HEX_DIRECTIONS-1` = 5 | 1.0e-1 | fine |
| `ptype` | `PLANET_TYPE_COUNT-1` = 7 | 7.1e-2 | fine |
| `unit_cell` | `GALAXY_CELLS-1` = 224 | 2.2e-3 | fine |
| `destination` | `GALAXY_CELLS` = 225 | 2.2e-3 | fine |
| `sector_id` | 395 | 1.3e-3 | fine, 2.6x margin |

Measured over 60 mid-game states: **0 of 21,150 decodes flip** under an fp32→fp16
round trip. The cast itself is accurate to 2.4e-4 max / 5.3e-7 mean, and the
observation is entirely within [0, 1].

**`PLANET_SLOT_ROWS` = 1800 is NOT an observation decode** and does not belong in
this table — `keyed_slot = cell * 8 + slot` is built from the integer
`cell_id`/`slot_id` action-factor *buffers*, never from a normalised float. A scan
that applies every candidate scale to every observation element manufactures
failures at scales nothing uses; do not size this risk that way.

So `ppo_obs_dtype_test`'s "fp16 storage error 6.3e-13" stands, and fp16 obs storage
remains settled. Do not pin `--obs_buffer_dtype=float32`.

#### FIXED: `observation_tensor_into` silently ignored any non-float32 buffer

`pyspiel.cc` declared it `py::array_t<float>`, and **`py::array_t<T>` defaults to
`py::array::forcecast`**: a non-float32 array was silently converted to a float32
*temporary*, the observation was written into the temporary, and the temporary was
discarded. So `observation_tensor_into(seat, float16_buf)` raised nothing and left
the caller's array **all zeros** (`nonzeros: 0 vs 981` for the same state into
fp32), which reads downstream as a legal all-padding observation. It also cost a
full dtype conversion of 37,596 elements per call.

This bit this session's own obs-writer benchmark (see above) and would bite anyone
reaching for a narrower wire dtype — the failure looks like a precision problem and
is actually a silent no-op.

Fixed by taking an untyped `py::array` and checking the dtype explicitly. Note that
merely dropping `forcecast` (`py::array_t<float, py::array::c_style>`) is **not
sufficient** — that rejects float64 and the integer dtypes but pybind still accepts
float16 and still drops it. The comparison has to be written out. Now rejects
float16 / float64 / int32 / non-contiguous with `TypeError`, and non-writeable with
`ValueError`; float32 is accepted and verified equal to `observation_tensor()`.
`rl_environment._ensure_obs_buffer` already allocates float32, so the training path
was never affected.

Requires a `make pyspiel` rebuild (~1 min incremental) and the resulting
`build/python/pyspiel.so` copied over the repo-root `pyspiel.so`, which is what
`import pyspiel` actually resolves to.

#### This fix and T2 are SUBSTITUTES, not additive — size them together

T2 hides `min(act+phi, env)`. Making `act` cheaper leaves T2 less to hide, so the
two cannot be added. At 1,024 envs, `update_epochs=4` (act+phi 43.5 ms, env 28.7
ms, learn 9.37 s):

| | rollout | update | speedup |
|---|---|---|---|
| today | 9.50s | 18.87s | — |
| `_last_decision` fixed only | 7.07s | 16.44s | 1.15x |
| T2 only | 5.83s | 15.20s | 1.24x |
| both | 3.93s | 13.30s | **1.42x** |

So T2 remains the bigger single item, the `act` cleanup is cheaper and lower-risk
in its first half, and doing both is worth 1.42x rather than the 1.15 × 1.24 =
1.43x one might naively multiply (they nearly coincide here only by accident —
at other act:env ratios the gap is large). **Never size these two independently.**


### `num_workers`: 12 is clearly too few; 16–24 is a plateau inside the noise

Measured at 1,024 envs, the long run's shape (`run_t0_workers.sh`). `learn` is
correctly invariant across worker counts — every real difference is the env phase:

| workers | env ms/step | rollout | learn | steady sps |
|---|---|---|---|---|
| 12 | 31.01 | 9.77s | 10.10s | 6,602 |
| 16 | 28.18 | 9.46s | 10.05s | 6,726 |
| 20 | 28.18 | 9.50s | 10.18s | 6,669 |
| 24 | 27.58 | 9.33s | 10.04s | 6,771 |

**12 → 16 is a real 9% cut in the env phase. 16 → 24 is a further 2%, which two
hot samples per rung cannot resolve** — the sps column is not even monotonic
(w20 dips below w16 and w24 comes out highest), which is the signature of noise
rather than a curve. Do not read w24 as "the optimum"; read 16–24 as flat.

**Chose 16**, not because it measured best but because it is within ~1% of the
best and BIG's 12 physical / 24 logical cores are *shared with two resident vLLM
workers* — a 2% throughput difference is not worth starving a co-resident
production workload of cores. Both arms of any A/B share the value anyway, so the
choice cannot bias a comparison.

The previously recommended `--num_workers=20` came off the 12 GB box's 32 cores;
it is not wrong here so much as indistinguishable from 16 and 24.

### `observation_tensor_into` is 5.44 µs — the 12 GB writer win carries

`tools/t0_obs_writer_check.py`: **5.69 µs at the opening, 5.44 µs mid-game**,
against the recorded ~5 µs. It carries. `observation_tensor` costs 268 µs, i.e.
**49x** `observation_tensor_into` — *twice* the ~24x the doc records, so the
"any `rl_environment` built without `observations_as_numpy=True` pays it" warning
matters more than written, not less.

**A first pass reported 15.16 µs and concluded the win did not carry. That was a
harness bug, not a machine difference**, and it is worth knowing how it happened:
the benchmark passed a **float64** buffer. `observation_tensor_into` was declared
`py::array_t<float>`, which defaults to pybind's `forcecast` — so the call
converted the buffer to a float32 temporary, wrote the observation into the
temporary, discarded it, and returned. The timing therefore included a
37,596-element dtype conversion *and* the call did nothing at all. See the
`observation_tensor_into` trap below; the binding now rejects non-float32 buffers.

Generalisation worth keeping: **three separate "the doc's number is wrong on BIG"
findings in this session were all instrumentation bugs** (this one, a
`float32`-vs-`float64` buffer; the fp16 id-decode scare, a scan over scales nothing
uses; and the synthetic index benchmarks that under-measured by 40-60x). When a
measurement contradicts a recorded figure, suspect the new harness first.

### T3: the league act curve on BIG is far WORSE than on the 12 GB box

The prediction was the opposite: "BIG's card is far wider, so the knee is
probably further right and a larger live set is likely free". **That reasoning is
backwards and the prediction is wrong.** Measured with `tools/t3_opponent_curve.py`,
which drives the real `_act_sparse` on a real agent with real observations and
real legal masks (mid-game, ~20–24 legal actions/row):

| K distinct policies | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 256 rows, BIG | 1.00x | **1.84x** | **3.65x** | 6.98x | 13.61x | 27.07x |
| 256 rows, 12 GB box | 1.00x | 1.00x | 1.12x | 2.05x | 4.26x | 8.32x |
| **1,024 rows, BIG** | 1.00x | 1.31x | 2.06x | 3.51x | 6.74x | — |

At 256 rows BIG's cost is ~0.87×K — **almost perfectly linear**, meaning each
group costs nearly the same regardless of how few rows it holds. That is the
signature of a **fixed-cost / launch-bound** regime, not a throughput-bound one.
Hence the inversion: a wider card makes the single big launch *cheaper*, so the
per-launch fixed cost becomes a *larger* fraction of it, and splitting into K
launches is relatively *worse*. Width helps the thing you are dividing, not the
divisions.

**Batch size is therefore not a detail in this measurement.** Going 256 → 1,024
rows nearly halves every ratio, because the groups get big enough to start
amortizing. Any future re-measure must use the batch size the run will actually
use; a 256-row curve does not describe a 1,024-env run.

`--compile_encoder` does **not** help the act path (K=1: 3.29 vs 3.36 ms at 256
rows; the whole compiled curve is within noise of eager). Consistent with T0,
where compile's entire win was in `learn`.

#### Why "largest K under 1.15x" cannot be applied here, and what replaced it

On BIG nothing above K=1 is under 1.15x, and the rule then returns a flag value
that must never be set: `--max_live_opponents=0` means **unbounded**
(`Matchmaker.__init__`'s default), the worst possible setting. The ratio is also the wrong
denominator — `_act_sparse` is a slice of `act`, which is a slice of the rollout,
which is a slice of the update, so a 2.06x act-path ratio is nowhere near a 2.06x
slowdown.

Converted to what it costs the update, at 1,024 envs (baseline 18.87 s/update,
128 steps, compiled encoder):

| K | `--max_live_opponents` | extra s/update | update | sps cost |
|---|---|---|---|---|
| 2 | 1 | 0.32s | 19.19s | −1.6% |
| 3 | 2 | 0.61s | 19.48s | −3.1% |
| 4 | 3 | 0.93s | 19.80s | −4.7% |
| 5 | **4 (current default)** | 1.31s | 20.18s | **−6.5%** |
| 6 | 5 | 1.61s | 20.48s | −7.9% |
| 8 | 7 | 2.55s | 21.42s | −11.9% |
| 12 | 11 | 3.86s | 22.73s | −17.0% |

**Keep `--max_live_opponents=4`.** It costs 6.5% of update time at 1,024 envs,
a reasonable price for league diversity. What changes is the *rationale*: it is
not free, and it is not "probably conservative on a big card". The cost is close
to **linear at ~1.6% of throughput per extra live opponent**, so raising it buys
diversity at a real, measurable price and lowering it saves very little. Do not
change it without deciding that trade deliberately — and note this table prices
only the throughput side; the league-quality side is unmeasured.

### FIXED: the roster prune collapsed to the two ends, deleting all mid-run history

`PolicyRoster.prune(keep_recent=4, keep_spaced=4)` is called after **every**
snapshot, so it is applied to its own output dozens of times. It selected the
"spaced" entries by **position in the surviving list**, which is not stable under
that iteration: the survivors are already collapsed toward the ends, so evenly
spacing by index re-selects the ends and squeezes the middle a little more each
round.

A real 1,622-update run finished holding `u100, u200, u1200, u1300, u1400, u1500,
u1600, u1622` — eight snapshots, a **1,000-update hole**, and no mid-run policy at
all. That silently defeats rating "mid and final" snapshots, which is the only way
to notice a run that peaked early and then regressed — precisely what the
`update_epochs=1` arm did.

Now spaced by **birth_update** across the older block, so the targets are
re-derived from the true age range on every call and survive repeated application.
Simulated over a 2,200-update run: `100, 700, 1600, 1800, 1900, 2000, 2100, 2200`
(largest gap 43% of the run) against the old `100, 200, 1200, …` (66%).

The existing test passed because it added 20 snapshots and pruned **once** — which
any spacing rule survives. `test_repeated_prune_keeps_a_genuine_mid_run_snapshot`
prunes after every snapshot, as the caller does, and asserts something survives in
the middle half of the range.

**Generalisation worth keeping:** a rule that is applied repeatedly to its own
output must be tested that way. Testing one application of an idempotent-looking
selection proves nothing about the fixed point it converges to.

### Operational: `runs/roster` is unloadable by the ladder

`runs/roster`'s 2026-08-13 checkpoints cannot be loaded by the current tree —
`actor.0.tail_mlp` is 1486 wide in the checkpoint against 2146 now,
`entity_fc` 256 against 128. This is a **size mismatch**, which
`load_state_dict` raises on *even with `strict=False`*, so
`roster_ladder._load_net_tolerant` cannot absorb it: that loader forgives
*missing* keys (aux heads), not resized ones. Every checkpoint and rating in
`runs/roster` is therefore dead weight, the same way the V2 observation change
orphaned the pre-V2 history (see `run_v2.sh`'s header).

Consequence for any ladder work: **generate a fresh roster from the current tree**
rather than reusing one from a previous encoder layout.
`run_ladder_sizing.sh` trains a throwaway one in ~1 minute.

### Ladder tournaments cost 1.39 s/game, and the default config is a 35-hour job

First measurement of ladder throughput on any box (`run_ladder_sizing.sh`):
**1.387 s per Eclipse game**, including net loading and the rating fit.

`roster_ladder` is a full round-robin — `p*(p-1)/2` pairs × 2 directions ×
`--ladder_games_per_dir` games — so cost grows **quadratically in snapshot count**.
Extrapolated for a two-arm tournament plus the three bot anchors:

| snapshots/arm | p | pairs | g/dir=32 | g/dir=64 | g/dir=128 |
|---|---|---|---|---|---|
| 2 | 9 | 36 | 0.9h | 1.8h | 3.5h |
| 3 | 11 | 55 | 1.4h | 2.7h | 5.4h |
| 4 | 13 | 78 | 1.9h | **3.8h** | 7.7h |
| 8 | 21 | 210 | 5.2h | 10.4h | 20.7h |
| 11 | 27 | 351 | 8.7h | 17.3h | **34.6h** |

**Two 5h arms at `--snapshot_every=100` produce ~9 and ~15 snapshots. Rating all of
them at the default `--ladder_games_per_dir=128` is a ~35-hour job** — longer than
the experiment it judges. Subsample the rosters (`tools/prune_roster.py` keeps
main plus evenly-spaced snapshots with the endpoints pinned) *before* rating.

Cut snapshots first, games/dir last: dropping games widens every rating CI, which
directly weakens any test phrased as "one arm's lower bound clears the other's
upper bound", whereas dropping near-adjacent snapshots costs almost no
information — they are nearly indistinguishable, which is why
`--ladder_min_sep` exists at all.


---

## Throughput findings, 2026-08-13 (12 GB box, merged as `5f5743be`)

All numbers from *paired* A/Bs (arms alternated in one session) — the box picked
up a load-average-45 background workload mid-session, so unpaired absolutes from
that window are worthless. See `docs/eclipse_observation_v2.md` for the
observation-writer half.

### The league had a silent throughput cliff

The most important finding, because nothing in the loss series, the ratings or
the diagnostics would ever have shown it. A `--league` smoke run had `act`
climbing **22.8 → 41.7 ms/step over 45 updates and still rising**, with `env` and
`learn` flat.

`PPO._act_sparse` groups the batch by policy id and runs
**one encoder forward per distinct policy**. Row count is unchanged; the launches
just get smaller and more numerous, and the cost is steeply superlinear:

| K distinct policies | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 256 rows, cost vs K=1 | 1.00x | 1.00x | **1.12x** | 2.05x | 4.26x | 8.32x |

`Matchmaker.sample_lineup` drew from the **entire roster**, so K grew for as long
as the run lasted. At `--snapshot_every=100` an 8h run reaches ~35 snapshots.

Fixed by bounding the live opponent set: `--max_live_opponents` (default 4, the
1.12x point; 0 restores the old behaviour) and `--live_opponent_refresh`
(default 2000 lineup samples). Every snapshot still enters play, clustered in
time rather than interleaved. `act` now plateaus (~21 ms flat from update 25 to
55). `league_test` asserts *both* halves — the cap, and that >20 of 30 snapshots
still enter play — because bounding without refreshing is a quality regression
instead of a speed one.

**Generalisation worth keeping:** any per-policy or per-group loop in the act
path is a throughput term that scales with league size. Check `np.unique(pids)`
before adding one.

### The env phase was not the engine

At 256 envs / 16 workers the env phase split `_run_step` 4.82 ms /
`_legal_indices` 0.32 ms / **buffer copies 7.32 ms**. So 59% of it was
`_collect`'s memcpy, and 65% of *that* was mmap/munmap of a fresh 38.5 MB array
every step (above glibc's mmap threshold), not the copy itself. Double-buffering
into two preallocated generations: env 13.5 → 7.9 ms, **+14% SPS**.

Two, not one: the PPO loop reads the *previous* step's observations after the
current step is collected (`last_obs_batch` → `_terminal_obs_for` inside
`post_step_np`), and never reaches further back. `async_vector_env_test` pins the
invariant in both directions.

### The timing instrument was measuring the wrong thing

`sps` and `learn_share` came from a **sum of phase durations**, which equals
elapsed time only while phases are strictly serialized. Any act/env overlap would
have made two slots cover the same real seconds, so the printed `sps` would have
gone *down* as the run got faster — i.e. the planned overlap work was
unmeasurable, on both machines. Per-update seconds now come from a wall-clock
bracket, with the phase sum reported beside it as an `overlap Nx` ratio that
reads 1.00x while serialized. **Treat pre-2026-08-13 SPS numbers in any doc as
phase-sum-derived.**


---

## Still open, none of it gating

- **`learn`'s remaining 35%.** `elementwise / copy` is the largest bucket now.
  The pointer head is compiled (`--compile_encoder` covers it as of 2026-08-14);
  `embedding_dense_backward` for `sector_embed` is ~7% and has no identified fix.
- **Peak learn memory is 1.54 MB/row, not the recorded 0.96**, and did not move
  when the 157 MB `(M, 4, 64)` intermediate was removed — so peak is set by
  something still unidentified. `torch.profiler(profile_memory=True)` via
  `tools/learn_profile.py` is wired up for whoever looks.
- **Env-count ceiling above 1,024.** Untested since the memory profile changed.
- **`--league` REDUCES act and learn** once opponents fill seats (fewer trainable
  rows shrink both the learn batch and the `_last_decision` loop). T3's cost curve
  measured `_act_sparse` in isolation at a fixed row count and therefore
  **overstates** league's true cost.

## Negative results — do not retry these blind

- **Pinning the H2D staging buffer.** ~~Isolated microbenchmark is compelling (38.5
  MB: 7.9 → 3.5 ms). Paired A/B in the real loop measured act **19.4 → 23.0
  ms/step, a net loss**.~~ **OVERTURNED 2026-08-14 — see "Pinning, done under the
  two conditions this entry asks for" below.** The original attempt staged obs into
  a pinned buffer *in addition to* the copy `_collect` already performs, so it paid
  a second 154 MB memcpy to save one transfer. Making the *existing* copy's
  destination pinned costs nothing extra (10.86 → 10.98 ms) and the transfer drops
  14.2 → 2.8 ms. The rest of this entry still holds: allocating pinned buffers
  *pre-fork* poisons every worker's inherited CUDA context — 7,900 → 3,000 SPS,
  with the *env* phase tripling, which is the tell that it is a fork problem and not
  a transfer one. Strictly post-fork, and prove it in a paired run.
- **Pooling `_last_decision`'s per-env observation copies.** Reusing preallocated
  destinations measured 12.40 ms against `.copy()`'s 11.88 ms — glibc already
  recycles chunks of that size. The only real fix remains the `(row, env)`
  reference design, which rewrites terminal attribution.
- **fp16 rollout buffer is settled and free.** `ppo_obs_dtype_test` measures the
  claim at the point of consumption rather than chasing divergent training runs:
  fp16 storage error **6.3e-13** against the bf16 autocast error **4.5e-3** that
  `--amp` already accepts, and combining them does not compound. Do not pin
  `--obs_buffer_dtype=float32`.


## Deliberately not doing

- **Shrinking the observation.** Full analysis in
  `docs/eclipse_observation_v2.md`. Costs a from-scratch retrain; not on this path.
- **fp16 on the wire.** Deprioritized as *redundant*, not unsafe — the H2D half is
  already captured by the pinned collect destination, and the copy half is hidden
  under the env phase. Measured safe (0 of 21,150 real obs decodes flip).
- **TF32 (`set_float32_matmul_precision('high')`).** ~0.4% of the update for a
  global numerics change, because autocast already owns the hot matmuls.

## Determinism, for anyone A/B-ing training runs

Comparing two training runs cannot resolve small numerical changes. At a fixed
seed, three master runs agreed **exactly** on `policy_loss`/`entropy` for updates
0–2 and then diverged chaotically from GPU reduction order alone; `aux_loss` is
never deterministic (spread 0.0022 at update 1). Judge a numerical change at the
point of consumption, or with a ladder over many seeds — never by diffing a loss
curve at update 20.


## Operational

- **A crashed run holds the GPU indefinitely.** Async workers block forever on
  their semaphores and nothing releases VRAM. A 91 GB zombie once made a clean
  config look like a fresh OOM. Check `nvidia-smi` before believing one. Note
  `--overlap_record` widens the window in which an exception can leave workers
  unawaited, though an exception already orphans the pool today.
- **Never gate a wait loop on `pgrep -f "<string in the loop's own cmdline>"`** —
  poll a file.
- **Real throughput is `final_steps / elapsed_seconds`**, never the logged SPS —
  except that with `--compile_encoder` the one-off ~30 s trace makes short runs
  look slow, so compare `STEADY` per-update seconds across rungs instead.
- **`observation_tensor_into` requires a float32, C-contiguous, writeable buffer**
  and now raises otherwise. It used to silently convert-and-discard, returning an
  all-zero observation; that invalidated one of this round's own measurements.
- **`observation_tensor()` costs ~49x `observation_tensor_into()`** (not the ~24x
  recorded). Any `rl_environment` built without `observations_as_numpy=True` pays it.
- **`--nn_norm` is a no-op** under `--encoder=spatial`; `--separate_critic`
  doubles the encoder forward *and* backward.
- **Re-check the compile targets** if the encoder or head entry points are
  refactored: `_encode_context` *and* `_pairs` both fall back to eager on any
  exception with only a warning. This trap has fired twice already.
- **`runs/roster` is unloadable** by the current tree (size mismatch, which
  `strict=False` does not forgive). Generate a fresh roster; do not try to rate it.

---

## Queued design work

Nothing below is in flight. Items 6/6b are explicitly out of scope this cycle;
Item 8 is queued behind a decision that has not been made.

### 7. Pointer/attention head — SHIPPED, and a null result on the ladder

> **Status: landed behind `--spatial_pointer` (default off), trained, and rated — it did not
> move the ladder. Keep this section for the traps, not as queued work.** What was measured
> when it landed:
>
> | check | result |
> |---|---|
> | `ActionFactorization.cell_id` populated | 7,650 / 11,117 actions, range [-1, 224] |
> | Cell-routing acceptance test | boosting cell 42 moves colony-actions targeting **cell 42 by 0.0448** vs **cell 100 by 0.0021** — **ratio 21.25** (was 0.60) |
> | `--spatial_pointer=False` unchanged | logits / value / features **bit-identical** to `git HEAD` on `runs/roster/main.pt` (`max\|diff\|=0.000e+00`) |
> | Test suites | `action_factors_test`, `ppo_pytorch_test`, `ppo_win_test`, `ppo_sparse_act_test`, `ppo_league_test`, `ppo_selfplay_pytorch_test`, `league_test` — all OK |
>
> Requires `--factored_actions` and `--encoder=spatial` (it raises otherwise — a flat trunk has
> no per-cell features to point at). New pieces: `head_logits`/`shared_and_cells` (`ppo.py`),
> `SpatialEclipseEncoder.forward_with_cells`, `SpatialFactoredActorHead`, and
> `EclipsePPOAgent.dense_logits`. **Trap, documented on the class:** `SpatialFactoredActorHead`
> inherits `rows_for`/`full_weight` from `FactoredActorHead` and those return the BASE term only
> — anything checking `hasattr(head, "rows_for")` will think it has the complete weight. The
> dense eval/ladder path must go through `dense_logits`, or the ladder would score a different
> (base-only) policy than the one being trained.
>
> `roster_ladder.py` threads `spatial_pointer` out of `arch.json` into `make_agent_fn`
> (verified: a pointer roster rebuilds as `SpatialFactoredActorHead` with `logits_for`). Without
> that thread the ladder rebuilds a base-only head and silently rates a *board-blind* policy
> instead of the trained one — if you add another arch flag, thread it here too.
>
> This has since been trained and judged — see the verdict immediately below.


**Ladder verdict: the pointer head was a null result.** Two separate tournaments
put it inside the CI of the same config without it, and their order flipped
between ladders — which is what "no real difference" looks like. The routing
*mechanism* is real and fixed (cell-boost ratio 0.60 → 21.25); it simply was not
the binding constraint at this scale. OpenAI's hide-and-seek work corroborates
this independently (encoder choice was ~9% of episodes for them, not pass/fail).
Keep the trap notes above — they are live for anyone touching the dense/eval path
— and do not build more spatial routing on the premise that this one paid off.

### 8. The ~22% blind actions — engine "which cell is unit u on" lookup (QUEUED — 100% gate before Item 6)

**Known gap, must be fixed.** The pointer head (Item 7) only routes actions whose string carries a
board coordinate. The following families carry **no hex coordinate** in their action strings, so
their `cell_id = -1` and the agent stays blind to them:

- `upgrade` (`UPGRADE_<ship>_SLOT<slot>_<part>` — ship/slot/part)
- `move_unit` (`MOVE_UNIT_<unit>_<dir>`) and `move_warp` (`MOVE_UNIT_<unit>_WARP`)

These are ~22% of actions. **Without a fix the agent is still partially blind even after Item 7**
— it must act on "upgrade the ship that usually sits on cell 42 / move unit 7" without being told
which hex that ship or unit occupies this game.

> **Correction note (precision, not a disagreement — the gap above is real and stays queued).**
> What is missing is the *action→cell routing*, not the underlying state. Unit positions **are**
> in the observation: the galaxy block carries per-cell my/enemy/NPC ship counts and damage
> (`observation.h`, block C), so the trunk can see where ships are. The defect is that
> `MOVE_UNIT_7_E` names a unit and a direction rather than a hex, so the factorization cannot
> assign it a `cell_id`, and the pointer head therefore has no per-cell term to add for it. The
> practical consequence is the same as stated — those actions get no board-conditioned logit —
> but the fix is a mapping from action id to the cell the unit currently occupies, not new
> observation content. That also means it need not be an action-string change: a per-decision
> `(action -> cell)` array supplied alongside the observation would do, which is likely cheaper
> than threading coordinates through the engine's action naming.

**The fix needs an engine-side lookup that does not exist today**: "which cell is unit/ship `u`
on" (`upgrade`/`move_unit` reference unit ids, not coordinates). Thread the resolved cell into the
action string (or into a parallel per-action array) so the factorization can set `cell_id`
like the coordinate-bearing families already do.

**Priority/when:** fix after the main pointer head (Item 7) proves out — Item 7 should land and be
validated on the ladder first, and only then take on the engine-side lookup. But it is **required
to reach 100% before Item 6 UI integration is picked up**: UI play-time integration must not
start until both Items 7 *and* 8 are complete, so the policy the UI exposes is not partly blind.
Item 6 remains out of scope this cycle regardless.

#### Engine ground truth (investigated 2026-08-07, with citations)

The 22% splits into two genuinely different problems, and only one of them is a gap:

| family | count | % of 11,117 | verdict |
|---|---|---|---|
| `upgrade` | 1600 | 14.4% | **not spatial — exclude by design** |
| `move_unit` | 768 | 6.9% | spatial, needs dynamic lookup |
| `move_warp` (unit-select) | 128 | 1.2% | spatial, source cell only |
| `MOVE_WARP_TO_<q>_<r>` | 225 | 2.0% | already coordinate-bearing → already fixed by Item 7 (`warp_dest`) |

- **`upgrade` targets a blueprint, not a hex.** `execute_upgrade` (`systems/actions/upgrade.cpp:178-188`)
  edits `player.blueprints[static_cast<size_t>(ship_type)]` — one template per ship *class*, shared
  by every ship of that class anywhere on the board (`state.h:116`). `can_upgrade`
  (`upgrade.cpp:36-150`) never reads `unit_registry` or `galaxy`. So there is no cell to point at;
  this is not blindness, it is a non-spatial decision. **The real spatial gap is 896 actions (8.1%),
  not 2,496 (22.4%).**
- **A static action→cell table is impossible in principle** for moves, on two independent axes:
  a unit's cell changes as it moves (`move.cpp:528`), *and* the index itself is unstable —
  `FlushDestroyedShips` (`systems/combat.cpp:334-343`) rebuilds `unit_registry` by compaction after
  every battle, so `unit_idx=5` refers to a different physical ship afterwards.
- **The cell is O(1) available**: `state.galaxy.FindSectorCoord(unit.sector_id)` is a cached array
  read (`galaxy.h:52,67-87`) already used by the observation writer (`observation.cpp:421`).
  Destination is also O(1) via `GetAdjacency` (`warped_universe/adjacency.h:12-32`).
- **`legal_move_steps` already computes the source cell and discards it**: `unit_can_contribute_legal_step`
  sets `source_cell` (`move.cpp:297-311`) purely for validation; `MoveStepOption` carries only
  `{unit_idx, direction}` (`move.h:53-57`), so it never escapes `MoveLegalActions()`.
- The existing `kCellMoveActiveUnit` observation flag does **not** help: it only fires for
  continuation steps of a move already begun (`observation.cpp:537-542`), whereas
  `active_unit_idx == 255` at exactly the first unit-choice decision.

#### Plan — append a `UNIT_CELLS` block, do NOT touch action strings

**The load-bearing constraint: do not change the observation layout in place, and do not thread
coordinates into action strings.** Either invalidates every existing checkpoint, including the
`roster:snap_u573` plateau baseline that the whole go/no-go test is measured against.

Appending a block at the **end** of the tensor is provably safe for the spatial encoder — verified
empirically: a net built for 24,714 floats fed a 24,842-float observation returns **bit-identical**
logits (`max|diff| = 0.0`), and parameter shapes are identical for either declared `input_shape`,
because `SpatialEclipseEncoder` never uses `in_features` (it slices by fixed named offsets and
ignores a trailing block). All three current rosters are `encoder=spatial`, so all survive.
**This does not hold for `encoder=flat`**, whose `Linear(in_features, width)` would break.

1. **Engine** (`observation.h`/`observation.cpp`): append `UNIT_CELLS`, 128 floats, one per
   `unit_registry` slot, holding that unit's linear cell id (normalized, with a clear sentinel for
   empty/invalid slots) via the existing `FindSectorCoord` call. 128 cached array reads per
   observation write — cheap next to the existing 0.141 ms/call.
2. **`obs_layout.py`**: add `UNIT_CELLS_START`/`SIZE` after `ACTION_STATES`, bump `TOTAL`. Every
   existing offset is unchanged by construction — assert that in the test.
3. **`action_factors.py`**: for `move_unit`/`move_warp`, record the `unit_idx` (and direction) in a
   parallel array, the same way `cell_id` is recorded — but resolved per-row at forward time, not
   statically.
4. **`SpatialFactoredActorHead`**: generalize the cell index from `cell_id[col]` (static) to a
   per-row resolution `cell_of(row, col)`, reading `UNIT_CELLS` out of the observation for move
   families and falling back to the static `cell_id` for the coordinate-bearing ones. The
   `(row, col)` gather in `logits_for` already has the right shape for this — this is the change
   Item 7's design anticipated.
5. **Point at the SOURCE cell first** (ponytail). Destination pointing (`GetAdjacency`) is likely
   more informative for "should I move *into* that hex", but costs 768 adjacency calls per
   observation instead of 128 lookups, and needs the warped-universe topology. Ship source-cell
   routing, measure, and only add destination if the ladder says moves are still the weak spot.

**Acceptance check** (mirror Item 7's): with two units on different hexes, boosting one unit's hex
must move that unit's `MOVE_UNIT_<i>_*` logits substantially more than the other's — and
`--spatial_pointer=False` must stay bit-identical.

**Sequencing:** do not start until the Item 7 ladder verdict is in. If the pointer head does not
move the rating, a second spatial-routing change is not the right next bet, and this work would be
building on an unvalidated premise.


### 6. Play-time search + the UI opponent — OUT OF SCOPE THIS CYCLE (design notes kept, do not pick up)

**The user chose training-only for this cycle.** Do not implement any of the below without
checking first — it is preserved as design research, not as a queued task.

**Deferred until training quality is good** — specifically: the agent developing strategies and
adapting to what opponents do, and a 4-trained-agents game producing *entertaining* games. Revisit
Item 6 once the board-blindness fix (above) and Items 2/3/4 deliver that level of play. Budget: a
few seconds per AI move.

Facts established, correcting Sprint C's premise: `explore_draw`, `combat_roll` and
`reputation_draw` **are** real chance nodes with fully explicit `ChanceOutcomes()`
(`eclipse.cc:1602-1653`) — the action id merely equals the outcome id. That is *better* for
search: MCTS takes a proper expectation, and `mcts.py` already samples `chance_outcomes()`. Only
`initial_setup` hides randomness behind a single fake outcome and consumes the shared `Game` RNG,
which play-time search never re-resolves.

- **Use vanilla MCTS, not IS-MCTS.** `EclipseState` does not override `ResampleFromInfostate`, so
  both IS-MCTS bots throw on a `kImperfectInformation` game. Fine here: no per-player private
  state, and bag *composition* is deducible from public play while the next draw is exactly what
  the explicit chance distributions model. Both `mcts.py` and the C++ `MCTSBot` are N-player and
  general-sum by construction.
- **Do not port anything from `alpha_zero/`.** It was generalized to N-player general-sum with a
  `num_players` value head, but it is JAX/Flax + orbax (not the PyTorch trained net) and was
  **never run on the real Eclipse game** — its tests use `tic_tac_toe` and `colored_trails`, and
  `examples/alpha_zero_eclipse.py` self-describes as a "starting point". The C++/LibTorch
  AlphaZero is hard-asserted 2-player-only (`alpha_zero_torch/alpha_zero.cc:507-508`).
  Instead: write a PyTorch-backed `Evaluator` (prior from the actor, value from the rank critic)
  and hand it to the generic Python `MCTSBot`.
- **UI wiring is easy.** `lobby.game_blob` *is* `EclipseState::Serialize()`'s JSON
  (`eclipse.cc:1657-1664`) and round-trips losslessly through `pyspiel.Game.deserialize_state()`
  to an identical observation tensor. `eclipse_ui_native.so` and `pyspiel.so` build from the same
  sources in the same CMake configuration (`open_spiel/CMakeLists.txt:298-299`).
  1. Fix `apps/eclipse_ui/run.sh:12` — `PYTHONPATH` points at `$REPO_ROOT/build/python`, which
     does not exist; the module is at `$REPO_ROOT/build/open_spiel/python`.
  2. Import `pyspiel` in `api/main.py` (it imports only `eclipse_ui_native` today) and deserialize
     the blob. `api/open_spiel_bridge.py` already has the helpers but is **dead code**.
  3. Replace `random.choice(legal)` in `_autoplay_ai` (`api/main.py`) with policy + search.
  4. Rebuild **both** targets after any `eclipse.cc` change — a stale `eclipse_ui_native.so`
     would silently diverge.
- Separate small issue: `broadcast_lobby` (`api/main.py`) sends the full state to every
  websocket client — an information leak for human-vs-human play in the UI.


### 6b. Actor-only MCTS plan (design agreed 2026-08-10) — build later, this is the notes to pick up

> **Decision made with the user (2026-08-10):** use **vanilla MCTS with an actor-ONLY evaluator**
> (no critic as leaf evaluator) for live UI play. Rejected using the shared-trunk `win` critic as
> the leaf value. Rationale recorded below.
>
> **The incumbent config was never a proven winner, it is just what was always run.** No ladder
> ever varied `--value_mode` — every run used the default `"win"` (see the `--value_mode` flag). And
> `separate_critic=False` is the baseline default, not a tested best: when `--separate_critic` was
> tested in isolation it scored BELOW baseline (`w3_sep:main` 0.9609 vs `baseline:main` 1.0288,
> wave3 ladder). The actual winner was `--ent_coef=0.05` alone (`w3_ent:main` 1.1071) — a
> training-time exploration knob, irrelevant at MCTS inference. Wave-3 lesson (ground rule 2, "do not
> infer strength from mean_episode_return / VP proxies") reinforces that value/util signals are
> decoupled from real strength, so handing MCTS the shared-trunk `win` critic as a ranker is an
> untested-calibration risk. Hence: **actor-only for now.**

**Building blocks (all already in repo — no MCTS from scratch):**
- `open_spiel/python/algorithms/async_mcts.py` — `MCTSBot` (line 362) with built-in
  `ThreadPoolExecutor` parallel leaf eval, `virtual_loss`, `batch_size`, `timeout`. Expects an
  `Evaluator` with **`prior_and_value(state) -> (prior_list, values_ndarray)`** (lines 181-185).
  This gives thread-parallel rollouts + virtual-loss for free, and solves the "can we parallelize
  the UCB-seq rollouts" question: keep selection sequential, batch the per-leaf evaluations.
- `EclipsePPOAgent.dense_logits(x)` — the (B, num_actions) eval/argmax
  entry point; routes through `SpatialFactoredActorHead` for spatial/factored configs (else plain
  `self.actor`). Relation to `get_action_and_value` (line 910): that does
  `CategoricalMasked(...).sample()`; single-state rollout just needs `dense_logits` + mask.
- Raw-state obs: `state.observation_tensor()` (template: `alpha_zero/evaluator.py`) →
  `np.float32` → `[None]` → `.to(device)` → `dense_logits` (pattern: `_argmax_over_legal` in `ppo_eclipse.py`).
- Chance nodes: real explicit `chance_outcomes()` in Eclipse (`eclipse.cc:1602-1653`); evaluator
  returns `state.chance_outcomes()` as prior for chance nodes.
- Existing UI targets (see "Item 6" above): `apps/eclipse_ui/run.sh` PYTHONPATH bug,
  `api/main.py`'s `random.choice(legal)`, `api/open_spiel_bridge.py` (dead-code helpers),
  pyspiel import in `api/main.py`.

**New code (minimal) — `ActorRolloutEvaluator(Evaluator)`** in `ppo_eclipse.py`:
- `prior_and_value(state)`:
  - chance node → `prior = state.chance_outcomes()`
  - else → `prior` = actor softmax over legal actions (this feeds UCT's `+c*prior*sqrt(...)`
    term, i.e. what makes search policy-guided); `value` = `_actor_rollout(state.clone())`.
- `_actor_rollout(state)` — full playout to terminal using the actor ONLY (no critic anywhere);
  at chance nodes sample `chance_outcomes()`; return `np.array(state.returns())`.
- Build `async_mcts.MCTSBot(game, uct_c, max_simulations, evaluator, batch_size, virtual_loss,
  timeout)`; feed `step(state)` into `_autoplay_ai` (`api/main.py`) using
  `pyspiel.deserialize_state` on the lobby blob.

**OPEN QUESTIONS to settle at build time:**
1. **Obs equivalence — RESOLVED, not a risk.** `dense_logits` was trained on `info_state`
   (`time_step.observations["info_state"]`), but with `ObservationType.OBSERVATION` (forced at all
   env construction site in `ppo_eclipse.py`) the dict key `"info_state"` actually holds
   `_state.observation_tensor(player_id)` (`rl_environment.Environment.get_time_step`) — the key name is a fixed
   misnomer in OpenSpiel. Eclipse `provides_observation_tensor=true`,
   `provides_information_state_tensor=false` (eclipse.cc:204-206), pure perfect-info. So training
   `info_state` and MCTS `state.observation_tensor()` are the SAME array. Phase 0: one-line assert
   `obs == info_state` to confirm, do not treat as a design risk.
2. **GIL/GPU contention in the thread pool.** Option A runs each rollout step's 64-wide MLP
   forward from worker threads → GIL-serialized for the model. Real win from `batch_size` threads
   is likely overlapping the C++ engine's `apply_action` (which releases the GIL). Start
   `batch_size` modest (4-8), measure single-thread vs threaded.
3. **Rollout depth / budget.** Eclipse runs to round 7-8 (hundreds of moves); full playout per
   leaf × `max_simulations` is the dominant cost. `async_mcts.MCTSBot` `timeout` (default 5.0s)
   bounds total search — matches the "few seconds/move" budget. Pick `max_simulations` + `timeout`
   + `uct_c`, confirm it holds at 1 game / 1 AI seat.
4. **Chance-node `value` in the `prior_and_value` contract.** async_mcts calls `prior_and_value`
   once per leaf; for a chance leaf we must still return a value. Confirm the async tree-policy
   deepens through chance nodes (as vanilla `mcts.py` does) or return weighted expectation.

**Phased execution:**
- **Phase 0 — correctness gate (before UI wiring):** standalone script: build
  `ActorRolloutEvaluator`, run `async_mcts.MCTSBot` on an all-AI headless game from
  `runs/roster/main.pt`; verify legal full games, `obs==info_state`, and beats random + beats own
  greedy argmax in a small match.
- **Phase 1 — perf:** measure sims/s single-threaded vs `batch_size ∈ {1,4,8}`; tune
  `max_simulations`/`timeout`/`uct_c` to the budget.
- **Phase 2 — UI wiring (todo Item 6):** fix run.sh PYTHONPATH, import pyspiel, deserialize blob,
  swap `_autoplay_ai`, rebuild both `.so` targets.
- **GATE — Item 8 (the ~22% blind-action engine bug, above) must be 100% first.** A blind
  action inside a rollout corrupts the search. Do not wire UI until this lands.


**Deferred MCTS note carried over from Observation V2:** if play-time search is
revisited after V2, reuse `async_mcts.MCTSBot` with prior-aware PUCT selection and
actor-only rollouts; do not use the unvalidated critic as a leaf value. Before any
raw-state rollout or UI wiring, determinize face-down discovery identities and bag
order from the public V2 ledger at each search root.

---

## Reference point: OpenAI hide-and-seek (Baker et al. 2019, arXiv 1909.07528)

Read because "they plateaued, kept training, and got better" was the motivating analogy.
It is true, and the scale is the point.

- **Plateau then phase transition is real**: phase 4 at ~110M episodes, then a **~270M-episode
  flat stretch** the authors believed was convergence ("we originally believed defending
  against ramp use would be the last stage"), then box-surfing at 380M episodes. Total for
  phase 4 alone: 132M episodes ≈ **31.7 billion frames**.
- **Batch size is a hard floor, not a dial** — the most transferable finding. To reach stage 4:
  32k batch = 167M episodes/98.8h; 64k = 132M/34h; 128k = 155M/19.8h; **16k and 8k NEVER
  CONVERGED**. Their batch was 64,000 chunks × 10 timesteps = **640,000 timesteps**. Ours is
  128 envs × 128 steps = **16,384** — ~39x smaller, and smaller than their failing configs.
  *This is the top untested lever here.*
- **Their autocurriculum is exactly our setup**: shared policy per team vs the *current*
  opponent. No league, no past-policy pool, no PBT, no shaped rewards. So we already have the
  mechanism; what we lack is scale.
- **Encoder architecture was a modest effect**: self-attention vs plain embed+pool was ~9%
  fewer episodes at ~10% more wall-clock. **This independently corroborates our null result on
  the pointer head** — routing/encoder choices were not pass/fail for them either. What *was*
  pass/fail: the centralized/omniscient value function (masked-V never passed stage 3) and
  environment randomization (removing random walls cut 6 stages → 2-4).
  - The omniscient-critic lever does **not** apply to Eclipse: there is no hidden per-player
    state (`InformationStateString == ObservationString`).
  - We already randomize races / NPC difficulty / warped universe per episode.
- **Their PPO hypers**: lr 3e-4, clip 0.2, **entropy 0.01 fixed with no annealing and no KL
  control**, γ **0.998**, λ 0.95, 4 optimization passes per chunk, separate policy/value nets.


## Deprioritized

- R-NaD / DeepNash, RND / curiosity, PBT, exploiters as the *main* driver — target a collapse the
  argmax eval contradicts.
- BC / DAgger warm-start from Greedy — Greedy is ~uniform-random for most decisions.
- MuZero / `mctx` / a JAX rewrite — the information fix and play-time search are cheaper and far
  more certain.
- Decision Transformer / offline sequence models — need a corpus that does not exist.
- SampleFactory / EnvPool migration — throughput was never the constraint.
