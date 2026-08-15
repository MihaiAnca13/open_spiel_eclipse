# What to test on BIG before committing to a long run

**Status 2026-08-14, after the pre-flight ran.** T0 and T3 are **done**, and the
throughput work they exposed is done: **18.87 → 12.44 s/update at 1,024 envs
(1.52x, sps 6,945 → 10,540; 10,704 real sps over a clean 30-min run)**. Detail in
`docs/eclipse_rl_todo.md` under "BIG re-baseline, 2026-08-14"; this file is only the
sequence and what is still open.

**T1 has RUN. It did not settle the config, and it surfaced a bigger problem: at
both epoch counts the run stops improving early, and at `update_epochs=1` it then
regresses.** Full numbers in `docs/eclipse_rl_todo.md` ("T1 results, 2026-08-14/15").
Headline:

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

The point of this sequence is that a long run is expensive and its failure modes
are quiet. The two that already bit this project — a terminal-attribution bug that
cost 408M steps, and a league throughput cliff invisible in every metric — were
both found by *running the thing*, not by reasoning about it. That held again: the
biggest win this round (the pointer head's gradient scatter, 43.6% of `learn`) was
invisible until something profiled it.

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

---

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

## T1. `update_epochs` 1 vs 4 — RUN, inconclusive

The largest open **quality** question and the only item that can make the long run
worse rather than merely slower. Needs no code.

`learn` runs `update_epochs × num_minibatches` forward+backward passes. Measured in
a smoke run at the production config: **ue=1 gives ~1.43x throughput** (10,875 vs
7,606 envstep/s). But samples/hour only converts to strength if one epoch learns as
much per sample.

Known and expected, and NOT disqualifying: `approx_kl` and `clipfrac` **will** fall
(a quarter as many gradient steps per batch is less policy movement, not more
stability), and the LR almost certainly wants to go up. `explained_variance` may
fall because the value head also gets a quarter of the updates — consider a
separate epoch count for the critic before concluding `update_epochs=1` loses.

**Note `learn` is now only ~35% of the update, down from 50%.** So ue=1's
throughput advantage is *smaller* than it was when this experiment was designed,
and the case for ue=4 is correspondingly stronger. Re-derive the ratio from the
arms rather than quoting 1.55x.

**Design.** Two arms, equal **wall-clock** (not equal steps — the whole point is
that one arm gets more samples per hour), both `--league`, seeds fixed. 4–6h per
arm. Then ONE ladder tournament rating both arms' final and mid snapshots.

    ./run_t1_update_epochs.sh 18000 4 16      # both arms, sequential
    ./run_t1_ladder.sh                         # one tournament, then the verdict

**Pass:** `update_epochs=1`'s rating lower bound clears `update_epochs=4`'s upper
bound. Anything less and keep 4. `tools/t1_verdict.py` applies this mechanically.

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

## What happened to T2 (act/env overlap)

**Superseded, and the env-group split is no longer worth building.** The overlap
was achieved without it: `AsyncVectorEnv.start_step`/`await_step` split the round
trip, and `PPO.flush_selfplay_record` defers the per-env bookkeeping (~29 ms/step at
1,024 envs), so the loop releases the workers and does that work while they run
(`--overlap_record`, on by default). Paired A/B: **update 13.42 → 12.44 s, sps
9,765 → 10,540, +7.9%**.

That is safe by an invariant the 12 GB work already established: workers write only
shared memory, `_collect` copies into one of two alternating *generation* buffers,
and a held `_StepArrays` points at a generation. Nothing in the deferred block
touches shm. `ppo_env_groups_test.py` pins it bitwise.

**The split is not worth building because the ceiling here is CPU contention, not
the act:env ratio.** Only ~15.5 ms of the 27.3 ms env step got hidden: the
bookkeeping is memcpy-heavy main-thread work competing with 16 env workers for 12
physical cores that the vLLM workers also use. An estimate treating the main thread
and the workers as independent predicted ~30% and was 4x optimistic. The group split
would need main-thread CPU for act-side work while workers run — the same wall — and
`T2 sizing` now reads 12% of wall clock, down from 26%.

If it is ever revisited, `ppo_env_groups_test.py` is the gate, and the old
change-surface analysis (11 row writes; `legal_*_packed` are list slots needing
fragment-append plus a global-index remap; `last_obs_batch` read by *global* env
index in `_terminal_obs_for`; `post_step_np`'s counters must move to the caller;
groups must be unions of whole worker shards) was verified accurate and is preserved
in git history at `6f7d9219`.

**Also dropped: the `(row, env)` reference design for `_last_decision`'s obs
copies.** Those copies are now inside the overlapped window, so removing them only
helps to the extent the window is not already CPU-saturated — and it is. It would
have rewritten terminal attribution, which is where the 408M-step bug lived. Bad
trade.

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

## Deliberately not doing

- **Shrinking the observation.** Full analysis in
  `docs/eclipse_observation_v2.md`. Costs a from-scratch retrain; not on this path.
- **fp16 on the wire.** Deprioritized as *redundant*, not unsafe — the H2D half is
  already captured by the pinned collect destination, and the copy half is hidden
  under the env phase. Measured safe (0 of 21,150 real obs decodes flip).
- **TF32 (`set_float32_matmul_precision('high')`).** ~0.4% of the update for a
  global numerics change, because autocast already owns the hot matmuls.

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
