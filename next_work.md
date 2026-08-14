# What to test on BIG before committing to a long run

**Status 2026-08-14.** The 12 GB work is merged (`5f5743be`). Its findings have
moved out of this file: observation/tensor facts to
`docs/eclipse_observation_v2.md`, throughput findings and negative results to
`docs/eclipse_rl_todo.md` ("Throughput findings, 2026-08-13"). Read those before
this. This file is now only the pre-flight sequence for **BIG** (RTX PRO 6000,
98 GB, 12 physical cores shared with two resident vLLM workers).

The point of this sequence is that a long run is expensive and its failure modes
are quiet. The two that already bit this project — a terminal-attribution bug
that cost 408M steps, and a league throughput cliff invisible in every metric —
were both found by *running the thing*, not by reasoning about it.

## Ground rules

1. **Every pre-2026-08-13 SPS number in any doc is phase-sum-derived and wrong
   under overlap.** Do not compare against them. Re-baseline first (T0).
2. **Judge strength only via ladder `rating` / `rating_ci`.** Never `elo`,
   `vp_all`, `mean_episode_return`, or vs-Greedy. Ratings are not comparable
   across ladder runs.
3. **Never A/B a training change by diffing a loss curve.** It is bit-identical
   for two updates and chaotic after; see the determinism note in
   `eclipse_rl_todo.md`.
4. Run arms **sequentially**. Two arms thrashed the GPU to ~45 SPS; three OOM'd.
5. BIG has 12 shared cores against the 12 GB box's 32, so its env phase is
   *relatively* more expensive. Expect the `_collect` fix to be worth **more**
   there, and the act/env overlap to be worth more too.

---

## T0. Re-baseline the throughput ladder — 1h, gates everything

Nothing downstream is interpretable until BIG has honest numbers on the merged
tree. Run the committed config with `--timing --timing_every=10` and record, per
rung, the **wall-clock** `rollout` / `learn` / `update` / `sps` and the new
`overlap Nx` field.

Rungs: committed config → device buffer → `--compile_encoder` → 1,024 envs.

Pass: `overlap` reads **1.00x** on every rung (nothing overlaps yet — if it does
not read 1.00x, the instrument is wrong and everything after this is noise), and
the act/env split is recorded. **The act:env ratio is the single number that
sizes T2**; on the 12 GB box it is now ~13:6.4.

Also confirm here: the 12 GB wins carry. Expect `observation_tensor_into` ≈ 5 µs
and the env phase materially below its pre-merge share.

## T1. `update_epochs` 1 vs 4 — the real experiment, and the expensive one

This is the largest open **quality** question and the only item that can make the
long run worse rather than merely slower. It needs no code.

BIG measured 1.55x throughput for `update_epochs=1` (5,735 → 8,863 at 1,024
envs) because `learn` runs `update_epochs × num_minibatches` forward+backward
passes. On the 12 GB box at the production config, `learn` is **65% of the
update** at `update_epochs=4` — so this is by far the biggest throughput lever
left, bigger than T2.

But samples/hour only converts to strength if one epoch learns as much per
sample. Known and expected: `approx_kl` and `clipfrac` **will** fall (a quarter
as many gradient steps per batch is less policy movement, not more stability),
and the LR almost certainly wants to go up. `explained_variance` may fall because
the value head also gets a quarter of the updates — consider a separate epoch
count for the critic before concluding `update_epochs=1` loses.

**Design.** Two arms, equal **wall-clock** (not equal steps — the whole point is
that one arm gets more samples per hour), both with `--league`, seeds fixed.
Recommended 4–6h per arm. Then one ladder tournament rating both arms' final and
mid snapshots together.

Pass: `update_epochs=1`'s rating lower bound clears `update_epochs=4`'s upper
bound. Anything less and keep 4 — it is what produced `long8h`, the strongest
model measured.

If it fails, **re-run once with a raised LR** before discarding it. A null result
at an LR tuned for 4x the gradient steps is not a result.

## T2. Act/env overlap — build only after T0 and T1

Worth `min(act, env)` per rollout step. On the 12 GB box that is ~22% of the
rollout but only ~7% of wall clock at `update_epochs=4`, because learn dominates.
**Its value is contingent on T1:** if `update_epochs=1` wins, learn collapses to
~16% of the update, rollout becomes the majority, and this item roughly triples
in value. If `update_epochs=4` stays, it is a minor item and arguably not worth
its risk. Sequence it after T1 deliberately.

**Do not extend it to overlapping envs with `learn()`.** That needs actors on
stale parameters — what OpenAI Five used 512 dedicated forward-pass GPUs to do
without contention. Out of scope on one card.

### V5 is the gate. Build it before the change, not after.

Fixed seed, two env groups vs one, same number of steps. Capture a golden
reference from today's single-group code and require **bitwise** equality after:
`obs`, `actions`, `logprobs`, `values`, `rewards`, `dones`, `trainable`,
`players`, `players_cpu`, `trainable_cpu`, `legal_rows_packed`,
`legal_cols_packed`, `total_steps_done`, `cur_batch_idx`, and `_extra_samples`
(count **and** content). Include an episode boundary inside the batch, and a
`--league` variant.

Reuse, do not rewrite: `ppo_selfplay_pytorch_test.py:215-275` already drives
`post_step_np` row by row with `cur_batch_idx` set manually — which is exactly
the "row advancement moved out" behaviour the split introduces — with a terminal
inside the batch. `async_vector_env_test.py:63-117` is the existing
async-vs-sequential equivalence test and becomes the two-group test almost
directly.

### The change surface is about twice what the old write-up claimed

- **Eleven row writes in `step_np`** (`ppo.py:933-954`), not five. Two of them —
  `legal_rows_packed` / `legal_cols_packed` — are plain Python **list slots**, so
  a second write *replaces* the first; they need fragment-append plus a
  global-index remap, not subsetting.
- **`last_obs_batch` / `last_seats` (`:904-905`) are read by *global* env index**
  in `_terminal_obs_for` (`:1198-1201`). Under a split, group B raises
  `IndexError` and group A returns a *plausible wrong row*, which flows into
  `_backfill_aux` as wrong-but-valid aux targets. Highest-risk silent path in the
  item. Do not fix it by making them full-width persistent buffers — that adds a
  38.5 MB copy per step. Store `(lo, obs, seats)` per group and index `env_idx - lo`.
- **`post_step_np`'s counters (`:1301-1302`) must move out** to the caller loop;
  called twice per logical step they double-count `total_steps_done` and advance
  `cur_batch_idx` twice, leaving half the buffer unfilled.
- `legal_rows` come out of the env layer as **global** indices; `_act_sparse`
  (`:1604-1606`) wants group-local while `_pack_legal_batch` (`:1427`) wants
  global. Today `batch == num_envs` so both readings coincide. Getting it wrong
  is silent.
- **Groups must be unions of whole worker shards.** A worker's single `publish`
  writes its whole row range, so a boundary inside one lets group A's step
  clobber rows group B has not collected. Assert it.

Safe as-is: the column-slice readers `_attribute_terminal:1053-1054`,
`_backfill_rank_labels:1146-1148`, `_backfill_aux:1174-1176`; `_compute_returns`;
the whole `_ObsRows` layer.

## T3. Re-tune `--max_live_opponents` for BIG — 30 min

The default of 4 was picked off the 12 GB cost curve (1.12x at K=4, 2.05x at
K=8). BIG's card is far wider, so the knee is probably further right and a larger
live set is likely free — which is strictly better league play.

Re-measure the curve there (256 rows through the production encoder, K = 1…32),
then set the flag to the largest K still under ~1.15x. Do **not** set 0.

## T4. Where the learn activations actually are — before any obs rewrite

`learn` is 65% of the update at `update_epochs=4`, and "memory-bound at a flat
0.96 MB/row" still has no identified cause. Nothing in the repo profiles memory.

Profile the merged encoder at a realistic minibatch with
`torch.profiler.profile(profile_memory=True)`. Record the share of: the
`(B, 64, 225)` conv/GroupNorm/act chain; **`unit_attn`** (8 heads × 128 × 128 =
131,072 elem/row *if* the attention probs materialize rather than going through
SDPA — a candidate nobody has looked at); the tail MLP; and the pointer head's
`_pairs` intermediates.

Note the existing 9.08 GB figure came from a scratch script that profiled the
**encoder only** — no pointer head, no optimizer state, no rollout buffer — so
the real peak is higher than recorded.

## T5. Env-count / minibatch sweep at the new memory profile — 1h

Re-derive the efficiency curve above 8,192 rows, which the 12 GB box cannot reach
(OOM at 16,384). Previous BIG measurement: 47,800 rows/s at 8,192 vs 43,100 at
16,384 and 41,280 at 32,768. Confirm it still holds, and find the env ceiling now
that host RAM is no longer the binding constraint (the rollout buffer is
device-resident fp16, so the old "17.5 GB host peak" note is stale).

---

## Only after T0–T3 pass

Then commit to the long run. Recommended starting point, to be overwritten by
whatever T1 and T3 decide:

```
--num_envs=1024 --num_steps=128 --num_workers=20 --num_minibatches=16 \
--update_epochs=<T1> --max_live_opponents=<T3> \
--amp --compile_encoder --lr_schedule=fixed --league \
--snapshot_every=100 --timing --timing_every=50
```

Keep the minibatch near 8,192 rows (most efficient measured on BIG). Raise
`num_envs` and `num_minibatches` together.

## Deliberately not doing

- **Shrinking the observation (the old items 3b/3c).** The env-step half of its
  payoff was bought for free by the writer fix, and its learn-activation half was
  never real. What remains is buffer/H2D/learn-input bytes — worth ~2x the env
  ceiling on a 12 GB card, worth much less on a 98 GB one. Full analysis and the
  two corrections (you cannot convolve a list; the slot block buys no learn
  memory) are in `docs/eclipse_observation_v2.md`. It costs a from-scratch
  retrain, so it is a BIG-card project and it is not on this critical path.
- **Eliminating the per-step `_last_decision` copies.** Pooling is a measured
  wash; the only real fix rewrites terminal attribution. Revisit after T2, which
  builds most of the test that would make it safe.
- **Overlapping envs with `learn()`** — see T2.

## Operational

- **A crashed run holds the GPU indefinitely.** Async workers block forever on
  their semaphores and nothing releases VRAM. A 91 GB zombie once made a clean
  config look like a fresh OOM. Check `nvidia-smi` before believing one.
- **Never gate a wait loop on `pgrep -f "<string in the loop's own cmdline>"`** —
  poll a file.
- **Real throughput is `final_steps / elapsed_seconds`**, never the logged SPS.
- **`observation_tensor()` costs ~24x `observation_tensor_into()`.** Any
  `rl_environment` built without `observations_as_numpy=True` pays it.
- **`--nn_norm` is a no-op** under `--encoder=spatial`; `--separate_critic`
  doubles the encoder forward *and* backward.
- **Re-check the compile target** if the encoder entry points are refactored:
  `_encode_context` falls back to eager on any exception with only a warning.
