# Stage 1 — Critic proof: State + big-machine commands (2026-08-17)

Repo: `/home/mihai/personal/open_spiel_eclipse`, branch `master` @ `360cb1f9` (clean tree).

## What is DONE and committed/pushed

| Commit | What |
|--------|------|
| `38671e25` | `feat(eclipse): frozen-policy diagnostic dataset schema` — frozen_dataset.py (+test): obs/seat/round/phase/rank soft target, terminal rank utility, 9 viewer-relative VP comps (4 seats), discounted return; whole-episode split; round-trip. |
| `89485ec8` | `feat(ppo): cell-attention critic readout (--critic_readout)` — CellAttentionCritic (1 fused-state + 1-query cross-attn over 225 h_cells, pre-LayerNorm, no C_PRESENT mask), 3 heads (unbounded scalar, 4 rank logits, 9 VP comps with frozen stats). Actor bitwise unchanged. `rank` default back-compat. ppo.py untouched. |
| `4f04b923` | `feat(eclipse): pre-V2 frozen-policy episode collector` — vendors the PRE-V2 encoder (git 92672cb8) so the frozen checkpoints actually load (missing actor==[], fuse [64,256]); feeds actor obs[:24714], records full 37596; fixes stale id() legal-mask cache + numpy aliasing. |
| `a03ea7ad` | `fix(eclipse): populate manifest code_revision/collection_ts on write` (setdefault bug). |
| `360cb1f9` | `feat(eclipse): gradient-norm scan -> picks --aux_coef` — tools/grad_norms.py, TDD additivity invariant. |

## CRITICAL finding (why the collector is pre-V2)

Every checkpoint in the repo (long_v2, long8h, _judge/baseline, roster, w3_*) was trained BEFORE
the V2 obs extension (commit `365a059a`, Aug 10). The current `SpatialEclipseEncoder` (fuse
`Linear(5*width)`=320, reads V2 regions, needs 37596 obs) CANNOT load them (fuse mismatch
`[64,256]` vs `[64,320]`). `tools/collect_frozen.py` now vendors the exact pre-V2 encoder to load
them; it feeds the frozen actor `obs[:24714]` (byte-identical pre-V2 prefix) and RECORDS the full
37596 row for the new critic. Empirically validated: loads clean, finite logits, plays to terminal.

## NOT done (user: do not run training/collection on this laptop — run on BIG/GPU 3)

- **T3 — real frozen dataset.** What exists at `runs/_stub_smoke/frozen_diag_stub/` is a stale
  STUB-policy dataset (`stub_policy:true`, `code_revision:null`) from a prior session — do NOT
  use it for the critic proof. No real dataset exists yet.
- **T6 real scan** (the committed tool is proven; numbers below are from stub data).
- **T7 short ladder.**

## Evidence for the committed tools

- Tests green (grad_norms 2, collector 8, schema 7, readout 8; plus back-compat
  ppo_win/ppo_sparse/action_factors 34). Run:
  `PYTHONPATH=build/open_spiel/python:. .venv/bin/python -m pytest tools/ open_spiel/python/eclipse/ open_spiel/python/pytorch/ppo_win_test.py open_spiel/python/pytorch/ppo_sparse_act_test.py open_spiel/python/eclipse/action_factors_test.py -q`
- grad_norms demo (CPU, STUB data — illustrative only, rerun on real data):
  policy 0.00% / value 44.08% / rank 18.15% / vp 37.77% of trunk grad at aux_coef=1;
  total trunk L2 46.9 (>> max_grad_norm 0.5 => clip SATURATING at aux_coef=1);
  recommended `--aux_coef≈0.66` to land VP ~25%.

---

## Big-machine commands (GPU 3, exact)

All below: `cd /home/mihai/personal/open_spiel_eclipse && CUDA_VISIBLE_DEVICES=3` and
`PYTHONPATH=build/open_spiel/python:. .venv/bin/python`.

### 1. Collect the real frozen dataset (T3)

```bash
mkdir -p runs/frozen_diag
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=build/open_spiel/python:. .venv/bin/python \
  tools/collect_frozen.py \
  --out runs/frozen_diag \
  --checkpoints "runs/long_v2/main.pt,runs/long_v2/snap_u2500.pt,runs/long8h/main.pt,runs/_judge/baseline/main.pt" \
  --episodes-per-policy 100 --workers 8 --device cuda --seed 0 --split_seed 0
# -> ~400 whole episodes, obs rows 37596-wide, manifest carries code_revision+collection_ts.
```

### 2. Real gradient-norm scan (T6) -> pick the aux_coef that feeds the ladder

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=build/open_spiel/python:. .venv/bin/python \
  tools/grad_norms.py --dataset runs/frozen_diag --batch 256
```
Read the printed VP share; the tool prints the recommended `--aux_coef`. Keep VP ~10-40% of the
trunk gradient and confirm the pre-clip total trunk L2 is not grossly >> 0.5 at that coef.

### 3. Short training ladder (T7) — the actual proof

Ladder verdict is the `run_long.sh`/`roster_ladder.py` gate (`IMPROVING/FLAT/REGRESSING`), NEVER
the loss curve. Short gates (1->10->30 min), no 2h/4h run.

12GB laptop (this machine), reduced envs — `--num_envs=128 --num_minibatches=4 --obs_buffer_device=auto`
(CPU fallback). Big machine uses the settled 1024-env config below.

```bash
# Replace AUX_COEF with the number from step 2.
AUX_COEF=0.66
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=build/open_spiel/python:. .venv/bin/python \
  -m open_spiel.python.examples.ppo_eclipse \
  --game='eclipse(players=4)' --seed=0 --cuda \
  --encoder=spatial --nn_activation=tanh --nn_width=64 --nn_depth=2 \
  --critic_readout=cell_attn \
  --factored_actions --league --max_live_opponents=4 \
  --ent_coef=0.05 --aux_target_mode=breakdown --aux_coef=$AUX_COEF \
  --num_envs=1024 --num_steps=128 --num_workers=16 --num_minibatches=16 --update_epochs=4 \
  --learning_rate=2.5e-4 --lr_schedule=fixed \
  --amp --compile_encoder --overlap_record --obs_buffer_device=cuda \
  --snapshot_every=25 --roster_keep_recent=0 --roster_keep_spaced=0 \
  --verdict_every_sec=0 --noeval_greedy --noeval_random --eval_games=8 \
  --timing --timing_every=100 --total_timesteps=100000000000 \
  --roster_dir=runs/frozen_ladder --run_dir=runs/frozen_ladder --track=frozen_ladder \
  --max_seconds=1800   # 30-min gate; use 60/600/1800 for the 1/10/30-min ladder
```

Notes:
- `--aux_target_mode=breakdown` routes the 9 VP targets through the cell_attn VP head
  (`_vp_breakdown_from_features`); the scan measured that objective. Use the scan value for AUX_COEF.
- The ladder rates the ACTOR (bitwise identical under both readouts), so a rank-default
  `roster_ladder.py` invocation still scores a cell_attn-trained roster correctly.
- Pre-register the command, seed(s), snapshot panel (early/peak/current), wall-clock caps, and the
  pass/fail threshold (positive lower-confidence bound on rating vs the rank-readout baseline arm).
  Stop on clear regression; A/B only by rating/CI, never by loss curve.

## Cleanup note
`runs/_stub_smoke/` holds the stale stub dataset (kept only as collector smoke reference; safe to
delete). No GPU collection/training is running on this laptop (idle, 134 MiB). Working tree clean.
