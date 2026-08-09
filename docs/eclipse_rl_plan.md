# Eclipse RL — active plan & progress tracker (2026-08-08)

Source-of-truth context lives in `docs/eclipse_rl_todo.md` (diagnosis, negative results,
measured constraints). This file is the **forward-looking plan + progress log** for the current
throughput/stability sprint. Update statuses here as work completes.

## Direction (user-confirmed 2026-08-08)

"Simplify. Fix the crash + speed loop, attack the VRAM/parallel-game ceiling, enable bigger
batch for training quality, and eventually evaluate a native step. Not the pointer/arch work."

The engine is already C++ via pybind11; env stepping is ~24% of step time, so a native stepper
is NOT the early win. The binding constraints are: the crash bug, redundant act-path encoder
compute, verdict-eval wall-clock waste, and the GPU-resident rollout obs buffer capping envs.

## Why the VRAM ceiling exists (the "1k parallel games" question)

The rollout obs buffer `self.obs` (`ppo.py:414`) is a GPU tensor: 128 steps x num_envs x
24,714 floats fp32 = **1.51 GiB parked on VRAM** at 128 envs. It does not need to be on GPU —
the learn loop only loads one minibatch at a time (`b_obs[mb_inds]`, `ppo.py:1633`). 1k envs
would need ~11.8 GiB of obs storage just to buffer one rollout epoch, and 10k envs ~120 GiB —
impossible while buffering full fp32 rollout obs. Fix: keep the obs rollout on CPU/shared
memory (or fp16) and stream minibatches to GPU. That lifts the 128-env cap *and* is the
prerequisite for the avoidable big-batch test.

## Work items

### 1. [x] Kill the crashed long_v2 run + orphans + supervisor/watchdog
- Started 10:09, crashed at 47.4M steps (CUDA gather OOB); main process + 16 workers held the
  GPU (7 GiB) until killed. GPU now free (134 MiB).
- Learned: a crashed-but-alive main process keeps the GPU hostage; pgrep-based supervisors
  report "RUNNING" while training is dead.

### 2. [x] Absorb NEXT_SESSION.md into eclipse_rl_todo.md; delete NEXT_SESSION.md
- Added "long_v2 result (2026-08-08)" section (crash, two bugs, verdict-eval 71% waste, VRAM
  bind) + redirected the "Current state" header away from Item 7 toward this sprint.

### 3. [x] Write this plan/tracker file
- The file you are reading.

### 4. [x] Fix the `chosen` sentinel OOB gather bug (`ppo.py` `_gumbel_sample`)
- `_gumbel_sample` seeded `chosen` with `num_actions` (ppo.py:1364); an empty legal segment or
  NaN logits left that sentinel, and `_log_prob_chosen` → `head_logits` gathered
  `head.bias[num_actions]` → OOB (killed long_v2). Fixed by `return chosen.clamp(max=num_actions-1)`
  so the gather can never go OOB; healthy rows unchanged. Added
  `test_sparse_empty_legal_row_does_not_oob` to ppo_sparse_act_test (the branch was previously
  untested). All tests green.

### 5. [x] Remove redundant encoder recompute on the act path
- `_act_sparse` and the learn minibatch loop ran the encoder once for actor features then AGAIN
  inside `value_from_obs`. With `separate_critic=False`, `critic_trunk is shared`, so the conv
  tower ran twice per decision. Added `value_from_actor_features = not separate_critic` to
  `EclipsePPOAgent` and made both paths read value off `features` via `value_from_features` when
  it is set. Verified bit-identical to `value_from_obs` (max diff 0.0) on a real shared-critic net.

### 6. [x] Cut verdict-eval wall-clock
- `--verdict_every_sec` default 1800 → **7200** (2h). Each verdict paused ~27 min; at 1800s they
  ate 71% of a 12h run. Default now fires evals 4x less often; the ladder is the judge anyway.

### 7. [x] Move the obs rollout buffer off GPU (CPU) to lift the 128-env cap
- `self.obs` was a GPU tensor (ppo.py:414) holding the full (steps, envs, obs) rollout on VRAM
  (~1.51 GiB at 128 envs, scaling 1:1 with num_envs). Now allocated on CPU; `step_np` stores the
  CPU-side obs directly (no GPU round-trip), and the learn minibatch loop streams one minibatch to
  GPU at a time (`mb_obs = b_obs[mb_inds].to(device)`). Avoids a device tensor in the extras concat.
- **Result: the 128-env OOM cap is lifted. Measured at the winning flags (`--amp --compile_encoder`,
  width 64/2): 128 envs = 4100 SPS, 192 = 3446, 256 = 3920, 384 = 3389 -- no OOM through 384.**
  (Was hard-OOM at 192 before.) A 3x wider parallel rollout on the same 12 GiB card.

### 8. [x] Verify with tests (ppo_sparse_act_test, ppo_win_test, ppo_pytorch_test, ppo_selfplay_pytorch_test, ppo_league_test, action_factors_test) — all green

### 9. [x] GPU-benchmark throughput + env-scaling after fixes (see item 7 result)

### 10. [in progress] 8h training run + monitor it (LAUNCHED 2026-08-09 08:54)
- `runs/long8h`, 256 envs / 128 steps (batch 32,768 = 2x baseline), `--ent_coef=0.05`,
  `--gamma=0.998`, `--amp --compile_encoder`, `--lr_schedule=fixed`, 8h target.
- Uses the CPU-obs offload (256 envs, was capped at 128). ~3700-3900 SPS at 256 envs.
- Monitoring: `supervise_long8h.sh` (PID 1810058, detached) ticks every 15 min to
  `runs/long8h_ticks.log`; `monitor_long8h.sh` is the per-tick reporter. It records a
  final line if the training PID exits, so a future session knows the run stopped.

### 10. RESULT (finished 2026-08-09 16:56, 137.6M steps, judged 17:29)
- Task requested: wait for the run, then run the judge. Done. Added `runs/long8h` to
  `run_judge.sh` and laddered it (128 games/pair) vs baseline + all wave arms in one
  tournament: `runs/long8h_ladder.json`.
- **Go/no-go: PASS by a wide margin.** long8h leads the whole ladder:
  - `long8h:u3900` rating +1.2195, CI [1.203, 1.236] — **#1 overall**.
  - `long8h:main` (u4200, final) rating +1.1431, CI [1.126, 1.164] — **#2 overall**.
  - Reference `baseline:u573` (plateau) +1.0419, CI upper 1.0604.
  - long8h lower bounds (1.203 / 1.126) clear baseline upper (1.0604) by ~0.06-0.14.
  This is the strongest model measured so far.
- **Caution — NON-MONOTONE regression:** u3900 (+1.2195) CI-clears final u4200 (+1.1431):
  the model peaked near u3900 then slipped slightly by u4200, but stayed far above baseline.
  Repeats the "train-longer-isn't-always-better" theme, small magnitude here.
- (Baseline arm picked u573 as its middle snapshot, matching the plateau reference, and w3_ent
  also marginally beat baseline u573 here: lower 1.0340 vs upper 1.0604 is NOT clear — so in this
  ladder the only CI-clear wins over the plateau are long8h's.)

## Guardrails (from docs — do not relearn)

- Judge strength ONLY via ladder `policies[].rating`/`rating_ci`, never `elo`/`vp_all`/
  `mean_episode_return`/vs-Greedy. Ratings are NOT comparable across ladder runs.
- Never gate a wait loop on `pgrep -f "<string in the loop's own cmdline>"` — poll a file.
- Run arms SEQUENTIALLY (two thrashed the GPU to ~45 SPS; three OOM'd).
- `--lr_schedule` defaults to fixed; don't use "kl" without watching control/lr.
- Real throughput = `final_steps/elapsed_seconds`, never the logged SPS.
