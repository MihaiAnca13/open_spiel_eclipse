# Handover prompt — paste this into a fresh session

> Copy everything inside the fenced block below as your first message in the new session.

```
Continue the Eclipse RL work in /home/mihai/personal/open_spiel_eclipse.

Read docs/eclipse_rl_todo.md first — it is the single source of truth and contains the
full diagnosis, measured constraints, and negative results. Then read the "Wave 3
results" section and runs/wave_ladder2.json.

WHERE THINGS STAND: wave 3's single-variable ladder (runs/wave_ladder2.json, 2048
games/policy) found exactly ONE change that beats the baseline: --ent_coef=0.05 alone,
rating 1.1071 [1.0817,1.1375] vs baseline:u573 1.0553 [1.0328,1.0810]. It clears by only
0.0007, so confirm it at higher --ladder_games_per_dir before trusting it. Everything
else lost: --separate_critic and --factored_actions each rate BELOW baseline,
--nn_width=256 is neutral, the spatial pointer head is a null result, and the bundle of
all of them is the WORST net on the board (0.8735) -- an interaction failure, not a bad
ingredient.

NOTE: ladder ratings are NOT comparable across runs (the margin fit is relative to the
pool, Random pinned at 0). baseline:u573 reads 1.0553 in wave_ladder2.json and 1.2204 in
wave_ladder.json. Compare within one tournament only.

NEVER infer strength from vp_all, mean_episode_return, or the vs-Greedy verdict. Two
wrong conclusions were reached that way in the last session, including calling
--ent_coef=0.05 harmful when it is in fact the only thing that won.

My goal this session: one long (8-12h) training run to find out where the plateau
actually sits when the agent gets 10x more steps than the ~16M our short arms managed.
Kill it early if the ladder clearly flattens.

Start by:
1. Confirming nothing is running on the GPU (nvidia-smi) — only ONE training run fits
   on this 12GB card at a time; two thrash to ~45 envstep/s.
2. GPU-benchmarking the three new throughput flags (--amp, --channels_last,
   --compile_encoder), which are implemented and CPU-verified but NEVER benchmarked on
   GPU. Measure envstep/s for each combination against the default before trusting any
   of them, then use the winning combination for the long run.
3. Deciding the batch size for the long run (see "biggest open lead" below), then
   launching it.

Guardrails, all learned the hard way — do not relearn them:
- Judge strength ONLY with the ladder's policies[].rating / rating_ci, never the elo
  field and never vp_all or the vs-Greedy verdict (Greedy is saturated: elo 282 vs
  main 1077). Use ./run_judge.sh — it builds trimmed rosters and runs one merged
  tournament across roster dirs.
- NEVER gate a wait loop on `pgrep -f "<string that is also in the loop's own command
  line>"` — the shell matches itself and spins forever. Poll for an output FILE.
- Run experiment arms SEQUENTIALLY. Three concurrent arms OOM'd; two thrashed.
- --lr_schedule now defaults to "fixed". Do not set it to "kl" without watching
  control/lr for the first 50 updates — that controller is still unvalidated and
  saturates lr_max within ~5 updates.
- Console logs are block-buffered. TensorBoard event files are the live source.

Biggest open lead (from the OpenAI hide-and-seek paper, Baker et al. 2019): their PPO
used a batch of 640,000 timesteps per optimization step, and batches below ~320,000
NEVER CONVERGED to the later skill phases — they didn't merely train slower, they got
stuck. Our batch is num_envs 128 x num_steps 128 = 16,384, roughly 39x smaller. Before
concluding anything about architecture, test whether batch size is the binding
constraint: raise --num_envs (and/or --num_steps) as far as VRAM allows and see if the
plateau moves. This is the single most promising untested lever.

Second cheap lead: --gamma is 0.99, but Eclipse pays its real reward only at the end,
~80 decisions per player. 0.99^80 ~ 0.45 discards half the terminal signal. Hide-and-seek
used 0.998. Worth one arm.

Do NOT do these without new evidence:
- Item 8 (routing move_unit/move_warp actions to their hex). It extends spatial routing,
  and spatial routing (Item 7, the pointer head) showed NO ladder effect. It is fully
  planned in docs/eclipse_rl_todo.md but must stay unbuilt until the pointer head is
  shown to help in a non-degraded regime.
- Item 6 (play-time MCTS / UI opponent). Explicitly out of scope.
- --ent_coef=0.05. It looks harmful (vp_all 9.3 vs baseline 15.9); hide-and-seek used a
  fixed 0.01, which is our default.

There is a large amount of UNCOMMITTED work in the tree (pointer head, throughput flags,
multi-roster ladder, doc rewrite, two deleted docs). Decide with me whether to commit it
before starting long runs.

Use sonnet subagents for research/investigation/implementation; you do synthesis and
verification. Verify subagent claims yourself — in the last session subagents twice
reported success on work that had a real bug in it.
```

---

## URGENT for the next long run: verdict evals eat 71% of wall-clock

Measured on `long_v2` at 7.75h elapsed / 47.4M steps:

- `charts/SPS` reads **5855**, but that is *per-update* throughput (16,384 steps / 2.80s).
- 2,894 updates x 2.80s = **2.25h of actual training compute inside 7.75h of wall-clock**.
- The average over wall-clock is **1700 steps/s** — 71% of the run is not training.
- Cause: **12 verdict evaluations**. `--verdict_every_sec` defaults to **1800** (30 min) and
  each eval plays `--eval_games=32` on `--eval_envs=64` against Random/Greedy, pausing
  training for roughly 27 minutes.

Consequence: a 12h run reaches ~73M steps instead of the ~200M the per-update rate implies.

**Fix for any future long run** (and especially on the cluster, where this waste scales):
set `--verdict_every_sec=7200` or higher, and/or cut `--eval_games`/`--eval_envs`. The
verdict is measured against Greedy anyway, which is saturated and worthless as a strength
signal — the ladder is the judge. There is little reason to pay 27 minutes every 30 minutes
for a number we have explicitly stopped trusting.

Beware: `charts/SPS` flatters the run. Always sanity-check steps-per-wall-clock as
`final_steps / elapsed_seconds`, not the logged SPS.

## Quick reference for whoever picks this up

### State of the world
| thing | value |
|---|---|
| Best policy | `runs/roster/snap_u573.pt`, rating **1.2204** [1.1941, 1.2490] |
| Best-policy config | width 64/2, tanh, `encoder=spatial`, `ent_coef=0.01`, `aux_target_mode=rank`, `aux_coef=0.1`, 128 envs x 128 steps, `num_minibatches=4` |
| Throughput | ~3.0-3.3k envstep/s (was ~13.6k with a flat trunk at the same obs, ~18.7k on the old 1,785-float obs) |
| Ladder outputs | `runs/wave_ladder.json` (wave 1/2), `runs/wave_ladder2.json` (wave 3) |

### Defaults as left (all deliberately conservative)
| flag | default | why |
|---|---|---|
| `--lr_schedule` | **`fixed`** (changed) | previously resolved to the unvalidated `kl` controller on any run that didn't pass it |
| `--spatial_pointer` | `False` | mechanism verified, but no ladder benefit |
| `--separate_critic` | `False` | no measured benefit alone |
| `--factored_actions` | `False` | unproven alone |
| `--ent_coef` | `0.01` | matches baseline and hide-and-seek |
| `--amp` / `--channels_last` / `--compile_encoder` | `False` | correctness-verified on CPU, **never GPU-benchmarked** |

### Scripts
- `run_wave1.sh` / `run_wave3.sh` — sequential experiment arms (wave 3 = single-variable attribution)
- `run_judge.sh [games] [out.json]` — builds trimmed rosters, runs one merged ladder over
  baseline + every arm dir that has a finished `main.pt`
- `run_item5_long.sh` — the older long-run resume script
