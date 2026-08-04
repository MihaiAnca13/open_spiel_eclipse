# PPO self-play for Eclipse (GPU-batched, many parallel games)

## Context

The goal is a good Eclipse-playing agent, trained with real GPU throughput and many games running in parallel. The original direction considered was extending AlphaZero's MCTS self-play with a custom GPU-batching inference server (à la KataGo/Leela Zero/MiniZero) — external research confirmed that architecture is indeed how serious AlphaZero implementations solve GPU utilization, but it doesn't fix a more fundamental problem: **Eclipse's ~11,000-action space is a documented poor fit for vanilla MCTS regardless of GPU throughput** (MCTS must explore actions before learning which are good, so efficiency degrades as the action space grows). Real systems facing comparably large action spaces made a different call — Jaipur (~25K actions) found PPO beat other RL algorithms; DouDiZhu (large card-game action space) uses PPO-based self-play.

**Correction from an earlier draft of this reasoning**: OpenAI Five/Dota was initially cited here as "PPO handles huge action spaces with no tree search," which is true but incomplete in a way that matters for Eclipse specifically. OpenAI Five did **not** rely on sparse terminal (win/loss) reward — it used dense, hand-shaped rewards (gold/experience advantage, kills, tower damage, etc.), explicitly because Dota's horizon (~20,000 timesteps) is so long that pure win/loss credit assignment is impractical for a policy-gradient method with no search. This is evidence *for* needing reward shaping on long-horizon problems, not evidence that plain PPO handles sparse terminal reward fine at scale.

### Why chess/Go are not a good comparison for the sparse-reward problem

Eclipse's scoring is genuinely sparse: only the final VP total (computed once, at the terminal state) matters — `Rewards()` is the zero vector at every non-terminal step (`kTerminal` reward model, confirmed). Chess and Go share that same sparse-terminal-reward shape (win/loss/draw only, no intermediate reward), so it's tempting to assume "AlphaZero/self-play already solved sparse reward for board games." **It didn't solve it the way PPO would need to.** Chess/Go's credit assignment is handled by MCTS itself during self-play: tree search produces well-shaped value estimates at every node via lookahead and backup, without needing any hand-designed intermediate reward. That mechanism is exactly what we ruled out for Eclipse (the ~11K action space makes vanilla tree search impractical). Plain PPO with GAE has to learn a value function purely by TD-bootstrapping across Eclipse's ~190 real decisions per game, with a reward signal that is zero everywhere except the very last step — a genuinely harder RL problem than what chess/Go's engines actually had to solve, and one this plan should validate is even learnable *before* investing in the full N-player self-play build (see the new Stage 0.5 below).

### What real precedent for this specific combination (sparse + long-horizon + huge action space + general-sum) actually shows

- **Reward shaping is the near-universal answer** across the precedent found (DouDiZhu's RARSMS reward shaping, real-time-strategy-game research on shaped intermediate rewards) — but naively done, it backfires: one RTS study found that rewarding "+1 per point of Victory-Points-equivalent gained this move" made agents greedy and short-sighted (building cheap immediate value while neglecting long-term setup), because per-move reward ignores the temporal structure of a long-horizon objective. The standard fix is **potential-based shaping** (`shaped_reward = γ·Φ(s') − Φ(s)` for some potential function Φ, e.g. "estimated final VP if the game ended now") — this is provably policy-invariant (doesn't change the optimal policy, unlike an arbitrary shaping bonus), while still giving much denser learning signal than raw terminal-only reward.
- **The closest real analog to Eclipse's actual profile** (complex, long-horizon, general-sum, N-player strategy game) is Diplomacy/Cicero — and it was explicitly **not** trained via pure self-play RL from a blank policy; Cicero warm-started its policy via imitation learning on human games, then refined with self-play RL. **This option is not available to us — we have no access to human/expert Eclipse demonstration data** — so potential-based reward shaping (Stage 0.5) is the primary tool instead, not a secondary one.
- **Honest caveat**: long-horizon 4X/economic-strategy-style credit assignment is an actively researched, not-fully-solved problem (research benchmarks like "Terra Nova" (arXiv:2511.15378), modeled on Civilization, exist precisely because of this), and RTS-style self-play training has a documented failure mode of collapsing to immediate aggression over long-term economic investment — a very plausible risk for Eclipse's economy/tech/expansion/combat balance too. There is no guaranteed recipe here; expect this to need real iteration, not just correct engineering.
- **Model-based RL (Gumbel MuZero / Sampled MuZero) is a real, deliberately-deferred alternative**, not a dismissed one. It's arguably a *more* directly-targeted fix for this exact problem than PPO+shaping: those methods were specifically designed to make search practical in huge action spaces (sampling a manageable subset of actions per node via Gumbel-top-k instead of exhaustive expansion), and a learned dynamics/reward model gives a much richer, self-supervised credit-assignment signal than a hand-designed shaping function — it doesn't depend on us guessing the right Φ at all. It's not being pursued now because: (1) no MuZero-family implementation exists anywhere in this OpenSpiel fork (confirmed by search — a from-scratch build); (2) it would need the same N-player/general-sum/imperfect-information generalization work AlphaZero already required, likely harder given MuZero's formalism assumes perfect information more centrally; (3) PPO is a substantially cheaper path to try first, and already exists in this repo. Revisit this specifically if Stage 0.5's sanity check comes back discouraging even after real effort on the shaping function.

Crucially, this repo already has a working, GPU-capable, vectorized PPO implementation (`open_spiel/python/pytorch/ppo.py` + `open_spiel/python/vector_env.SyncVectorEnv`) that batches its neural-net forward/backward passes across `num_envs` parallel game environments *natively* — the entire "build a custom batching inference server" problem that MCTS self-play would require is simply not needed here; vectorized RL already solves it.

Investigation of this PPO stack against Eclipse's specific characteristics found:
- **Already fine, no changes needed:** action masking (`CategoricalMasked` already masks illegal actions out of the full ~11K logits — works at this scale, just a fixed compute/memory cost, not a correctness issue); chance-node handling (`rl_environment.Environment._sample_external_events()` transparently resolves chance nodes before returning control to the training loop — Eclipse's dice/tile/reputation draws need no special handling); GPU device placement and batched forward passes across `num_envs` (already implemented, `torch.device`-based).
- **The real gap:** `pytorch/ppo.py` is explicitly single-agent-only (module docstring: *"Currently only supports the single-agent case"*). It hardcodes one fixed `player_id` at construction and assumes a single scalar reward per environment per step. For a real 4-player game this breaks in two ways: (1) on any timestep where the fixed `player_id` isn't actually `current_player()`, `legal_actions_mask` is all-zero, producing a uniform-random illegal action applied with no legality check; (2) `vector_env`'s per-player reward vector (`(num_envs, num_players)`) doesn't fit the reward buffer's expected `(num_envs,)` shape. This needs the same kind of N-player generalization AlphaZero already received (prior work, complete) — informed by, but not identical to, existing precedent:
  - The standard OpenSpiel turn-based self-play idiom (found in `independent_tabular_qlearning.py`, `dqn_breakthrough_pytorch.py`): `while not time_step.last(): player_id = time_step.observations["current_player"]; agent_output = agents[player_id].step(time_step); time_step = env.step([agent_output.action])`, then `for agent in agents: agent.step(time_step)` at termination so every seat's last transition gets closed out with its own slot of the terminal reward vector — not just whichever seat made the final move. These examples use **N separate agent instances** (independent learners), not one shared policy.
  - The precedent for **one shared set of weights dispatched by seat** is AlphaZero's own self-play loop (`alpha_zero.py`): one `model`/`az_evaluator`, `num_players` thin bot wrappers all deferring to it.
  - Eclipse's `GameType.reward_model` is `kTerminal` (confirmed): intermediate `Rewards()` are always the zero vector, and the full per-player payoff arrives simultaneously exactly at the terminal state (`Rewards() == Returns()` there). This means the standard "close out every seat's last transition using the terminal reward vector" idiom is directly usable, with no extra "reward accrued while it wasn't my turn" bookkeeping needed (that gap would only matter for games with non-terminal, bystander-visible intermediate rewards, which Eclipse doesn't have).
- **A real but much smaller throughput consideration than MCTS had:** `SyncVectorEnv.step()` advances each of the `num_envs` game states sequentially in Python (CPU-bound) — only the neural-net forward/backward pass is GPU-batched. This is far cheaper per step than MCTS (one `apply_action` call vs. hundreds of tree simulations per decision), so it should scale to a much higher `num_envs` before becoming a bottleneck, but this should still be measured, not assumed.

## Stage 0 — Light setup + profiling

1. Install a CUDA-enabled `torch` build into `.venv` (PPO uses PyTorch, not JAX — separate from the AlphaZero stack's jax dependency).
2. Microbenchmark forward-pass latency at `num_envs` = 1, 64, 128, 256 on CPU vs GPU, at a realistic Eclipse-sized network (obs=1785, actions≈11K) — expect a much clearer GPU win here than the earlier MCTS single-example case, since PPO naturally does one large batched matmul per step rather than many tiny sequential calls.
3. Measure `SyncVectorEnv` env-stepping throughput on Eclipse alone (no NN involved) at increasing `num_envs`, to find where the sequential Python env-stepping loop becomes the bottleneck relative to GPU forward-pass time. If it caps out well before "hundreds," note whether `vector_env.py` has (or would need) an async/multi-process variant — this can be a later addition, not a blocker for the first working version.

## Stage 0.5 — Define the best shaped-reward function, using the rulebook and existing code

We have no access to human/expert demonstration data for Eclipse, so imitation-learning warm-start (the Cicero/Diplomacy precedent) is **not viable** and is dropped as an option. That makes potential-based reward shaping the primary, load-bearing tool for the sparse-reward/long-horizon credit-assignment problem — this must happen early and cheaply, before committing to Stage 1's full N-player PPO generalization.

### Stage 0.5 execution notes (this session)

- **Φ is already computable exactly from live state**: `compute_player_score(state, pid).total_vp` (open_spiel/games/eclipse/systems/scoring.cpp) is "current VP if the game ended now", species-aware (Planta/Draco/minor-species), computed at any time. The rulebook re-derivation was unnecessary; the scoring code is the oracle.
- **Unix enabled it for Python cheaply**: `Player.score` was dead (set 0 at setup, never updated), so the observation tensor advertised a constant-0 "score" (a real bug: nets could never see VP). Changed `ObservationTensor` (eclipse.cc) to write every seat's live `total_vp` into the self slot (B0+10) and each opponent slot (Bn+0), one galaxy pass per call, tensor shape unchanged. So both `Φ(s)` and `Φ(s')` for the acting seat are readable from the two consecutive observation tensors (self slot, and the seat's opponent block as seen by the next viewer).
- **Pilot verdict (16 envs, 64-wide→128-wide MLP, ~82k steps ≈ 400+ episodes, GPU)**, `open_spiel/python/examples/ppo_eclipse.py`:
  - `--phi=banked` (potential = banked VPs): **no learnable signal**. Mean terminal return stayed 0.00 and 0/200 episodes ever banked VP across two seeds and 40 updates. Because early policies almost never complete a scoring sequence, no trajectory in the batch ever changes `total_vp`, so the shaped reward is silent everywhere → no gradient. Random/naive play degenerates to ~0-VP games (~170 steps, round 9, everyone 0).
  - `--phi=soft` (banked VP + in-progress presence terms: colony ships, disks on sectors, orbitals/monoliths, ambassadors — all already in the observation, same terms visible in both self and opponent blocks): **model-liveable**. Nonzero-VP episodes emerge (upd35: mean 1.52, 81/200; seed2: 23/200) and mean return climbs steadily.
  - `--phi=none` (no shaping, pure terminal return): also learnable, seed1 reached 4.59/198/200 but seed2 lagged (0.39/23) — higher variance than soft at this tiny budget.
  - Robust takeaways: banked-only Φ is a dead end at this scale; both soft-Φ and no-shaping break the degeneracy; this is exactly the "discouraging → revisit Φ" branch the plan anticipated, and it fired productively. Longer runs / better potentials (or a learned potential / value warm-up) are the natural Stage 3 refinements. The dense-but-biassed nature of the hand-rolled soft weights is a known risk (presence-hoarding ≠ VP) and should be re-tuned or replaced before production runs.
- **Dependency found**: the pilot itself required the Stage 1 core (per-call seat dispatch, vectorized self-play, all-seats terminal closeout); that is now built and tested (see Stage 1), so Stage 0.5's gate and Stage 1's core landed together.

1. **Define the shaping potential Φ(s) using both the rulebook and the existing scoring code**: read `07-eclipse-second-dawn-for-the-galaxy-rulebook.pdf` (repo root) for the actual VP-scoring categories and their relative weight/timing in a real game (reputation tracks, tech, colonization, combat, ambassadors, etc.), cross-referenced against the already-implemented scoring logic (`open_spiel/games/eclipse/systems/scoring.{h,cpp}`) to make sure Φ(s) — "current VP total if the game ended right now" — is actually computable from live game state, not just the rulebook's end-of-game procedure. The goal of this step is specifically to find the *best* definition of Φ, not just *a* workable one — e.g. deciding whether to weight partial/in-progress VP sources (tech not yet scored, board position not yet converted to reputation) versus only counting already-banked VP, since that choice directly determines whether the shaped reward encourages long-term setup or short-term grabs.
2. Derive `shaped_reward = γ·Φ(s') − Φ(s)` per step — this is provably policy-invariant (doesn't change the optimal policy, unlike an arbitrary bonus), and is the standard fix for the "naive per-move reward → myopic agent" failure mode found in RTS research (rewarding raw ΔVP each move made agents greedy/short-sighted in that study).
3. **Run a cheap sanity check** before building full self-play: using a small dataset of played-out games (random or heuristic play is fine for this), check whether a value function can learn anything predictive of final VP from partial game states, using only the shaped reward from step 2. This is far cheaper than standing up full N-player self-play (Stage 1) and answers the most important open question early: is this reward signal learnable at all.
4. If the sanity check is discouraging (value function learns nothing better than a constant baseline even with shaping), that's a real signal to revisit Φ's definition — or reconsider the model-based-RL alternative noted below — rather than assuming more `num_envs`/compute will fix a credit-assignment problem.

## Stage 1 — Generalize `PPO` to N-player shared-policy self-play

**Built and tested in this session.** Summary of what landed (details in the code):
- `open_spiel/python/pytorch/ppo.py`: `player_id` no longer fixed — `step()` dispatches each env's `current_player` (shared network, one batched forward per row). Per-(env,seat) GAE: advantage chains through a seat's *own* decision values (not cross-seat bootstrapping, which would mix value-to-different-seats). Terminal attribution: the acting seat's payoff lands on its own done row; every other seat is closed out via an independent terminal sample carrying that seat's slot of the terminal returns vector — collected across batches, so a seat whose last decision was in a previous rollout batch is still closed out. `post_step(reward, done, shaped_reward=...)` supports per-step potential-based shaping while keeping terminal payoff attribution intact. A batch-size guard (`batch` kept a whole multiple of `num_minibatches`) avoids single-sample-minibatch NaN in advantage normalization.
- Tests: `open_spiel/python/pytorch/ppo_selfplay_pytorch_test.py` (colored_trails smoke + per-seat terminal-reward attribution invariant); original single-agent `ppo_pytorch_test.py` still passes (backward compatible). 
- A/B sanity on colored_trails vs Eclipse deferred; the Eclipse pilot in Stage 0.5 exercises the same code path.
- **Known caveat (documented in plan reasoning):** the speedup of `SyncVectorEnv.step()` (sequential Python CPU-side env stepping) has not been re-measured at high num_envs for Eclipse yet — fine for Stage 2/3 pilot scale.

The original Stage 1 checklist, all done:
- [x] `player_id` passed per-call (`get_action_and_value`/`step`) so one shared network acts for whichever seat is to move.
- [x] Vectorized self-play loop: read each env's `current_player`, gather that seat's obs + mask across all `num_envs` into one batch, one shared-network forward pass (GPU batching preserved regardless of which seats act), apply per-env actions.
- [x] Rollout/reward bookkeeping: (env, seat) pending transitions; at terminal, close out **all** seats with their own slot of the terminal returns vector (surviving batch boundaries via the independent-sample path).
- [x] Value/advantage = plain scalar "value to the seat currently acting", per-seat GAE subsequences; no per-player value vector.
- [x] Tests: synthetic multi-env multi-seat attribution invariant + colored_trails self-play smoke test; single-agent regression still green.

## Stage 2 — Eclipse-specific wiring

**Built as part of the Stage 0.5 pilot** — `open_spiel/python/examples/ppo_eclipse.py`:
- Shared-policy self-play loop over `num_envs` Eclipse games (`ObservationType.OBSERVATION`, obs 1785, ~11K actions, `CategoricalMasked` over the full action space).
- Configurable potential: `--phi=banked|soft|none` with soft-Φ weights (colony ships / disks on sectors / structures / ambassadors). `--shaping` toggles it entirely.
- Shapes from the observation "score" slots: `Φ(s)` from the acting seat's self slot, `Φ(s')` from that seat's opponent block as seen by the next viewer in the following obs — no extra C++↔Python traffic; terminal transition unshaped (true payoff).
- Eval-style logging: per-update steps, episode count, mean terminal return, nonzero-VP episode count.
- Caveat: the ~11K-logit actor head makes the learn step heavy on CPU; GPU (`--cuda`) is the intended path. Env stepping (`SyncVectorEnv`) remains sequential per env — fine at pilot num_envs, worth re-measuring (Stage 0 item) before scaling.

## Stage 3 — Hyperparameter search & pilot runs

Same philosophy as before: short pilot runs judged against logged diagnostics (episode return per player, entropy, KL/clip-fraction — standard PPO diagnostics, extended with per-player return tracking for the general-sum multiplayer case) before committing to long runs. Revisit throughput/timing estimates once Stage 0 and Stage 3 give real numbers, rather than guessing now.

## Phase: Win-utility value head + learned auxiliary heads (done)

Moves the value target off raw VP (which rewards "cashing out a big score", not "finishing first") onto **rank-utility**, and replaces the hand-tuned soft-Φ shaping weights with supervised auxiliary heads.

- `open_spiel/python/pytorch/ppo.py`: `rank_utility()`/`rank_of()` map a per-seat payoff vector to a scalar target via the rank distribution `(1.0, 0.5, 0.0, -0.5)`; new `value_mode="win"` vs legacy `"vp"`; terminal targets (both sparse+batch paths) become rank-utility for win mode; `post_step`/`post_step_np` close out every seat's last decision with that seat's rank-utility target; `aux_targets`/`aux_mask` buffers back-fill terminal-derived aux targets onto every stored row of a seat when its episode closes; aux loss (masked MSE, `aux_coef`) added in `_learn_core`.
- `open_spiel/python/examples/ppo_eclipse.py`: `EclipsePPOAgent` now has a **4-output rank critic** (softmax → expected rank-utility, `rank_value`), an `aux_heads.final_vp` head (predicts terminal VP/200), and `value_from_features` for the sparse learn path; `--phi` gained `learned` (potential = the network's own win value, `_phi_wins`); `--value_mode` default `win`.
- Decision trail: the 4-output rank critic is the value head *and* the learned potential — B/A-style shaping replaces the hand-rolled soft-Φ weights for the **default post-A1 runs**, with `--phi=banked` choosing banked-VP potential (conservative: the shaped signal stays policy-invariant) and `--phi=soft|none` still available.
- Tests: `open_spiel/python/pytorch/ppo_win_test.py` (6 tests incl. rank-utility ties `[40,20,40,30]`→seat3 rank 3), plus regressions (`ppo_selfplay_pytorch_test.py`, `ppo_pytorch_test.py`).
- Pilot A (`clever-badger-3100`): `--value_mode=win --phi=banked`, 128 envs / 524k steps → nonzero-VP episodes 0→35/200 and mean terminal return 0.0→0.40, validating win+aux mechanics end-to-end.

## Phase: League / population self-play with exploiters (done)

- `open_spiel/python/pytorch/ppo.py`: `setup_league(networks, lineup, train_pid)`; `_act_batch` routes each (env, seat) row through its lineup policy's network; `step`/`step_np` mark `trainable` rows = train_pid seats and skip non-trainable seats in `_last_decision`/aux backfill and extra-sample closeout (`.get()` guard); `_learn_core` **drops non-trainable rows** before any loss math (advantages/returns/obs/aux all filtered; packed-legal sparse structures remapped to the filtered index space with extras appended after main rows — bug fixed: `keep` must include the appended extras and `extra_base` must be main-only row count, else zero-legal-count rows → NaN entropy).
- New `open_spiel/python/examples/league.py`: `PolicyRoster` (JSON index + per-net `.pt`, `record_main`, `add_snapshot` (`snap_u<update>`), `add_exploiter`, `prune`, `load_net` [now `.to(device)`-safe], `opponent_ids`) and `Matchmaker` (`sample_lineup` keeps `train_pid` on seat 0, draws opponents/selfplay by fraction, `lineups`).
- `open_spiel/python/examples/ppo_eclipse.py`: `--league` (matchmaking + snapshots + optional `--eval_squad` ladder argmax vs snapshots), and sequential-exploiter mode `--exploit_victim=<id>` (one trainable policy vs a frozen victim, head-to-head argmax eval at closeout, `--exploit_promote` folds it into the roster when win-rate ≥ 0.5; `--exploit_lr` boosts LR).
- Decision trail: exploiters are **sequential** (one at a time), not concurrent — single 12GB GPU; heuristics (rule-based dev/bot opponents) are **excluded from the league in v1**, deferred to v2; FCM is out of scope (no engine exists); league mode is optional — default single-network self-play is unchanged (`value_mode="vp"` stays raw-VP compatible).
- Tests: `open_spiel/python/examples/league_test.py` (roster roundtrip, missing net → None, prune bounds, matchmaker lineup shapes/mixed) and `open_spiel/python/pytorch/ppo_league_test.py` (multi-network colored_trails: main-only gradients — frozen snap unchanged after `learn`, no extras close out non-trainable seats, trainable flags per seat).
- Smoke (Eclipse): league mode with `--snapshot_every=1` produces `main.pt` + `snap_u1.pt` + `roster.json`; exploiter mode trained vs frozen `snap_u1`, head-to-head 16/16 win-rate, promoted → `expl_u1_vsnap_u1.pt` (role=exploiter, win_rate=1.0).

## Phase: Per-episode randomized setup (races / NPC difficulty / warped universe)

Goal: the agent learns to play *all* races and against *all* races (plus both NPC difficulties and module layouts) instead of the fixed all-Terran Easy configuration. Races were already observable — own species one-hot at obs `B0+0..6` and each opponent's at `Bn+3..9` — so no observation/tensor change is needed.

- `open_spiel/games/eclipse/` (C++): new opt-in game params `randomize_races` (+ `race_alien_prob`, default 0.8), `randomize_npc_difficulty`, `randomize_warped` (+ `warped_prob` 0.5). New `RandomizeSetupForEpisode()` (`systems/setup.{h,cpp}`) draws the per-episode config; races use a **unique alien draft** (the 6 non-Terran species at most once per episode) with **Terran as with-replacement filler**; the draw is made in `EclipseState::ResolveChanceEvent(initial_setup)` (`eclipse.cc`) from the seeded per-instance game RNG and written back to `setup_config_` so serialization/info-string reflect the real draw. Works uniformly for sync/async/league/exploiter/eval. The **C++ engine defaults stay off** (a bare `eclipse(...)` load stays deterministic all-Terran/Easy for other consumers and C++ tests).
- `open_spiel/python/examples/ppo_eclipse.py`: flags `--randomize_races --race_alien_prob --randomize_npc_difficulty --randomize_warped --warped_prob`; a `_randomized_game_string()` helper builder; **per-env `rng_seed`** (`FLAGS.seed + i`) so each game instance gets a fresh seeded RNG (fixes the latent identical-board/every-env issue and diversifies starting tech/discovery/market per env). Eval (squad/head2head) uses the same randomized string so eval matches training. **Training defaults: all three toggles are ON** (`--no*` flags disable) — a default run now trains on the full race/difficulty/warped distribution.
- `open_spiel/python/async_vector_env.py`: accepts `game_strs` (per-env list) alongside `game_str`; workers build each env from its own string/seed.
- Tests: `eclipse_test.cc` `SetupRandomizationTest` (determinism for fixed seed, alien uniqueness + Terran filler, variety across seeds, all three difficulties reachable, warped flips, serialization round-trip preserves the draw); `eclipse_test` 76/76 green; ppo_eclipse sync + async smokes green.
- Caveat: `--randomize_races` broadens the state space the shared policy must absorb; expect to revisit `--nn_width/depth` if learning slows.

## Stage 3.5 — Staged roll-out of the league (pending)

Short staged runs before long production runs, judged on the same diagnostics + league-specific signals:
1. `--league` only (main vs frozen snapshots, matchmaker lineups, `--eval_squad`) — verify main learns to beat increasingly-old snapshots.
2. `--exploit_victim=<snapshot>` exploiters run sequentially, promoted on win — watch for policy-collapse/repeated-exploit patterns and use `prune` to keep the roster small.
3. Long run: league + exploiters together with heuristics excluded (v2 adds them).

## Stage 4 (optional, future, out of scope for now)

If pure-PPO play quality falls short after real tuning, consider wrapping the trained policy with a light MCTS at **play time only** (not training time) for a strength boost — no training-time batching infrastructure required for that, since it's just inference-time search on top of an already-good learned policy/value function.

## Verification per stage

- **Stage 0**: written summary of measured numbers (**deferred**; env stepping sanity-run informally, GPU forward not yet benchmarked at scale).
- **Stage 0.5**: **done** — chosen Φ(s) documented above with code justification and the pilot verdict (banked-only: not learnable; soft/none: learnable; recorded in the Stage 0.5 notes).
- **Stage 1**: **done** — attribution unit test + colored_trails smoke test added and green; single-agent regression green.
- **Stage 2**: **done (smoke)** — ppo_eclipse.py runs end-to-end on GPU; shapes/masking correct over the full ~11K actions; used for the Stage 0.5 pilots.
- **Stage 3**: pilot-run diagnostics compared across hyperparameter sweeps — underway (banked/soft/none A/B above is a first pass; win-utility + aux-heads Pilot A and league/exploiter smokes landed).
- **Win/rank + aux heads**: **done** — `ppo_win_test.py` 6/6, regressions green.
- **League self-play + exploiters**: **done** — `league_test.py` 5/5, `ppo_league_test.py` 2/2, Eclipse league + exploiter smokes green.
- **Stage 3.5**: staged roll-out of the league on Eclipse — pending.

### Critical files
- Modified: `open_spiel/python/pytorch/ppo.py` (N-player self-play generalization, per-seat GAE + terminal closeout + shaped reward hook; win/rank value + aux heads; league/multi-policy support), `open_spiel/games/eclipse/eclipse.cc` (write live per-seat `total_vp` into the observation score slots; per-episode setup-randomization game params + draw wiring), `open_spiel/games/eclipse/eclipse.h` + `systems/setup.{h,cpp}` (params getters; `RandomizeSetupForEpisode`), `open_spiel/python/examples/ppo_eclipse.py` (Eclipse agent rank critic + aux heads + `--phi=learned`, `--league`/exploiter/eval-squad wiring, per-env game strings + `--randomize_*` flags), `open_spiel/python/async_vector_env.py` (per-env `game_strs`).
- Reference only, no modification expected: `open_spiel/python/rl_environment.py`, `open_spiel/python/vector_env.py` (both already correctly handle chance nodes, per-player observations, and per-player rewards), `open_spiel/python/examples/independent_tabular_qlearning.py` and `dqn_breakthrough_pytorch.py` (turn-dispatch idiom precedent), `open_spiel/python/algorithms/alpha_zero/alpha_zero.py` (one-shared-model-dispatched-by-seat precedent).
- New: `open_spiel/python/examples/ppo_eclipse.py`, `open_spiel/python/pytorch/ppo_selfplay_pytorch_test.py`, `open_spiel/python/pytorch/ppo_win_test.py`, `open_spiel/python/pytorch/ppo_league_test.py`, `open_spiel/python/examples/league.py`, `open_spiel/python/examples/league_test.py`.

Stages 2-4 should be refined once Stage 0/1 are actually built and measured — this is a living plan, not a locked spec.

## Note on the existing AlphaZero/MCTS work

Prior to this plan, `open_spiel/python/algorithms/alpha_zero/*` was extended to correctly support N-player, general-sum training (Eclipse-ready, tested, working). That work is **not being deleted or reverted** — it's correct and complete, just not the currently-chosen training path, for the reasons in Context above (Eclipse's ~11K action space is a poor fit for vanilla MCTS regardless of GPU throughput).

It's being kept because: (1) Stage 4 above (wrapping a trained PPO policy with a light MCTS at play time) could directly reuse the AlphaZero evaluator/model/MCTS integration work; (2) if PPO self-play underperforms after real tuning, AlphaZero is a validated fallback.

**To avoid confusing future sessions about which path is active**: a follow-up (non-implementation) task should add a short pointer — e.g. a note at the top of `open_spiel/python/algorithms/alpha_zero/alpha_zero.py` or a small README in that directory — stating that PPO self-play (`open_spiel/python/pytorch/ppo.py`) is the active Eclipse training path, and linking back to this plan/decision. Not done in this session per instruction (docs-only, no code edits this session).

---

# Sprint A corrections (2026-08-04) — READ THIS BEFORE TRUSTING ANYTHING ABOVE

Several conclusions recorded above were measured through a saturated metric, on a
truncated action space, against a degenerate objective. They are corrected here
rather than edited in place, so the reasoning trail stays intact.

## What was actually wrong

1. **The objective's global optimum was mutual bankruptcy.** `rank_utility` gave
   tied seats the *best* placement slot, so `[0,0,0,0]` paid every seat +1.0 --
   the maximum. Since ~95% of unskilled games ended with all four players
   eliminated by bankruptcy in round 1-2, training began at the objective's
   maximum. The grid's "scoring decreases under training" (`phi=none` 5.3 -> 1.67,
   `phi=banked` 0.30 -> 0.05) was this, working as specified.
2. **Every strength metric shared the same tie rule.** `rank = 1 + sum(rewards >
   main_best)` scores a 0-0-0-0 game as rank 1 for main. Hence
   `squad/main_win_rate` = 0.97 at 1.7M steps and 0.95 at 408M, and four of six
   grid cells reporting an identical 8/8 against both fixed baselines. **The
   fixed-baseline columns in the Sprint-1 table carry no information.** Only
   `nonzero_episodes` and the own-squad column did.
3. **The async env silently dropped ~2/3 of every legal action set.**
   `_probe_max_legal` sized the shared buffer from *initial* states only (13-15)
   while real states reach 133, and `publish` truncated with `min(...)`. Measured:
   20% of decisions truncated, 68% of legal entries dropped, systematically the
   highest id blocks -- MOVE and UPGRADE. Every league and grid run used
   `--num_workers>0`, so **the agent could not move fleets or upgrade ships.**
   This also explains why the early 16-env *sync* pilots reached ~4.6 mean VP
   while the 128-env async runs sat at 0.33 with 4000x the compute.
4. **`--phi=learned` is not a potential function.** It differences the critic's
   value at the *next acting seat's* state against the mover's own -- two
   different players' values in a competitive game. The Sprint-1 verdict that
   selected it as default was produced by (2) on top of (3).
5. **The aux head was both dead and gradient-eating.** Its `/200` normalization
   was dropped, so the target was raw VP: ~0 (aux_loss 1e-5) in the 408M run, and
   8-11 against pg_loss ~1e-3 in the grid, where global grad-norm clipping then
   rescaled the policy gradient toward zero.
6. **Non-acting seats' GAE chains crossed episode boundaries.** Only the seat that
   made the final move got `done=1`; the other three bootstrapped into the next
   episode's values *and* were duplicated as extra samples with a contradictory
   target.
7. **The engine violated the rulebook on elimination.** Eliminated players scored
   0.0 although the rulebook says they count their score, making "died in round
   2" indistinguishable from "solvent through round 8".

## What changed (all committed, all tested)

Tie-averaged (fractional-rank) utility, exactly constant-sum, plus a clamped
`--rank_vp_beta` escape bonus that can only break ties; `max_legal` defaults to
the full action space and truncation now raises; `--phi=telescope` (new default)
differences a banked-VP potential across each seat's *own* consecutive decisions,
which is the only variant that telescopes against the per-own-decision gamma in
the self-play GAE; aux targets bounded and scale-guarded at startup, with a new
`losses/aux_share` diagnostic; terminal attribution rewritten (write-or-extra,
never both); eliminated players score their snapshot VP; and a batch of smaller
fixes (sync-path NaN from a leaked loop variable, `--exploit_lr` being reverted by
LR annealing, no checkpointing without `--league`, `--resume` losing optimizer
state, 182MB of unused mask buffer, `ACTION_MOVE_START` off by 640).

## Measured effect

Random play, 30 games, after the objective + engine fixes:

| | before | after |
|---|---|---|
| games with all four seats at 0 | 19/20 (95%) | 3/30 (10%) |
| episodes with no gradient at all | ~95% | 10% |
| within-game utility spread | 0 for 95% of games | 1.008 mean |
| sum of utilities per game | outcome-dependent (up to 4.0) | exactly 1.0 |

First short training run (128 envs / 16 workers, telescope phi, 0.8M steps) --
compare against the 408M-step run's terminal values:

| | 408M-step run | this run @0.8M |
|---|---|---|
| `nonzero_episodes` (of 200) | 23 | 197 |
| `mean_episode_return` (seat-0 VP) | 0.33 | 10.22 |
| `explained_variance` | 0.69 (on a constant) | 0.42 and rising |
| SPS | ~10k | 16-19k |

## Still open

- **`aux_coef` — investigated, left alone, tune in Sprint B.** `losses/aux_share`
  reads ~0.83, and an explicit gradient-norm measurement on rows carrying aux
  targets confirms the split is aux 82% / policy 11% / value 6%: the shared trunk
  is trained mostly as a rank predictor. Two corrections to the obvious reading,
  both measured:
  (a) the mechanism is *not* grad-norm clipping crowding out the policy gradient
  -- the combined norm (~0.34) stays under `--max_grad_norm` (0.5), so clipping
  never engages; it would be direct gradient dominance on the shared trunk.
  (b) an A/B at `--aux_coef=0.01` cut the aux share to 0.31 and changed nothing:
  `approx_kl` and `clipfrac` stayed at ~0, and VP at matched steps was slightly
  *worse* (11.97 vs 12.72 @1.65M). So aux dominance is not what is holding the
  policy back, and the default is unchanged.
- **Low `approx_kl` / `clipfrac` is probably not a pathology.** They sit near 0 in
  every configuration tried, but entropy falls 1.96 -> 0.78 and VP climbs
  1.75 -> 13.7 over the same window, so the policy is plainly learning; the trust
  region simply never binds. That points at the step size being conservative
  (a candidate for raising `--learning_rate`) rather than at broken updates.
  Contrast the 408M-step run, where entropy was *flat* at 0.53 -- that was the
  real no-learning signature, not the clipfrac value.
- **Sprint B partially done.** B2 (episode-health diagnostics) and B1 (batched
  evaluator) have landed; the re-adjudication of the six grid cells (B3) and Elo
  over the roster have **not**. The Sprint-1 ranking should not be reused until
  B3 runs.
  - Correction to the B1 estimate above: the old per-game evaluator was assumed
    to be ~100x slower than necessary. Measured, it is **2.3x** per game (20 ms
    vs 9 ms) -- Eclipse env stepping is cheap enough that the per-decision
    1-sample forward was not the dominant cost. The real win is that 64+ games
    per eval is now routine (0.6 s) instead of 8 being the practical ceiling, so
    the reported interval is narrow enough to separate configurations.
  - Sanity check that the metric now behaves: an *untrained* network scores
    utility +0.254 against a chance level of +0.250, CI [+0.172, +0.332] --
    correctly inconclusive. Under the old tie rule the same situation reported
    wins.
  - `mean_elim_round` is the metric to watch. Random play sits at 1.25/8 with a
    wipeout rate of 1.00; a 17k-step smoke already moves it to 1.95 and 0.94.
- Sprint C (capacity, factored action head, distributional rank critic,
  play-time search) is untouched.
