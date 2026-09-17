# Eclipse PPO readiness: next work

This is the remaining work before an expensive Eclipse PPO self-play run can
produce credible evidence of a strong four-player policy. The target is one
policy that can later occupy three opponent seats. UI and application work are
out of scope.

## Do not start the long run yet

Resolved this pass, verified with a real (tiny) end-to-end training run —
rollout collection, `learn()`, and a second update complete with finite
losses under the default `--factored_actions --encoder=spatial
--critic_readout=cell_attn` configuration, which previously could not get
past startup:

- ~~The selected `pyspiel` extension lacks `PlayerId`~~. The `build/` tree now
  builds `PlayerId` and reports the expected 37,804-length observation.
- ~~PPO auxiliary targets survive reuse of rollout rows~~. `clear_batch` now
  zeroes `aux_targets`/`aux_mask` alongside the other per-batch labels.
- ~~Reward shaping has incompatible implementations~~. Collapsed to one
  same-seat, privacy-safe potential (telescoped banked VP across each seat's
  own consecutive decisions); `--phi=none` is the unshaped control.
- ~~The default rank critic is bounded to `[-0.5, 1]`... `cell_attn` reports
  `(None, inf)`~~. `cell_attn` is now the default readout and reports
  `(-inf, inf)` (an intentionally unbounded linear head).

Also found and fixed by actually running a full step+learn cycle for the
first time (previously unreachable behind an action-factorization crash,
below): `action_factors.py`'s part-name lookup didn't match the engine's
action-string format; `CellAttentionCritic.key_norm` normalized over the
wrong dimension, crashing any `--nn_width != 64`; `PPO._sparse_minibatch`
assumed plain-tensor `features` when the candidate actor head passes a
structured context; `head_logits()` silently returned a non-float32 dtype
under autocast that broke the log-prob/entropy reductions built on top of it.

Not yet re-run: a longer pilot to confirm these hold under real load (many
envs, many updates, checkpoint/resume). Do that before anything longer.

## Game rules that must be fixed or explicitly scoped out

All five resolved this pass, each with a regression test exercised through
the public State/Action API where practical:

- ~~**Ancient Labs**~~: researching it now draws and resolves one Discovery
  Tile (rulebook p.10), reusing `apply_discovery_reward`. Tiles that read
  "place ... in the Sector where found" (Monolith/Orbital/Cruiser/Warp
  Portal) resolve against the first Sector the player Controls, falling back
  to the tile's VP value if they control none.
- ~~**Elimination**~~: a player left with no Ships and no Sectors under their
  Control at the end of the Combat Phase is now eliminated too, independent
  of the Upkeep bankruptcy path.
- ~~**Soliton Missile**~~: `PlaceablePartIds`' loop bound was off by one,
  silently excluding the last entry of `SHIP_PART_TABLE`.
- ~~**Diplomacy non-progress loop**~~: a decline changed nothing else, so the
  proposer could re-propose to the same partner forever without ever taking
  a main Action. A decline now blocks re-proposing that exact pair until the
  turn genuinely ends, bounding the cycle to at most `player_count^2`
  declines.
- ~~**Special ship parts**~~: Jump Drive and Morph Shield had no behavioral
  implementation beyond their table entries. Confirmed against the source
  (not the rulebook PDF, which doesn't cover Discovery ship parts) and
  implemented: Jump Drive lets a ship enter an adjacent Sector on every Move
  regardless of Wormhole Connections, still subject to every other Move rule
  (pinning, an actual Sector must be there); Morph Shield heals 1 damage,
  unconditionally, on every still-alive ship in the current engagement after
  each Combat Round resolves.

~~Add randomized full-game tests that assert every game reaches the normal
round-eight ending below the safety cap.~~ Thirty deterministic uniform-random
four-player playouts verify that even early universal bankruptcy continues to
round 9 without hitting `MaxGameLength()`. Twelve additional seeded playouts
limit each surviving player to one random main Action per round, randomize all
sub-decisions, and require round-eight completion with survivors after at least
32 main Actions. The complete Eclipse engine regression suite passes.

## Make credit assignment and optimization measurable

Before tuning model size, prove the learning data is correct.

1. ~~Create deterministic, multi-batch trajectory tests with passing, early
   elimination, and terminal closeout. Assert which action of each seat gets
   the terminal target and that no stale action/log-probability is optimized.~~
   Cross-batch terminal extras now pin the original per-seat observation,
   legal set, action, and log-probability.
2. ~~Test each shaping mode against exact expected rewards in both synchronous
   and asynchronous collection. Assert identical results for identical game
   transitions.~~ The unshaped and same-seat telescope modes now share a
   scripted exact-reward parity test.
3. ~~Log the fraction of returns outside the critic range, terminal-target
   variance, all-tied outcome rate, eliminations, normal-round endings, and
   safety-cap endings. Abort a pilot on non-finite values, invalid legal-action
   sets, or any safety-cap terminal.~~ Terminal current-round capture now
   distinguishes normal round-eight completion from the move-count backstop;
   malformed legal sets and non-finite training data fail immediately.
4. ~~Decide the objective deliberately.~~ PPO keeps the tie-aware rank-utility
   target, while first-place rate remains the checkpoint/evaluation headline;
   tie-aware utility and VP remain diagnostics.

## Make runs recoverable

- Always write final network weights, optimizer state, counters, and exact
  architecture metadata on ordinary completion as well as snapshot cadence.
- Restore optimizer state only when it belongs to the exact resumed run and
  checkpoint. An arbitrary `--resume` weight file must not silently receive
  Adam moments and schedules from `roster_dir/train_state.pt`.
- ~~Have async workers report exceptions to the parent and make the parent fail
  with context rather than block indefinitely on a semaphore.~~ Worker failures
  now carry their traceback to the parent; a learner exception also closes the
  worker pool before it is re-raised.
- Evict opponent modules that no longer appear in current league lineups;
  roster pruning alone does not reclaim already loaded networks.

## Evaluate the product scenario

The recurring verdict must evaluate one candidate seat against three opponents,
not the current two-candidate-seat versus two-bot arrangement. Every scheduled
held-out scenario must complete; do not take merely the first environments to
finish.

For each candidate:

- Rotate the candidate through all four seats.
- Use one policy on the remaining three seats, then mixed historical roster
  lineups to test exploitability and non-transitivity.
- Fix and record setup and chance seeds. Report first-place rate, tie-aware
  per-seat utility, VP, game length, eliminations, and bootstrap confidence
  intervals.
- Require the lower confidence bound of one-seat utility to exceed four-player
  chance utility (`0.25`) against several held-out opponents.

Use the existing
[`ffa_metagame.py`](../open_spiel/python/eclipse/ffa_metagame.py) machinery for
reproducible full four-policy profiles and AlphaRank. The existing 2v2 ladder is
useful population telemetry, but cannot establish the one-versus-three claim.

## Pilot acceptance gate

Run several independent short seeds only after the preceding blockers are
closed. Promote to a long run only when every seed shows:

- the native build, PPO unit tests, engine regression tests, and one async
  multi-worker update all pass;
- finite losses and returns, correct checkpoint/resume behavior, and no worker
  hang;
- no safety-cap termination, with healthy round completion and no unexplained
  collapse into universal bankruptcy;
- stable improvement on the held-out one-seat-versus-three suite over a
  documented baseline and older snapshots.

Do not add a larger network, recurrence, search, or a more elaborate league
until this pilot identifies a bottleneck they address. The existing structured
encoder, candidate action head, roster, matcher, ladder, and FFA evaluator are
enough to obtain a trustworthy baseline once the correctness work above is
complete.
