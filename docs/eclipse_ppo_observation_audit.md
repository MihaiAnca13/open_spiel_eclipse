# Eclipse PPO readiness: next work

This is the remaining work before an expensive Eclipse PPO self-play run can
produce credible evidence of a strong four-player policy. The target is one
policy that can later occupy three opponent seats. UI and application work are
out of scope.

## Do not start the long run yet

The present local training path cannot be trusted or launched as-is:

- The selected `pyspiel` extension lacks `PlayerId`, required by
  `rl_environment.py`, and reports an observation length of 37,788 while the
  current Python layout expects 37,804. Rebuild and validate one coherent
  native/Python artifact before any PPO test or training command.
- PPO auxiliary targets survive reuse of rollout rows. A new unfinished
  trajectory can train on final targets from the previous batch. Clear
  `aux_targets` and `aux_mask` together with the other per-batch labels in
  [`ppo.py`](../open_spiel/python/pytorch/ppo.py).
- Reward shaping has incompatible implementations. The synchronous and async
  paths scale the same VP potential differently; `telescope` produces zero;
  and cross-viewer potential reads treat a player's hidden reputation as a
  loss when another player becomes the viewer. Define one same-seat,
  privacy-safe shaping transition, including terminal and rollout-boundary
  behavior. Keep unshaped PPO as the control.
- The default rank critic is bounded to `[-0.5, 1]`, but the default terminal
  VP bonus can exceed `1`. The `cell_attn` alternative reports `(None, inf)`
  and currently crashes PPO's bounds diagnostic. Pick a working value head
  whose range matches every return that the selected reward design can emit.

## Game rules that must be fixed or explicitly scoped out

These are reachable mechanics or termination paths that change the game the
agent would learn:

- **Ancient Labs:** researching it must immediately draw and resolve a
  Discovery Tile. Its effect is absent from
  [`research.cpp`](../open_spiel/games/eclipse/systems/actions/research.cpp).
- **Elimination:** a player with neither ships nor controlled sectors at the
  end of combat must be eliminated. Only bankruptcy elimination is currently
  implemented in [`eclipse.cc`](../open_spiel/games/eclipse/eclipse.cc).
- **Soliton Missile:** the final ship part is excluded from
  `PlaceablePartIds`, so an owned Soliton Missile can never be installed.
  Fix the bounds in
  [`upgrade.cpp`](../open_spiel/games/eclipse/systems/actions/upgrade.cpp).
- **Diplomacy non-progress loop:** proposing and declining diplomacy does not
  advance the turn. Players can repeat it until the generic 1,000-move cap
  ends and scores an unfinished game. A safety cutoff must be a failure, not a
  scored terminal result; resolve the action progression so normal games end
  through round-eight cleanup.
- **Special ship parts:** Jump Drive and Morph Shield have no behavioral
  implementation beyond their table entries. Confirm their intended effects
  against the source rules, implement them, or exclude them from training
  configurations until they are correct.

Add compact regression tests for each item, plus randomized full-game tests
that assert every game reaches the normal round-eight ending below the safety
cap.

## Make credit assignment and optimization measurable

Before tuning model size, prove the learning data is correct.

1. Create deterministic, multi-batch trajectory tests with passing, early
   elimination, and terminal closeout. Assert which action of each seat gets
   the terminal target and that no stale action/log-probability is optimized.
2. Test each shaping mode against exact expected rewards in both synchronous
   and asynchronous collection. Assert identical results for identical game
   transitions.
3. Log the fraction of returns outside the critic range, terminal-target
   variance, all-tied outcome rate, eliminations, normal-round endings, and
   safety-cap endings. Abort a pilot on non-finite values, invalid legal-action
   sets, or any safety-cap terminal.
4. Decide the objective deliberately. The current rank-utility table rewards
   guaranteed second place more than a sufficiently risky win strategy. If
   first-place probability is the product objective, report it and select
   policies by it; retain tie-aware utility and VP as diagnostic measures.

## Make runs recoverable

- Always write final network weights, optimizer state, counters, and exact
  architecture metadata on ordinary completion as well as snapshot cadence.
- Restore optimizer state only when it belongs to the exact resumed run and
  checkpoint. An arbitrary `--resume` weight file must not silently receive
  Adam moments and schedules from `roster_dir/train_state.pt`.
- Have async workers report exceptions to the parent and make the parent fail
  with context rather than block indefinitely on a semaphore.
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
