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

~~Longer pilot to confirm these hold under real load~~ (256 envs, 50→100 updates,
checkpoint/resume): found and fixed a Diplomacy rearrange loop bug
(slot_is_returnable ignored already-empty slots, causing the proposer and
partner to ping-pong "returning nothing" forever under real PPO load until
the safety-cap move limit — invisible to uniform-random and one-action
regression tests). Pilot completed cleanly post-fix with `normal_end=1.00`
throughout; checkpoint/resume verified working correctly.

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

The FFA collector now writes this report directly: terminal VP, first-place
rate, reached round, eliminations, normal-ending rate, and tie-aware utility
are preserved in input order, with bootstrap samples formed from matched
setup/chance replicates across all four candidate-seat rotations. Homogeneous
opponent trios and genuinely mixed historical trios are reported separately.

The first full evidence artifact is remote-only at
`runs/pilot_2026-09-18-v2/ffa_product_32.{json,npz}`: all 8,192 scheduled
games (four policies, all profiles, 32 fixed setup/chance pairs) completed
normally. For `main` as the candidate, rotated one-seat utility was 0.912
[0.857, 0.957] vs `snap_u25` x3, 0.691 [0.623, 0.771] vs `snap_u50` x3, and
0.365 [0.315, 0.421] vs `snap_u75` x3; mixed historical trios were 0.644
[0.621, 0.668]. Each lower bound clears four-player chance utility (0.25).
This establishes the recovery pilot's internal progression, not a promotion
to a long training run: its opponents are its own earlier snapshots, and
independent training seeds remain required.

**2026-09-18 second independent seed:**
`runs/pilot_2026-09-18-seed2/phase1.log` completed the same 256-environment,
50-update pilot, then `phase2.log` resumed `main` with its optimizer/counters
from update 50 and completed at update 100. Both phases reported only normal
round-eight endings (`normal_end=1.00`), no universal-bankruptcy collapse, no
safety-cap ending, and finite losses/returns; `roster/snap_u100.pt` and
`train_state.pt` are present. The first cold compiled-encoder step took about
four and a half minutes and retained up to 88 GiB on GPU 2, but the process
completed and released the device.

**2026-09-18 three-seed one-vs-three result:** seeds 2 and 3 completed the same
two-phase 50→100 pilot and the same fixed 32-replicate FFA suite (8,192 profiles
each, all completed). All three seeds are healthy: no traceback, `normal_end=1.00`,
`safety_cap=0`, finite losses, and a full snapshot roster. Rotated one-seat utility
for `main`, with four-player chance utility at 0.25:

| `main` vs | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| 3x `snap_u25` | 0.912 [0.857, 0.957] | 0.834 [0.774, 0.883] | 0.939 [0.895, 0.975] |
| 3x `snap_u50` | 0.691 [0.623, 0.771] | 0.736 [0.680, 0.788] | 0.592 [0.501, 0.679] |
| 3x `snap_u75` | 0.365 [0.315, 0.421] | 0.367 [0.308, 0.433] | 0.309 [0.252, 0.354] |

Every lower bound clears 0.25, every snapshot loses to 3x `main`, and AlphaRank
puts essentially all mass on the all-`main` profile in each seed. The margin
shrinks monotonically as the opponent snapshot gets later, in all three seeds —
the signature of continuing improvement rather than a single lucky arm.

One consistent artifact: `snap_u75` against 3x `main` scores positive (0.191,
0.100, 0.189) but does not clear chance in any seed. This is the odd-one-out
advantage against three identical opponents, reproducible across seeds, and
reads as a property of the one-vs-three protocol rather than a defect.

This closes the observation-audit pilot gate. It does **not** close the separate
`docs/eclipse_rl_todo.md` blocker ("why does this stop learning"): these pilots
run to update 100, and the documented plateau/collapse is in the update
100→1700 range, which no pilot here exercises. See below.

**2026-09-19, engine and trainer fixes found by running at 1,024 envs.** Five
decision nodes could offer no legal action at all (each crashed or could crash a
run): diplomacy `choose_pop_track` (no Population Cube to give — `can_propose_
diplomacy` lacked the guard `can_form_minor_species` already had),
`choose_rearrange` (now resolved automatically, preferring a lossless swap and
giving up the cheapest tile only when forced), `choose_accept` (now always
offers DECLINE), move `choose_warp_destination` (now offers MOVE_STOP like
`choose_move`), and upkeep `choose_return_track` (unreachable by invariant, so
it got an assert rather than a fallback in `d3a28d25` — every escape hatch
would have lost a Population Cube that has already left the board). A kept Reputation tile now fills a free slot
and the tile it displaces goes back to the bag rather than out of the game,
which matters because `reputation_draw` is a chance node whose probabilities
come from the bag. Separately, a move-capped game no longer pays
(`--stalled_game_penalty`; flattening payoffs instead would have *rewarded* a
losing seat, since an all-tie scores the mean of the rank table), stalls are
counted rather than instantly fatal (`--safety_cap_abort_rate`), and a looping
game is now captured at 900 moves while it still exists. The largest find is the
free MOVE action — see `eclipse_rl_todo.md`, "a likely cause of stops learning
after update 100".

**2026-09-20/27 — the late regime is measured: still learning at u763.**
The "one medium diagnostic run carried past update 100" was run on the post-free-MOVE
engine (`0c32d7c7`) as three arms: `runs/main_v3` (control, to u763), `runs/lr_anneal`
(`--lr_schedule=anneal`, u650) and `runs/league_refresh` (`--live_opponent_refresh=250`,
u625). About 18h, no crashes, `normal_end=1.00`, `safety_cap=0` throughout.

A 256-replicate FFA along one lineage (65,536 games) finds each later snapshot
CI-clear above each earlier one, with a strictly transitive order. u763 over u625 is
+0.033 [+0.017, +0.050] on the symmetric one-vs-three edge. The full table is in
`eclipse_rl_todo.md`. Conclusions:

1. **The late-regime blocker is closed.** On the fixed engine this stack keeps
   improving past u100 through u763, which the T1 runs never did. The free-MOVE fix
   is the likeliest cause of the difference.
2. **The 32-replicate FFA was underpowered, not flat.** One chunk's gain (~+0.03)
   sits inside its ±0.06 CIs. The same limit applies to its reading that LR anneal
   and league refresh made no difference: that is only established to within ~±0.06.
3. **No intransitivity was found.** It isn't in the lineage, and the three u625s
   are too alike to show any. There is no case yet for PSRO over plain self-play.
4. **Entropy collapse is moot for now.** The run isn't stalled, and measured entropy
   sat at 0.79-0.86.

**Chunk gate.** It used to compare newest vs oldest, and `prune_roster.py` pins
`snap_u25`, so IMPROVING only meant "beats a 25-update policy". At 24 games/pair
it also lacked the resolution to see one chunk's gain: re-scored newest-vs-previous,
11 of the 12 gates read FLAT. Fixed 2026-09-27. `run_long.sh` pins the previous
gate's `main` (`prune_roster.py --pin_birth`), `gate_report.py --baseline_age` makes
RECENT newest-vs-one-chunk-ago the headline, and `GATE_GAMES` defaults to 96.

**Next.** The stack is adequate, so the lever is throughput and hours. Measure the
unmeasured throughput patches `cf6fa5d9` and `527ed459` on GPU, then resume
`main_v3` from u763 under the fixed gate. Pick a peak with a powered lineage FFA,
not the gate. For sizing, 256 replicates of a 4-policy suite took 2h07m on GPU 3.

**Where the evidence lives** (all remote on `behemoth`, under
`~/mihai/open_spiel_eclipse`; `runs/` is gitignored, so none of this is in git):

- `runs/{main_v3,lr_anneal,league_refresh}/gates.log` — per-chunk gate verdicts
  and rating tables, one block per chunk; the quickest way to re-read the runs.
- `runs/{...}/train.log` — full trainer output, including `[timing uN]` lines and
  the `resumed optimizer + counters: ... updates=N` line that gives the true
  global age (the `[update N]` counter in this log is chunk-local, not the age).
- `runs/{...}/gate_N.json` — machine-readable ladder results per gate.
- `runs/{...}/snap_u*.pt` — full snapshot history at `--snapshot_every=25`,
  nothing pruned (34 snapshots for `main_v3`).
- `runs/crossarm/lineage_256.{json,npz}` — the powered `main_v3` lineage FFA
  (u375/u525/u625/u763, 256 replicates); the `.json` holds `one_vs_three` and
  `alpharank`, the `.npz` the raw per-profile utilities.
- `runs/crossarm/ffa_u625_32.{json,npz}` — the underpowered 32-replicate cross-arm
  FFA (three arms at u625 plus `ctrl763`).
- `runs/crossarm/roster.json` — built by `tools/stage_crossarm_roster.py`, which
  indexes policies across separate run dirs without copying weights (the FFA
  collector takes four ids from ONE roster dir, so cross-arm needs this).

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
  documented baseline and older snapshots. **As of 2026-09-20 this last
  criterion is known to FAIL beyond ~u625**: no arm beats an older snapshot of
  its own run on the FFA suite (see the 2026-09-20 result above). The pilot gate
  passes at u100; the long run does not hold it at u625.

Do not add a larger network, recurrence, search, or a more elaborate league
until this pilot identifies a bottleneck they address. The existing structured
encoder, candidate action head, roster, matcher, ladder, and FFA evaluator are
enough to obtain a trustworthy baseline once the correctness work above is
complete.
